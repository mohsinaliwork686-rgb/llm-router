import asyncio
from datetime import UTC, datetime

import pytest

from llm_router.core.budget import BudgetGuard
from llm_router.core.chat import ChatMessage, ChatRequest, Completion, Usage
from llm_router.core.config import parse_config
from llm_router.core.pricing import worst_case_micros
from llm_router.core.providers import Failure, ProviderError
from llm_router.core.router import AllFailed, OverBudget, RequestRejected, Router
from llm_router.core.spend import MemorySpendStore
from llm_router.core.tokens import estimate_messages

pytestmark = pytest.mark.anyio

REQUEST = ChatRequest(messages=(ChatMessage("user", "Summarise the meeting notes."),))
PROMPT_TOKENS = estimate_messages(REQUEST.contents())


def done(text="ok", input_tokens=10, output_tokens=5):
    return Completion(text, "stop", Usage(input_tokens, output_tokens), model="served")


def fail(kind, status=None, retry_after=None, billed=False):
    return ProviderError(
        kind, f"failed {status}", status=status, retry_after=retry_after, may_have_billed=billed
    )


class Scripted:
    """A provider that plays back a list of results: a Completion, an exception, or a coroutine
    function to await (for slow or hanging calls)."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    async def complete(self, model, request, max_tokens):
        self.calls.append((model.name, max_tokens))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return await step()
        return step

    async def close(self):
        pass


def make_router(large, small, *, budget=None, retry=None, policies=None, clients=None, rules=None):
    data = {
        "providers": {"p_large": {"type": "simulated"}, "p_small": {"type": "simulated"}},
        "models": {
            "large": {
                "provider": "p_large",
                "model": "large-1",
                "input_per_million": 4,
                "output_per_million": 20,
                "max_output_tokens": 200,
            },
            "small": {
                "provider": "p_small",
                "model": "small-1",
                "input_per_million": 1,
                "output_per_million": 5,
                "max_output_tokens": 100,
            },
        },
        "policies": policies or {"default": ["large", "small"]},
        "default_policy": "default",
        "budget": budget or {"daily_usd": 10},
        "retry": {
            "attempts_per_model": 2,
            "backoff_ms": 100,
            "timeout_seconds": 1,
            **(retry or {}),
        },
        "clients": clients or [],
        "rules": rules or [],
    }
    config = parse_config(data)
    store = MemorySpendStore()
    clock = lambda: datetime(2026, 10, 14, 12, tzinfo=UTC)  # noqa: E731
    guard = BudgetGuard(config.budget, config.clients, store, clock=clock)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    router = Router(
        config,
        {"p_large": large, "p_small": small},
        guard,
        sleep=fake_sleep,
        jitter=lambda: 0.5,
    )
    return router, guard, sleeps, config


async def spent(guard):
    status = await guard.status()
    return status["global"]["spent_usd"], status["global"]["reserved_usd"]


async def test_first_model_answers():
    large, small = Scripted(done(input_tokens=100, output_tokens=50)), Scripted()
    router, guard, sleeps, _ = make_router(large, small)

    result = await router.complete("default", REQUEST)

    assert result.model.name == "large"
    assert result.route.policy == "default"
    assert result.cost_micros == 100 * 4 + 50 * 20
    assert [a.outcome for a in result.attempts] == ["ok"]
    assert small.calls == []
    assert sleeps == []
    assert await spent(guard) == ("0.001400", "0.000000")


async def test_429_then_success_retries_the_same_model_after_a_backoff():
    large = Scripted(fail(Failure.RETRY, 429), done())
    router, _, sleeps, _ = make_router(large, Scripted())

    result = await router.complete("default", REQUEST)

    assert result.model.name == "large"
    assert [a.outcome for a in result.attempts] == ["failed", "ok"]
    # backoff_ms 100 on the first retry, with jitter 0.5: halfway between 0.05 and 0.1.
    assert sleeps == [pytest.approx(0.075)]


async def test_500_on_every_attempt_moves_to_the_next_model():
    large = Scripted(fail(Failure.RETRY, 500), fail(Failure.RETRY, 500))
    small = Scripted(done())
    router, _, sleeps, _ = make_router(large, small)

    result = await router.complete("default", REQUEST)

    assert result.model.name == "small"
    assert [(a.model, a.outcome) for a in result.attempts] == [
        ("large", "failed"),
        ("large", "failed"),
        ("small", "ok"),
    ]
    assert len(sleeps) == 1  # between the two large attempts, not before moving on


async def test_backoff_grows_exponentially():
    large = Scripted(*[fail(Failure.RETRY, 503)] * 3, done())
    router, _, sleeps, _ = make_router(large, Scripted(), retry={"attempts_per_model": 4})
    await router.complete("default", REQUEST)
    assert sleeps == [pytest.approx(0.075), pytest.approx(0.15), pytest.approx(0.3)]


async def test_timeout_then_success_and_the_timed_out_call_counts_as_spent():
    async def hang():
        await asyncio.Event().wait()

    large = Scripted(hang, done(input_tokens=10, output_tokens=5))
    router, guard, _, config = make_router(large, Scripted(), retry={"timeout_seconds": 0.05})

    result = await router.complete("default", REQUEST)

    assert [a.outcome for a in result.attempts] == ["failed", "ok"]
    assert result.attempts[0].detail == "no answer within 0.05 seconds"
    worst = worst_case_micros(config.models["large"], PROMPT_TOKENS, 200)
    assert result.attempts[0].cost_micros == worst
    status = await guard.status()
    assert status["global"]["spent_usd"] == f"{(worst + 10 * 4 + 5 * 20) / 1e6:.6f}"


async def test_400_is_returned_at_once():
    large = Scripted(fail(Failure.REJECT, 400))
    small = Scripted(done())
    router, guard, _, _ = make_router(large, small)

    with pytest.raises(RequestRejected) as caught:
        await router.complete("default", REQUEST)

    assert caught.value.status == 400
    assert small.calls == []
    assert len(large.calls) == 1
    assert await spent(guard) == ("0.000000", "0.000000")


async def test_401_skips_the_model_without_retrying():
    large = Scripted(fail(Failure.SKIP, 401))
    small = Scripted(done())
    router, _, sleeps, _ = make_router(large, small)

    result = await router.complete("default", REQUEST)

    assert len(large.calls) == 1
    assert result.model.name == "small"
    assert sleeps == []


async def test_every_candidate_failing_gives_one_clean_error():
    large = Scripted(fail(Failure.SKIP, 401))
    small = Scripted(fail(Failure.RETRY, 503), fail(Failure.RETRY, 503))
    router, guard, _, _ = make_router(large, small)

    with pytest.raises(AllFailed) as caught:
        await router.complete("default", REQUEST)

    assert [(a.model, a.status) for a in caught.value.attempts] == [
        ("large", 401),
        ("small", 503),
        ("small", 503),
    ]
    assert str(caught.value).startswith("every model failed (large: failed 401; small:")
    # Nothing was billed, and every reservation was given back.
    assert await spent(guard) == ("0.000000", "0.000000")


async def test_a_short_retry_after_is_respected():
    large = Scripted(fail(Failure.RETRY, 429, retry_after=2.0), done())
    router, _, sleeps, _ = make_router(large, Scripted())
    await router.complete("default", REQUEST)
    assert sleeps == [2.0]


async def test_a_long_retry_after_moves_on_instead_of_waiting():
    large = Scripted(fail(Failure.RETRY, 429, retry_after=120.0))
    small = Scripted(done())
    router, _, sleeps, _ = make_router(large, small)

    result = await router.complete("default", REQUEST)

    assert result.model.name == "small"
    assert sleeps == []


async def test_a_model_over_the_cap_falls_through_to_a_cheaper_one():
    # A $0.001 cap: the large model's worst case (about $0.004) cannot fit, the small one can.
    large, small = Scripted(), Scripted(done(input_tokens=10, output_tokens=5))
    router, _, _, _ = make_router(large, small, budget={"daily_usd": "0.001"})

    result = await router.complete("default", REQUEST)

    assert large.calls == []
    assert result.model.name == "small"
    assert [a.outcome for a in result.attempts] == ["over_budget", "ok"]


async def test_every_model_over_the_cap_is_over_budget():
    large, small = Scripted(), Scripted()
    router, _, _, _ = make_router(large, small, budget={"daily_usd": "0.0001"})

    with pytest.raises(OverBudget) as caught:
        await router.complete("default", REQUEST)

    assert large.calls == [] and small.calls == []
    assert caught.value.exceeded.scope == "global"
    assert [a.outcome for a in caught.value.attempts] == ["over_budget", "over_budget"]


async def test_a_named_model_gets_no_fallback():
    large = Scripted(fail(Failure.RETRY, 500), fail(Failure.RETRY, 500))
    small = Scripted(done())
    router, _, _, _ = make_router(large, small)

    with pytest.raises(AllFailed):
        await router.complete("large", REQUEST)
    assert small.calls == []


async def test_max_tokens_is_capped_at_the_model_limit():
    large = Scripted(fail(Failure.SKIP, 401))
    small = Scripted(done())
    router, _, _, _ = make_router(large, small)
    request = ChatRequest(messages=REQUEST.messages, max_tokens=150)

    await router.complete("default", request)

    assert large.calls == [("large", 150)]
    assert small.calls == [("small", 100)]


async def test_client_caps_apply():
    router, _, _, _ = make_router(
        Scripted(),
        Scripted(),
        clients=[{"name": "app", "key_env": "APP_KEY", "daily_usd": "0.0001"}],
    )
    with pytest.raises(OverBudget) as caught:
        await router.complete("default", REQUEST, client="app")
    assert caught.value.exceeded.scope == "client 'app'"


async def test_warn_mode_passes_the_warning_on():
    small = Scripted(done())
    router, _, _, _ = make_router(
        Scripted(), small, budget={"daily_usd": "0.0001", "on_exceeded": "warn"}
    )
    result = await router.complete("small", REQUEST)
    assert result.budget_warning is not None


async def test_an_unexpected_adapter_error_skips_to_the_next_model():
    large = Scripted(RuntimeError("bug in the adapter"))
    small = Scripted(done())
    router, _, _, _ = make_router(large, small)

    result = await router.complete("default", REQUEST)

    assert result.model.name == "small"
    assert result.attempts[0].detail == "unexpected error: RuntimeError"


async def test_a_cancelled_call_counts_its_reservation_as_spent():
    started = asyncio.Event()

    async def hang():
        started.set()
        await asyncio.Event().wait()

    router, guard, _, config = make_router(Scripted(hang), Scripted())
    task = asyncio.create_task(router.complete("default", REQUEST))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    worst = worst_case_micros(config.models["large"], PROMPT_TOKENS, 200)
    assert await spent(guard) == (f"{worst / 1e6:.6f}", "0.000000")


async def test_auto_routes_by_rules():
    large, small = Scripted(), Scripted(done())
    router, _, _, _ = make_router(
        large,
        small,
        policies={"default": ["large", "small"], "cheap": ["small"]},
        rules=[{"when": {"tag": "batch"}, "use": "cheap"}],
    )
    result = await router.complete("auto", REQUEST, tags={"batch"})
    assert result.route.policy == "cheap"
    assert large.calls == []
