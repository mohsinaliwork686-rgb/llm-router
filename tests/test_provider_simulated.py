import pytest

from llm_router.core.chat import ChatMessage, ChatRequest
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers import Failure, ProviderError
from llm_router.core.providers.simulated import SimulatedProvider

pytestmark = pytest.mark.anyio

MODEL = ModelConfig(
    name="sim",
    provider="sim",
    model="sim-1",
    input_per_million=1,
    output_per_million=5,
    max_output_tokens=1000,
)
REQUEST = ChatRequest(messages=(ChatMessage("user", "Hello there, router."),))


async def test_answers_with_the_configured_reply_and_reports_usage():
    provider = SimulatedProvider(ProviderConfig(type="simulated", reply="Hi from the simulator."))
    completion = await provider.complete(MODEL, REQUEST, max_tokens=100)
    assert completion.text == "Hi from the simulator."
    assert completion.finish_reason == "stop"
    assert completion.model == "sim-1"
    # 20 characters at 4 per token, plus 4 for the message and 3 for the prompt.
    assert completion.usage.input_tokens == 5 + 4 + 3
    assert completion.usage.output_tokens == 6


async def test_truncates_to_max_tokens():
    provider = SimulatedProvider(ProviderConfig(type="simulated", reply="x" * 100))
    completion = await provider.complete(MODEL, REQUEST, max_tokens=5)
    assert completion.text == "x" * 20
    assert completion.finish_reason == "length"
    assert completion.usage.output_tokens == 5


async def test_fails_the_first_n_calls_then_succeeds():
    provider = SimulatedProvider(ProviderConfig(type="simulated", fail_with=503, fail_times=2))
    for _ in range(2):
        with pytest.raises(ProviderError) as caught:
            await provider.complete(MODEL, REQUEST, max_tokens=10)
        assert caught.value.kind is Failure.RETRY
        assert caught.value.status == 503
    await provider.complete(MODEL, REQUEST, max_tokens=10)


async def test_fail_times_zero_fails_every_call():
    provider = SimulatedProvider(ProviderConfig(type="simulated", fail_with=401))
    for _ in range(5):
        with pytest.raises(ProviderError) as caught:
            await provider.complete(MODEL, REQUEST, max_tokens=10)
        assert caught.value.kind is Failure.SKIP


async def test_latency_waits_through_the_injected_sleep():
    waited = []

    async def fake_sleep(seconds):
        waited.append(seconds)

    provider = SimulatedProvider(ProviderConfig(type="simulated", latency_ms=250), sleep=fake_sleep)
    await provider.complete(MODEL, REQUEST, max_tokens=10)
    assert waited == [0.25]
