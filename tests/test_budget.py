import asyncio
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from llm_router.core.budget import BudgetExceeded, BudgetGuard
from llm_router.core.config import BudgetConfig, ClientConfig
from llm_router.core.spend import MemorySpendStore, Totals

pytestmark = pytest.mark.anyio

DOLLAR = 1_000_000  # micro-dollars


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def make_guard(
    daily_usd="10",
    clients=(),
    on_exceeded="reject",
    timezone="UTC",
    clock=None,
    store=None,
):
    budget = BudgetConfig(
        daily_usd=None if daily_usd is None else Decimal(daily_usd),
        on_exceeded=on_exceeded,
        timezone=timezone,
    )
    client_configs = [
        ClientConfig(
            name=name,
            key_env=f"KEY_{name.upper()}",
            daily_usd=None if cap is None else Decimal(cap),
        )
        for name, cap in clients
    ]
    clock = clock or FakeClock(datetime(2026, 10, 14, 12, 0, tzinfo=UTC))
    store = store or MemorySpendStore()
    return BudgetGuard(budget, client_configs, store, clock=clock), store


async def test_under_the_cap_is_reserved():
    guard, _ = make_guard()
    reservation = await guard.reserve(3 * DOLLAR)
    assert reservation.amount == 3 * DOLLAR
    assert reservation.day == "2026-10-14"
    assert reservation.warning is None


async def test_exactly_at_the_cap_is_allowed():
    guard, _ = make_guard()
    await guard.reserve(10 * DOLLAR)


async def test_over_the_cap_is_refused_with_the_reason():
    guard, _ = make_guard()
    held = await guard.reserve(6 * DOLLAR)
    await guard.settle(held, 4 * DOLLAR)
    await guard.reserve(3 * DOLLAR)
    with pytest.raises(BudgetExceeded) as caught:
        await guard.reserve(3 * DOLLAR + 1)
    assert caught.value.scope == "global"
    assert str(caught.value) == (
        "global daily cap of $10.000000 would be exceeded: $4.000000 spent, "
        "$3.000000 reserved by calls in flight, and this request could cost up to $3.000001"
    )


async def test_a_request_bigger_than_the_whole_cap_says_so():
    guard, _ = make_guard(daily_usd="1")
    with pytest.raises(BudgetExceeded, match="could cost up to \\$2.000000 on its own"):
        await guard.reserve(2 * DOLLAR)


async def test_refusal_holds_nothing():
    guard, store = make_guard()
    with pytest.raises(BudgetExceeded):
        await guard.reserve(11 * DOLLAR)
    status = await guard.status()
    assert status["global"]["reserved_usd"] == "0.000000"


async def test_settling_for_less_frees_the_difference():
    guard, _ = make_guard()
    reservation = await guard.reserve(10 * DOLLAR)
    assert await guard.settle(reservation, 2 * DOLLAR) == 0
    await guard.reserve(8 * DOLLAR)
    with pytest.raises(BudgetExceeded):
        await guard.reserve(1)


async def test_a_failed_call_releases_its_reservation():
    guard, _ = make_guard()
    reservation = await guard.reserve(10 * DOLLAR)
    await guard.release(reservation)
    await guard.reserve(10 * DOLLAR)


async def test_running_over_the_reservation_is_recorded_and_reported():
    guard, _ = make_guard()
    reservation = await guard.reserve(1 * DOLLAR)
    shortfall = await guard.settle(reservation, 1 * DOLLAR + 250)
    assert shortfall == 250
    status = await guard.status()
    assert status["global"]["spent_usd"] == "1.000250"


async def test_a_reservation_finishes_once():
    guard, _ = make_guard()
    reservation = await guard.reserve(DOLLAR)
    await guard.settle(reservation, DOLLAR)
    with pytest.raises(RuntimeError):
        await guard.settle(reservation, DOLLAR)
    with pytest.raises(RuntimeError):
        await guard.release(reservation)


async def test_client_cap_inside_the_global_one():
    guard, _ = make_guard(daily_usd="10", clients=[("billing", "2"), ("support", None)])
    await guard.reserve(2 * DOLLAR, client="billing")
    with pytest.raises(BudgetExceeded) as caught:
        await guard.reserve(1, client="billing")
    assert caught.value.scope == "client 'billing'"
    # Billing being at its own cap does not stop another client.
    await guard.reserve(5 * DOLLAR, client="support")


async def test_global_cap_still_binds_a_client_under_its_own_cap():
    guard, _ = make_guard(daily_usd="10", clients=[("billing", "5"), ("support", None)])
    await guard.reserve(8 * DOLLAR, client="support")
    with pytest.raises(BudgetExceeded) as caught:
        await guard.reserve(3 * DOLLAR, client="billing")
    assert caught.value.scope == "global"


async def test_the_narrower_cap_is_reported_when_both_are_hit():
    guard, _ = make_guard(daily_usd="10", clients=[("billing", "5")])
    with pytest.raises(BudgetExceeded) as caught:
        await guard.reserve(20 * DOLLAR, client="billing")
    assert caught.value.scope == "client 'billing'"


async def test_unknown_client_is_an_error():
    guard, _ = make_guard(clients=[("billing", "5")])
    with pytest.raises(KeyError):
        await guard.reserve(1, client="stranger")


async def test_no_cap_tracks_spend_without_limiting_it():
    guard, _ = make_guard(daily_usd=None)
    reservation = await guard.reserve(1_000 * DOLLAR)
    await guard.settle(reservation, 1_000 * DOLLAR)
    status = await guard.status()
    assert status["global"] == {
        "limit_usd": None,
        "spent_usd": "1000.000000",
        "reserved_usd": "0.000000",
        "remaining_usd": None,
    }


async def test_warn_mode_lets_the_request_through_and_flags_it():
    guard, _ = make_guard(daily_usd="1", on_exceeded="warn")
    first = await guard.reserve(DOLLAR)
    assert first.warning is None
    second = await guard.reserve(DOLLAR)
    assert second.warning is not None
    assert second.warning.startswith("global daily cap of $1.000000 would be exceeded")
    status = await guard.status()
    assert status["global"]["reserved_usd"] == "2.000000"
    assert status["global"]["remaining_usd"] == "0.000000"


async def test_free_calls_pass_even_with_the_cap_used_up():
    guard, _ = make_guard(daily_usd="1")
    reservation = await guard.reserve(DOLLAR)
    await guard.settle(reservation, DOLLAR)
    await guard.reserve(0)


async def test_the_day_rolls_over_at_midnight_in_the_configured_timezone():
    # Karachi is UTC+5 all year, so its midnight is 19:00 UTC.
    clock = FakeClock(datetime(2026, 10, 14, 18, 59, 59, tzinfo=UTC))
    guard, _ = make_guard(daily_usd="1", timezone="Asia/Karachi", clock=clock)
    assert guard.today() == "2026-10-14"
    reservation = await guard.reserve(DOLLAR)
    await guard.settle(reservation, DOLLAR)
    with pytest.raises(BudgetExceeded):
        await guard.reserve(1)

    clock.advance(seconds=1)
    assert guard.today() == "2026-10-15"
    await guard.reserve(DOLLAR)


async def test_the_day_follows_daylight_saving_changes():
    # New York leaves daylight saving on 1 November 2026: midnight is 04:00 UTC before
    # the change and 05:00 UTC after it.
    clock = FakeClock(datetime(2026, 11, 1, 3, 59, tzinfo=UTC))
    guard, _ = make_guard(timezone="America/New_York", clock=clock)
    assert guard.today() == "2026-10-31"
    clock.advance(minutes=1)
    assert guard.today() == "2026-11-01"
    clock.now = datetime(2026, 11, 2, 4, 59, tzinfo=UTC)
    assert guard.today() == "2026-11-01"
    clock.advance(minutes=1)
    assert guard.today() == "2026-11-02"


async def test_a_call_spanning_midnight_settles_on_the_day_it_reserved():
    clock = FakeClock(datetime(2026, 10, 14, 23, 59, 59, tzinfo=UTC))
    guard, store = make_guard(clock=clock)
    reservation = await guard.reserve(DOLLAR)
    clock.advance(seconds=5)
    await guard.settle(reservation, DOLLAR)
    totals = await store.totals(["2026-10-14/global", "2026-10-15/global"])
    assert totals == {"2026-10-14/global": Totals(DOLLAR, 0), "2026-10-15/global": Totals(0, 0)}


async def test_naive_clock_is_refused():
    guard, _ = make_guard(clock=lambda: datetime(2026, 10, 14, 12, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        await guard.reserve(1)


async def test_status_reports_every_cap():
    guard, _ = make_guard(daily_usd="10", clients=[("billing", "2"), ("support", None)])
    done = await guard.reserve(DOLLAR, client="billing")
    await guard.settle(done, 400_000)
    await guard.reserve(DOLLAR // 2, client="support")

    assert await guard.status() == {
        "day": "2026-10-14",
        "timezone": "UTC",
        "on_exceeded": "reject",
        "global": {
            "limit_usd": "10.000000",
            "spent_usd": "0.400000",
            "reserved_usd": "0.500000",
            "remaining_usd": "9.100000",
        },
        "clients": {
            "billing": {
                "limit_usd": "2.000000",
                "spent_usd": "0.400000",
                "reserved_usd": "0.000000",
                "remaining_usd": "1.600000",
            },
            "support": {
                "limit_usd": None,
                "spent_usd": "0.000000",
                "reserved_usd": "0.500000",
                "remaining_usd": None,
            },
        },
    }


async def test_concurrent_calls_cannot_jointly_exceed_the_cap():
    """Many simulated calls race: reserve, wait a random while, then settle or fail.

    A store wrapper checks after every single operation that spent plus reserved never passes
    the cap, so a breach at any moment fails the test, not only one visible at the end.
    """
    cap = 10 * DOLLAR
    inner = MemorySpendStore()
    breaches = []

    class CheckingStore:
        async def _check(self):
            totals = (await inner.totals(["2026-10-14/global"]))["2026-10-14/global"]
            if totals.committed + totals.held > cap:
                breaches.append(totals)

        async def reserve(self, buckets, amount):
            await asyncio.sleep(0)
            result = await inner.reserve(buckets, amount)
            await self._check()
            return result

        async def settle(self, keys, held, actual):
            await asyncio.sleep(0)
            await inner.settle(keys, held, actual)
            await self._check()

        async def release(self, keys, held):
            await inner.release(keys, held)
            await self._check()

        async def totals(self, keys):
            return await inner.totals(keys)

    guard, _ = make_guard(daily_usd="10", store=CheckingStore())
    rng = random.Random(4)
    outcomes = {"settled": 0, "failed": 0, "refused": 0}

    async def call():
        worst_case = rng.randint(DOLLAR // 10, DOLLAR)
        try:
            reservation = await guard.reserve(worst_case)
        except BudgetExceeded:
            outcomes["refused"] += 1
            return
        await asyncio.sleep(rng.random() / 1000)
        if rng.random() < 0.2:
            await guard.release(reservation)
            outcomes["failed"] += 1
        else:
            await guard.settle(reservation, rng.randint(0, worst_case))
            outcomes["settled"] += 1

    await asyncio.gather(*(call() for _ in range(400)))

    assert breaches == []
    assert outcomes["settled"] > 0 and outcomes["refused"] > 0
    final = (await inner.totals(["2026-10-14/global"]))["2026-10-14/global"]
    assert final.held == 0
    assert final.committed <= cap
