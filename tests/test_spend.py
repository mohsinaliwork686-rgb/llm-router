"""Contract tests every spend store must pass: in memory, and Redis when REDIS_URL is set."""

import asyncio
import threading

import pytest

from llm_router.core.spend import Bucket, MemorySpendStore, Refusal, Totals

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "redis"])
async def store(request, redis_store_factory):
    if request.param == "redis":
        if redis_store_factory is None:
            pytest.skip("set REDIS_URL to run the Redis tests")
        yield redis_store_factory()
        return
    store = MemorySpendStore()
    yield store
    await store.close()


async def test_unknown_buckets_are_empty(store):
    assert await store.totals(["a", "b"]) == {"a": Totals(0, 0), "b": Totals(0, 0)}


async def test_reserve_holds_in_every_bucket(store):
    assert await store.reserve([Bucket("a", 100), Bucket("b", None)], 40) is None
    assert await store.totals(["a", "b"]) == {"a": Totals(0, 40), "b": Totals(0, 40)}


async def test_reserve_up_to_exactly_the_limit(store):
    assert await store.reserve([Bucket("a", 100)], 60) is None
    assert await store.reserve([Bucket("a", 100)], 40) is None
    assert await store.totals(["a"]) == {"a": Totals(0, 100)}


async def test_reserve_past_the_limit_is_refused_and_holds_nothing(store):
    await store.reserve([Bucket("a", 100)], 60)
    refusal = await store.reserve([Bucket("b", None), Bucket("a", 100)], 41)
    assert refusal == Refusal(key="a", limit=100, committed=0, held=60)
    # All or nothing: bucket b, which had room, was not touched either.
    assert await store.totals(["a", "b"]) == {"a": Totals(0, 60), "b": Totals(0, 0)}


async def test_refusal_names_the_first_bucket_in_the_way(store):
    refusal = await store.reserve([Bucket("narrow", 10), Bucket("wide", 5)], 20)
    assert refusal is not None
    assert refusal.key == "narrow"


async def test_committed_spend_counts_against_the_limit(store):
    await store.reserve([Bucket("a", 100)], 50)
    await store.settle(["a"], held=50, actual=80)
    assert await store.reserve([Bucket("a", 100)], 21) == Refusal("a", 100, 80, 0)
    assert await store.reserve([Bucket("a", 100)], 20) is None


async def test_settle_swaps_the_hold_for_the_real_cost(store):
    await store.reserve([Bucket("a", None), Bucket("b", None)], 50)
    await store.settle(["a", "b"], held=50, actual=12)
    assert await store.totals(["a", "b"]) == {"a": Totals(12, 0), "b": Totals(12, 0)}


async def test_release_drops_the_hold(store):
    await store.reserve([Bucket("a", 100)], 50)
    await store.release(["a"], held=50)
    assert await store.totals(["a"]) == {"a": Totals(0, 0)}


async def test_zero_cost_is_never_refused(store):
    await store.reserve([Bucket("a", None)], 500)
    await store.settle(["a"], held=500, actual=500)
    # Bucket a is over a limit of 100 now (spend tracked while uncapped), yet free calls pass.
    assert await store.reserve([Bucket("a", 100)], 0) is None


async def test_negative_amounts_are_refused(store):
    with pytest.raises(ValueError):
        await store.reserve([Bucket("a", 100)], -1)
    with pytest.raises(ValueError):
        await store.settle(["a"], held=0, actual=-1)
    with pytest.raises(ValueError):
        await store.release(["a"], held=-1)


async def test_concurrent_reservations_cannot_jointly_exceed_the_limit(store):
    limit, amount = 1_000, 7

    async def attempt():
        await asyncio.sleep(0)
        return await store.reserve([Bucket("a", limit)], amount)

    results = await asyncio.gather(*(attempt() for _ in range(500)))
    accepted = sum(result is None for result in results)
    assert accepted == limit // amount
    assert (await store.totals(["a"]))["a"] == Totals(0, accepted * amount)


def test_memory_store_is_safe_across_threads():
    store = MemorySpendStore()
    limit, amount = 1_000, 7
    accepted = []
    start = threading.Barrier(8)

    def worker():
        start.wait()
        count = 0
        for _ in range(200):
            if asyncio.run(store.reserve([Bucket("a", limit)], amount)) is None:
                count += 1
        accepted.append(count)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(accepted) == limit // amount
    assert asyncio.run(store.totals(["a"]))["a"].held == (limit // amount) * amount
