"""What only a shared store has to prove: several routers, one cap."""

import asyncio

import pytest

from llm_router.core.redis_store import BUCKET_TTL_SECONDS, RedisSpendStore
from llm_router.core.spend import Bucket, SpendStoreError, Totals

pytestmark = pytest.mark.anyio


@pytest.fixture
def make_store(redis_store_factory):
    if redis_store_factory is None:
        pytest.skip("set REDIS_URL to run the Redis tests")
    return redis_store_factory


async def test_several_routers_cannot_jointly_exceed_a_shared_cap(make_store):
    # Four stores on separate connections stand in for four router processes.
    routers = [make_store() for _ in range(4)]
    limit, amount = 1_000, 7

    async def attempt(store):
        return await store.reserve([Bucket("2026-10-15/global", limit)], amount)

    results = await asyncio.gather(*(attempt(routers[i % 4]) for i in range(600)))

    accepted = sum(result is None for result in results)
    assert accepted == limit // amount
    totals = await routers[0].totals(["2026-10-15/global"])
    assert totals["2026-10-15/global"] == Totals(0, accepted * amount)


async def test_one_routers_spend_is_visible_to_another(make_store):
    first, second = make_store(), make_store()
    await first.reserve([Bucket("day/global", 100)], 60)
    await first.settle(["day/global"], held=60, actual=55)

    refusal = await second.reserve([Bucket("day/global", 100)], 50)

    assert refusal is not None
    assert (refusal.committed, refusal.held) == (55, 0)


async def test_buckets_expire(make_store):
    store = make_store()
    await store.reserve([Bucket("day/global", None)], 5)
    ttl = await store._client.ttl(store._key("day/global"))
    assert 0 < ttl <= BUCKET_TTL_SECONDS


async def test_large_amounts_stay_exact(make_store):
    store = make_store()
    big = 9_000_000_000_000  # nine million dollars in micro-dollars
    await store.reserve([Bucket("day/global", None)], big)
    await store.settle(["day/global"], held=big, actual=big - 1)
    assert (await store.totals(["day/global"]))["day/global"] == Totals(big - 1, 0)


async def test_an_unreachable_redis_is_a_store_error():
    store = RedisSpendStore.from_url("redis://127.0.0.1:1/0")
    try:
        with pytest.raises(SpendStoreError):
            await store.reserve([Bucket("day/global", 100)], 1)
        with pytest.raises(SpendStoreError):
            await store.totals(["day/global"])
        with pytest.raises(SpendStoreError):
            await store.settle(["day/global"], held=1, actual=1)
    finally:
        await store.close()
