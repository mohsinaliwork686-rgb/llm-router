"""Spend kept in Redis, so several routers can share one set of caps.

Each bucket is a Redis hash with two fields, committed and held. Reserving runs as one Lua script:
Redis executes a script start to finish with nothing else in between, so checking every bucket and
holding the amount in all of them is a single atomic step, even with many routers calling at once.

Buckets expire a few days after their last write. If a router dies mid-call, its held amount stays
until then. That errs safe: the cap gets tighter, never looser.

A single Redis instance (or a primary with replicas) is expected. Redis Cluster is not supported,
because one reservation touches several keys that would need to live on the same node.
"""

from __future__ import annotations

from collections.abc import Sequence

import redis.asyncio as redis
from redis.exceptions import RedisError

from llm_router.core.spend import Bucket, Refusal, SpendStoreError, Totals

#: Long enough to cover a day in any timezone, plus a call that runs over midnight.
BUCKET_TTL_SECONDS = 3 * 24 * 60 * 60
#: How long a request waits for a free Redis connection before giving up.
POOL_WAIT_SECONDS = 10

# KEYS: one hash per bucket. ARGV[1]: the amount. ARGV[2 .. #KEYS+1]: each bucket's limit, or -1
# for none. ARGV[#KEYS+2]: the expiry in seconds. Returns {0} when held, or {index, committed, held}
# for the first bucket that would go over its limit.
_RESERVE = """
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[#KEYS + 2])
if amount > 0 then
  for i, key in ipairs(KEYS) do
    local limit = tonumber(ARGV[i + 1])
    if limit >= 0 then
      local totals = redis.call('HMGET', key, 'committed', 'held')
      local committed = tonumber(totals[1] or '0')
      local held = tonumber(totals[2] or '0')
      if committed + held + amount > limit then
        return {i, committed, held}
      end
    end
  end
end
for _, key in ipairs(KEYS) do
  redis.call('HINCRBY', key, 'held', amount)
  redis.call('EXPIRE', key, ttl)
end
return {0}
"""


class RedisSpendStore:
    def __init__(self, client: redis.Redis, key_prefix: str = "llm-router:") -> None:
        self._client = client
        self._prefix = key_prefix
        self._reserve = client.register_script(_RESERVE)

    @classmethod
    def from_url(
        cls, url: str, key_prefix: str = "llm-router:", max_connections: int = 50
    ) -> RedisSpendStore:
        # A blocking pool: when every connection is busy, a request waits its turn instead of
        # failing with "too many connections", which would turn a traffic spike into errors.
        pool = redis.BlockingConnectionPool.from_url(
            url,
            max_connections=max_connections,
            timeout=POOL_WAIT_SECONDS,
            decode_responses=True,
        )
        return cls(redis.Redis.from_pool(pool), key_prefix)

    async def reserve(self, buckets: Sequence[Bucket], amount: int) -> Refusal | None:
        _check_amount(amount)
        if not buckets:
            return None
        keys = [self._key(bucket.key) for bucket in buckets]
        limits = [-1 if bucket.limit is None else bucket.limit for bucket in buckets]
        try:
            result = await self._reserve(keys=keys, args=[amount, *limits, BUCKET_TTL_SECONDS])
        except RedisError as error:
            raise SpendStoreError(f"could not reserve spend in Redis: {error}") from error
        index = int(result[0])
        if index == 0:
            return None
        bucket = buckets[index - 1]
        assert bucket.limit is not None
        return Refusal(bucket.key, bucket.limit, int(result[1]), int(result[2]))

    async def settle(self, keys: Sequence[str], held: int, actual: int) -> None:
        _check_amount(held)
        _check_amount(actual)
        await self._apply(keys, held=-held, committed=actual)

    async def release(self, keys: Sequence[str], held: int) -> None:
        _check_amount(held)
        await self._apply(keys, held=-held, committed=0)

    async def totals(self, keys: Sequence[str]) -> dict[str, Totals]:
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                for key in keys:
                    pipe.hmget(self._key(key), "committed", "held")
                rows = await pipe.execute()
        except RedisError as error:
            raise SpendStoreError(f"could not read spend from Redis: {error}") from error
        return {
            key: Totals(int(committed or 0), int(held or 0))
            for key, (committed, held) in zip(keys, rows, strict=True)
        }

    async def close(self) -> None:
        await self._client.aclose()

    async def _apply(self, keys: Sequence[str], *, held: int, committed: int) -> None:
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                for key in keys:
                    full = self._key(key)
                    pipe.hincrby(full, "held", held)
                    if committed:
                        pipe.hincrby(full, "committed", committed)
                    pipe.expire(full, BUCKET_TTL_SECONDS)
                await pipe.execute()
        except RedisError as error:
            raise SpendStoreError(f"could not record spend in Redis: {error}") from error

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"


def _check_amount(amount: int) -> None:
    if amount < 0:
        raise ValueError(f"amounts cannot be negative, got {amount}")
