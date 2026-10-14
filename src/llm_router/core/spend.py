"""Where spend is kept.

Spend is tracked in buckets, one per cap per day: today's global total, today's total for one
client, and so on. Each bucket holds two numbers, in micro-dollars:

- committed: money actually spent, from the providers' reported usage;
- held: worst-case amounts reserved by calls still in flight.

The one operation that must be atomic is reserve. It checks every bucket a request touches and
either holds the amount in all of them or in none, so two requests can never both squeeze into
the same last dollar. Settle and release only ever move money in a safe direction.

Two stores implement the same interface: one in memory, for a single router process, and one in
Redis, for several routers sharing caps.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Bucket:
    key: str
    #: The cap in micro-dollars, or None to track spend in this bucket without limiting it.
    limit: int | None


@dataclass(frozen=True)
class Totals:
    committed: int = 0
    held: int = 0


@dataclass(frozen=True)
class Refusal:
    """Why a reservation was refused: the first bucket it would have pushed over its cap."""

    key: str
    limit: int
    committed: int
    held: int


class SpendStore(Protocol):
    async def reserve(self, buckets: Sequence[Bucket], amount: int) -> Refusal | None:
        """Hold `amount` in every bucket, or in none and say which cap was in the way.

        A reservation that cannot cost anything (amount 0) is never refused.
        """

    async def settle(self, keys: Sequence[str], held: int, actual: int) -> None:
        """Replace a reservation of `held` with the real cost, `actual`."""

    async def release(self, keys: Sequence[str], held: int) -> None:
        """Drop a reservation of `held` without spending anything."""

    async def totals(self, keys: Sequence[str]) -> dict[str, Totals]: ...

    async def close(self) -> None: ...


class MemorySpendStore:
    """Spend kept in this process. Caps are per process, and spend is lost on restart.

    A threading lock rather than an asyncio one: nothing inside it awaits, and it also keeps the
    store correct if it is ever called from more than one thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._committed: dict[str, int] = defaultdict(int)
        self._held: dict[str, int] = defaultdict(int)

    async def reserve(self, buckets: Sequence[Bucket], amount: int) -> Refusal | None:
        _check_amount(amount)
        with self._lock:
            if amount > 0:
                for bucket in buckets:
                    committed, held = self._committed[bucket.key], self._held[bucket.key]
                    if bucket.limit is not None and committed + held + amount > bucket.limit:
                        return Refusal(bucket.key, bucket.limit, committed, held)
            for bucket in buckets:
                self._held[bucket.key] += amount
        return None

    async def settle(self, keys: Sequence[str], held: int, actual: int) -> None:
        _check_amount(held)
        _check_amount(actual)
        with self._lock:
            for key in keys:
                self._held[key] -= held
                self._committed[key] += actual

    async def release(self, keys: Sequence[str], held: int) -> None:
        _check_amount(held)
        with self._lock:
            for key in keys:
                self._held[key] -= held

    async def totals(self, keys: Sequence[str]) -> dict[str, Totals]:
        with self._lock:
            return {
                key: Totals(self._committed.get(key, 0), self._held.get(key, 0)) for key in keys
            }

    async def close(self) -> None:
        return None


def _check_amount(amount: int) -> None:
    if amount < 0:
        raise ValueError(f"amounts cannot be negative, got {amount}")
