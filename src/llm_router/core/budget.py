"""The spending cap: reserve the worst case before a call, settle the real cost after it.

Checking "have we spent the budget yet?" before each call is not a cap. The call that passes the
check can cost more than what is left, and ten calls arriving together all pass the same check.
So instead:

1. Before a call, reserve its worst-case cost. The store checks it against every cap that applies
   (the global one, and the client's own) and holds it in all of them, in one atomic step. If it
   does not fit, the call is refused without ever reaching a provider.
2. After the call, settle: swap the reservation for the real cost. A call that failed releases its
   reservation instead.

A day is a calendar day in the configured timezone. The clock is passed in, so tests move time
forward rather than wait for midnight.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from llm_router.core.config import BudgetConfig, ClientConfig
from llm_router.core.pricing import format_usd, usd_to_micros
from llm_router.core.spend import Bucket, Refusal, SpendStore, Totals

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class BudgetExceeded(Exception):
    """A request would push spend past a cap. The router answers it with HTTP 429."""

    def __init__(self, scope: str, refusal: Refusal, requested: int) -> None:
        self.scope = scope
        self.refusal = refusal
        self.requested = requested
        super().__init__(_refusal_message(scope, refusal, requested))


@dataclass
class Reservation:
    day: str
    keys: tuple[str, ...]
    amount: int
    client: str | None
    #: Set when on_exceeded is "warn" and this request went over a cap it would otherwise hit.
    warning: str | None = None
    finished: bool = field(default=False, repr=False)


class BudgetGuard:
    def __init__(
        self,
        budget: BudgetConfig,
        clients: Sequence[ClientConfig],
        store: SpendStore,
        clock: Clock = utc_now,
    ) -> None:
        self._budget = budget
        self._zone = ZoneInfo(budget.timezone)
        self._store = store
        self._clock = clock
        self._global_limit = _micros_or_none(budget.daily_usd)
        self._client_limits = {client.name: _micros_or_none(client.daily_usd) for client in clients}

    def today(self) -> str:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("the budget clock must return timezone-aware datetimes")
        return now.astimezone(self._zone).date().isoformat()

    async def reserve(self, amount: int, client: str | None = None) -> Reservation:
        """Hold `amount` micro-dollars against today's caps, or raise BudgetExceeded."""
        day = self.today()
        buckets = self._buckets(day, client)
        refusal = await self._store.reserve(buckets, amount)
        warning = None
        if refusal is not None:
            scope = self._scope(refusal.key)
            if self._budget.on_exceeded == "reject":
                raise BudgetExceeded(scope, refusal, amount)
            # Warn mode: let it through, but still record it so the totals stay true.
            unlimited = [Bucket(bucket.key, None) for bucket in buckets]
            await self._store.reserve(unlimited, amount)
            warning = _refusal_message(scope, refusal, amount)
        return Reservation(
            day=day,
            keys=tuple(bucket.key for bucket in buckets),
            amount=amount,
            client=client,
            warning=warning,
        )

    async def settle(self, reservation: Reservation, actual: int) -> int:
        """Record the real cost of a finished call.

        Returns how far the real cost ran over the reservation, which should be zero. The money
        was spent either way, so it is recorded in full; the caller reports any shortfall.
        """
        _finish(reservation)
        await self._store.settle(reservation.keys, reservation.amount, actual)
        return max(0, actual - reservation.amount)

    async def release(self, reservation: Reservation) -> None:
        """Give back a reservation for a call that cost nothing, such as one that was refused."""
        _finish(reservation)
        await self._store.release(reservation.keys, reservation.amount)

    async def status(self) -> dict[str, Any]:
        """Today's spend against each cap, for GET /budget."""
        day = self.today()
        global_key = _global_key(day)
        client_keys = {name: _client_key(day, name) for name in self._client_limits}
        totals = await self._store.totals([global_key, *client_keys.values()])
        return {
            "day": day,
            "timezone": self._budget.timezone,
            "on_exceeded": self._budget.on_exceeded,
            "global": _summary(self._global_limit, totals[global_key]),
            "clients": {
                name: _summary(self._client_limits[name], totals[key])
                for name, key in client_keys.items()
            },
        }

    def _buckets(self, day: str, client: str | None) -> list[Bucket]:
        buckets = []
        # The client's own cap is checked first: when both would be exceeded, the narrower one
        # is the more useful thing to tell the caller.
        if client is not None:
            if client not in self._client_limits:
                raise KeyError(f"unknown client {client!r}")
            buckets.append(Bucket(_client_key(day, client), self._client_limits[client]))
        buckets.append(Bucket(_global_key(day), self._global_limit))
        return buckets

    @staticmethod
    def _scope(key: str) -> str:
        _, kind, *rest = key.split("/", 2)
        return "global" if kind == "global" else f"client {rest[0]!r}"


def _global_key(day: str) -> str:
    return f"{day}/global"


def _client_key(day: str, client: str) -> str:
    return f"{day}/client/{client}"


def _micros_or_none(usd: Decimal | None) -> int | None:
    return None if usd is None else usd_to_micros(usd)


def _finish(reservation: Reservation) -> None:
    if reservation.finished:
        raise RuntimeError("this reservation was already settled or released")
    reservation.finished = True


def _summary(limit: int | None, totals: Totals) -> dict[str, str | None]:
    remaining = None if limit is None else max(0, limit - totals.committed - totals.held)
    return {
        "limit_usd": None if limit is None else format_usd(limit),
        "spent_usd": format_usd(totals.committed),
        "reserved_usd": format_usd(totals.held),
        "remaining_usd": None if remaining is None else format_usd(remaining),
    }


def _refusal_message(scope: str, refusal: Refusal, requested: int) -> str:
    cap = format_usd(refusal.limit)
    if requested > refusal.limit:
        return (
            f"{scope} daily cap is ${cap}, and this request could cost up to "
            f"${format_usd(requested)} on its own"
        )
    return (
        f"{scope} daily cap of ${cap} would be exceeded: ${format_usd(refusal.committed)} spent, "
        f"${format_usd(refusal.held)} reserved by calls in flight, and this request could cost "
        f"up to ${format_usd(requested)}"
    )
