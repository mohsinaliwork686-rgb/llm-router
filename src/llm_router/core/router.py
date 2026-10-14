"""The fallback loop: try each candidate model in order until one answers.

For every attempt:

1. reserve the worst case the call could cost, or skip this model if it would cross a cap;
2. call the provider, with a timeout;
3. settle the real cost on success. On failure, give the reservation back, unless the provider may
   already have done the work (a timeout, a connection dropped mid-reply). Then it counts as spent.

Then, by failure kind: RETRY tries the same model again after a backoff, up to
attempts_per_model times; SKIP moves to the next model; REJECT stops, because the request itself is
the problem and every other model would refuse it too.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from llm_router.core.budget import BudgetExceeded, BudgetGuard, Reservation
from llm_router.core.chat import ChatRequest, Completion
from llm_router.core.config import ModelConfig, RouterConfig
from llm_router.core.policy import Route, route
from llm_router.core.pricing import cost_micros, output_limit, worst_case_micros
from llm_router.core.providers.base import Failure, Provider, ProviderError
from llm_router.core.tokens import estimate_messages

log = logging.getLogger(__name__)

Outcome = Literal["ok", "failed", "over_budget"]


@dataclass(frozen=True)
class Attempt:
    model: str
    outcome: Outcome
    detail: str = ""
    status: int | None = None
    cost_micros: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "outcome": self.outcome,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RoutedCompletion:
    completion: Completion
    model: ModelConfig
    route: Route
    attempts: tuple[Attempt, ...]
    cost_micros: int
    #: Set when on_exceeded is "warn" and this call went over a cap.
    budget_warning: str | None = None
    #: How far the real cost ran over the reservation. Should always be zero.
    shortfall_micros: int = 0


class RoutingFailed(Exception):
    """No candidate produced an answer."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


class RequestRejected(RoutingFailed):
    """A provider refused the request itself. The caller's problem: HTTP 400."""

    def __init__(self, message: str, attempts: list[Attempt], status: int | None) -> None:
        super().__init__(message, attempts)
        self.status = status


class OverBudget(RoutingFailed):
    """Every candidate would have crossed a cap. HTTP 429."""

    def __init__(self, first: BudgetExceeded, attempts: list[Attempt]) -> None:
        super().__init__(str(first), attempts)
        self.exceeded = first


class AllFailed(RoutingFailed):
    """Every candidate failed. HTTP 502."""


class Router:
    def __init__(
        self,
        config: RouterConfig,
        providers: Mapping[str, Provider],
        guard: BudgetGuard,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._providers = providers
        self._guard = guard
        self._sleep = sleep
        self._jitter = jitter
        self._monotonic = monotonic

    async def complete(
        self,
        requested: str,
        request: ChatRequest,
        *,
        client: str | None = None,
        tags: Collection[str] = (),
    ) -> RoutedCompletion:
        prompt_tokens = estimate_messages(request.contents())
        chosen = route(self._config, requested, tags=tags, prompt_tokens=prompt_tokens)
        attempts: list[Attempt] = []
        first_refusal: BudgetExceeded | None = None

        for model in chosen.candidates:
            max_tokens = output_limit(model, request.max_tokens)
            worst_case = worst_case_micros(model, prompt_tokens, max_tokens)
            provider = self._providers[model.provider]

            for attempt_number in range(1, self._config.retry.attempts_per_model + 1):
                try:
                    reservation = await self._guard.reserve(worst_case, client)
                except BudgetExceeded as refused:
                    # A cheaper model further down the list may still fit, so keep going.
                    attempts.append(Attempt(model.name, "over_budget", str(refused)))
                    first_refusal = first_refusal or refused
                    break

                started = self._monotonic()
                result = await self._call(provider, model, request, max_tokens, reservation)
                seconds = self._monotonic() - started

                if isinstance(result, Completion):
                    cost = cost_micros(model, result.usage.input_tokens, result.usage.output_tokens)
                    shortfall = await self._guard.settle(reservation, cost)
                    if shortfall:
                        log.warning(
                            "%s cost %d micro-dollars more than its reservation",
                            model.name,
                            shortfall,
                        )
                    attempts.append(Attempt(model.name, "ok", cost_micros=cost, seconds=seconds))
                    return RoutedCompletion(
                        completion=result,
                        model=model,
                        route=chosen,
                        attempts=tuple(attempts),
                        cost_micros=cost,
                        budget_warning=reservation.warning,
                        shortfall_micros=shortfall,
                    )

                error = result
                charged = worst_case if error.may_have_billed else 0
                if error.may_have_billed:
                    await self._guard.settle(reservation, worst_case)
                else:
                    await self._guard.release(reservation)
                attempts.append(
                    Attempt(
                        model.name,
                        "failed",
                        error.message,
                        status=error.status,
                        cost_micros=charged,
                        seconds=seconds,
                    )
                )
                log.info("%s attempt %d failed: %s", model.name, attempt_number, error.message)

                if error.kind is Failure.REJECT:
                    raise RequestRejected(error.message, attempts, error.status)
                if (
                    error.kind is Failure.SKIP
                    or attempt_number == self._config.retry.attempts_per_model
                ):
                    break
                delay = self._backoff(attempt_number, error.retry_after)
                if delay is None:
                    break  # the provider wants a longer pause than we will wait; try the next model
                await self._sleep(delay)

        if attempts and first_refusal and all(a.outcome == "over_budget" for a in attempts):
            raise OverBudget(first_refusal, attempts)
        raise AllFailed(_summary(attempts), attempts)

    async def _call(
        self,
        provider: Provider,
        model: ModelConfig,
        request: ChatRequest,
        max_tokens: int,
        reservation: Reservation,
    ) -> Completion | ProviderError:
        timeout = self._config.retry.timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                return await provider.complete(model, request, max_tokens)
        except TimeoutError:
            return ProviderError(
                Failure.RETRY, f"no answer within {timeout:g} seconds", may_have_billed=True
            )
        except ProviderError as error:
            return error
        except asyncio.CancelledError:
            # The caller went away mid-call. The provider may still finish and bill, so count the
            # reservation as spent before letting the cancellation through.
            await asyncio.shield(self._guard.settle(reservation, reservation.amount))
            raise
        except Exception as error:  # a bug in an adapter must not take down the request
            log.exception("unexpected error from %s", model.name)
            return ProviderError(
                Failure.SKIP, f"unexpected error: {type(error).__name__}", may_have_billed=True
            )

    def _backoff(self, attempt_number: int, retry_after: float | None) -> float | None:
        """Exponential backoff with jitter: somewhere between half and all of base * 2^(n-1)."""
        base = self._config.retry.backoff_ms / 1000 * 2 ** (attempt_number - 1)
        delay = base / 2 + self._jitter() * base / 2
        if retry_after is not None:
            if retry_after > self._config.retry.max_retry_after_seconds:
                return None
            delay = max(delay, retry_after)
        return delay


def _summary(attempts: list[Attempt]) -> str:
    if not attempts:
        return "no model was tried"
    tried = "; ".join(f"{a.model}: {a.detail}" for a in attempts)
    return f"every model failed ({tried})"
