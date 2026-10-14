"""What every provider adapter promises, and how failures are described.

An adapter turns a ChatRequest into one call to its provider and returns a Completion, or raises
ProviderError saying what kind of failure it was. The router decides what to do next from that
kind alone, so it never needs to know which provider it is talking to.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Protocol

from llm_router.core.chat import ChatRequest, Completion
from llm_router.core.config import ModelConfig


class Failure(StrEnum):
    #: Worth trying the same model again: rate limits, server errors, timeouts, dropped connections.
    RETRY = "retry"
    #: Something wrong with this model or provider, such as a bad key. Move to the next model.
    SKIP = "skip"
    #: Something wrong with the request itself. Every model would refuse it, so stop and say so.
    REJECT = "reject"


class ProviderError(Exception):
    def __init__(
        self,
        kind: Failure,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        may_have_billed: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        #: Seconds the provider asked us to wait, from a Retry-After header.
        self.retry_after = retry_after
        #: True when the provider may have done the work before the failure, as with a timeout.
        #: The router then counts the reservation as spent instead of giving it back.
        self.may_have_billed = may_have_billed


class Provider(Protocol):
    async def complete(
        self, model: ModelConfig, request: ChatRequest, max_tokens: int
    ) -> Completion: ...

    async def close(self) -> None: ...


def classify_status(status: int) -> Failure:
    """Map an HTTP error status to what the router should do about it.

    404 skips rather than rejects: the client asked the router for a policy, so a provider that
    does not know the model name points at the router's config, not at the client's request.
    """
    if status in (408, 409, 429) or status >= 500:
        return Failure.RETRY
    if status in (401, 402, 403, 404):
        return Failure.SKIP
    return Failure.REJECT


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After header, which is either a number or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        seconds = (when - (now or datetime.now(UTC))).total_seconds()
    if math.isnan(seconds) or math.isinf(seconds):
        return None
    return max(0.0, seconds)
