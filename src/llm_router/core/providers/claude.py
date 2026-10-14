"""Adapter for Claude, through the official Anthropic SDK.

It translates between the router's neutral shape and the Messages API:

- leading system messages become the top-level `system` prompt;
- `stop` becomes `stop_sequences`;
- temperature and top_p are sent only for models that accept them (see drop_sampling_params);
- the reply's text blocks are joined, and `stop_reason` maps onto OpenAI's finish reasons.

The SDK's own retries are switched off: the router does the retrying, across models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anthropic
import httpx2

from llm_router.core.chat import ChatRequest, Completion, FinishReason, Usage
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers.base import (
    Failure,
    ProviderError,
    classify_status,
    parse_retry_after,
)

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}


class ClaudeProvider:
    def __init__(
        self,
        config: ProviderConfig,
        env: Mapping[str, str],
        timeout: float,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._missing_key: str | None = None
        options: dict[str, Any] = {"max_retries": 0, "timeout": timeout}
        if config.api_key_env:
            key = env.get(config.api_key_env)
            if key:
                options["api_key"] = key
            else:
                self._missing_key = config.api_key_env
        if config.base_url:
            options["base_url"] = config.base_url
        if http_client is not None:
            options["http_client"] = http_client
        self._client = anthropic.AsyncAnthropic(**options)

    async def complete(
        self, model: ModelConfig, request: ChatRequest, max_tokens: int
    ) -> Completion:
        if self._missing_key:
            raise ProviderError(Failure.SKIP, f"{self._missing_key} is not set")

        params: dict[str, Any] = {"model": model.model, "max_tokens": max_tokens}
        system, messages = _split_system(request)
        params["messages"] = messages
        if system:
            params["system"] = system
        if request.stop:
            params["stop_sequences"] = list(request.stop)
        sampling = {}
        if not model.drop_sampling_params:
            if request.temperature is not None:
                sampling["temperature"] = request.temperature
            if request.top_p is not None:
                sampling["top_p"] = request.top_p
        if sampling:
            # The 1.x SDK no longer takes these as arguments; models that still accept them get
            # them in the request body directly.
            params["extra_body"] = sampling

        try:
            message = await self._client.messages.create(**params)
        except anthropic.APITimeoutError as error:
            raise ProviderError(
                Failure.RETRY, "Claude did not answer in time", may_have_billed=True
            ) from error
        except anthropic.APIConnectionError as error:
            never_sent = isinstance(error.__cause__, httpx2.ConnectError | httpx2.ConnectTimeout)
            raise ProviderError(
                Failure.RETRY,
                f"connection to Claude failed: {error.__cause__ or error!r}",
                may_have_billed=not never_sent,
            ) from error
        except anthropic.APIStatusError as error:
            raise ProviderError(
                classify_status(error.status_code),
                f"HTTP {error.status_code}: {error.message}",
                status=error.status_code,
                retry_after=parse_retry_after(error.response.headers.get("retry-after")),
            ) from error
        except TypeError as error:
            # Raised before anything is sent, most often because no credentials were found.
            raise ProviderError(Failure.SKIP, f"Claude client error: {error}") from error

        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        # Cached input is still input. The router never asks for caching, so these are normally
        # zero, but counting them keeps the cost honest if a provider caches on its own.
        input_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return Completion(
            text=text,
            finish_reason=_FINISH.get(message.stop_reason or "end_turn", "stop"),
            usage=Usage(input_tokens=input_tokens, output_tokens=usage.output_tokens),
            model=message.model,
        )

    async def close(self) -> None:
        await self._client.close()


def _split_system(request: ChatRequest) -> tuple[str, list[dict[str, str]]]:
    """Leading system messages become the system prompt; the rest are the conversation."""
    messages = list(request.messages)
    system_parts = []
    while messages and messages[0].role == "system":
        system_parts.append(messages.pop(0).content)
    if any(message.role == "system" for message in messages):
        raise ProviderError(
            Failure.REJECT,
            "system messages are only supported at the start of the conversation for Claude",
        )
    if not messages:
        raise ProviderError(Failure.REJECT, "the conversation needs at least one user message")
    return "\n\n".join(system_parts), [{"role": m.role, "content": m.content} for m in messages]
