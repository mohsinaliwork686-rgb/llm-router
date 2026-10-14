"""Adapter for anything that speaks the OpenAI chat-completions API.

That covers OpenAI itself and most servers that copy its API: Ollama, vLLM, LM Studio, and many
hosted providers. One HTTP call per attempt; retries are the router's job, not this adapter's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx2

from llm_router.core.chat import ChatRequest, Completion, FinishReason, Usage
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers.base import (
    Failure,
    ProviderError,
    classify_status,
    parse_retry_after,
)
from llm_router.core.tokens import estimate_messages, estimate_text

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
}


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ProviderConfig,
        env: Mapping[str, str],
        timeout: float,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if not config.base_url:
            raise ValueError("an openai_compatible provider needs a base_url")
        self._config = config
        self._missing_key: str | None = None
        headers = {}
        if config.api_key_env:
            key = env.get(config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
            else:
                self._missing_key = config.api_key_env
        self._client = httpx2.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def complete(
        self, model: ModelConfig, request: ChatRequest, max_tokens: int
    ) -> Completion:
        if self._missing_key:
            raise ProviderError(Failure.SKIP, f"{self._missing_key} is not set")

        body: dict[str, Any] = {
            "model": model.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            self._config.max_tokens_field: max_tokens,
        }
        if not model.drop_sampling_params:
            if request.temperature is not None:
                body["temperature"] = request.temperature
            if request.top_p is not None:
                body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = list(request.stop)

        try:
            response = await self._client.post("chat/completions", json=body)
        except (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout) as error:
            # The request never reached the provider, so nothing was billed.
            raise ProviderError(Failure.RETRY, f"could not connect: {error!r}") from error
        except httpx2.TimeoutException as error:
            raise ProviderError(
                Failure.RETRY, "the provider did not answer in time", may_have_billed=True
            ) from error
        except httpx2.TransportError as error:
            raise ProviderError(
                Failure.RETRY, f"connection failed mid-request: {error!r}", may_have_billed=True
            ) from error

        if response.status_code >= 400:
            raise ProviderError(
                classify_status(response.status_code),
                _error_message(response),
                status=response.status_code,
                retry_after=parse_retry_after(response.headers.get("retry-after")),
            )
        return _parse(response, model, request)

    async def close(self) -> None:
        await self._client.aclose()


def _parse(response: httpx2.Response, model: ModelConfig, request: ChatRequest) -> Completion:
    try:
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        finish = _FINISH.get(choice.get("finish_reason") or "stop", "stop")
        usage = data.get("usage")
        if usage:
            counted = Usage(int(usage["prompt_tokens"]), int(usage["completion_tokens"]))
        else:
            # Some servers leave usage out. The estimate errs high, so the cap stays safe.
            counted = Usage(estimate_messages(request.contents()), estimate_text(text))
        served_by = data.get("model") or model.model
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise ProviderError(
            Failure.SKIP,
            "the provider sent a response the router could not read",
            may_have_billed=True,
        ) from error
    if counted.input_tokens < 0 or counted.output_tokens < 0:
        raise ProviderError(
            Failure.SKIP, "the provider reported negative usage", may_have_billed=True
        )
    return Completion(text=text, finish_reason=finish, usage=counted, model=served_by)


def _error_message(response: httpx2.Response) -> str:
    try:
        error = response.json().get("error")
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    detail = str(message) if message else response.text[:200] or response.reason_phrase
    return f"HTTP {response.status_code}: {detail}"
