"""A provider that calls nothing. It answers with canned text, or fails on purpose.

This is a feature, not test scaffolding: it lets anyone try fallback and the spending cap end to
end with no API keys, and it is what the demo runs on. Configure it per provider:

    providers:
      flaky:
        type: simulated
        fail_with: 503       # fail with this HTTP status...
        fail_times: 2        # ...for the first two calls, then succeed. 0 means fail every time.
        latency_ms: 300      # wait this long before answering
        reply: "Hello from the simulator."
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from llm_router.core.chat import ChatRequest, Completion, Usage
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers.base import ProviderError, classify_status

#: Real tokenizers average about four characters of English per token. The simulator reports that,
#: so its real cost comes in under the router's deliberately high reservation, as it would live.
CHARS_PER_TOKEN = 4


class SimulatedProvider:
    def __init__(
        self,
        config: ProviderConfig,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self.calls = 0

    async def complete(
        self, model: ModelConfig, request: ChatRequest, max_tokens: int
    ) -> Completion:
        self.calls += 1
        config = self._config
        if config.latency_ms:
            await self._sleep(config.latency_ms / 1000)
        always = config.fail_times == 0
        if config.fail_with is not None and (always or self.calls <= config.fail_times):
            raise ProviderError(
                classify_status(config.fail_with),
                f"simulated failure {config.fail_with} from {model.model}",
                status=config.fail_with,
            )

        input_tokens = sum(_tokens(content) + 4 for content in request.contents()) + 3
        text = config.reply
        finish = "stop"
        if _tokens(text) > max_tokens:
            text = text[: max_tokens * CHARS_PER_TOKEN]
            finish = "length"
        return Completion(
            text=text,
            finish_reason=finish,
            usage=Usage(input_tokens=input_tokens, output_tokens=_tokens(text)),
            model=model.model,
        )

    async def close(self) -> None:
        return None


def _tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)
