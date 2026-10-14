"""The router's own shape for a chat request and its answer.

The HTTP layer translates the OpenAI wire format into these, and each provider adapter translates
these into whatever its provider speaks. Keeping one neutral shape in the middle means an adapter
never needs to know about any other adapter's format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    #: What the client asked for. The router caps it at the model's limit before sending it on.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()

    def contents(self) -> list[str]:
        return [message.content for message in self.messages]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: FinishReason
    usage: Usage
    #: The model the provider says served the call, which may carry a version suffix.
    model: str
