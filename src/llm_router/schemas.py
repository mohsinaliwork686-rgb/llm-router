"""The OpenAI chat-completions wire format, as far as the router supports it.

Requests are validated strictly. Anything the router cannot honour (tools, images, streaming,
several choices) is refused with a clear 400 rather than silently ignored, because a request that
quietly loses its tools gives a confidently wrong answer.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llm_router.core.chat import ChatMessage, ChatRequest
from llm_router.core.router import RoutedCompletion

#: Accepted in a request only when unset or empty. Each changes the answer in a way the router
#: cannot reproduce on every provider.
UNSUPPORTED_FIELDS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "audio",
    "modalities",
    "prediction",
)


class WireMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant"]
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _text_only(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, list):
            texts = []
            for part in value:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind != "text" or not isinstance(part.get("text"), str):
                    raise ValueError(f"only text content is supported, got a {kind!r} part")
                texts.append(part["text"])
            return "\n".join(texts)
        return value

    @model_validator(mode="after")
    def _no_tool_calls(self) -> WireMessage:
        if (self.model_extra or {}).get("tool_calls"):
            raise ValueError("tool calls are not supported")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[WireMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1

    @model_validator(mode="after")
    def _supported(self) -> ChatCompletionRequest:
        if self.stream:
            raise ValueError("streaming is not supported yet; send stream: false")
        if self.n != 1:
            raise ValueError("n must be 1; the router returns one choice per request")
        if isinstance(self.stop, list) and len(self.stop) > 4:
            raise ValueError("stop takes at most 4 sequences")
        extras = self.model_extra or {}
        refused = [name for name in UNSUPPORTED_FIELDS if extras.get(name)]
        response_format = extras.get("response_format")
        if response_format and response_format != {"type": "text"}:
            refused.append("response_format")
        if refused:
            raise ValueError(f"not supported by llm-router: {', '.join(refused)}")
        return self

    def to_chat_request(self) -> ChatRequest:
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())
        return ChatRequest(
            # "developer" is OpenAI's newer name for a system message.
            messages=tuple(
                ChatMessage("system" if m.role == "developer" else m.role, m.content)
                for m in self.messages
            ),
            max_tokens=self.max_completion_tokens or self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
        )


def completion_body(routed: RoutedCompletion) -> dict[str, Any]:
    completion = routed.completion
    usage = completion.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": completion.finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


ErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "budget_exceeded",
    "upstream_error",
    "service_unavailable",
]


def error_body(
    message: str, error_type: ErrorType, code: str | None = None, **extra: Any
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": code, **extra}}
