import json

import httpx2
import pytest

from llm_router.core.chat import ChatMessage, ChatRequest
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers import Failure, ProviderError
from llm_router.core.providers.claude import ClaudeProvider

pytestmark = pytest.mark.anyio

HAIKU = ModelConfig(
    name="haiku",
    provider="anthropic",
    model="claude-haiku-4-5",
    input_per_million=1,
    output_per_million=5,
    max_output_tokens=8192,
)
OPUS = ModelConfig(
    name="opus",
    provider="anthropic",
    model="claude-opus-5-5",
    input_per_million=4,
    output_per_million=20,
    max_output_tokens=16000,
    drop_sampling_params=True,
)
REQUEST = ChatRequest(
    messages=(
        ChatMessage("system", "You are terse."),
        ChatMessage("system", "Answer in English."),
        ChatMessage("user", "Hello"),
        ChatMessage("assistant", "Hi."),
        ChatMessage("user", "How are you?"),
    ),
    temperature=0.3,
    stop=("END",),
)


def message_body(**overrides):
    body = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "Fine, thanks."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    body.update(overrides)
    return body


def make(handler, api_key_env="CLAUDE_KEY"):
    config = ProviderConfig(type="anthropic", api_key_env=api_key_env)
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return ClaudeProvider(config, {"CLAUDE_KEY": "sk-test"}, timeout=5, http_client=client)


def error_response(status, error_type, message, headers=None):
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    return httpx2.Response(status, json=body, headers=headers or {})


async def test_translates_the_request_and_the_reply():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=message_body())

    completion = await make(handler).complete(HAIKU, REQUEST, max_tokens=300)

    assert seen["path"] == "/v1/messages"
    assert seen["key"] == "sk-test"
    assert seen["body"] == {
        "model": "claude-haiku-4-5",
        "max_tokens": 300,
        "system": "You are terse.\n\nAnswer in English.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "How are you?"},
        ],
        "stop_sequences": ["END"],
        "temperature": 0.3,
    }
    assert completion.text == "Fine, thanks."
    assert completion.finish_reason == "stop"
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (20, 4)
    assert completion.model == "claude-haiku-4-5"


async def test_sampling_params_are_left_out_for_models_that_refuse_them():
    def handler(request):
        body = json.loads(request.content)
        assert "temperature" not in body and "top_p" not in body
        return httpx2.Response(200, json=message_body(model="claude-opus-5-5"))

    await make(handler).complete(OPUS, REQUEST, max_tokens=300)


async def test_no_system_prompt_sends_no_system_field():
    def handler(request):
        assert "system" not in json.loads(request.content)
        return httpx2.Response(200, json=message_body())

    request = ChatRequest(messages=(ChatMessage("user", "Hello"),))
    await make(handler).complete(HAIKU, request, max_tokens=50)


async def test_only_text_blocks_make_the_reply():
    body = message_body(
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."},
        ]
    )
    completion = await make(lambda r: httpx2.Response(200, json=body)).complete(
        HAIKU, REQUEST, max_tokens=50
    )
    assert completion.text == "Part one. Part two."


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
    ],
)
async def test_stop_reasons_map_to_finish_reasons(stop_reason, expected):
    body = message_body(stop_reason=stop_reason)
    completion = await make(lambda r: httpx2.Response(200, json=body)).complete(
        HAIKU, REQUEST, max_tokens=50
    )
    assert completion.finish_reason == expected


async def test_cached_input_counts_as_input():
    body = message_body()
    body["usage"].update(cache_creation_input_tokens=100, cache_read_input_tokens=1000)
    completion = await make(lambda r: httpx2.Response(200, json=body)).complete(
        HAIKU, REQUEST, max_tokens=50
    )
    assert completion.usage.input_tokens == 20 + 100 + 1000


@pytest.mark.parametrize(
    ("status", "error_type", "kind"),
    [
        (429, "rate_limit_error", Failure.RETRY),
        (500, "api_error", Failure.RETRY),
        (529, "overloaded_error", Failure.RETRY),
        (401, "authentication_error", Failure.SKIP),
        (404, "not_found_error", Failure.SKIP),
        (400, "invalid_request_error", Failure.REJECT),
    ],
)
async def test_api_errors(status, error_type, kind):
    provider = make(lambda r: error_response(status, error_type, "nope"))
    with pytest.raises(ProviderError) as caught:
        await provider.complete(HAIKU, REQUEST, max_tokens=50)
    assert caught.value.kind is kind
    assert caught.value.status == status
    assert "nope" in caught.value.message
    assert caught.value.may_have_billed is False


async def test_retry_after_is_read():
    provider = make(lambda r: error_response(429, "rate_limit_error", "slow", {"retry-after": "3"}))
    with pytest.raises(ProviderError) as caught:
        await provider.complete(HAIKU, REQUEST, max_tokens=50)
    assert caught.value.retry_after == 3.0


async def test_connect_failure_was_not_billed():
    def handler(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as caught:
        await make(handler).complete(HAIKU, REQUEST, max_tokens=50)
    assert caught.value.kind is Failure.RETRY
    assert caught.value.may_have_billed is False


async def test_timeout_may_have_billed():
    def handler(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as caught:
        await make(handler).complete(HAIKU, REQUEST, max_tokens=50)
    assert caught.value.kind is Failure.RETRY
    assert caught.value.may_have_billed is True


async def test_system_message_mid_conversation_is_rejected_before_calling():
    def handler(request):
        raise AssertionError("must not be called")

    request = ChatRequest(
        messages=(ChatMessage("user", "Hi"), ChatMessage("system", "Now be formal."))
    )
    with pytest.raises(ProviderError) as caught:
        await make(handler).complete(HAIKU, request, max_tokens=50)
    assert caught.value.kind is Failure.REJECT


async def test_only_system_messages_is_rejected():
    request = ChatRequest(messages=(ChatMessage("system", "Be brief."),))
    with pytest.raises(ProviderError) as caught:
        await make(lambda r: None).complete(HAIKU, request, max_tokens=50)
    assert caught.value.kind is Failure.REJECT


async def test_missing_key_skips_without_calling():
    def handler(request):
        raise AssertionError("must not be called")

    with pytest.raises(ProviderError) as caught:
        await make(handler, api_key_env="NOT_SET").complete(HAIKU, REQUEST, max_tokens=50)
    assert caught.value.kind is Failure.SKIP
    assert caught.value.message == "NOT_SET is not set"
