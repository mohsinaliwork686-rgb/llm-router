import json

import httpx2
import pytest

from llm_router.core.chat import ChatMessage, ChatRequest
from llm_router.core.config import ModelConfig, ProviderConfig
from llm_router.core.providers import Failure, ProviderError
from llm_router.core.providers.openai_compatible import OpenAICompatibleProvider

pytestmark = pytest.mark.anyio

MODEL = ModelConfig(
    name="llama",
    provider="local",
    model="llama3.1:8b",
    input_per_million=0,
    output_per_million=0,
    max_output_tokens=4096,
)
REQUEST = ChatRequest(
    messages=(ChatMessage("system", "Be brief."), ChatMessage("user", "Hello")),
    temperature=0.2,
    top_p=0.9,
    stop=("END",),
)


def ok_body(**overrides):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "llama3.1:8b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }
    body.update(overrides)
    return body


def make(handler, **config):
    provider_config = ProviderConfig(
        type="openai_compatible", base_url="http://local.test/v1", **config
    )
    env = {"LOCAL_KEY": "secret-key"}
    return OpenAICompatibleProvider(
        provider_config, env, timeout=5, transport=httpx2.MockTransport(handler)
    )


async def test_sends_the_openai_shape_and_parses_the_reply():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=ok_body())

    provider = make(handler, api_key_env="LOCAL_KEY")
    completion = await provider.complete(MODEL, REQUEST, max_tokens=256)

    assert seen["url"] == "http://local.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"] == {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["END"],
    }
    assert completion.text == "Hi!"
    assert completion.finish_reason == "stop"
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (12, 3)
    assert completion.model == "llama3.1:8b"


async def test_max_completion_tokens_field():
    def handler(request):
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 64
        assert "max_tokens" not in body
        return httpx2.Response(200, json=ok_body())

    provider = make(handler, max_tokens_field="max_completion_tokens")
    await provider.complete(MODEL, REQUEST, max_tokens=64)


async def test_sampling_params_dropped_when_the_model_refuses_them():
    def handler(request):
        body = json.loads(request.content)
        assert "temperature" not in body and "top_p" not in body
        return httpx2.Response(200, json=ok_body())

    model = MODEL.model_copy(update={"drop_sampling_params": True})
    await make(handler).complete(model, REQUEST, max_tokens=64)


async def test_no_key_configured_sends_no_auth_header():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx2.Response(200, json=ok_body())

    await make(handler).complete(MODEL, REQUEST, max_tokens=64)


async def test_missing_key_skips_without_calling():
    def handler(request):
        raise AssertionError("must not be called")

    provider = make(handler, api_key_env="NOT_SET")
    with pytest.raises(ProviderError) as caught:
        await provider.complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.kind is Failure.SKIP
    assert "NOT_SET is not set" in caught.value.message


@pytest.mark.parametrize(
    ("finish", "expected"),
    [("stop", "stop"), ("length", "length"), ("content_filter", "content_filter"), (None, "stop")],
)
async def test_finish_reasons(finish, expected):
    body = ok_body()
    body["choices"][0]["finish_reason"] = finish
    provider = make(lambda request: httpx2.Response(200, json=body))
    assert (await provider.complete(MODEL, REQUEST, max_tokens=64)).finish_reason == expected


async def test_missing_usage_falls_back_to_the_high_estimate():
    body = ok_body()
    del body["usage"]
    provider = make(lambda request: httpx2.Response(200, json=body))
    completion = await provider.complete(MODEL, REQUEST, max_tokens=64)
    assert completion.usage.input_tokens > 0
    assert completion.usage.output_tokens == 1


async def test_null_content_becomes_empty_text():
    body = ok_body()
    body["choices"][0]["message"]["content"] = None
    provider = make(lambda request: httpx2.Response(200, json=body))
    assert (await provider.complete(MODEL, REQUEST, max_tokens=64)).text == ""


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, Failure.RETRY),
        (500, Failure.RETRY),
        (503, Failure.RETRY),
        (401, Failure.SKIP),
        (403, Failure.SKIP),
        (404, Failure.SKIP),
        (400, Failure.REJECT),
        (422, Failure.REJECT),
    ],
)
async def test_error_statuses(status, kind):
    error = {"error": {"message": "nope", "type": "x"}}
    provider = make(lambda request: httpx2.Response(status, json=error))
    with pytest.raises(ProviderError) as caught:
        await provider.complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.kind is kind
    assert caught.value.status == status
    assert caught.value.message == f"HTTP {status}: nope"
    assert caught.value.may_have_billed is False


async def test_retry_after_is_read():
    provider = make(lambda request: httpx2.Response(429, headers={"retry-after": "2"}, text="slow"))
    with pytest.raises(ProviderError) as caught:
        await provider.complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.retry_after == 2.0
    assert caught.value.message == "HTTP 429: slow"


async def test_connect_failure_retries_and_was_not_billed():
    def handler(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as caught:
        await make(handler).complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.kind is Failure.RETRY
    assert caught.value.may_have_billed is False


@pytest.mark.parametrize("error", [httpx2.ReadTimeout, httpx2.RemoteProtocolError])
async def test_failures_after_sending_may_have_billed(error):
    def handler(request):
        raise error("gone", request=request)

    with pytest.raises(ProviderError) as caught:
        await make(handler).complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.kind is Failure.RETRY
    assert caught.value.may_have_billed is True


async def test_unreadable_response_skips_and_may_have_billed():
    provider = make(lambda request: httpx2.Response(200, text="<html>not json</html>"))
    with pytest.raises(ProviderError) as caught:
        await provider.complete(MODEL, REQUEST, max_tokens=64)
    assert caught.value.kind is Failure.SKIP
    assert caught.value.may_have_billed is True
