import httpx2
import openai
import pytest
from fastapi.testclient import TestClient

from llm_router.core.config import ConfigError, parse_config
from llm_router.core.spend import MemorySpendStore, SpendStoreError
from llm_router.main import build_store, create_app
from tests.api_helpers import KEYS, WITH_CLIENTS, chat, config_data


def make_client(**overrides):
    app = create_app(parse_config(config_data(**overrides)), env=KEYS)
    return TestClient(app)


def test_open_mode_answers_in_the_openai_shape():
    with make_client() as client:
        response = chat(client)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "sim-fast"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the simulator."},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ]
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert response.headers["x-router-model"] == "fast"
    assert response.headers["x-router-attempts"] == "1"
    assert "x-router-policy" not in response.headers
    assert float(response.headers["x-router-cost-usd"]) > 0


def test_fallback_shows_in_the_headers():
    with make_client() as client:
        response = chat(client, model="resilient")
    assert response.status_code == 200
    assert response.headers["x-router-model"] == "fast"
    assert response.headers["x-router-policy"] == "resilient"
    assert response.headers["x-router-attempts"] == "3"


def test_tags_header_drives_auto_routing():
    with make_client() as client:
        response = chat(client, model="auto", headers={"X-Router-Tags": "batch, nightly"})
    assert response.status_code == 200
    assert response.headers["x-router-policy"] == "resilient"


def test_every_candidate_failing_is_one_clean_502():
    with make_client() as client:
        response = chat(client, model="dead")

    assert response.status_code == 502
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    assert [a["status"] for a in error["attempts"]] == [503, 503]
    assert "Traceback" not in response.text


def test_a_request_the_provider_rejects_is_a_400_and_stops_there():
    with make_client() as client:
        response = chat(client, model="strict")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert [a["model"] for a in error["attempts"]] == ["fussy"]


def test_over_budget_is_a_429_with_the_reason():
    with make_client(budget={"daily_usd": "0.000001"}) as client:
        response = chat(client)

    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == "budget_exceeded"
    assert error["code"] == "daily_cap_reached"
    assert error["scope"] == "global"
    assert "daily cap" in error["message"]


def test_unknown_model_lists_what_exists():
    with make_client() as client:
        response = chat(client, model="gpt-9")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert "Policies: dead, resilient, strict" in error["message"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"stream": True}, "streaming is not supported"),
        ({"n": 2}, "n must be 1"),
        ({"tools": [{"type": "function"}]}, "not supported by llm-router: tools"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": 3}, "temperature"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages.0.role"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            "only text content is supported, got a 'image_url' part",
        ),
    ],
)
def test_unsupported_or_invalid_requests_are_400s(body, message):
    with make_client() as client:
        response = chat(client, **body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert message in response.json()["error"]["message"]


def test_body_that_is_not_json():
    with make_client() as client:
        response = client.post("/v1/chat/completions", content=b"{nope")
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "the request body is not valid JSON"


def test_harmless_extra_fields_are_accepted():
    with make_client() as client:
        response = chat(client, user="u-1", seed=4, response_format={"type": "text"}, tools=[])
    assert response.status_code == 200


def test_missing_or_wrong_key_is_401():
    with make_client(**WITH_CLIENTS) as client:
        missing = chat(client)
        wrong = chat(client, headers={"Authorization": "Bearer guess"})
        right = chat(client, headers={"Authorization": "Bearer billing-secret"})

    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
    assert right.status_code == 200


def test_auth_is_checked_before_the_body():
    with make_client(**WITH_CLIENTS) as client:
        response = client.post("/v1/chat/completions", content=b"{nope")
    assert response.status_code == 401


def test_per_client_cap_inside_the_global_one():
    clients = {
        "clients": [
            {"name": "billing", "key_env": "KEY_BILLING", "daily_usd": "0.000001"},
            {"name": "support", "key_env": "KEY_SUPPORT"},
        ]
    }
    with make_client(**clients) as client:
        billing = chat(client, headers={"Authorization": "Bearer billing-secret"})
        support = chat(client, headers={"Authorization": "Bearer support-secret"})
    assert billing.status_code == 429
    assert billing.json()["error"]["scope"] == "client 'billing'"
    assert support.status_code == 200


def test_a_client_key_missing_from_the_environment_fails_startup():
    config = parse_config(config_data(**WITH_CLIENTS))
    with pytest.raises(ConfigError, match="KEY_SUPPORT is not set"):
        create_app(config, env={"KEY_BILLING": "billing-secret"})


def test_two_clients_cannot_share_a_key():
    config = parse_config(config_data(**WITH_CLIENTS))
    with pytest.raises(ConfigError, match="support has the same key as billing"):
        create_app(config, env={"KEY_BILLING": "same", "KEY_SUPPORT": "same"})


def test_models_lists_everything_routable():
    with make_client() as client:
        body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [
        "auto",
        "dead",
        "resilient",
        "strict",
        "broken",
        "fast",
        "fussy",
    ]


def test_budget_shows_todays_spend():
    with make_client() as client:
        chat(client)
        status = client.get("/budget").json()
    assert status["global"]["limit_usd"] == "1.000000"
    assert float(status["global"]["spent_usd"]) > 0
    assert status["global"]["reserved_usd"] == "0.000000"


def test_a_client_sees_only_its_own_budget():
    with make_client(**WITH_CLIENTS) as client:
        unauthenticated = client.get("/budget")
        status = client.get("/budget", headers={"Authorization": "Bearer support-secret"}).json()
    assert unauthenticated.status_code == 401
    assert list(status["clients"]) == ["support"]
    assert "global" in status


def test_healthz_needs_no_key():
    with make_client(**WITH_CLIENTS) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.anyio
async def test_the_official_openai_sdk_works_unchanged():
    app = create_app(parse_config(config_data(**WITH_CLIENTS)), env=KEYS)
    requests = []

    async def count(request):
        requests.append(request.url.path)

    http_client = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), event_hooks={"request": [count]}
    )
    sdk = openai.AsyncOpenAI(
        base_url="http://router.test/v1", api_key="billing-secret", http_client=http_client
    )

    raw = await sdk.chat.completions.with_raw_response.create(
        model="resilient",
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
        max_tokens=50,
        temperature=0.2,
    )
    completion = raw.parse()
    assert completion.choices[0].message.content == "Hello from the simulator."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.completion_tokens > 0
    assert raw.headers["x-router-model"] == "fast"

    models = await sdk.models.list()
    assert "resilient" in [model.id for model in models.data]

    # A 502 after the router's own retries is not retried again by the SDK.
    requests.clear()
    with pytest.raises(openai.InternalServerError):
        await sdk.chat.completions.create(model="dead", messages=[{"role": "user", "content": "x"}])
    assert requests == ["/v1/chat/completions"]

    wrong_key = sdk.with_options(api_key="nope")
    with pytest.raises(openai.AuthenticationError):
        await wrong_key.chat.completions.create(
            model="fast", messages=[{"role": "user", "content": "x"}]
        )
    await http_client.aclose()


class DownStore(MemorySpendStore):
    async def reserve(self, buckets, amount):
        raise SpendStoreError("connection refused")

    async def totals(self, keys):
        raise SpendStoreError("connection refused")


def test_an_unreachable_spend_store_is_a_503_and_no_provider_is_called():
    app = create_app(parse_config(config_data()), env=KEYS, store=DownStore())
    with TestClient(app) as client:
        response = chat(client)
        budget = client.get("/budget")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable"
    assert budget.status_code == 503


def test_the_store_is_built_from_config():
    assert isinstance(build_store(parse_config(config_data()), env={}), MemorySpendStore)
    redis_config = parse_config(config_data(store={"type": "redis"}))
    with pytest.raises(ConfigError, match="REDIS_URL is not set"):
        build_store(redis_config, env={})
    store = build_store(redis_config, env={"REDIS_URL": "redis://localhost:6379/0"})
    assert type(store).__name__ == "RedisSpendStore"


@pytest.mark.anyio
async def test_two_routers_sharing_redis_enforce_one_cap(redis_store_factory):
    if redis_store_factory is None:
        pytest.skip("set REDIS_URL to run the Redis tests")
    # Each call reserves a worst case of 510 micro-dollars and settles at 44. Under a $0.0007 cap
    # that admits exactly five calls in total. Two routers with separate stores would admit ten.
    data = config_data(budget={"daily_usd": "0.0007"})
    apps = [create_app(parse_config(data), env=KEYS, store=redis_store_factory()) for _ in range(2)]
    clients = [
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://router")
        for app in apps
    ]
    payload = {"model": "fast", "messages": [{"role": "user", "content": "Hi there"}]}

    statuses = []
    for i in range(20):
        response = await clients[i % 2].post("/v1/chat/completions", json=payload)
        statuses.append(response.status_code)

    assert statuses.count(200) == 5
    assert statuses.count(429) == 15
    # Both routers report the same total: it is one cap, not two.
    for client in clients:
        assert (await client.get("/budget")).json()["global"]["spent_usd"] == "0.000220"
    for client in clients:
        await client.aclose()
