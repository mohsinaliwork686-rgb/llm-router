from fastapi.testclient import TestClient

from llm_router.core.config import parse_config
from llm_router.main import create_app
from tests.api_helpers import KEYS, WITH_CLIENTS, chat, config_data


def make(**overrides):
    app = create_app(parse_config(config_data(**overrides)), env=KEYS)
    return app, TestClient(app)


def sample(app, name, **labels):
    return app.state.metrics.registry.get_sample_value(name, labels) or 0.0


def test_a_fallback_shows_every_attempt_and_the_answer():
    app, client = make()
    with client:
        chat(client, model="resilient")

    assert sample(app, "llm_router_requests_total", result="ok") == 1
    assert sample(app, "llm_router_attempts_total", model="broken", outcome="failed") == 2
    assert sample(app, "llm_router_attempts_total", model="fast", outcome="ok") == 1
    assert sample(app, "llm_router_tokens_total", model="fast", direction="output") == 7
    assert sample(app, "llm_router_cost_usd_total", model="fast", client="-") > 0
    assert sample(app, "llm_router_cost_usd_total", model="broken", client="-") == 0
    assert sample(app, "llm_router_attempt_duration_seconds_count", model="broken") == 2
    assert sample(app, "llm_router_request_duration_seconds_count", result="ok") == 1


def test_each_kind_of_failure_has_its_own_result():
    app, client = make(**WITH_CLIENTS)
    auth = {"Authorization": "Bearer support-secret"}
    with client:
        chat(client)  # no key
        chat(client, model="gpt-9", headers=auth)  # unknown model
        chat(client, model="strict", headers=auth)  # provider rejects the request
        chat(client, model="dead", headers=auth)  # every candidate fails
        chat(client, headers={"Authorization": "Bearer billing-secret"}, max_tokens=50)

    assert sample(app, "llm_router_requests_total", result="unauthorized") == 1
    assert sample(app, "llm_router_requests_total", result="invalid") == 1
    assert sample(app, "llm_router_requests_total", result="rejected") == 1
    assert sample(app, "llm_router_requests_total", result="failed") == 1
    assert sample(app, "llm_router_requests_total", result="ok") == 1
    assert sample(app, "llm_router_cost_usd_total", model="fast", client="billing") > 0


def test_over_budget_counts_the_refused_attempts():
    app, client = make(budget={"daily_usd": "0.000001"})
    with client:
        chat(client, model="resilient")
    assert sample(app, "llm_router_requests_total", result="over_budget") == 1
    assert sample(app, "llm_router_attempts_total", model="broken", outcome="over_budget") == 1
    assert sample(app, "llm_router_attempts_total", model="fast", outcome="over_budget") == 1
    assert sample(app, "llm_router_attempt_duration_seconds_count", model="fast") == 0


def test_the_metrics_endpoint_serves_prometheus_text_with_fresh_spend():
    app, client = make(**WITH_CLIENTS)
    with client:
        chat(client, headers={"Authorization": "Bearer billing-secret"})
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'llm_router_requests_total{result="ok"} 1.0' in text
    assert 'llm_router_limit_usd{scope="global"} 1.0' in text
    assert 'llm_router_limit_usd{scope="client:billing"} 0.5' in text
    assert 'llm_router_spent_usd{scope="client:billing"}' in text
    assert sample(app, "llm_router_spent_usd", scope="global") > 0
    # support has no cap of its own, so it has no limit series.
    assert 'llm_router_limit_usd{scope="client:support"}' not in text


def test_unknown_model_names_never_become_labels():
    app, client = make()
    with client:
        for name in ("a", "b", "c"):
            chat(client, model=name)
        text = client.get("/metrics").text
    assert 'model="a"' not in text
    assert sample(app, "llm_router_requests_total", result="invalid") == 3
