"""Shared config and request helpers for the HTTP tests."""

KEYS = {"KEY_BILLING": "billing-secret", "KEY_SUPPORT": "support-secret"}


def config_data(**overrides):
    data = {
        "providers": {
            "sim": {"type": "simulated", "reply": "Hello from the simulator."},
            "down": {"type": "simulated", "fail_with": 503},
            "picky": {"type": "simulated", "fail_with": 400},
        },
        "models": {
            "fast": {
                "provider": "sim",
                "model": "sim-fast",
                "input_per_million": 1,
                "output_per_million": 5,
                "max_output_tokens": 100,
            },
            "broken": {
                "provider": "down",
                "model": "sim-broken",
                "input_per_million": 1,
                "output_per_million": 5,
                "max_output_tokens": 100,
            },
            "fussy": {
                "provider": "picky",
                "model": "sim-fussy",
                "input_per_million": 1,
                "output_per_million": 5,
                "max_output_tokens": 100,
            },
        },
        "policies": {
            "resilient": ["broken", "fast"],
            "dead": ["broken"],
            "strict": ["fussy", "fast"],
        },
        "rules": [{"when": {"tag": "batch"}, "use": "resilient"}],
        "default_policy": "dead",
        "budget": {"daily_usd": 1},
        "retry": {"attempts_per_model": 2, "backoff_ms": 0},
    }
    data.update(overrides)
    return data


WITH_CLIENTS = {
    "clients": [
        {"name": "billing", "key_env": "KEY_BILLING", "daily_usd": "0.5"},
        {"name": "support", "key_env": "KEY_SUPPORT"},
    ]
}


def chat(client, model="fast", headers=None, **body):
    payload = {"model": model, "messages": [{"role": "user", "content": "Hi there"}], **body}
    return client.post("/v1/chat/completions", json=payload, headers=headers or {})
