"""Shared test helpers. Nothing here touches the network."""

from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_router.core.config import RouterConfig, parse_config

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "router.yaml"


def minimal_config_data() -> dict[str, Any]:
    """The smallest config worth testing against: two simulated models and one policy."""
    return {
        "providers": {"sim": {"type": "simulated"}},
        "models": {
            "small": {
                "provider": "sim",
                "model": "sim-small",
                "input_per_million": 1,
                "output_per_million": 5,
                "max_output_tokens": 1000,
            },
            "large": {
                "provider": "sim",
                "model": "sim-large",
                "input_per_million": 4,
                "output_per_million": 20,
                "max_output_tokens": 2000,
            },
        },
        "policies": {"default": ["small", "large"]},
        "default_policy": "default",
    }


@pytest.fixture
def example_data() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


@pytest.fixture
def example_config(example_data: dict[str, Any]) -> RouterConfig:
    return parse_config(example_data)


@pytest.fixture
def minimal_data() -> dict[str, Any]:
    return copy.deepcopy(minimal_config_data())


@pytest.fixture
def anyio_backend() -> str:
    # anyio ships with httpx and starlette; its pytest plugin runs the async tests on asyncio.
    return "asyncio"


def redis_url() -> str | None:
    """The Redis to test against. CI sets REQUIRE_REDIS so a missing Redis fails, not skips."""
    url = os.environ.get("REDIS_URL")
    if not url and os.environ.get("REQUIRE_REDIS"):
        pytest.fail("REQUIRE_REDIS is set but REDIS_URL is not")
    return url


@pytest.fixture
async def redis_store_factory():
    """Makes Redis stores that share one key prefix, as several routers would, and deletes every
    key they wrote afterwards. None when there is no Redis to test against."""
    url = redis_url()
    if not url:
        yield None
        return
    from llm_router.core.redis_store import RedisSpendStore

    prefix = f"llm-router-test:{uuid.uuid4().hex}:"
    stores = []

    def make():
        store = RedisSpendStore.from_url(url, key_prefix=prefix)
        stores.append(store)
        return store

    yield make
    cleanup = make()
    async for key in cleanup._client.scan_iter(match=f"{prefix}*"):
        await cleanup._client.delete(key)
    for store in stores:
        await store.close()
