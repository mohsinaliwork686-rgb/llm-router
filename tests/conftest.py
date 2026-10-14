"""Shared test helpers. Nothing here touches the network."""

from __future__ import annotations

import copy
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
