from datetime import UTC, datetime

import pytest

from llm_router.core.config import ConfigError, parse_config
from llm_router.core.providers import Failure, build_providers
from llm_router.core.providers.base import classify_status, parse_retry_after
from llm_router.core.providers.claude import ClaudeProvider
from llm_router.core.providers.openai_compatible import OpenAICompatibleProvider
from llm_router.core.providers.simulated import SimulatedProvider


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (408, Failure.RETRY),
        (409, Failure.RETRY),
        (429, Failure.RETRY),
        (500, Failure.RETRY),
        (529, Failure.RETRY),
        (401, Failure.SKIP),
        (402, Failure.SKIP),
        (403, Failure.SKIP),
        (404, Failure.SKIP),
        (400, Failure.REJECT),
        (413, Failure.REJECT),
        (422, Failure.REJECT),
    ],
)
def test_classify_status(status, kind):
    assert classify_status(status) is kind


def test_retry_after_seconds_and_dates():
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("0.5") == 0.5
    assert parse_retry_after("Wed, 14 Oct 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Wed, 14 Oct 2026 11:00:00 GMT", now=now) == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after("nan") is None


def test_build_providers_makes_one_adapter_per_type(example_data):
    example_data["providers"]["sim"] = {"type": "simulated"}
    config = parse_config(example_data)
    providers = build_providers(config, env={})
    assert isinstance(providers["anthropic"], ClaudeProvider)
    assert isinstance(providers["local"], OpenAICompatibleProvider)
    assert isinstance(providers["sim"], SimulatedProvider)


def test_new_provider_fields_are_checked(minimal_data):
    minimal_data["providers"]["sim"]["max_tokens_field"] = "max_completion_tokens"
    with pytest.raises(ConfigError) as caught:
        parse_config(minimal_data)
    assert caught.value.problems == [
        "providers.sim: max_tokens_field only applies to type openai_compatible"
    ]
