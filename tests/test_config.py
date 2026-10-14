from decimal import Decimal

import pytest

from llm_router.core.config import ConfigError, load_config, parse_config
from tests.conftest import EXAMPLE


def problems_for(data) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        parse_config(data)
    return caught.value.problems


def test_example_file_loads_into_typed_objects():
    config = load_config(EXAMPLE)

    assert set(config.policies) == {"cheap", "balanced", "best", "long-context"}
    haiku = config.models["haiku"]
    assert haiku.name == "haiku"
    assert haiku.model == "claude-haiku-4-5"
    assert haiku.input_per_million == Decimal("1.00")
    assert config.budget.daily_usd == Decimal("25.00")
    assert config.clients[0].daily_usd == Decimal("5.00")
    assert config.clients[1].daily_usd is None
    assert config.rules[1].when.prompt_tokens_over == 150000


def test_yaml_floats_become_exact_decimals(minimal_data):
    minimal_data["models"]["small"]["input_per_million"] = 0.1
    config = parse_config(minimal_data)
    assert config.models["small"].input_per_million == Decimal("0.1")


def test_defaults_when_sections_are_left_out(minimal_data):
    config = parse_config(minimal_data)
    assert config.rules == []
    assert config.clients == []
    assert config.budget.daily_usd is None
    assert config.budget.on_exceeded == "reject"
    assert config.budget.timezone == "UTC"
    assert config.retry.attempts_per_model == 2


def test_every_broken_reference_is_reported_at_once(minimal_data):
    minimal_data["models"]["small"]["provider"] = "simm"
    minimal_data["policies"]["default"] = ["small", "lage"]
    minimal_data["rules"] = [{"when": {"tag": "urgent"}, "use": "best"}]
    minimal_data["default_policy"] = "fastest"

    assert problems_for(minimal_data) == [
        "models.small.provider: unknown provider 'simm'",
        "policies.default[1]: unknown model 'lage'",
        "rules[0].use: unknown policy 'best'",
        "default_policy: unknown policy 'fastest'",
    ]


def test_message_lists_each_problem_on_its_own_line(minimal_data):
    minimal_data["policies"]["default"] = ["nope"]
    with pytest.raises(ConfigError) as caught:
        parse_config(minimal_data)
    assert str(caught.value).splitlines() == [
        "invalid router config (1 problem(s)):",
        "  - policies.default[0]: unknown model 'nope'",
    ]


def test_negative_price_names_the_field(minimal_data):
    minimal_data["models"]["large"]["output_per_million"] = -1
    [problem] = problems_for(minimal_data)
    assert problem.startswith("models.large.output_per_million:")
    assert "greater than or equal to 0" in problem


def test_missing_field_names_the_field(minimal_data):
    del minimal_data["models"]["small"]["max_output_tokens"]
    assert problems_for(minimal_data) == ["models.small.max_output_tokens: Field required"]


def test_unknown_keys_are_typos_not_silence(minimal_data):
    minimal_data["models"]["small"]["input_per_milion"] = 1
    [problem] = problems_for(minimal_data)
    assert problem.startswith("models.small.input_per_milion:")


def test_policy_with_no_models(minimal_data):
    minimal_data["policies"]["empty"] = []
    assert problems_for(minimal_data) == ["policies.empty: a policy needs at least one model"]


def test_a_name_cannot_be_both_model_and_policy(minimal_data):
    minimal_data["policies"]["small"] = ["large"]
    assert problems_for(minimal_data) == ["'small' is both a model and a policy; rename one"]


def test_auto_is_reserved(minimal_data):
    minimal_data["policies"]["auto"] = ["small"]
    assert problems_for(minimal_data) == [
        "policies.auto: 'auto' is reserved for rule-based routing"
    ]


def test_rule_needs_a_condition(minimal_data):
    minimal_data["rules"] = [{"when": {}, "use": "default"}]
    [problem] = problems_for(minimal_data)
    assert problem.startswith("rules.0.when: a rule needs a condition")


def test_openai_compatible_needs_base_url(minimal_data):
    minimal_data["providers"]["remote"] = {"type": "openai_compatible"}
    assert problems_for(minimal_data) == [
        "providers.remote: type openai_compatible needs a base_url"
    ]


def test_failure_settings_only_for_simulated(minimal_data):
    minimal_data["providers"]["claude"] = {"type": "anthropic", "fail_with": 500}
    assert problems_for(minimal_data) == [
        "providers.claude: fail_with and fail_times only apply to type simulated"
    ]


def test_unknown_timezone(minimal_data):
    minimal_data["budget"] = {"daily_usd": 10, "timezone": "Mars/Olympus"}
    assert problems_for(minimal_data) == ["budget.timezone: unknown timezone 'Mars/Olympus'"]


def test_zero_budget_is_rejected(minimal_data):
    minimal_data["budget"] = {"daily_usd": 0}
    [problem] = problems_for(minimal_data)
    assert problem.startswith("budget.daily_usd:")


def test_duplicate_clients(minimal_data):
    minimal_data["clients"] = [
        {"name": "app", "key_env": "KEY_A"},
        {"name": "app", "key_env": "KEY_A"},
    ]
    assert problems_for(minimal_data) == [
        "clients[1].name: duplicate client 'app'",
        "clients[1].key_env: 'KEY_A' is used by two clients",
    ]


def test_top_level_must_be_a_mapping():
    assert problems_for(["not", "a", "mapping"]) == [
        "the config must be a mapping at the top level"
    ]


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("providers: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid YAML"):
        load_config(path)
