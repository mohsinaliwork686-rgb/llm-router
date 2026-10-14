import pytest

from llm_router.core.config import parse_config
from llm_router.core.policy import RoutingError, routable_names, route


def names(result) -> list[str]:
    return [model.name for model in result.candidates]


def test_policy_name_tries_its_models_in_order(example_config):
    result = route(example_config, "cheap")
    assert names(result) == ["llama", "haiku"]
    assert result.policy == "cheap"
    assert result.rule is None


def test_model_name_uses_exactly_that_model(example_config):
    result = route(example_config, "opus")
    assert names(result) == ["opus"]
    assert result.policy is None


def test_auto_matches_a_tag_rule(example_config):
    result = route(example_config, "auto", tags={"urgent"}, prompt_tokens=10)
    assert names(result) == ["opus", "sonnet"]
    assert result.policy == "best"
    assert result.rule == 0


def test_auto_matches_a_size_rule(example_config):
    result = route(example_config, "auto", prompt_tokens=200_000)
    assert result.policy == "long-context"
    assert result.rule == 1


def test_auto_size_rules_are_strict_bounds(example_config):
    # prompt_tokens_over: 150000 means more than 150,000, not 150,000 itself.
    assert route(example_config, "auto", prompt_tokens=150_000).policy == "balanced"
    assert route(example_config, "auto", prompt_tokens=150_001).policy == "long-context"
    # prompt_tokens_under: 500 means fewer than 500.
    assert route(example_config, "auto", prompt_tokens=499).policy == "cheap"
    assert route(example_config, "auto", prompt_tokens=500).policy == "balanced"


def test_auto_falls_back_to_the_default(example_config):
    result = route(example_config, "auto", tags={"something-else"}, prompt_tokens=2_000)
    assert result.policy == "balanced"
    assert result.rule is None


def test_rules_are_evaluated_in_order(example_config):
    # Both the tag rule (first) and the size rule (second) match; the first one wins.
    result = route(example_config, "auto", tags={"urgent"}, prompt_tokens=200_000)
    assert result.policy == "best"


def test_a_rule_needs_every_condition_to_hold(minimal_data):
    minimal_data["policies"]["solo"] = ["large"]
    minimal_data["rules"] = [
        {"when": {"tag": "batch", "prompt_tokens_over": 1_000}, "use": "solo"},
    ]
    config = parse_config(minimal_data)
    assert route(config, "auto", tags={"batch"}, prompt_tokens=5_000).policy == "solo"
    assert route(config, "auto", tags={"batch"}, prompt_tokens=10).policy == "default"
    assert route(config, "auto", tags=set(), prompt_tokens=5_000).policy == "default"


def test_auto_without_a_match_or_default_is_an_error(minimal_data):
    del minimal_data["default_policy"]
    config = parse_config(minimal_data)
    with pytest.raises(RoutingError, match="matched no rule"):
        route(config, "auto")


def test_unknown_name_lists_what_exists(example_config):
    with pytest.raises(RoutingError) as caught:
        route(example_config, "gpt-9")
    assert str(caught.value) == (
        "unknown model 'gpt-9'. Policies: balanced, best, cheap, long-context. "
        "Models: haiku, llama, opus, sonnet. Or 'auto' to route by rules."
    )


def test_names_are_case_sensitive(example_config):
    with pytest.raises(RoutingError):
        route(example_config, "Cheap")


def test_routable_names(example_config):
    assert routable_names(example_config) == [
        "auto",
        "balanced",
        "best",
        "cheap",
        "long-context",
        "haiku",
        "llama",
        "opus",
        "sonnet",
    ]


def test_routing_is_deterministic(example_config):
    first = route(example_config, "auto", tags={"urgent"}, prompt_tokens=42)
    assert all(
        route(example_config, "auto", tags={"urgent"}, prompt_tokens=42) == first
        for _ in range(100)
    )
