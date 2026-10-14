"""Turn a request into an ordered list of models to try.

Pure: no I/O, no clock, no randomness. Given the same config and request it always returns the
same answer, which is what makes routing easy to reason about and to test.

The request's model field decides the route:

- a policy name tries that policy's models in order;
- a model name uses exactly that model, with no fallback;
- "auto" checks the rules in order, and the first rule that matches picks the policy. If none
  matches, default_policy is used.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from llm_router.core.config import AUTO, ModelConfig, RouterConfig, Rule


class RoutingError(Exception):
    """The request names nothing the router can route to. The caller's mistake: an HTTP 400."""


@dataclass(frozen=True)
class Route:
    candidates: tuple[ModelConfig, ...]
    #: The policy that produced the candidates, or None when one model was named directly.
    policy: str | None = None
    #: For "auto", the index of the rule that matched, or None when the default was used.
    rule: int | None = None


def rule_matches(rule: Rule, tags: Collection[str], prompt_tokens: int) -> bool:
    """A rule matches when every condition it sets holds."""
    when = rule.when
    return all(
        (
            when.tag is None or when.tag in tags,
            when.prompt_tokens_over is None or prompt_tokens > when.prompt_tokens_over,
            when.prompt_tokens_under is None or prompt_tokens < when.prompt_tokens_under,
        )
    )


def route(
    config: RouterConfig,
    requested: str,
    *,
    tags: Collection[str] = (),
    prompt_tokens: int = 0,
) -> Route:
    if requested == AUTO:
        return _auto(config, tags, prompt_tokens)
    if requested in config.policies:
        return _policy(config, requested)
    if requested in config.models:
        return Route(candidates=(config.models[requested],))
    raise RoutingError(_unknown_message(config, requested))


def _policy(config: RouterConfig, name: str, rule: int | None = None) -> Route:
    candidates = tuple(config.models[member] for member in config.policies[name])
    return Route(candidates=candidates, policy=name, rule=rule)


def _auto(config: RouterConfig, tags: Collection[str], prompt_tokens: int) -> Route:
    for index, rule in enumerate(config.rules):
        if rule_matches(rule, tags, prompt_tokens):
            return _policy(config, rule.use, rule=index)
    if config.default_policy is None:
        raise RoutingError(
            "model 'auto' matched no rule, and the router has no default_policy to fall back on"
        )
    return _policy(config, config.default_policy)


def _unknown_message(config: RouterConfig, requested: str) -> str:
    policies = ", ".join(sorted(config.policies)) or "none"
    models = ", ".join(sorted(config.models)) or "none"
    return (
        f"unknown model {requested!r}. Policies: {policies}. Models: {models}. "
        f"Or 'auto' to route by rules."
    )


def routable_names(config: RouterConfig) -> list[str]:
    """Everything a client may put in the model field, for GET /v1/models."""
    return [AUTO, *sorted(config.policies), *sorted(config.models)]
