"""Load and validate the router's YAML configuration.

Everything is checked on startup, including references between sections: a policy naming a
model that does not exist, a model naming a provider that does not exist, a rule pointing at
a missing policy. Problems are reported together, each with the path to the entry that is
wrong, rather than surfacing on whichever request first trips over them. Shape problems (a
missing field, a negative price) come first; references are checked once the shapes are valid.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

#: The one model name a client may send that is neither a model nor a policy.
AUTO = "auto"


class ConfigError(Exception):
    """The configuration cannot be used. The message lists every problem found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"invalid router config ({len(problems)} problem(s)):\n{lines}")


class _Strict(BaseModel):
    # Unknown keys are almost always typos ("input_per_milion"), so they are errors.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(_Strict):
    type: Literal["openai_compatible", "anthropic", "simulated"]
    base_url: str | None = None
    #: The environment variable holding this provider's key. For type anthropic it may be left
    #: out, and the SDK then finds credentials itself (ANTHROPIC_API_KEY, or an `ant` profile).
    api_key_env: str | None = None
    #: Only for type openai_compatible. Newer OpenAI models want max_completion_tokens.
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # Only for type "simulated": how to behave on purpose.
    fail_with: int | None = Field(default=None, ge=400, le=599)
    fail_times: int = Field(default=0, ge=0)
    reply: str = "This is a simulated reply."
    latency_ms: int = Field(default=0, ge=0)


class ModelConfig(_Strict):
    name: str
    provider: str
    model: str
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    max_output_tokens: int = Field(gt=0)
    #: Leave out temperature and top_p when calling this model. Newer Claude models refuse them,
    #: and most OpenAI-style clients send them by habit.
    drop_sampling_params: bool = False


class RuleCondition(_Strict):
    tag: str | None = None
    prompt_tokens_over: int | None = Field(default=None, ge=0)
    prompt_tokens_under: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _not_empty(self) -> RuleCondition:
        conditions = (self.tag, self.prompt_tokens_over, self.prompt_tokens_under)
        if all(condition is None for condition in conditions):
            raise ValueError(
                "a rule needs a condition: tag, prompt_tokens_over or prompt_tokens_under"
            )
        return self


class Rule(_Strict):
    when: RuleCondition
    use: str


class BudgetConfig(_Strict):
    #: The cap across every client. None means spend is tracked but not limited.
    daily_usd: Decimal | None = Field(default=None, gt=0)
    #: "reject" refuses over-budget requests. "warn" lets them through and flags them, which is
    #: useful for a week of watching what a new cap would have blocked before turning it on.
    on_exceeded: Literal["reject", "warn"] = "reject"
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown timezone {value!r}") from error
        return value


class ClientConfig(_Strict):
    name: str
    key_env: str
    daily_usd: Decimal | None = Field(default=None, gt=0)


class RetryConfig(_Strict):
    attempts_per_model: int = Field(default=2, ge=1)
    backoff_ms: int = Field(default=200, ge=0)
    timeout_seconds: float = Field(default=60, gt=0)
    #: Honour a provider's Retry-After when it is at most this long; otherwise move on.
    max_retry_after_seconds: float = Field(default=5, ge=0)


class RouterConfig(_Strict):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    policies: dict[str, list[str]]
    rules: list[Rule] = Field(default_factory=list)
    default_policy: str | None = None
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    clients: list[ClientConfig] = Field(default_factory=list)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def _references(self) -> RouterConfig:
        problems: list[str] = []

        for name, provider in self.providers.items():
            if provider.type == "openai_compatible" and not provider.base_url:
                problems.append(f"providers.{name}: type openai_compatible needs a base_url")
            if provider.type != "openai_compatible" and provider.max_tokens_field != "max_tokens":
                problems.append(
                    f"providers.{name}: max_tokens_field only applies to type openai_compatible"
                )
            if provider.type != "simulated" and (provider.fail_with or provider.fail_times):
                problems.append(
                    f"providers.{name}: fail_with and fail_times only apply to type simulated"
                )

        for name, model in self.models.items():
            if model.provider not in self.providers:
                problems.append(f"models.{name}.provider: unknown provider {model.provider!r}")

        for name, members in self.policies.items():
            if not members:
                problems.append(f"policies.{name}: a policy needs at least one model")
            for index, member in enumerate(members):
                if member not in self.models:
                    problems.append(f"policies.{name}[{index}]: unknown model {member!r}")

        # A client's "model" field can name either, so the two namespaces must not overlap.
        for clash in sorted(set(self.policies) & set(self.models)):
            problems.append(f"{clash!r} is both a model and a policy; rename one")
        for section, names in (("models", self.models), ("policies", self.policies)):
            if AUTO in names:
                problems.append(f"{section}.{AUTO}: 'auto' is reserved for rule-based routing")

        for index, rule in enumerate(self.rules):
            if rule.use not in self.policies:
                problems.append(f"rules[{index}].use: unknown policy {rule.use!r}")
        if self.default_policy is not None and self.default_policy not in self.policies:
            problems.append(f"default_policy: unknown policy {self.default_policy!r}")

        seen_names: set[str] = set()
        seen_keys: set[str] = set()
        for index, client in enumerate(self.clients):
            if client.name in seen_names:
                problems.append(f"clients[{index}].name: duplicate client {client.name!r}")
            if client.key_env in seen_keys:
                problems.append(
                    f"clients[{index}].key_env: {client.key_env!r} is used by two clients"
                )
            seen_names.add(client.name)
            seen_keys.add(client.key_env)

        if problems:
            raise ConfigError(problems)
        return self


def _format_validation_error(error: ValidationError) -> list[str]:
    problems = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"]) or "(top level)"
        message = item["msg"].removeprefix("Value error, ")
        problems.append(f"{path}: {message}")
    return problems


def parse_config(data: Any) -> RouterConfig:
    """Validate already-parsed YAML. Raises ConfigError listing every problem."""
    if not isinstance(data, dict):
        raise ConfigError(["the config must be a mapping at the top level"])

    # Each model carries its own name, so the rest of the code never needs the dict key.
    models = data.get("models")
    if isinstance(models, dict):
        named = {
            name: {**spec, "name": name} if isinstance(spec, dict) else spec
            for name, spec in models.items()
        }
        data = {**data, "models": named}

    # ConfigError is not a ValueError, so pydantic lets the reference check's error through as is.
    try:
        return RouterConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_format_validation_error(error)) from None


def load_config(path: Path | str) -> RouterConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError([f"config file not found: {path}"])
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError([f"{path} is not valid YAML: {error}"]) from None
    return parse_config(data)
