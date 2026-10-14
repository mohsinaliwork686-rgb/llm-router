"""Provider adapters: one per provider type, all behind the same interface."""

from __future__ import annotations

from collections.abc import Mapping

from llm_router.core.config import ConfigError, RouterConfig
from llm_router.core.providers.base import Failure, Provider, ProviderError
from llm_router.core.providers.openai_compatible import OpenAICompatibleProvider
from llm_router.core.providers.simulated import SimulatedProvider

__all__ = ["Failure", "Provider", "ProviderError", "build_providers"]


def build_providers(config: RouterConfig, env: Mapping[str, str]) -> dict[str, Provider]:
    """One adapter per configured provider. Missing keys surface per call, not here, so a router
    with one unconfigured provider still serves the others."""
    timeout = config.retry.timeout_seconds
    providers: dict[str, Provider] = {}
    for name, provider in config.providers.items():
        if provider.type == "simulated":
            providers[name] = SimulatedProvider(provider)
        elif provider.type == "openai_compatible":
            providers[name] = OpenAICompatibleProvider(provider, env, timeout)
        else:
            try:
                from llm_router.core.providers.claude import ClaudeProvider
            except ImportError:
                raise ConfigError(
                    [
                        f"providers.{name}: type anthropic needs the Anthropic SDK; "
                        "install it with: pip install 'llm-router[anthropic]'"
                    ]
                ) from None
            providers[name] = ClaudeProvider(provider, env, timeout)
    return providers
