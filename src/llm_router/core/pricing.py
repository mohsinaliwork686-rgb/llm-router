"""Turn token counts into money.

Money is held as integer micro-dollars: millionths of a dollar. Integers add up exactly, where a
day of tiny float additions drifts, and Redis can increment them atomically.

The units line up neatly. A price of X dollars per million tokens is X micro-dollars per token,
so a call's cost is just tokens times price. Prices are Decimals, so the product is exact, and it
is rounded up once at the end: a fraction of a micro-dollar is never dropped in the caller's favour.
"""

from __future__ import annotations

import math
from decimal import Decimal

from llm_router.core.config import ModelConfig

MICROS_PER_USD = 1_000_000


def cost_micros(model: ModelConfig, input_tokens: int, output_tokens: int) -> int:
    """What a call with these token counts costs on this model, rounded up to a micro-dollar."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    exact = input_tokens * model.input_per_million + output_tokens * model.output_per_million
    return math.ceil(exact)


def output_limit(model: ModelConfig, requested: int | None) -> int:
    """How many output tokens a call may produce: what was asked for, capped by the model's limit.

    This is the number sent to the provider as max_tokens, so it is also a hard bound on the
    output side of the worst-case cost.
    """
    if requested is None:
        return model.max_output_tokens
    if requested <= 0:
        raise ValueError("max_tokens must be positive")
    return min(requested, model.max_output_tokens)


def worst_case_micros(model: ModelConfig, prompt_tokens: int, max_output_tokens: int) -> int:
    """The most a call can cost: the estimated prompt, plus every output token it is allowed."""
    return cost_micros(model, prompt_tokens, max_output_tokens)


def usd_to_micros(usd: Decimal) -> int:
    """Dollars to micro-dollars. Rounds down, so a cap is never loosened by rounding."""
    return math.floor(usd * MICROS_PER_USD)


def micros_to_usd(micros: int) -> Decimal:
    return Decimal(micros).scaleb(-6)


def format_usd(micros: int) -> str:
    """Six decimal places, so small costs stay visible: 1234 micro-dollars is '0.001234'."""
    return f"{micros_to_usd(micros):.6f}"
