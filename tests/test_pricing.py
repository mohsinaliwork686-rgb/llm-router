from decimal import Decimal

import pytest

from llm_router.core.config import ModelConfig
from llm_router.core.pricing import (
    cost_micros,
    format_usd,
    micros_to_usd,
    output_limit,
    usd_to_micros,
    worst_case_micros,
)


def model(input_price: str, output_price: str, max_output: int = 8192) -> ModelConfig:
    return ModelConfig(
        name="m",
        provider="p",
        model="m",
        input_per_million=Decimal(input_price),
        output_per_million=Decimal(output_price),
        max_output_tokens=max_output,
    )


HAIKU = model("1.00", "5.00")
OPUS = model("4.00", "20.00")
FREE = model("0", "0")


def test_hand_worked_costs():
    # 1,000 input tokens at $1/M is $0.001; 500 output at $5/M is $0.0025. Total $0.0035.
    assert cost_micros(HAIKU, 1_000, 500) == 3_500
    # 12,345 in at $4/M plus 678 out at $20/M: 49,380 + 13,560 micro-dollars.
    assert cost_micros(OPUS, 12_345, 678) == 62_940


def test_a_million_tokens_costs_the_listed_price():
    assert cost_micros(HAIKU, 1_000_000, 0) == usd_to_micros(Decimal("1.00"))
    assert cost_micros(HAIKU, 0, 1_000_000) == usd_to_micros(Decimal("5.00"))


def test_fractions_of_a_micro_dollar_round_up():
    cheap = model("0.15", "0.60")
    # 7 tokens at 0.15 micro-dollars each is 1.05, which must not round down to 1.
    assert cost_micros(cheap, 7, 0) == 2
    assert cost_micros(cheap, 0, 1) == 1


def test_zero_tokens_cost_nothing():
    assert cost_micros(OPUS, 0, 0) == 0


def test_free_models_cost_nothing():
    assert cost_micros(FREE, 1_000_000, 1_000_000) == 0


def test_negative_tokens_are_refused():
    with pytest.raises(ValueError):
        cost_micros(HAIKU, -1, 0)


def test_no_drift_over_a_million_small_additions():
    one_call = cost_micros(HAIKU, 3, 2)  # 13 micro-dollars
    total = 0
    for _ in range(1_000_000):
        total += one_call
    assert total == 13_000_000
    assert micros_to_usd(total) == Decimal("13")
    # The float version of the same sum does drift, which is why money is not a float.
    float_total = 0.0
    for _ in range(1_000_000):
        float_total += 0.000013
    assert float_total != 13.0


def test_output_limit_uses_the_model_maximum_by_default():
    assert output_limit(HAIKU, None) == 8192


def test_output_limit_never_exceeds_the_model_maximum():
    assert output_limit(HAIKU, 100) == 100
    assert output_limit(HAIKU, 1_000_000) == 8192


def test_output_limit_rejects_non_positive():
    with pytest.raises(ValueError):
        output_limit(HAIKU, 0)


def test_worst_case_prices_every_allowed_output_token():
    assert worst_case_micros(HAIKU, 2_000, 8192) == 2_000 * 1 + 8192 * 5


def test_usd_conversion_rounds_caps_down():
    assert usd_to_micros(Decimal("25")) == 25_000_000
    assert usd_to_micros(Decimal("0.0000019")) == 1


def test_format_keeps_small_amounts_visible():
    assert format_usd(1_234) == "0.001234"
    assert format_usd(0) == "0.000000"
    assert format_usd(25_000_000) == "25.000000"
