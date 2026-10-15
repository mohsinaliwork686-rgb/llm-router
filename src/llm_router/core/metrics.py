"""Prometheus metrics: what was asked, what answered, what it cost, how long it took.

Labels only ever carry names from the config (models, policies, clients) or a small fixed set of
results, never anything a caller typed, so a stream of bad requests cannot blow up the number of
time series.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from llm_router.core.pricing import MICROS_PER_USD
from llm_router.core.router import Attempt, RoutedCompletion

#: Seconds. Model calls range from well under a second to over a minute.
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 120)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        # Each app gets its own registry, so tests and several apps in one process never clash.
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "llm_router_requests",
            "Chat requests by result: ok, invalid, unauthorized, rejected, over_budget, failed, "
            "unavailable.",
            ["result"],
            registry=r,
        )
        self.request_seconds = Histogram(
            "llm_router_request_duration_seconds",
            "Time to answer a chat request, fallback included.",
            ["result"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.attempts = Counter(
            "llm_router_attempts",
            "Calls the router tried, per model: ok, failed, or over_budget (not called).",
            ["model", "outcome"],
            registry=r,
        )
        self.attempt_seconds = Histogram(
            "llm_router_attempt_duration_seconds",
            "Time each provider call took, successful or not.",
            ["model"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "llm_router_tokens",
            "Tokens used, as reported by the provider.",
            ["model", "direction"],
            registry=r,
        )
        self.cost = Counter(
            "llm_router_cost_usd",
            "Money spent, including timed-out calls counted at their worst case.",
            ["model", "client"],
            registry=r,
        )
        self.shortfall = Counter(
            "llm_router_reservation_shortfall_usd",
            "How far real costs ran over their reservations. Should stay at zero; anything else "
            "means the token estimate ran low.",
            ["model"],
            registry=r,
        )
        self.spent = Gauge("llm_router_spent_usd", "Spent today, per cap.", ["scope"], registry=r)
        self.reserved = Gauge(
            "llm_router_reserved_usd", "Held by calls in flight, per cap.", ["scope"], registry=r
        )
        self.limit = Gauge(
            "llm_router_limit_usd", "Today's cap, where one is set.", ["scope"], registry=r
        )

    def request_done(self, result: str, seconds: float) -> None:
        self.requests.labels(result).inc()
        self.request_seconds.labels(result).observe(seconds)

    def attempts_made(self, attempts: Iterable[Attempt], client: str | None) -> None:
        for attempt in attempts:
            self.attempts.labels(attempt.model, attempt.outcome).inc()
            if attempt.outcome == "over_budget":
                continue
            self.attempt_seconds.labels(attempt.model).observe(attempt.seconds)
            if attempt.cost_micros:
                self.cost.labels(attempt.model, client or "-").inc(
                    attempt.cost_micros / MICROS_PER_USD
                )

    def answered(self, routed: RoutedCompletion) -> None:
        usage = routed.completion.usage
        self.tokens.labels(routed.model.name, "input").inc(usage.input_tokens)
        self.tokens.labels(routed.model.name, "output").inc(usage.output_tokens)
        if routed.shortfall_micros:
            self.shortfall.labels(routed.model.name).inc(routed.shortfall_micros / MICROS_PER_USD)

    def budget_status(self, status: Mapping[str, Any]) -> None:
        """Refresh the spend gauges from GET /budget's numbers, just before a scrape."""
        scopes = {"global": status["global"]}
        scopes.update({f"client:{name}": row for name, row in status["clients"].items()})
        for scope, row in scopes.items():
            self.spent.labels(scope).set(float(row["spent_usd"]))
            self.reserved.labels(scope).set(float(row["reserved_usd"]))
            if row["limit_usd"] is not None:
                self.limit.labels(scope).set(float(row["limit_usd"]))

    def render(self) -> bytes:
        return generate_latest(self.registry)
