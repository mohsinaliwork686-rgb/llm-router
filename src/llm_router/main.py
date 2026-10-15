"""The HTTP service: routes and wiring only. The logic lives in llm_router.core."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from llm_router import __version__
from llm_router.core.budget import BudgetGuard, Clock, utc_now
from llm_router.core.config import ConfigError, RouterConfig, load_config
from llm_router.core.policy import RoutingError, routable_names
from llm_router.core.pricing import format_usd
from llm_router.core.providers import Provider, build_providers
from llm_router.core.router import AllFailed, OverBudget, RequestRejected, Router, RoutingFailed
from llm_router.core.spend import MemorySpendStore, SpendStore, SpendStoreError
from llm_router.schemas import ChatCompletionRequest, ErrorType, completion_body, error_body

log = logging.getLogger("llm_router")


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        error_type: ErrorType,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = error_body(message, error_type, code, **extra)
        self.headers = dict(headers or {})


def client_keys(config: RouterConfig, env: Mapping[str, str]) -> dict[str, str]:
    """Map each client's key to its name. A missing key fails startup rather than locking a client
    out at its first request."""
    problems = []
    keys: dict[str, str] = {}
    for index, client in enumerate(config.clients):
        key = env.get(client.key_env, "")
        if not key:
            problems.append(f"clients[{index}]: environment variable {client.key_env} is not set")
        elif key in keys:
            problems.append(f"clients[{index}]: {client.name} has the same key as {keys[key]}")
        else:
            keys[key] = client.name
    if problems:
        raise ConfigError(problems)
    return keys


def build_store(config: RouterConfig, env: Mapping[str, str]) -> SpendStore:
    settings = config.store
    if settings.type == "memory":
        return MemorySpendStore()
    url = env.get(settings.url_env, "")
    if not url:
        raise ConfigError([f"store: environment variable {settings.url_env} is not set"])
    try:
        from llm_router.core.redis_store import RedisSpendStore
    except ImportError:
        raise ConfigError(
            [
                "store: type redis needs the Redis client; "
                "install it with: pip install 'llm-router[redis]'"
            ]
        ) from None
    return RedisSpendStore.from_url(url, settings.key_prefix, settings.max_connections)


def create_app(
    config: RouterConfig,
    *,
    env: Mapping[str, str] | None = None,
    providers: Mapping[str, Provider] | None = None,
    store: SpendStore | None = None,
    clock: Clock = utc_now,
    router_options: Mapping[str, Any] | None = None,
) -> FastAPI:
    env = os.environ if env is None else env
    keys = client_keys(config, env)
    providers = dict(providers) if providers is not None else build_providers(config, env)
    store = store if store is not None else build_store(config, env)
    guard = BudgetGuard(config.budget, config.clients, store, clock=clock)
    router = Router(config, providers, guard, **(router_options or {}))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not config.clients:
            log.warning("no clients are configured, so the router accepts requests without a key")
        yield
        for provider in providers.values():
            await provider.close()
        await store.close()

    app = FastAPI(title="llm-router", version=__version__, lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(error.body, status_code=error.status, headers=error.headers)

    def authenticate(request: Request) -> str | None:
        """The calling client's name, or None when the router runs open."""
        if not keys:
            return None
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and token:
            for key, name in keys.items():
                if hmac.compare_digest(token.encode(), key.encode()):
                    return name
        raise ApiError(
            401, "missing or unknown API key", "authentication_error", code="invalid_api_key"
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        client = authenticate(request)
        try:
            payload = json.loads(await request.body())
        except ValueError:
            raise ApiError(
                400, "the request body is not valid JSON", "invalid_request_error"
            ) from None
        try:
            body = ChatCompletionRequest.model_validate(payload)
        except ValidationError as error:
            raise ApiError(400, _validation_message(error), "invalid_request_error") from None

        tags = {tag.strip() for tag in request.headers.get("x-router-tags", "").split(",")}
        tags.discard("")
        try:
            routed = await router.complete(
                body.model, body.to_chat_request(), client=client, tags=tags
            )
        except RoutingError as error:
            raise ApiError(
                400, str(error), "invalid_request_error", code="model_not_found"
            ) from None
        except RoutingFailed as error:
            raise _routing_failure(error) from None
        except SpendStoreError:
            log.exception("spend store unavailable")
            raise ApiError(
                503,
                "the spend store is unavailable, so the router cannot check its caps",
                "service_unavailable",
            ) from None

        headers = {
            "x-router-model": routed.model.name,
            "x-router-attempts": str(_calls(routed.attempts)),
            "x-router-cost-usd": format_usd(routed.cost_micros),
        }
        if routed.route.policy:
            headers["x-router-policy"] = routed.route.policy
        if routed.budget_warning:
            headers["x-router-budget-warning"] = routed.budget_warning
        log.info(
            "client=%s requested=%s served_by=%s attempts=%s cost_usd=%s",
            client or "-",
            body.model,
            routed.model.name,
            headers["x-router-attempts"],
            headers["x-router-cost-usd"],
        )
        return JSONResponse(completion_body(routed), headers=headers)

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        authenticate(request)
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "llm-router"}
                for name in routable_names(config)
            ],
        }

    @app.get("/budget")
    async def budget(request: Request) -> dict[str, Any]:
        client = authenticate(request)
        try:
            status = await guard.status()
        except SpendStoreError:
            raise ApiError(503, "the spend store is unavailable", "service_unavailable") from None
        if client is not None:
            # A client sees the shared cap and its own, never another client's spending.
            status["clients"] = {client: status["clients"][client]}
        return status

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.state.router = router
    app.state.guard = guard
    return app


def _calls(attempts) -> int:
    return sum(1 for attempt in attempts if attempt.outcome != "over_budget")


def _routing_failure(error: RoutingFailed) -> ApiError:
    attempts = [attempt.as_dict() for attempt in error.attempts]
    # The router has already retried. Asking the OpenAI SDKs not to retry again stops one request
    # turning into three rounds of fallback.
    headers = {"x-router-attempts": str(_calls(error.attempts)), "x-should-retry": "false"}
    if isinstance(error, RequestRejected):
        return ApiError(
            400, str(error), "invalid_request_error", headers=headers, attempts=attempts
        )
    if isinstance(error, OverBudget):
        return ApiError(
            429,
            str(error),
            "budget_exceeded",
            code="daily_cap_reached",
            headers=headers,
            scope=error.exceeded.scope,
        )
    assert isinstance(error, AllFailed)
    return ApiError(502, str(error), "upstream_error", headers=headers, attempts=attempts)


def _validation_message(error: ValidationError) -> str:
    problems = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        problems.append(f"{path}: {message}" if path else message)
    return "; ".join(problems)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="One endpoint, many models, and a daily spending cap.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("LLM_ROUTER_CONFIG", "router.yaml"),
        help="path to the YAML config (default: $LLM_ROUTER_CONFIG or router.yaml)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on")
    parser.add_argument("--check", action="store_true", help="validate the config and exit")
    parser.add_argument("--version", action="version", version=f"llm-router {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        config = load_config(args.config)
        app = create_app(config)
    except ConfigError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
    if args.check:
        print(
            f"config ok: {len(config.models)} models, {len(config.policies)} policies, "
            f"{len(config.clients)} clients"
        )
        return

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
