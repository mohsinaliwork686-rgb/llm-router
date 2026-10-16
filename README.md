# llm-router

[![ci](https://github.com/mohsinaliwork686-rgb/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinaliwork686-rgb/llm-router/actions/workflows/ci.yml)

**One endpoint, many models.** A small service that sits between your application and its model
providers. It routes each request by policy, falls back when a provider fails, and enforces a daily
spending cap that holds even under concurrent load: over-budget requests are refused *before* the
money is spent, not reported after.

It speaks the OpenAI chat-completions API, so an existing application adopts it by changing one
line, its base URL.

---

## The problem

An application that calls language models directly is hard-wired to one provider. When that
provider rate-limits or goes down, the application goes down with it. And when someone ships a
loop that calls the most expensive model ten thousand times, the bill shows up a month later.

The usual answer to the second problem is a dashboard. A dashboard tells you what you spent after
you spent it. That is not a limit.

## What it does

- **Routes by policy.** `model: "cheap"` tries that policy's models in order. `model: "haiku"`
  uses exactly that model. `model: "auto"` lets rules on tags and prompt size pick the policy.
- **Falls back cleanly.** It retries what can recover (rate limits, server errors, timeouts),
  skips what cannot (a revoked key), and stops at once on a request every model would refuse.
- **Enforces a hard daily cap**, across all traffic and per client, and shares it between
  several routers through Redis.
- **Shows its work** in response headers, at `GET /budget`, and as Prometheus metrics.
- **Talks to** Claude through the official Anthropic SDK, to anything OpenAI-compatible (OpenAI,
  Ollama, vLLM, LM Studio and most hosted providers), and to a simulated provider for demos and
  tests.

## Try it in a minute, with no API keys

```console
$ docker compose up
```

That starts Redis, the router on simulated providers, and a short demo that drives it. If port
8080 is taken on your machine, run `ROUTER_PORT=8090 docker compose up`. To see only the demo's
output, start the router first with `docker compose up -d router`, then run
`docker compose run --rm demo`. This is its real output:

```text
1. Fallback: ask the "resilient" policy for an answer
-----------------------------------------------------
   Its models, in order: frontier (down, HTTP 503), backup (key revoked, HTTP 401),
   budget (healthy).

   answer       Hello! This answer came from a simulated model.
   x-router-policy      resilient
   x-router-model       budget
   x-router-attempts    4
   x-router-cost-usd    0.000026

   4 attempts: frontier was retried once (a 503 can be temporary), backup was
   skipped at once (a bad key will not fix itself), and budget answered.

2. The cap: ten expensive requests at the same moment
-----------------------------------------------------
   Daily cap $1.000000, spent so far $0.000026.
   Each request may write up to 4,000 tokens at $75 per million: a worst case of
   about $0.30, although each one really costs a fraction of a cent.

   answered     3
   refused      7  (HTTP 429, before reaching the model)

   why: global daily cap of $1.000000 would be exceeded: $0.000026 spent, $0.900810 reserved by calls in flight, and this request could cost up to $0.300270

   Worst cases are reserved before any call starts, in one atomic step in Redis,
   so concurrent requests cannot jointly break the cap. The answered ones then
   settled at their real cost and released the rest of their reservation.

3. Where the money went
-----------------------
   limit_usd      $1.000000
   spent_usd      $0.003401
   reserved_usd   $0.000000
   remaining_usd  $0.996599

4. The same story in Prometheus metrics (GET /metrics)
------------------------------------------------------
   llm_router_requests_total{result="ok"} 4.0
   llm_router_requests_total{result="over_budget"} 7.0
   llm_router_attempts_total{model="frontier",outcome="failed"} 2.0
   llm_router_attempts_total{model="backup",outcome="failed"} 1.0
   llm_router_attempts_total{model="budget",outcome="ok"} 1.0
   llm_router_attempts_total{model="premium",outcome="over_budget"} 7.0
   llm_router_attempts_total{model="premium",outcome="ok"} 3.0
   llm_router_spent_usd{scope="global"} 0.003401
```

The demo checks what it prints and exits non-zero if the router misbehaves, and CI runs it on
every push.

## Use it from your application

With the OpenAI SDK, change the base URL and pick a policy:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-router-key")

reply = client.chat.completions.create(
    model="cheap",  # a policy, a single model, or "auto"
    messages=[{"role": "user", "content": "Summarise this support ticket."}],
)
```

For `model="auto"`, tag the request so the rules can match it:
`extra_headers={"X-Router-Tags": "urgent"}`.

Or with curl, where the headers show what happened:

```console
$ curl -i http://localhost:8080/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model": "resilient", "messages": [{"role": "user", "content": "Hi"}]}'
HTTP/1.1 200 OK
x-router-model: budget
x-router-attempts: 4
x-router-cost-usd: 0.000022
x-router-policy: resilient
content-type: application/json

{"id":"chatcmpl-d0d1220c9573449491f3e910dbe9aaef","object":"chat.completion","created":1792149267,"model":"sim-budget","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! This answer came from a simulated model."},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":8,"completion_tokens":12,"total_tokens":20}}
```

## Run it for real

```console
$ git clone https://github.com/mohsinaliwork686-rgb/llm-router && cd llm-router
$ pip install ".[anthropic,redis]"
$ cp examples/router.yaml router.yaml        # then edit models, prices, policies, clients
$ export ANTHROPIC_API_KEY=...  ROUTER_KEY_BILLING=...  ROUTER_KEY_SUPPORT=...
$ llm-router --config router.yaml            # listens on 127.0.0.1:8080
```

Or with Docker, mounting your config:

```console
$ docker build -t llm-router .
$ docker run -p 8080:8080 -v "$PWD/router.yaml:/config/router.yaml" \
    -e ANTHROPIC_API_KEY -e ROUTER_KEY_BILLING -e ROUTER_KEY_SUPPORT llm-router
```

## Configuration

One YAML file. [examples/router.yaml](examples/router.yaml) is a full example with Claude and a
local Ollama model; [examples/demo.yaml](examples/demo.yaml) is the demo's.

| Section | What it sets |
|---------|--------------|
| `providers` | Where models live. `type` is `anthropic`, `openai_compatible` (needs a `base_url`) or `simulated`. `api_key_env` names the environment variable holding the key |
| `models` | Each model's provider, the provider's own model id, prices in dollars per million input and output tokens, and `max_output_tokens`. `drop_sampling_params: true` leaves out `temperature` and `top_p` for models that refuse them, such as current Claude models |
| `policies` | Named, ordered lists of models to try |
| `rules`, `default_policy` | For `model: "auto"`: the first rule whose conditions all hold (`tag`, `prompt_tokens_over`, `prompt_tokens_under`) picks the policy, otherwise the default |
| `budget` | `daily_usd` for the global cap, `on_exceeded` (`reject`, or `warn` to watch what a new cap would block), and the `timezone` whose midnight starts a new day |
| `clients` | Each client's `name`, the environment variable holding its key, and an optional `daily_usd` cap inside the global one. Leave the section out to run without keys, for local use |
| `store` | `memory` for one router process, or `redis` to share caps between routers (the URL comes from `REDIS_URL`) |
| `retry` | `attempts_per_model`, `backoff_ms`, `timeout_seconds`, and `max_retry_after_seconds` |

**Prices are yours to set.** They live in the config, never in the code, because prices change.
Check them against your provider's pricing page.

The whole file is checked at startup, and every problem is reported at once with the path to the
entry that is wrong:

```console
$ llm-router --check --config router.yaml
invalid router config (2 problem(s)):
  - policies.cheap[1]: unknown model 'hiaku'
  - rules[0].use: unknown policy 'fastest'
```

Unknown keys are errors too, so a typo such as `input_per_milion` is caught instead of silently
ignored.

## How the spending cap works

Checking "have we spent the budget yet?" before each call is not a cap. The call that passes the
check can cost more than what is left, and ten calls arriving together all pass the same check.

llm-router **reserves before it spends**:

1. Before calling a model, it works out the **worst case** the call could cost: the prompt's
   tokens at the input price, plus every output token the call is allowed (`max_tokens`, capped
   at the model's limit) at the output price.
2. In one atomic step it checks that today's spend, plus everything already reserved, plus this
   worst case, stays within every cap that applies: the global one and the client's own. If not,
   the call is refused without reaching the provider.
3. After the call it **settles**: the reservation is swapped for the real cost from the
   provider's reported usage. A failed call gives its reservation back.

Details that make it hold:

- **Money is counted in integer micro-dollars.** A day of tiny float additions drifts; integers do
  not, and Redis increments them atomically.
- **The output side of the worst case is exact**, because the router sends the provider the same
  `max_tokens` it reserved for. The input side uses a token estimate built to run high (about
  three characters per token, where English averages four). If an estimate ever runs low, the
  overrun is recorded and counted in `llm_router_reservation_shortfall_usd_total` rather than
  hidden.
- **A call that may have been billed counts as spent.** A timeout, or a connection lost after the
  request was sent, keeps its whole reservation, because the provider may have done the work. A
  refused connection gives it back.
- **A model over the cap is skipped, not the whole request**, so a cheaper model further down the
  policy can still answer. The caller gets a 429 only when no candidate fits.
- **It fails closed.** If the spend store cannot be reached, new calls are refused with a 503
  rather than sent without a check.

The trade-off is deliberate: near the limit, a request can be refused although its real cost
would have fitted. A cap that is usually respected is not a cap.

## How fallback works

| What happened | What the router does |
|---------------|----------------------|
| 429, 5xx, a timeout, a dropped connection | Retries the same model with exponential backoff and jitter, up to `attempts_per_model`. A short `Retry-After` is honoured; a long one moves on to the next model instead of waiting |
| 401, 402, 403, 404 | Skips to the next model. A bad key, or a model name the provider does not know, is a problem with that provider or the config, not the request |
| 400, 413, 422 | Returns the error to the caller at once. A malformed request fails on every model, so trying more only costs time and money |
| Every candidate failed | One `502` listing each attempt and why it failed |

The `502` and `429` responses carry `x-should-retry: false`. The router has already retried, and
this stops the OpenAI SDKs from retrying on top, which would turn one request into three rounds of
fallback.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | The proxy |
| `GET /v1/models` | Everything a client may put in `model` |
| `GET /budget` | Today's spend against each cap. A client sees the global cap and its own, never another client's |
| `GET /metrics` | Prometheus metrics |
| `GET /healthz` | Liveness, with no key needed |

Every answer carries `X-Router-Model`, `X-Router-Attempts` and `X-Router-Cost-USD`, plus
`X-Router-Policy` when a policy chose the model, and `X-Router-Budget-Warning` when `on_exceeded`
is `warn` and a cap was crossed.

## Metrics

| Metric | What it counts |
|--------|----------------|
| `llm_router_requests_total{result}` | Requests by result: `ok`, `invalid`, `unauthorized`, `rejected`, `over_budget`, `failed`, `unavailable` |
| `llm_router_attempts_total{model,outcome}` | Each model the router tried: `ok`, `failed`, or `over_budget` (not called) |
| `llm_router_request_duration_seconds`, `llm_router_attempt_duration_seconds` | Latency of whole requests, and of each provider call |
| `llm_router_tokens_total{model,direction}` | Input and output tokens, as the provider reported them |
| `llm_router_cost_usd_total{model,client}` | Money spent |
| `llm_router_spent_usd`, `llm_router_reserved_usd`, `llm_router_limit_usd` `{scope}` | Today's spend, what is reserved now, and the cap, globally and per client |
| `llm_router_reservation_shortfall_usd_total{model}` | How far real costs ran over their reservations. Should stay at zero |

Labels only ever carry names from the config, never anything a caller typed, so bad requests
cannot multiply the number of series. Keep `/metrics` on an internal network: it shows spend per
client.

## Running several routers

With `store: {type: redis}`, every router reads and writes the same caps. Reserving runs as one Lua
script, which Redis executes start to finish with nothing in between, so the check and the hold
are a single atomic step however many routers call at once. The tests check this: four stores
making 600 simultaneous reservations admit exactly as many as the cap allows, and a deliberately
racy version of the same store fails that test.

Spend is kept per day and expires a few days later. If a router dies mid-call, its reservation
stays held until then, which only makes the cap stricter. A single Redis instance (or a primary
with replicas) is expected; Redis Cluster is not supported.

## Providers

- **`anthropic`** uses the official Anthropic SDK. Leading system messages become the system
  prompt, `stop` becomes `stop_sequences`, and a refusal comes back as
  `finish_reason: "content_filter"`. Claude's server-side refusal fallback is deliberately not
  enabled: it can hand the request to a model with different prices, which the cap's arithmetic
  would not see.
- **`openai_compatible`** posts to `{base_url}/chat/completions`. Set
  `max_tokens_field: max_completion_tokens` for newer OpenAI models that require it.
- **`simulated`** calls nothing. `fail_with` and `fail_times` make it fail on purpose (the first
  N calls, or every call when `fail_times` is 0), `latency_ms` slows it down, and `reply` sets its
  answer.

## What it does not do

- **No streaming, yet.** With streaming, part of an answer has already gone out when a provider
  fails, so falling back cleanly is a different problem. It is the next thing to build.
- **No tools, images or multiple choices.** Requests that use them are refused with a clear 400
  instead of being forwarded with those parts silently dropped.
- **No billing or accounts.** Per-client caps, yes; invoicing, no.

## Development

```console
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e ".[anthropic,redis,dev]"
$ pytest                                         # the Redis tests skip without a Redis
$ docker run -d -p 6379:6379 redis:7-alpine
$ REDIS_URL=redis://localhost:6379/0 pytest      # all 236 tests
$ ruff check . && ruff format --check .
```

No test touches the network: providers are faked at the HTTP layer and time is passed in. CI runs
lint, the tests on Python 3.11 to 3.14 against a real Redis, the tests on macOS and Windows, and
the Docker demo.

```text
src/llm_router/
  main.py              the HTTP service: routes and wiring only
  schemas.py           the OpenAI request and response shapes
  core/
    config.py          load and validate the YAML
    policy.py          request -> ordered list of models. Pure
    tokens.py          the conservative token estimate
    pricing.py         tokens -> integer micro-dollars
    budget.py          reserve, settle, release; the day boundary
    spend.py           the spend store interface, and the in-memory store
    redis_store.py     the shared store, with the atomic Lua reserve
    router.py          the fallback loop
    metrics.py         Prometheus metrics
    providers/         one adapter per provider type
```

The design, and the reasoning behind each decision, is in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## License

MIT. See [LICENSE](LICENSE).
