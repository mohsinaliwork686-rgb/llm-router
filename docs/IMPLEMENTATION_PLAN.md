# Implementation plan — llm-router

> Written Wednesday 14 October 2026, before any code.
> Week 4 of the 10-week programme. Monday and Tuesday were missed, so the five days of work
> run over three: full scope, longer days. Nothing in this plan was cut for time.

## 1. Problem

An application that calls language models directly ends up hard-wired to one provider and one
model. When that provider rate-limits or has an outage, the application fails. When someone
writes a loop that calls the most expensive model ten thousand times, the bill arrives a month
later.

The usual answers to the second problem are dashboards and alerts. They tell you what you spent
after you spent it. That is not a limit.

`llm-router` is a small HTTP service that sits between an application and its model providers.
The application sends every request to the router instead; the router decides which model handles
it, falls back to another when one fails, records what every call cost, and **refuses to spend
past a daily cap**, before the money is gone rather than after.

## 2. Non-goals

- **No streaming in version one.** Decided on the Monday brief, before any days were missed:
  with streaming, part of the answer has already been sent when a provider fails, so falling back
  cleanly is a different and much harder problem. It is the stretch goal.
- **No web interface.** Configuration is a file; observation is `/metrics`.
- **No billing, invoicing or multi-tenant accounts.** Per-key caps, yes; customer billing, no.
- **No prompt caching or response caching.** That is week 7 (`semcache`), which will plug in here.
- **No fine-tuning, no model hosting.** It routes to models; it does not run them.

## 3. Surface

The application changes one line: its base URL.

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="my-router-key")
client.chat.completions.create(model="cheap", messages=[...])
```

The router speaks the OpenAI chat-completions shape, because almost every client library and
tool already speaks it. That costs nothing and means any existing application can adopt the router
without code changes.

The `model` field picks how the request is routed:

| `model` | Meaning |
|---------|---------|
| a **policy** name, e.g. `cheap` | try that policy's models, in order |
| a **model** name, e.g. `haiku` | use exactly that model, no fallback |
| `auto` | evaluate the routing rules and let the first match pick a policy |

Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | the proxy itself |
| `GET /v1/models` | the policies and models a client may ask for |
| `GET /budget` | today's spend against each cap |
| `GET /metrics` | Prometheus metrics |
| `GET /healthz` | liveness |

Every response carries `X-Router-Model`, `X-Router-Policy`, `X-Router-Attempts` and
`X-Router-Cost-USD` headers, so a caller can see what happened without reading logs.

## 4. Configuration

One YAML file. Prices live here, set by whoever runs the router, rather than baked into the code
— prices change, and a router that quotes last quarter's prices undercounts every request.

```yaml
providers:
  anthropic:
    type: anthropic
    api_key_env: ANTHROPIC_API_KEY
  local:
    type: openai_compatible          # Ollama, vLLM, LM Studio, and most hosted providers
    base_url: http://localhost:11434/v1

models:
  haiku:
    provider: anthropic
    model: claude-haiku-4-5
    input_per_million: 1.00
    output_per_million: 5.00
    max_output_tokens: 8192
  llama:
    provider: local
    model: llama3.1:8b
    input_per_million: 0
    output_per_million: 0
    max_output_tokens: 4096

policies:
  cheap: [llama, haiku]
  best:  [opus, sonnet]

rules:                               # for model="auto"; first match wins
  - when: { tag: urgent }
    use: best
  - when: { prompt_tokens_over: 50000 }
    use: long-context
default_policy: cheap

budget:
  daily_usd: 25.00                   # across every client
  on_exceeded: reject
  timezone: UTC

clients:                             # optional; without it the router runs open, for local use
  - name: billing-app
    key_env: ROUTER_KEY_BILLING
    daily_usd: 5.00                  # a per-client cap inside the global one

retry:
  attempts_per_model: 2
  backoff_ms: 200
  timeout_seconds: 60
```

The config is validated completely on startup. A policy naming a model that does not exist, a
model naming a provider that does not exist, a rule pointing at a missing policy, a negative
price — each fails startup with a message giving the path to the wrong entry, such as
`policies.cheap[1]: unknown model 'hiaku'`, rather than failing the first request that happens to
touch it. Every problem is listed at once, so fixing a config is one pass, not one error at a time.

## 5. Architecture

```
  client --> POST /v1/chat/completions
                 |
                 v
          authenticate        which client is this? (or open mode)
                 |
                 v
          policy engine       request -> ordered list of candidate models. Pure.
                 |
                 v
          for each candidate:
             budget guard     reserve the worst-case cost; refuse if it would cross a cap
             provider call    with a timeout; retry with backoff on 429, 5xx, timeouts
             settle           replace the reservation with the real cost
                 |            on failure: release the reservation, try the next candidate
                 v
          usage recorder      tokens, cost, latency, outcome  -->  /metrics
```

| Module | Job |
|--------|-----|
| `core/config.py` | Load and validate the YAML into typed objects |
| `core/policy.py` | Turn a request into an ordered candidate list. Pure, no I/O |
| `core/pricing.py` | Tokens to cost, in integer micro-dollars |
| `core/budget.py` | Reservations, settlement and the cap decision. Clock passed in |
| `core/spend.py` | Where spend is stored: in memory, or Redis. One interface |
| `core/tokens.py` | A deliberately conservative token estimate for reservations |
| `core/providers/` | One adapter per provider type behind a shared interface. All network calls live here |
| `core/router.py` | The fallback loop: candidates, retries, backoff, settlement |
| `core/metrics.py` | Prometheus counters and histograms |
| `main.py` | The FastAPI app: routes and wiring only |

Everything under `core/` imports without FastAPI, so it is tested with no server and no network.

## 6. The spending cap, done properly

This is the part that makes the project worth building, so it gets the most care.

**The naive version** checks "have we spent the budget yet?" before each call. It overshoots in
two ways:

1. The request that passes the check can itself cost more than what is left.
2. Ten requests arriving together all see the same "not yet", and all go through.

**The version built here reserves before it spends:**

1. Before calling a model, work out the **worst case** the call could cost: the estimated input
   tokens at the input price, plus the request's `max_tokens` (or the model's configured maximum)
   at the output price.
2. Atomically check that today's spend, plus everything currently reserved, plus this worst case,
   stays under every cap that applies — the global one and the client's own. If not, refuse
   *without calling the provider*.
3. Otherwise hold that reservation, make the call, then **settle**: swap the reservation for the
   real cost from the provider's reported usage. A failed call releases its reservation.

Because the worst case is reserved up front and the check-and-reserve is a single atomic step,
the cap cannot be exceeded, even by concurrent requests. The cost is that a request near the limit
can be refused while its *actual* cost would have fitted. That trade is deliberate: a cap that is
usually respected is not a cap.

Two details that make it hold:

- **Money is stored as integer micro-dollars**, not floating-point dollars. A day of tiny requests
  added as floats drifts; integers do not, and Redis increments them atomically.
- **The token estimate for reservations errs high** (about three characters per token, where
  English is closer to four). An estimate that runs low would let a request reserve less than it
  can cost. Settlement uses the provider's real counts, so erring high only costs headroom.

The day boundary follows the configured timezone, and the clock is passed in, so tests can move
time forward instead of waiting.

## 7. Fallback

For each candidate model in order:

- **Retryable failures** — HTTP 429, 5xx, timeouts, dropped connections — are retried on the same
  model up to `attempts_per_model` times, with exponential backoff and random jitter. A `Retry-After`
  header is respected when it is short; when it is long, the router moves on instead of waiting.
- **Provider-specific failures** — 401 and 403, a bad key for *that* provider — skip straight to
  the next model.
- **Client errors** — 400, 404, 422 — are returned immediately. A malformed request fails on every
  provider, so trying the next one only wastes time and money.

If every candidate fails, the caller gets one clean `502` listing each attempt and why it failed,
never a stack trace.

## 8. Providers

| Type | Speaks to |
|------|-----------|
| `openai_compatible` | OpenAI, and anything that copies its API: Ollama, vLLM, LM Studio, most hosted providers |
| `anthropic` | Claude, through the official Anthropic SDK, translating between the two request shapes |
| `simulated` | Nothing. Returns canned answers, or fails on purpose with a chosen status |

The `simulated` provider is a real feature, not test scaffolding. It lets someone try fallback and
the spending cap end to end, with no API keys, and it is what the README demo runs on.

## 9. Build order

| Day | Work |
|-----|------|
| **Wed** | This plan and the README pitch, committed first. Then packaging, config loading and validation, the policy engine, pricing, token estimation, the budget guard with in-memory spend. Tests for each as it lands |
| **Thu** | The provider interface and all three adapters, the fallback loop, the FastAPI app, client keys and per-client caps, the Redis spend store, metrics. CI with a real Redis service |
| **Fri** | Dockerfile and Compose, a no-keys demo on the simulated provider, the README with real output, publish, update the portfolio and roadmap |

## 10. Test plan

**No test touches the network.** Providers are faked at the HTTP layer; time is a parameter.

| Area | Must work | Must not break |
|------|-----------|----------------|
| config | a full valid file loads into typed objects | each broken reference fails startup with a message naming it; negative prices; a policy with no models |
| policy | policy name, model name, `auto` with tag rules and size rules, the default | unknown name gives a clear 400 listing what exists; rules evaluated in order |
| pricing | cost against hand-worked figures | zero tokens; free models; no float drift over a million small additions |
| tokens | the estimate errs high on English | empty input; unicode |
| budget | under, exactly at, and over the cap; per-client cap inside the global one | **concurrent requests cannot jointly exceed the cap**; a failed call releases its reservation; the day rolls over at midnight in the configured timezone |
| spend stores | memory and Redis pass the same contract tests | Redis check-and-reserve is atomic |
| fallback | 429 then success; 500 then success; timeout then success | 400 returns at once; 401 skips the model; every candidate failing gives one clean 502 |
| HTTP | the OpenAI request and response shape round-trips | missing or wrong client key is 401; an over-budget request is 429 with the reason |

## 11. Risks

| Risk | Mitigation |
|------|------------|
| The cap is exceeded under concurrency | Reserve-then-settle with an atomic check; a dedicated concurrency test |
| A reservation estimate runs low | The estimator errs high on purpose, and settlement uses real counts |
| Translating between the OpenAI and Anthropic shapes loses information | Map the fields both sides share, reject the rest clearly, test the round trip |
| Prices go stale | Prices live in the operator's config, never in code |
| Fitting five days into three overruns | Scope is not cut to fit. The core logic comes first so the cap is never at risk, and if Friday runs over, the remaining work continues on Monday rather than being dropped |

## 12. Definition of done

- [ ] Public repo, description and topics
- [ ] This plan, committed before the code
- [ ] README with the problem, a real demo on the simulated provider, configuration, and non-goals
- [ ] Every row of the test plan covered, with no network access
- [ ] CI green, including a real Redis service
- [ ] `docker compose up` runs the demo with no API keys
- [ ] At least 10 commits, one logical change each
