# llm-router

**One endpoint, many models.** Routes each request by policy, falls back when a provider fails,
and stops spending at a daily cap — before the money is gone, not after.

> Week 4 of a 10-week build. In progress — see
> [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the design.

---

## The problem

An application that calls language models directly is hard-wired to one provider. When that
provider rate-limits or goes down, the application goes down with it. And when someone ships a
loop that calls the most expensive model ten thousand times, the bill shows up a month later.

The usual answer to the second problem is a dashboard. A dashboard tells you what you spent after
you spent it. That is not a limit.

## What it does

Point your application at the router instead of at a provider — one line, the base URL. The
router speaks the OpenAI chat-completions format, so existing clients work unchanged.

- **Routes** each request by policy: cheapest first, best first, or by rules on tags and prompt size
- **Falls back** to the next model when one is rate-limited, erroring or timing out
- **Prices** every call and exposes tokens, cost and latency at `/metrics`
- **Enforces a hard daily cap**, globally and per client

## A cap that cannot be exceeded

Most spend limits check "have we hit the budget?" before each call. That overshoots: the call it
lets through can cost more than what is left, and ten calls arriving at once all pass the same
check.

llm-router **reserves the worst-case cost of a call before making it**, in one atomic step, and
settles the real cost afterwards. If the worst case would cross a cap, the request is refused
without ever reaching the provider. Money is counted in integer micro-dollars, so a day of tiny
requests cannot drift.

The output side of the worst case is exact, because the router sends the provider the same
`max_tokens` it reserved for. The input side uses a token estimate built to run high (about three
characters per token, where English averages four). The real cost is settled from the provider's
own counts, and if an estimate ever runs low the overrun is recorded and reported rather than
hidden.

## Status

| Day | Work | |
|-----|------|-|
| Wed | Plan, README, config, routing policy, pricing, the spending cap, the three providers, fallback, the HTTP service with client keys | done |
| Thu | Redis spend store shared between routers, Prometheus metrics, CI on Python 3.11 to 3.14 with a real Redis | done |
| Fri | Docker, a no-keys demo, publish | |

## Non-goals

- **No streaming in version one.** Fallback mid-stream is a different problem; it is the stretch goal.
- **No web interface.** Configuration is a file; observation is `/metrics`.
- **No billing or accounts.** Per-client caps, yes. Invoicing, no.

## License

MIT — see [LICENSE](LICENSE).
