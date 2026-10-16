"""Walk through what llm-router does, against a running router on the demo config.

Run by `docker compose up`, or by hand: ROUTER_URL=http://localhost:8080 python examples/demo.py
It checks what it shows, and exits non-zero if the router does not behave as described.
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx2

ROUTER_URL = os.environ.get("ROUTER_URL", "http://localhost:8080")
QUESTION = [{"role": "user", "content": "Say hello in one short sentence."}]


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


async def chat(client: httpx2.AsyncClient, model: str, **extra) -> httpx2.Response:
    return await client.post(
        "/v1/chat/completions", json={"model": model, "messages": QUESTION, **extra}
    )


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"\nDEMO FAILED: {message}", file=sys.stderr)
        raise SystemExit(1)


async def main() -> None:
    async with httpx2.AsyncClient(base_url=ROUTER_URL, timeout=30) as client:
        heading('1. Fallback: ask the "resilient" policy for an answer')
        print("   Its models, in order: frontier (down, HTTP 503), backup (key revoked, HTTP 401),")
        print("   budget (healthy).")
        response = await chat(client, "resilient")
        check(response.status_code == 200, f"expected 200, got {response.status_code}")
        print(f"\n   answer       {response.json()['choices'][0]['message']['content']}")
        for name in ("x-router-policy", "x-router-model", "x-router-attempts", "x-router-cost-usd"):
            print(f"   {name:<20} {response.headers.get(name)}")
        check(response.headers["x-router-model"] == "budget", "expected the budget model to answer")
        check(response.headers["x-router-attempts"] == "4", "expected 2 + 1 + 1 attempts")
        print("\n   4 attempts: frontier was retried once (a 503 can be temporary), backup was")
        print("   skipped at once (a bad key will not fix itself), and budget answered.")

        heading("2. The cap: ten expensive requests at the same moment")
        before = (await client.get("/budget")).json()["global"]
        print(f"   Daily cap ${before['limit_usd']}, spent so far ${before['spent_usd']}.")
        print("   Each request may write up to 4,000 tokens at $75 per million: a worst case of")
        print("   about $0.30, although each one really costs a fraction of a cent.")
        results = await asyncio.gather(
            *(chat(client, "premium", max_tokens=4000) for _ in range(10))
        )
        answered = [r for r in results if r.status_code == 200]
        refused = [r for r in results if r.status_code == 429]
        print(f"\n   answered     {len(answered)}")
        print(f"   refused      {len(refused)}  (HTTP 429, before reaching the model)")
        check(len(answered) + len(refused) == 10, "every request should be answered or refused")
        check(len(refused) > 0, "the cap should refuse some of the burst")
        print(f"\n   why: {refused[0].json()['error']['message']}")
        print("\n   Worst cases are reserved before any call starts, in one atomic step in Redis,")
        print("   so concurrent requests cannot jointly break the cap. The answered ones then")
        print("   settled at their real cost and released the rest of their reservation.")

        heading("3. Where the money went")
        after = (await client.get("/budget")).json()["global"]
        for key in ("limit_usd", "spent_usd", "reserved_usd", "remaining_usd"):
            print(f"   {key:<14} ${after[key]}")
        check(float(after["spent_usd"]) <= float(after["limit_usd"]), "spend passed the cap")
        check(after["reserved_usd"] == "0.000000", "nothing should still be reserved")

        heading("4. The same story in Prometheus metrics (GET /metrics)")
        metrics = (await client.get("/metrics")).text
        shown = ("llm_router_requests_total", "llm_router_attempts_total", "llm_router_spent_usd")
        for line in metrics.splitlines():
            if line.startswith(shown):
                print(f"   {line}")
    print("\nDemo finished. The router keeps running on localhost:8080 (or your ROUTER_PORT):")
    print("try it with curl or the OpenAI SDK. The README has the rest.")


if __name__ == "__main__":
    asyncio.run(main())
