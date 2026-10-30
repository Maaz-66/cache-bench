#!/usr/bin/env python3
"""
cache_bench.py - measure provider-side prompt-cache behaviour from your own runs.

Providers:
  anthropic  uses cache_control on the system block (reads ANTHROPIC_API_KEY)
  openai     relies on automatic prefix caching   (reads OPENAI_API_KEY)
  dry        no API calls; only checks which variants change the prefix bytes

Every cache number comes from the `usage` fields the provider returns.
Nothing here is estimated except `est_prefix_tokens` (chars/4, sanity check only).

Experiments (each one gets a fresh nonce so runs cannot contaminate each other):
  seq_stable       N sequential calls, byte-identical prefix        (baseline)
  par_cold         N parallel calls, identical prefix, no warm-up   (fan-out race)
  par_warm         1 warm-up call, then N-1 parallel calls          (fan-out after warm-up)
  ts_top           timestamp injected at the top of the system prompt
  ts_user          timestamp injected in the user message instead
  tools_reordered  tool list order shuffled per call
  keys_shuffled    JSON key order inside tool schemas shuffled per call
  ttl_gap          call, pause --ttl-gap seconds, call again        (only if --ttl-gap > 0)

Usage:
  python cache_bench.py --provider dry --n 4
  python cache_bench.py --provider anthropic --model <cheap-model> --n 8 \
      --price-in 1.0 --price-read 0.1 --price-write 1.25
  python cache_bench.py --provider openai --model <cheap-model> --n 8 \
      --price-in 0.15 --price-read 0.075 --ttl-gap 360

Prices are USD per 1M tokens. Copy them from the provider's CURRENT pricing page.
Without --price-in the cost columns stay 0.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# request material (deterministic, so the "stable" prefix is byte-identical)
# ----------------------------------------------------------------------------
VOCAB = ("policy control access audit retention encryption boundary schema tenant "
         "record escalate review incident baseline telemetry mapping constraint "
         "workflow approval evidence integrity logging session").split()


def filler_text(words, seed=7):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(VOCAB) for _ in range(words))


def base_tools(n=10):
    tools = []
    for i in range(n):
        tools.append({
            "name": f"tool_{i}",
            "description": f"Utility {i}: " + filler_text(40, seed=100 + i),
            "schema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": filler_text(12, seed=200 + i)},
                    "mode": {"type": "string", "enum": ["fast", "safe", "audit"]},
                    "limit": {"type": "integer"},
                },
                "required": ["target"],
            },
        })
    return tools


def shuffle_keys(obj, rnd):
    if isinstance(obj, dict):
        items = list(obj.items())
        rnd.shuffle(items)
        return {k: shuffle_keys(v, rnd) for k, v in items}
    if isinstance(obj, list):
        return [shuffle_keys(v, rnd) for v in obj]
    return obj


def build(variant, i, nonce, words, rnd):
    system = f"[session-nonce {nonce}]\n" + filler_text(words)
    tools = base_tools()
    user = f"Question {i}: reply with the single word OK."
    if variant == "ts_top":
        system = f"Current time: {time.time_ns()}\n" + system
    elif variant == "ts_user":
        user = f"Current time: {time.time_ns()}\n" + user
    elif variant == "tools_reordered":
        rnd.shuffle(tools)
    elif variant == "keys_shuffled":
        tools = [shuffle_keys(t, rnd) for t in tools]
    return system, tools, user


def prefix_hash(system, tools):
    # No sort_keys on purpose: key order is part of the bytes the provider sees.
    blob = json.dumps({"tools": tools, "system": system},
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# provider callers - each returns dict(fresh, read, write, out, ms)
# ----------------------------------------------------------------------------
async def call_anthropic(client, model, system, tools, user):
    t0 = time.perf_counter()
    r = await client.messages.create(
        model=model,
        max_tokens=8,
        tools=[{"name": t["name"], "description": t["description"],
                "input_schema": t["schema"]} for t in tools],
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    u = r.usage
    return dict(fresh=u.input_tokens,
                read=getattr(u, "cache_read_input_tokens", 0) or 0,
                write=getattr(u, "cache_creation_input_tokens", 0) or 0,
                out=u.output_tokens,
                ms=(time.perf_counter() - t0) * 1000)


async def call_openai(client, model, system, tools, user):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        max_completion_tokens=16,  # reasoning models may need a larger value
        tools=[{"type": "function",
                "function": {"name": t["name"], "description": t["description"],
                             "parameters": t["schema"]}} for t in tools],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    u = r.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    return dict(fresh=u.prompt_tokens - cached, read=cached, write=0,
                out=u.completion_tokens, ms=(time.perf_counter() - t0) * 1000)


async def call_dry(system, tools, user):
    return dict(fresh=0, read=0, write=0, out=0, ms=0.0)


# ----------------------------------------------------------------------------
# experiment runner
# ----------------------------------------------------------------------------
async def run_experiment(name, variant, mode, args, caller):
    nonce = uuid.uuid4().hex[:12]
    rnd = random.Random(1234)
    n = 2 if mode == "ttl" else args.n
    reqs = [build(variant, i, nonce, args.words, rnd) for i in range(n)]

    async def one(i):
        system, tools, user = reqs[i]
        row = {
            "experiment": name, "mode": mode, "call": i,
            "prefix_sha256": prefix_hash(system, tools)[:16],
            "est_prefix_tokens": (len(system) + len(json.dumps(tools))) // 4,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            u = await caller(system, tools, user)
        except Exception as e:  # keep going, log the failure
            row["error"] = f"{type(e).__name__}: {e}"[:200]
            return row
        total = u["fresh"] + u["read"] + u["write"]
        priced = args.price_in > 0
        row.update(
            fresh=u["fresh"], cache_read=u["read"], cache_write=u["write"],
            total_in=total, out=u["out"], latency_ms=round(u["ms"]),
            hit_rate=round(u["read"] / total, 4) if total else 0.0,
            cost=round((u["fresh"] * args.price_in + u["read"] * args.price_read
                        + u["write"] * args.price_write) / 1e6, 6) if priced else 0.0,
            cost_uncached=round(total * args.price_in / 1e6, 6) if priced else 0.0,
        )
        return row

    rows = []
    if mode == "par":
        rows = list(await asyncio.gather(*[one(i) for i in range(n)]))
    elif mode == "par_warm":
        rows.append(await one(0))
        await asyncio.sleep(args.gap)
        rows += list(await asyncio.gather(*[one(i) for i in range(1, n)]))
    elif mode == "ttl":
        rows.append(await one(0))
        print(f"  [ttl_gap] sleeping {args.ttl_gap:.0f}s ...", flush=True)
        await asyncio.sleep(args.ttl_gap)
        rows.append(await one(1))
    else:  # seq
        for i in range(n):
            rows.append(await one(i))
            await asyncio.sleep(args.gap)  # cache entries can lag slightly
    return rows


def summarize(all_rows, dry):
    by_exp = {}
    for r in all_rows:
        by_exp.setdefault(r["experiment"], []).append(r)
    print(f"\n{'experiment':16} {'calls':>5} {'errs':>4} {'distinct_prefixes':>17} "
          f"{'hit%':>7} {'cost':>10} {'uncached':>10}")
    for name, rs in by_exp.items():
        ok = [r for r in rs if "error" not in r]
        distinct = len({r["prefix_sha256"] for r in rs})
        scored = ok if name == "par_cold" else [r for r in ok if r["call"] >= 1]
        hit = (statistics.mean(r["hit_rate"] for r in scored) * 100) if scored else float("nan")
        cost = sum(r.get("cost", 0) for r in ok)
        unc = sum(r.get("cost_uncached", 0) for r in ok)
        print(f"{name:16} {len(rs):>5} {len(rs) - len(ok):>4} {distinct:>17} "
              f"{hit:>7.1f} {cost:>10.5f} {unc:>10.5f}")
    if dry:
        print("\n(dry run: only 'distinct_prefixes' is meaningful. 1 = byte-identical prefix; "
              "N = every call changed the bytes.)")
    else:
        print("\nhit% = mean cached-read share of input tokens (call>=1; all calls for par_cold).")
        print("cost vs uncached includes any cache-write premium you passed via --price-write.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["anthropic", "openai", "dry"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=8, help="calls per experiment")
    ap.add_argument("--words", type=int, default=2500,
                    help="filler words in system prompt; check reported input tokens "
                         "clear the provider's minimum cacheable length")
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between sequential calls")
    ap.add_argument("--ttl-gap", type=float, default=0.0,
                    help="pause (s) for the TTL test, e.g. 360; 0 skips it")
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-read", type=float, default=0.0)
    ap.add_argument("--price-write", type=float, default=0.0)
    ap.add_argument("--only", default=None, help="comma-separated experiment names")
    ap.add_argument("--out", default="bench_results")
    args = ap.parse_args()

    if args.provider != "dry" and not args.model:
        sys.exit("--model is required for anthropic/openai")

    if args.provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic()
        caller = lambda s, t, u: call_anthropic(client, args.model, s, t, u)
    elif args.provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        caller = lambda s, t, u: call_openai(client, args.model, s, t, u)
    else:
        caller = call_dry

    experiments = [
        ("seq_stable", "stable", "seq"),
        ("par_cold", "stable", "par"),
        ("par_warm", "stable", "par_warm"),
        ("ts_top", "ts_top", "seq"),
        ("ts_user", "ts_user", "seq"),
        ("tools_reordered", "tools_reordered", "seq"),
        ("keys_shuffled", "keys_shuffled", "seq"),
    ]
    if args.ttl_gap > 0:
        experiments.append(("ttl_gap", "stable", "ttl"))
    if args.only:
        wanted = set(args.only.split(","))
        experiments = [e for e in experiments if e[0] in wanted]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{args.out}_{args.provider}_{stamp}"
    all_rows = []

    async def go():
        for name, variant, mode in experiments:
            print(f"running {name} ...", flush=True)
            all_rows.extend(await run_experiment(name, variant, mode, args, caller))

    asyncio.run(go())

    fields = ["experiment", "mode", "call", "prefix_sha256", "est_prefix_tokens",
              "fresh", "cache_read", "cache_write", "total_in", "out", "hit_rate",
              "latency_ms", "cost", "cost_uncached", "ts_utc", "error"]
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    with open(base + ".jsonl", "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    with open(base + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "python": platform.python_version(),
                   "started_utc": stamp}, f, indent=2)

    summarize(all_rows, args.provider == "dry")
    digest = hashlib.sha256(open(base + ".csv", "rb").read()).hexdigest()
    print(f"\nlogs: {base}.csv / .jsonl / .meta.json")
    print(f"csv sha256: {digest}")


if __name__ == "__main__":
    main()
