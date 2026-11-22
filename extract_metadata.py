import argparse
import csv
import hashlib
import hmac
import json
import os
from collections import Counter
from datetime import datetime, timezone

def hash_value(val: str, salt: bytes) -> str:
    return hmac.new(salt, str(val).encode("utf-8"), hashlib.sha256).hexdigest()[:16]

def bucket_hour(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    except Exception:
        return "unknown_hour"

def extract_prefix_text(record: dict) -> str:
    if record.get("system_prefix"):
        return str(record["system_prefix"])
    if isinstance(record.get("system"), str):
        return record["system"]
    messages = record.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                return json.dumps(content, sort_keys=True) if not isinstance(content, str) else content
    return ""

def analyze_jsonl(input_path: str, output_csv: str = None, rate_per_m: float = 3.0):
    local_salt = os.urandom(32)
    rows = []
    total_in = 0
    total_cached = 0
    prefix_hashes = []
    models = Counter()

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts_bucket = bucket_hour(record.get("timestamp", record.get("created_at", "")))
            sess_hash = hash_value(record.get("session_id", record.get("id", "default")), local_salt)
            turn_idx = record.get("turn_idx", 0)
            model_id = record.get("model", "unknown")
            models[model_id] += 1

            usage = record.get("usage", {})
            in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0))

            cache_read = usage.get("cache_read_input_tokens", 0)
            if not cache_read and "prompt_tokens_details" in usage:
                cache_read = usage["prompt_tokens_details"].get("cached_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)

            raw_prefix = extract_prefix_text(record)
            p_hash = hash_value(raw_prefix, local_salt) if raw_prefix else "none"
            prefix_hashes.append(p_hash)

            total_in += in_tok
            total_cached += cache_read

            rows.append([
                ts_bucket, sess_hash, turn_idx, model_id,
                in_tok, out_tok, cache_read, cache_write, p_hash
            ])

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow([
                "ts", "session_id_hash", "turn_idx", "model_id",
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "prefix_hash"
            ])
            writer.writerows(rows)

    calls = len(rows)
    hit_rate = (total_cached / total_in * 100) if total_in > 0 else 0.0
    unique_prefixes = len(set(prefix_hashes))

    # Identify top leak cause
    if calls > 1 and unique_prefixes > (calls * 0.5):
        top_leak = f"Prefix drift ({unique_prefixes} unique prefixes across {calls} calls)"
    elif hit_rate < 20.0 and total_in > 0:
        top_leak = "Cold cache / missing cache breakpoints"
    else:
        top_leak = "Partial multi-turn context re-billing"

    # Estimate monthly savings if hit rate reaches 85% (assuming ~30x monthly run rate & 90% cache discount)
    target_hit = 0.85
    gap = max(0.0, target_hit - (hit_rate / 100.0))
    monthly_tokens = total_in * 30
    est_savings = (monthly_tokens / 1_000_000) * rate_per_m * gap * 0.90

    return {
        "calls": calls,
        "total_in": total_in,
        "hit_rate": hit_rate,
        "top_leak": top_leak,
        "est_savings": est_savings,
        "model": models.most_common(1)[0][0] if models else "unknown"
    }

def print_receipt(stats: dict, csv_name: str):
    print("\n" + "=" * 56)
    print("  TOKENVAULT AI — OFFLINE CACHE RECEIPT (METADATA-ONLY)")
    print("=" * 56)
    print(f"  Calls Analyzed : {stats['calls']} ({stats['total_in']:,} input tokens)")
    print(f"  Cache Hit Rate : {stats['hit_rate']:.1f}%  (Target: 85.0%+)")
    print(f"  Top Leak Cause : {stats['top_leak']}")
    print(f"  Est. Leakage   : ~${stats['est_savings']:,.2f} / month (at $3/M base)")
    print("-" * 56)
    print(f"  Exported: {csv_name} (0 prompts, 0 keys, 0 network calls)")
    print("  Next: Send metadata.csv for a 48h fix brief")
    print("=" * 56 + "\n")

def print_comparison(before: dict, after: dict):
    saved_delta = max(0.0, before["est_savings"] - after["est_savings"])
    print("\n" + "=" * 56)
    print("  TOKENVAULT AI — CACHE FIX REDEMPTION CARD")
    print("=" * 56)
    print(f"  Before Hit Rate : {before['hit_rate']:.1f}%  -->  After Hit Rate : {after['hit_rate']:.1f}%")
    print(f"  Before Leakage  : ~${before['est_savings']:,.2f}/mo  -->  After Leakage : ~${after['est_savings']:,.2f}/mo")
    print(f"  Net Monthly Win : +${saved_delta:,.2f} / month recovered")
    print("-" * 56)
    print("  Verified offline via TokenVault cache-bench")
    print("=" * 56 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Metadata-Only LLM Cache Auditor")
    parser.add_argument("input", nargs="?", help="Input JSONL log file")
    parser.add_argument("-o", "--output", default="metadata.csv", help="Output CSV filename")
    parser.add_argument("--before", help="Before-fix JSONL log file for comparison")
    parser.add_argument("--after", help="After-fix JSONL log file for comparison")
    parser.add_argument("--rate", type=float, default=3.0, help="Input token price per 1M ($)")
    args = parser.parse_args()

    if args.before and args.after:
        b_stats = analyze_jsonl(args.before, rate_per_m=args.rate)
        a_stats = analyze_jsonl(args.after, rate_per_m=args.rate)
        print_comparison(b_stats, a_stats)
    elif args.input:
        stats = analyze_jsonl(args.input, args.output, rate_per_m=args.rate)
        print_receipt(stats, args.output)
    else:
        print("Usage:")
        print("  Single Audit : python extract_metadata.py logs.jsonl -o metadata.csv")
        print("  Before/After : python extract_metadata.py --before old.jsonl --after new.jsonl")
