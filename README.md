# cache-bench

Measures which request changes break prompt-cache prefix matching, and audits production logs offline using metadata only.

## 1. Audit Your Production Logs (Offline & Metadata-Only)

Run `extract_metadata.py` inside your own network. It makes **zero network calls** and strips all prompts, completions, API keys, and PII—exporting only token counts, hour-bucketed timestamps, and locally salted hashes.

```bash
python extract_metadata.py logs.jsonl -o metadata.csv
```

### Example Cache Receipt Output
```text
========================================================
  TOKENVAULT AI — OFFLINE CACHE RECEIPT (METADATA-ONLY)
========================================================
  Calls Analyzed : 1,240 (18,450,000 input tokens)
  Cache Hit Rate : 14.2%  (Target: 85.0%+)
  Top Leak Cause : Prefix drift (412 unique prefixes across 1240 calls)
  Est. Leakage   : ~\$1,175.00 / month (at \$3/M base)
--------------------------------------------------------
  Exported: metadata.csv (0 prompts, 0 keys, 0 network calls)
  Next: Send metadata.csv for a 48h fix brief
========================================================
```

### Verify Before/After Fixes
```bash
python extract_metadata.py --before old_logs.jsonl --after new_logs.jsonl
```

## 2. Synthetic Prefix Benchmark

```bash
pip install anthropic openai
python cache_bench.py --provider dry --n 4   # no API calls
```

**What it tests:** `seq_stable`, `par_cold`, `par_warm`, `ts_top`, `ts_user`, `tools_reordered`, `keys_shuffled`, `ttl_gap`

### Dry-mode result (4 calls each)
- **ts_top:** 4 prefixes
- **tools_reordered:** 4 prefixes
- **keys_shuffled:** 4 prefixes
- **ts_user:** 1 prefix
- **seq_stable / par_cold / par_warm:** 1 prefix

**Note:** Dry mode hashes the request prefix only. It makes no API calls and reports no hit rates. Live mode reads provider usage fields.

## Limits
Synthetic requests, not your production traffic. Verify provider cache rules in their current docs.
