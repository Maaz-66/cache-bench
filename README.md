---
# cache-bench

Measures which request changes break prompt-cache prefix matching, and (with a live API key) what that does to cached-token counts.

## Quick start
```bash
pip install anthropic openai
python cache_bench.py --provider dry --n 4        # no API calls
```

## What it tests
`seq_stable`, `par_cold`, `par_warm`, `ts_top`, `ts_user`, `tools_reordered`, `keys_shuffled`, `ttl_gap`

## Dry-mode result (4 calls each)
* **ts_top:** 4 prefixes
* **tools_reordered:** 4 prefixes
* **keys_shuffled:** 4 prefixes
* **ts_user:** 1 prefix
* **seq_stable / par_cold / par_warm:** 1 prefix

> **Note:** Dry mode hashes the request prefix only. It makes no API calls and reports no hit rates. Live mode reads provider usage fields.

## Limits
Synthetic requests, not your production traffic. Verify provider cache rules in their current docs.
---
