import csv
import hashlib
import hmac
import json
import os
import sys
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
    # 1. Check if explicitly provided as system_prefix or system (Anthropic style)
    if record.get("system_prefix"):
        return str(record["system_prefix"])
    if isinstance(record.get("system"), str):
        return record["system"]
    
    # 2. Check inside standard messages array (OpenAI / standard chat format)
    messages = record.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                return json.dumps(content, sort_keys=True) if not isinstance(content, str) else content
    return ""

def process_logs(input_path: str, output_path: str):
    # Generates a random local salt per run; never leaves client machine
    local_salt = os.urandom(32)
    
    rows_written = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "ts", "session_id_hash", "turn_idx", "model_id",
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "prefix_hash"
        ])
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            ts_bucket = bucket_hour(record.get("timestamp", record.get("created_at", "")))
            sess_hash = hash_value(record.get("session_id", record.get("id", "default")), local_salt)
            turn_idx = record.get("turn_idx", 0)
            model_id = record.get("model", "unknown")
            
            usage = record.get("usage", {})
            in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0))
            out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0))
            
            # Extract cache metrics across Anthropic / OpenAI schemas
            cache_read = usage.get("cache_read_input_tokens", 0)
            if not cache_read and "prompt_tokens_details" in usage:
                cache_read = usage["prompt_tokens_details"].get("cached_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            
            # Auto-extract system prefix from either dedicated field or messages array
            raw_prefix = extract_prefix_text(record)
            prefix_hash = hash_value(raw_prefix, local_salt) if raw_prefix else "none"
            
            writer.writerow([
                ts_bucket, sess_hash, turn_idx, model_id,
                in_tok, out_tok, cache_read, cache_write, prefix_hash
            ])
            rows_written += 1
            
    print(f"Done. Exported {rows_written} metadata-only rows to {output_path}.")
    print("Inspect the CSV in Excel/Sheets to verify zero text is present before sharing.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_metadata.py <input_logs.jsonl> <output_audit.csv>")
        sys.exit(1)
    process_logs(sys.argv[1], sys.argv[2])
