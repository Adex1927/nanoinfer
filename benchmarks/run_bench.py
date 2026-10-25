#!/usr/bin/env python3
import sys
import os
import subprocess
import csv
from collections import defaultdict

def run_benchmark(model_path, tokens=30):
    cmd = ["./model_test", model_path]
    csv_file = "probe.csv"
    env = os.environ.copy()
    env["NANO_PROBE_CSV"] = csv_file
    env["NANO_N"] = str(tokens)

    print(f"Running benchmark on {model_path} ({tokens} tokens)...")
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("Benchmark run failed!")
        print(res.stderr)
        sys.exit(1)

    print("Run completed successfully. Parsing probe data...")

    # Parse probe.csv
    data = defaultdict(list)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[(row["name"], row["phase"])].append(float(row["ms"]))

    return data, res.stdout

def format_table(data):
    # Print markdown table
    print("\n### Performance Profile (Probe Summary)\n")
    print("| Component / Stage | Phase | Samples | Total (ms) | Avg (ms) | Min (ms) | Max (ms) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Order by specific keys if present, otherwise sort
    key_order = [
        ("prefill_total", "prefill"),
        ("prefill", "prefill"),
        ("decode:forward_token", "decode"),
        ("all_layers", "decode"),
        ("layer:ffn", "decode"),
        ("layer:qkv_proj", "decode"),
        ("layer:attn_out_proj", "decode"),
        ("layer:mha", "decode"),
        ("layer:rope", "decode"),
        ("layer:attn_norm", "decode"),
        ("final_norm_and_lm_head", "decode"),
        ("embedding_lookup", "decode"),
        ("decode:argmax", "decode"),
    ]

    seen = set()
    rows = []
    for k in key_order:
        if k in data:
            seen.add(k)
            rows.append(k)
    for k in sorted(data.keys()):
        if k not in seen:
            rows.append(k)

    lines = []
    for name, phase in rows:
        vals = data[(name, phase)]
        n = len(vals)
        tot = sum(vals)
        avg = tot / n
        mn = min(vals)
        mx = max(vals)
        line = f"| `{name}` | `{phase}` | {n} | {tot:.2f} | **{avg:.3f}** | {mn:.3f} | {mx:.3f} |"
        print(line)
        lines.append(line)

    # Compute tokens/sec
    if ("decode:forward_token", "decode") in data:
        avg_decode_ms = sum(data[("decode:forward_token", "decode")]) / len(data[("decode:forward_token", "decode")])
        tok_sec = 1000.0 / avg_decode_ms
        print(f"\n**Decode Throughput:** {tok_sec:.2f} tokens/sec ({avg_decode_ms:.1f} ms/token)")

    return lines

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "/Users/trishabh/workspace/llama.cpp/models/tinyllama.gguf"
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    data, stdout = run_benchmark(model, tokens)
    format_table(data)
