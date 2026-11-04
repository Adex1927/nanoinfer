#!/usr/bin/env python3
"""
Benchmark end-to-end prefill latency across batch sizes B in {1, 2, 4, 8, 16, 32, 64}.

Uses an exact power-of-two prompt length (default: 64 tokens) so every batch
size divides the prompt evenly without partial-tail skew, and runs one warmup
pass first so mmap page faults do not skew B=1.

Usage:
  python3 tools/bench_prefill.py -m ../llama.cpp/models/tinyllama.gguf
  python3 tools/bench_prefill.py -m ../llama.cpp/models/tinyllama.gguf -p 256
"""
import argparse
import os
import re
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(REPO_ROOT, "model_test")


def run_batch(bin_path, model_path, tokens, batch_size):
    env = os.environ.copy()
    env["NANO_BATCH"] = str(batch_size)
    env["NANO_N"] = "1"
    env["NANO_PROBE_CSV"] = "/tmp/nanoinfer_bench_prefill.csv"
    cmd = [bin_path, model_path] + tokens
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env)
    out = res.stderr

    def grab(name):
        m = re.search(rf"{re.escape(name)}\s+prefill\s+\d+\s+([\d.]+)", out)
        return float(m.group(1)) if m else 0.0

    return (
        grab("prefill_total"),
        grab("layer:ffn"),
        grab("layer:qkv_proj"),
        grab("layer:attn_out_proj"),
        grab("layer:mha"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to GGUF model")
    ap.add_argument("-p", "--prompt-len", type=int, default=64,
                    help="Number of prompt tokens (default: 64, divisible by all B)")
    ap.add_argument("--bin", default=DEFAULT_BIN, help="Path to model_test binary")
    args = ap.parse_args()

    tokens = [str(300 + (i % 200)) for i in range(args.prompt_len)]

    # Warm up OS page cache / mmap tables
    run_batch(args.bin, args.model, tokens, 16)

    print(f"=== End-to-End Prefill Benchmark ({args.prompt_len} tokens, warmed-up mmap) ===")
    print("  B | Total (ms) |   ms/tok |    tok/s |  FFN (ms) |  QKV (ms) |  OutProj | MHA (ms) | Speedup")
    print("-" * 93)

    b1_tot = None
    for b in [1, 2, 4, 8, 16, 32, 64]:
        if b > args.prompt_len:
            continue
        tot, ffn, qkv, outp, mha = run_batch(args.bin, args.model, tokens, b)
        if b1_tot is None:
            b1_tot = tot
        print(f"{b:3d} | {tot:10.2f} | {tot/args.prompt_len:8.3f} | {args.prompt_len*1000.0/tot:8.1f} | "
              f"{ffn:9.2f} | {qkv:9.2f} | {outp:8.2f} | {mha:8.2f} | {b1_tot/tot:6.2f}x")


if __name__ == "__main__":
    main()

