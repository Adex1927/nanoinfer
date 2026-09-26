#!/usr/bin/env python3
"""
Generate and dump top-K logits from llama.cpp's llama-debug binary.

Usage:
  # All-in-one: run llama-debug and dump logits
  python3 dump_logits.py --model model.gguf --prompt "Hello"

  # Just read an existing logits .bin file
  python3 dump_logits.py --bin /tmp/logits/llamacpp-foo.bin --model model.gguf

  # Options
  python3 dump_logits.py --model model.gguf --prompt "" --top 20 --check 450 2760
"""

import argparse
import glob
import os
import struct
import subprocess
import sys
import tempfile


# ── Default paths (edit these for your setup) ──
DEFAULT_LLAMA_DEBUG = os.path.expanduser(
    "~/workspace/llama.cpp/build/bin/llama-debug"
)


def read_vocab(gguf_path):
    """Read the tokenizer.ggml.tokens array from a GGUF file."""
    with open(gguf_path, "rb") as f:
        magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))

        def read_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", errors="replace")

        def skip_val(t):
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            if t == 8:
                read_str()
            elif t == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                for _ in range(cnt):
                    skip_val(et)
            elif t in sizes:
                f.read(sizes[t])

        for _ in range(n_kv):
            key = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            if key == "tokenizer.ggml.tokens":
                et, cnt = struct.unpack("<IQ", f.read(12))
                return [read_str() for _ in range(cnt)]
            else:
                skip_val(vtype)
    return None


def run_llama_debug(model_path, prompt, llama_debug_bin):
    """Run llama-debug --save-logits and return path to the .bin file."""
    outdir = tempfile.mkdtemp(prefix="nanoinfer_logits_")

    cmd = [
        llama_debug_bin,
        "-m", model_path,
        "-p", prompt,
        "--save-logits",
        "--logits-output-dir", outdir,
        "--temp", "0",
        "-n", "1",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"llama-debug failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[:500], file=sys.stderr)
        sys.exit(1)

    # Find the logits .bin file (ignore tokens.bin)
    bins = [f for f in glob.glob(os.path.join(outdir, "*.bin")) if not f.endswith("-tokens.bin")]
    if not bins:
        print(f"ERROR: no logits .bin file found in {outdir}", file=sys.stderr)
        sys.exit(1)

    # Check for tokens file
    tokens_file = glob.glob(os.path.join(outdir, "*-tokens.bin"))
    prompt_tokens = []
    if tokens_file:
        with open(tokens_file[0], "rb") as f:
            tdata = f.read()
            prompt_tokens = list(struct.unpack(f"<{len(tdata)//4}i", tdata))

    return bins[0], prompt_tokens


def dump_logits(logits_bin, vocab, top_k, check_ids):
    """Read a binary logits file and print the top-K tokens."""
    with open(logits_bin, "rb") as f:
        data = f.read()

    n_floats = len(data) // 4
    all_logits = struct.unpack(f"<{n_floats}f", data)

    n_vocab = len(vocab) if vocab else 32000
    if n_floats < n_vocab:
        print(f"ERROR: file has {n_floats} floats, need at least {n_vocab}", file=sys.stderr)
        sys.exit(1)

    logits = all_logits[-n_vocab:]

    # Top-K
    indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_k} logits ({os.path.basename(logits_bin)}):")
    print(f"{'Rank':>4}  {'Token':>6}  {'Logit':>10}  {'Text'}")
    print("-" * 50)
    for rank, (tid, score) in enumerate(indexed[:top_k]):
        name = repr(vocab[tid]) if vocab and tid < len(vocab) else "?"
        print(f"  #{rank+1:<2}  {tid:>6}  {score:>10.4f}  {name}")

    # Check specific tokens
    if check_ids:
        print(f"\nChecked tokens:")
        for tid in check_ids:
            if tid < len(logits):
                name = repr(vocab[tid]) if vocab and tid < len(vocab) else "?"
                print(f"  token {tid:>6} = {logits[tid]:>10.4f}  {name}")

    print(f"\nStats: min={min(logits):.4f}  max={max(logits):.4f}  mean={sum(logits)/len(logits):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate and dump top-K logits from llama.cpp")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", "-p", default=None, help="Prompt to evaluate (runs llama-debug)")
    parser.add_argument("--bin", default=None, help="Path to existing .bin logits file (skip llama-debug)")
    parser.add_argument("--top", type=int, default=10, help="Top-K tokens to show (default: 10)")
    parser.add_argument("--check", type=int, nargs="+", help="Extra token IDs to check")
    parser.add_argument("--llama-debug", default=DEFAULT_LLAMA_DEBUG, help="Path to llama-debug binary")
    args = parser.parse_args()

    if args.bin is None and args.prompt is None:
        parser.error("Provide either --prompt (to run llama-debug) or --bin (to read existing logits)")

    # Get logits file
    prompt_tokens = []
    if args.bin:
        logits_bin = args.bin
    else:
        logits_bin, prompt_tokens = run_llama_debug(args.model, args.prompt, args.llama_debug)

    # Read vocab for token names
    vocab = read_vocab(args.model)

    if prompt_tokens:
        tok_reprs = [f"{tid}:{repr(vocab[tid]) if vocab and tid < len(vocab) else '?'}" for tid in prompt_tokens]
        print(f"\nEvaluated Prompt Tokens ({len(prompt_tokens)}): {' '.join(tok_reprs)}")

    dump_logits(logits_bin, vocab, args.top, args.check)


if __name__ == "__main__":
    main()
