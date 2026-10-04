#!/usr/bin/env python3
"""
Run greedy generation with llama.cpp (llama-completion) and optionally
compare directly against nanoinfer's model_test.

Usage:
  # Run llama.cpp greedy generation for any prompt:
  python3 tools/compare_greedy.py -m ../llama.cpp/models/tinyllama.gguf -p "World" -n 20

  # Compare llama.cpp directly with nanoinfer:
  python3 tools/compare_greedy.py -m ../llama.cpp/models/tinyllama.gguf -p "Hello" -n 20 --compare
"""

import argparse
import os
import re
import subprocess
import sys

DEFAULT_LLAMA_DIR = os.path.expanduser("~/workspace/llama.cpp")
DEFAULT_COMPLETION = os.path.join(DEFAULT_LLAMA_DIR, "build/bin/llama-completion")
DEFAULT_TOKENIZE = os.path.join(DEFAULT_LLAMA_DIR, "build/bin/llama-tokenize")
DEFAULT_NANOINFER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_test")


def tokenize_prompt(prompt, model_path, tokenize_bin):
    """Use llama-tokenize to get exact token IDs for any prompt string."""
    if not os.path.isfile(tokenize_bin):
        return []
    cmd = [tokenize_bin, "-m", model_path, "-p", prompt]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []

    tokens = []
    for line in res.stdout.splitlines():
        # Match lines like: "     1 -> '<s>'"
        m = re.match(r"^\s*(\d+)\s*->", line)
        if m:
            tokens.append(int(m.group(1)))
    return tokens


def run_llamacpp(model_path, prompt, n_predict, completion_bin):
    """Run llama-completion with greedy sampling (--temp 0)."""
    if not os.path.isfile(completion_bin):
        print(f"Error: llama-completion binary not found at {completion_bin}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        completion_bin,
        "-m", model_path,
        "-p", prompt,
        "--temp", "0",
        "-n", str(n_predict),
        "--no-warmup",
        "-no-cnv",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"llama-completion failed with returncode {res.returncode}:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

    return res.stdout


def run_nanoinfer(model_path, token_ids, n_predict, nanoinfer_bin):
    """Run nanoinfer's model_test with the given token IDs."""
    if not os.path.isfile(nanoinfer_bin):
        print(f"Error: nanoinfer binary not found at {nanoinfer_bin}. Run `make` first.", file=sys.stderr)
        return None

    cmd = [nanoinfer_bin, model_path] + [str(t) for t in token_ids]
    env = os.environ.copy()
    env["NANO_N"] = str(n_predict)

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"nanoinfer failed with returncode {res.returncode}:\n{res.stderr}", file=sys.stderr)
        return None

    return res.stdout


def parse_nanoinfer_output(output):
    """Extract generated text and generated token IDs from model_test output."""
    gen_text = ""
    gen_tokens = []
    
    # Extract "Generated text: <text>\n(<N> tokens)"
    text_match = re.search(r"Generated text: (.*?)(?:\n\(\d+ tokens\))", output, re.DOTALL)
    if text_match:
        gen_text = text_match.group(1).strip("\r\n")

    # Extract "Generated token IDs: 2018 29889 ..."
    tokens_match = re.search(r"Generated token IDs:\s*([0-9 ]+)", output)
    if tokens_match:
        gen_tokens = [int(x) for x in tokens_match.group(1).split()]

    return gen_text, gen_tokens


def main():
    parser = argparse.ArgumentParser(description="Run greedy generation with llama.cpp and compare with nanoinfer")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", "-p", default="World", help="Input prompt text (default: 'World')")
    parser.add_argument("-n", "--n-predict", type=int, default=20, help="Number of tokens to generate (default: 20)")
    parser.add_argument("--compare", action="store_true", help="Also run nanoinfer and compare outputs side-by-side")
    parser.add_argument("--completion-bin", default=DEFAULT_COMPLETION, help="Path to llama-completion binary")
    parser.add_argument("--tokenize-bin", default=DEFAULT_TOKENIZE, help="Path to llama-tokenize binary")
    parser.add_argument("--nanoinfer-bin", default=DEFAULT_NANOINFER, help="Path to nanoinfer model_test binary")
    args = parser.parse_args()

    print(f"Prompt: {repr(args.prompt)}")
    print(f"Generating {args.n_predict} tokens (greedy, temp=0)...\n")

    # 1. llama.cpp output
    llama_out = run_llamacpp(args.model, args.prompt, args.n_predict, args.completion_bin)
    
    print("=" * 60)
    print(" [llama.cpp output]")
    print("=" * 60)
    print(llama_out.strip("\r\n"))
    print("=" * 60)

    # 2. Optional nanoinfer comparison
    if args.compare:
        token_ids = tokenize_prompt(args.prompt, args.model, args.tokenize_bin)
        if not token_ids:
            # Fallback for default "World" if tokenizer is unavailable
            token_ids = [1, 2787] if args.prompt == "World" else []

        if not token_ids:
            print("\nCould not tokenize prompt for nanoinfer (llama-tokenize unavailable).", file=sys.stderr)
            return

        print(f"\nPrompt token IDs: {token_ids}")
        nano_raw = run_nanoinfer(args.model, token_ids, args.n_predict, args.nanoinfer_bin)
        if nano_raw:
            nano_text, nano_tokens = parse_nanoinfer_output(nano_raw)
            print("=" * 60)
            print(" [nanoinfer output]")
            print("=" * 60)
            print(nano_text)
            if nano_tokens:
                print(f"\nGenerated tokens ({len(nano_tokens)}): {nano_tokens}")
            print("=" * 60)

            # Compare generated text
            # llama-completion prefixes prompt text, so check if nano_text matches completion portion
            clean_llama = llama_out.strip()
            clean_nano = nano_text.strip()

            if clean_nano and (clean_nano in clean_llama or clean_llama.endswith(clean_nano)):
                print("\n✅ MATCH: nanoinfer generated text matches llama.cpp!")
            else:
                print("\n⚠️ Outputs differ (check above).")


if __name__ == "__main__":
    main()
