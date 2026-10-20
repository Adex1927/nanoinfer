#!/usr/bin/env python3
"""
Run greedy generation with llama.cpp (llama-completion) and optionally
compare directly against nanoinfer's model_test.

The prompt comes from -p/--prompt if given, otherwise from a prompt file
(default: prompt.txt in the repo root, the same file model_test reads).

In file mode nanoinfer reads and tokenizes the file itself, so the comparison
also checks nanoinfer's tokenizer against llama.cpp's.

Usage:
  # Greedy generation with llama.cpp on prompt.txt:
  python3 tools/compare_greedy.py -m ../llama.cpp/models/tinyllama.gguf -n 20

  # Compare llama.cpp with nanoinfer on prompt.txt:
  python3 tools/compare_greedy.py -m ../llama.cpp/models/tinyllama.gguf -n 20 --compare

  # A different prompt file, or an inline prompt:
  python3 tools/compare_greedy.py -m ... -f other.txt --compare
  python3 tools/compare_greedy.py -m ... -p "Hello" --compare
"""

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LLAMA_DIR = os.path.expanduser("~/workspace/llama.cpp")
DEFAULT_COMPLETION = os.path.join(DEFAULT_LLAMA_DIR, "build/bin/llama-completion")
DEFAULT_TOKENIZE = os.path.join(DEFAULT_LLAMA_DIR, "build/bin/llama-tokenize")
DEFAULT_NANOINFER = os.path.join(REPO_ROOT, "model_test")
DEFAULT_PROMPT_FILE = os.path.join(REPO_ROOT, "prompt.txt")


def read_prompt_file(path):
    """Read a prompt file like llama.cpp's -f: drop exactly one trailing newline."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if text.endswith("\n"):
        text = text[:-1]
    return text


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


def run_llamacpp(model_path, prompt, prompt_file, n_predict, completion_bin):
    """Run llama-completion with greedy sampling (--temp 0).

    If prompt_file is set it is passed with -f (same newline handling as
    nanoinfer); otherwise the inline prompt is passed with -p.
    """
    if not os.path.isfile(completion_bin):
        print(f"Error: llama-completion binary not found at {completion_bin}", file=sys.stderr)
        sys.exit(1)

    cmd = [completion_bin, "-m", model_path]
    cmd += ["-f", prompt_file] if prompt_file else ["-p", prompt]
    cmd += [
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


def run_nanoinfer(model_path, token_ids, prompt_file, n_predict, nanoinfer_bin):
    """Run nanoinfer's model_test.

    With prompt_file, model_test reads and tokenizes it itself (no token IDs on
    the command line). Otherwise the given token IDs are passed.
    """
    if not os.path.isfile(nanoinfer_bin):
        print(f"Error: nanoinfer binary not found at {nanoinfer_bin}. Run `make` first.", file=sys.stderr)
        return None

    env = os.environ.copy()
    env["NANO_N"] = str(n_predict)
    if prompt_file:
        cmd = [nanoinfer_bin, model_path]
        env["NANO_PROMPT_FILE"] = os.path.abspath(prompt_file)
    else:
        cmd = [nanoinfer_bin, model_path] + [str(t) for t in token_ids]

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"nanoinfer failed with returncode {res.returncode}:\n{res.stderr}", file=sys.stderr)
        return None

    return res.stdout


def parse_nanoinfer_output(output):
    """Extract prompt token IDs, generated text and generated token IDs from model_test output."""
    gen_text = ""
    gen_tokens = []
    prompt_tokens = []

    # "Prompt tokens (12): 1 450 7483 ..."
    prompt_match = re.search(r"Prompt tokens \(\d+\):([0-9 ]*)", output)
    if prompt_match:
        prompt_tokens = [int(x) for x in prompt_match.group(1).split()]

    # Extract "Generated text: <text>\n(<N> tokens)"
    text_match = re.search(r"Generated text: (.*?)(?:\n\(\d+ tokens\))", output, re.DOTALL)
    if text_match:
        gen_text = text_match.group(1).strip("\r\n")

    # Extract "Generated token IDs: 2018 29889 ..."
    tokens_match = re.search(r"Generated token IDs:\s*([0-9 ]+)", output)
    if tokens_match:
        gen_tokens = [int(x) for x in tokens_match.group(1).split()]

    return prompt_tokens, gen_text, gen_tokens


def main():
    parser = argparse.ArgumentParser(description="Run greedy generation with llama.cpp and compare with nanoinfer")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt", "-p", default=None, help="Inline prompt text (overrides the prompt file)")
    parser.add_argument("--prompt-file", "-f", default=DEFAULT_PROMPT_FILE,
                        help="Prompt file used when -p is not given (default: prompt.txt in the repo root)")
    parser.add_argument("-n", "--n-predict", type=int, default=20, help="Number of tokens to generate (default: 20)")
    parser.add_argument("--compare", action="store_true", help="Also run nanoinfer and compare outputs side-by-side")
    parser.add_argument("--completion-bin", default=DEFAULT_COMPLETION, help="Path to llama-completion binary")
    parser.add_argument("--tokenize-bin", default=DEFAULT_TOKENIZE, help="Path to llama-tokenize binary")
    parser.add_argument("--nanoinfer-bin", default=DEFAULT_NANOINFER, help="Path to nanoinfer model_test binary")
    args = parser.parse_args()

    if args.prompt is not None:
        prompt_file = None
        prompt = args.prompt
    else:
        prompt_file = args.prompt_file
        if not os.path.isfile(prompt_file):
            print(f"Error: prompt file not found: {prompt_file} (use -p or -f)", file=sys.stderr)
            sys.exit(1)
        prompt = read_prompt_file(prompt_file)

    if prompt_file:
        print(f"Prompt file: {prompt_file}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Generating {args.n_predict} tokens (greedy, temp=0)...\n")

    # 1. llama.cpp output
    llama_out = run_llamacpp(args.model, prompt, prompt_file, args.n_predict, args.completion_bin)

    print("=" * 60)
    print(" [llama.cpp output]")
    print("=" * 60)
    print(llama_out.strip("\r\n"))
    print("=" * 60)

    # 2. Optional nanoinfer comparison
    if args.compare:
        # Reference token IDs from llama.cpp's tokenizer
        ref_ids = tokenize_prompt(prompt, args.model, args.tokenize_bin)
        if prompt_file is None and not ref_ids:
            print("\nCould not tokenize prompt for nanoinfer (llama-tokenize unavailable).", file=sys.stderr)
            return

        if prompt_file is None:
            print(f"\nPrompt token IDs: {ref_ids}")
        nano_raw = run_nanoinfer(args.model, ref_ids, prompt_file, args.n_predict, args.nanoinfer_bin)
        if nano_raw:
            nano_prompt_ids, nano_text, nano_tokens = parse_nanoinfer_output(nano_raw)

            # In file mode nanoinfer tokenized the text itself: check it.
            if prompt_file:
                print(f"\nnanoinfer prompt tokens: {nano_prompt_ids}")
                if ref_ids:
                    print(f"llama.cpp prompt tokens: {ref_ids}")
                    if nano_prompt_ids == ref_ids:
                        print("✅ Prompt tokenization matches")
                    else:
                        print("⚠️ Prompt tokenization DIFFERS")
                else:
                    print("(llama-tokenize unavailable; skipping tokenization check)")

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
