#!/usr/bin/env python3
"""
Bare-minimum correctness tests for nanoinfer against llama.cpp.

  1. Tokenizer: nanoinfer's text -> token IDs must match llama-tokenize exactly.
  2. Logits:    for each prompt, nanoinfer's next-token logits (after the same
                token IDs) must match llama.cpp's full logit vector closely.

Exit code is 0 only if everything passes.

Usage:
  make
  python3 tools/test_nanoinfer.py -m ../llama.cpp/models/tinyllama.gguf
  python3 tools/test_nanoinfer.py -m ... --tokenizer-only     # fast (no forward pass)
  python3 tools/test_nanoinfer.py -m ... --tol 0.05

Notes:
  * By default llama.cpp runs on its default device (Metal GPU on a Mac). Use
    --cpu to force the CPU backend (-ngl 0). The two backends use different
    numerics, so the reference itself moves by ~0.1-0.2 in logits between them;
    see the --tol help.
  * Small differences are expected from summation order / activation precision;
    a near-tie in the top-1 is reported but not failed.
  * Prompts passed to llama.cpp must not contain backslashes (llama.cpp
    processes escapes in -p).
"""

import argparse
import glob
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from array import array

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_BIN = os.path.expanduser("~/workspace/llama.cpp/build/bin")
DEFAULT_NANOINFER = os.path.join(REPO_ROOT, "model_test")
DEFAULT_PROMPT_FILE = os.path.join(REPO_ROOT, "prompt.txt")

# No trailing newline (model_test strips one, like llama.cpp -f) and no backslashes.
TOKENIZER_CASES = [
    "Hello",
    "Hello world",
    "  two leading spaces",
    "trailing spaces  ",
    "multiple   inner   spaces",
    "Line one\nLine two",
    "Numbers: 12345 and 3.14159",
    "Punctuation!? (yes) [no] {maybe}; 'quote' \"dq\"",
    "def f(x):\n    return x * 2",
    "naïve café résumé",
    "日本語のテキスト",
    "emoji 😀 test",
    "UPPER lower MiXeD",
    "The capital of France is Paris, which is famous for",
]

LOGIT_PROMPTS = [
    "Hello",
    "def fibonacci(n):\n    if n <= 1:",
]

TOP_K = 10


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, ok, name, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1


def run_nano(nano_bin, model, args=(), env_extra=None):
    env = os.environ.copy()
    env.update(env_extra or {})
    res = subprocess.run([nano_bin, model, *args], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"nanoinfer failed ({res.returncode}):\n{res.stderr[-500:]}")
    return res.stdout


# ── 1. tokenizer ──

def llama_tokenize(model, text):
    res = subprocess.run(
        [os.path.join(LLAMA_BIN, "llama-tokenize"), "-m", model, "-p", text, "--ids", "--log-disable"],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed:\n{res.stderr[-500:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def nano_tokenize(nano_bin, model, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        out = run_nano(nano_bin, model, env_extra={"NANO_TOKENIZE_ONLY": "1", "NANO_PROMPT_FILE": path})
    finally:
        os.unlink(path)
    m = re.search(r"Prompt tokens \(\d+\):([0-9 ]*)", out)
    return [int(x) for x in m.group(1).split()] if m else None


def test_tokenizer(r, nano_bin, model):
    print("\n[tokenizer] nanoinfer vs llama-tokenize")
    for text in TOKENIZER_CASES:
        ref = llama_tokenize(model, text)
        got = nano_tokenize(nano_bin, model, text)
        detail = "" if got == ref else f"\n         llama.cpp: {ref}\n         nanoinfer: {got}"
        r.check(got == ref, repr(text), detail)


# ── 2. logits ──

def llama_logits(model, prompt, cpu):
    """Run llama-debug; return (prompt_token_ids, last-position logits)."""
    outdir = tempfile.mkdtemp(prefix="nanoinfer_test_")
    cmd = [os.path.join(LLAMA_BIN, "llama-debug"), "-m", model, "-p", prompt,
           "--save-logits", "--logits-output-dir", outdir, "--temp", "0", "-n", "1"]
    if cpu:
        cmd += ["-ngl", "0"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"llama-debug failed ({res.returncode}):\n{res.stderr[-500:]}")

    bins = [f for f in glob.glob(os.path.join(outdir, "*.bin")) if not f.endswith("-tokens.bin")]
    toks = glob.glob(os.path.join(outdir, "*-tokens.bin"))
    if not bins or not toks:
        raise RuntimeError(f"llama-debug produced no logits/tokens files in {outdir}")

    with open(toks[0], "rb") as f:
        data = f.read()
    ids = list(struct.unpack(f"<{len(data)//4}i", data))

    logits = array("f")
    with open(bins[0], "rb") as f:
        logits.frombytes(f.read())
    return ids, logits


def nano_topk(nano_bin, model, ids, k):
    """Run nanoinfer on token IDs; return {token_id: logit} for the top-k next-token logits."""
    out = run_nano(nano_bin, model, [str(i) for i in ids],
                   {"NANO_N": "1", "NANO_TOPK": str(k),
                    "NANO_PROBE_CSV": os.path.join(tempfile.gettempdir(), "nano_probe_test.csv")})
    m = re.search(r"^step 0:(.*)$", out, re.MULTILINE)
    if not m:
        raise RuntimeError("could not find 'step 0:' top-k line in nanoinfer output")
    return {int(a): float(b) for a, b in (p.split(":") for p in m.group(1).split())}


def test_logits(r, nano_bin, model, prompts, tol, k, n_vocab, cpu):
    print(f"\n[logits] nanoinfer vs llama.cpp ({'CPU' if cpu else 'default device'}), top-{k}, tol={tol}")
    for prompt in prompts:
        ids, ref_all = llama_logits(model, prompt, cpu)
        ref = ref_all[-n_vocab:]  # last position's logits
        nano = nano_topk(nano_bin, model, ids, k)

        ref_top = sorted(range(n_vocab), key=lambda i: ref[i], reverse=True)[:k]
        nano_top = sorted(nano, key=lambda i: nano[i], reverse=True)

        label = f"{prompt!r} ({len(ids)} tokens)"
        max_diff = max(abs(nano[i] - ref[i]) for i in nano)
        r.check(max_diff <= tol, f"{label}: max |logit diff| over nano top-{k}", f"= {max_diff:.4f}")

        overlap = len(set(nano_top) & set(ref_top))
        r.check(overlap >= k - 2, f"{label}: top-{k} set overlap", f"= {overlap}/{k}")

        gap = ref[ref_top[0]] - ref[nano_top[0]]
        if nano_top[0] == ref_top[0]:
            r.check(True, f"{label}: top-1 token", f"= {nano_top[0]}")
        else:
            r.check(gap <= tol, f"{label}: top-1 token (near-tie allowed)",
                    f"nano={nano_top[0]} llama={ref_top[0]} llama gap={gap:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Minimal nanoinfer correctness tests vs llama.cpp")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--nanoinfer-bin", default=DEFAULT_NANOINFER)
    ap.add_argument("--tol", type=float, default=0.1, help="max allowed |logit diff| (default 0.1)")
    ap.add_argument("--n-vocab", type=int, default=32000, help="vocabulary size (default 32000)")
    ap.add_argument("--cpu", action="store_true", help="run the llama.cpp reference on CPU only (-ngl 0)")
    ap.add_argument("--tokenizer-only", action="store_true", help="skip the (slow) logit tests")
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE,
                    help="also use this file as a logit-test prompt (default: prompt.txt)")
    args = ap.parse_args()

    if not os.path.isfile(args.nanoinfer_bin):
        sys.exit(f"nanoinfer binary not found: {args.nanoinfer_bin} (run `make`)")

    r = Results()
    test_tokenizer(r, args.nanoinfer_bin, args.model)

    if not args.tokenizer_only:
        prompts = list(LOGIT_PROMPTS)
        if args.prompt_file and os.path.isfile(args.prompt_file):
            with open(args.prompt_file, encoding="utf-8") as f:
                text = f.read()
            prompts.append(text[:-1] if text.endswith("\n") else text)
        test_logits(r, args.nanoinfer_bin, args.model, prompts, args.tol, TOP_K, args.n_vocab, args.cpu)

    print(f"\n{r.passed} passed, {r.failed} failed")
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
