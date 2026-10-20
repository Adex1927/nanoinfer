# nanoinfer benchmark metrics

What we measure, why it matters, how to compute it, and where we stand.
Baseline numbers are from one run of `./model_test tinyllama.gguf` (TinyLlama 1.1B
Q4_K_M, prompt `<s> World`, 50 greedy tokens, probes on, single thread).
Treat them as a reference point, not a benchmark result: one run, one machine.

Data source: `probe.csv` (columns `name,phase,token_pos,layer,ms`), written by
`probe_dump_csv()` and analysed with `tools/plot_probe.py`.

**Status legend:** ✅ computable from `probe.csv` today · 🔧 needs small new
instrumentation · ⏳ needs features we don't have yet (batching, concurrency).

---

## 1. User-facing latency

These are what a user feels. Report them as distributions, not just means.

| Metric | Definition | Computation | Status | Baseline |
|---|---|---|---|---|
| **TTFT** (time to first token) | Prompt submitted → first token emitted. Responsiveness. | `prefill` + first `decode:argmax` | ✅ | ~2.73 s (2-token prompt) |
| **TPOT / ITL** (time per output token / inter-token latency) | Gap between consecutive streamed tokens. Reading speed. | `decode:forward_token` per `token_pos` | ✅ | mean 1041 ms |
| **Tail latency** | p50 / p95 / p99 of TPOT. Stutter is noticed more than the average. | percentiles of `decode:forward_token` | ✅ | p50 1037 ms, p99 1153 ms |
| **Decode throughput** | Output tokens per second for one stream. | `1000 / mean TPOT` | ✅ | ~0.96 tok/s |
| **Prefill throughput** | Prompt tokens processed per second. Matters for long prompts / RAG. | `n_prompt / prefill_total` | ✅ | ~0.7 tok/s (token-by-token prefill) |
| **End-to-end latency** | Request → last token. | `TTFT + (N-1) × TPOT` | ✅ | ~55 s for 50 tokens |

Notes:
- TTFT scales with prompt length; always record the prompt length next to it.
- Report TPOT against context length (`token_pos`) to see attention cost grow.
- Our prefill runs one token at a time, so prefill throughput ≈ decode throughput.
  Batched prefill is a major future win, since prefill can be compute-bound.
- **Tokenizer overhead:** `tokenize()` in `tokenizer.cpp` currently uses a naive greedy BPE implementation with $O(N^2)$ complexity (rescanning all adjacent pairs on every merge and shifting `std::vector` elements). It is fast for short prompts (<2 ms), but will become a bottleneck on long inputs. `llama.cpp` avoids this using a priority queue and doubly-linked list for $O(N \log N)$ merges.

## 2. Hardware efficiency

These tell us how far we are from what the hardware can do. They point at the
next optimization.

| Metric | Definition | Computation | Status |
|---|---|---|---|
| **MBU** (memory-bandwidth utilization) | Fraction of peak memory bandwidth used during decode. | `bytes_read_per_token / TPOT / peak_bandwidth` | 🔧 |
| **Roofline TPOT floor** | Best possible decode latency if purely memory-bound. | `model_bytes / peak_bandwidth` | 🔧 |
| **Achieved FLOP/s vs peak** | Compute efficiency; mostly relevant to batched prefill. | `FLOPs / time / peak_FLOPs` | 🔧 |
| **Peak memory (RSS)** | Highest resident memory during the run. | `getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS) | 🔧 |
| **KV-cache size per token** | Memory a token of context costs. Bounds context length and concurrent sessions. | `n_layers × 2 × kv_dim × sizeof(float)` | ✅ |
| **Stage breakdown** | Share of a decode step per stage (FFN, QKV, out-proj, attention, LM head, embedding). | sum of `layer:*` per `token_pos`, plus `embedding_lookup`, `final_norm_and_lm_head` | ✅ |

How to compute the 🔧 items:
- **Bytes read per token.** Decode reads each weight once. Sum the tensor sizes
  (from `TensorInfo`: element count × bytes per element for its quant type).
  For a Q4_K_M 1.1B model this is roughly the file size, ~638 MB.
- **Peak bandwidth.** Look up the machine's spec, or measure with a simple
  large-array copy/stream test. Record which one you used.
- **Why MBU matters.** Decode is memory-bound: each token streams all weights.
  The floor is `model_bytes / bandwidth` (for example, 638 MB at 100 GB/s ≈ 6 ms;
  substitute your machine's real bandwidth). Our TPOT is ~1041 ms, so we are far
  below 1% MBU. The time is going into float32 dequantization and temporary
  buffers, not into memory bandwidth.

Reference values for TinyLlama in this implementation:
- KV cache: `22 × 2 × 256 × 4 B` = **45,056 B ≈ 44 KiB per token** (`kv_dim` = 4 KV heads × 64).
- Largest temporary buffers: LM head and embedding table dequantized to float32,
  `32000 × 2048 × 4 B` ≈ **262 MB each**, allocated per token.

Observed stage breakdown (mean per decode token, summed over 22 layers):

| Stage | Time | Share |
|---|---|---|
| FFN | ~750 ms | ~72% |
| QKV projections | ~110 ms | ~11% |
| Attention output projection | ~87 ms | ~8% |
| LM head (final norm + output) | ~74 ms | ~7% |
| Embedding lookup | ~20 ms | ~2% |
| Attention (mha), RoPE, norms | < 1 ms | < 0.1% |

Attention time grows linearly with context (about 0.7 µs per position per layer
call in our run) but is negligible at short context. Re-measure with a long
generation (for example `NANO_N=500`).

## 3. Quality guardrails

Speed numbers only count while the output stays correct.

| Metric | Definition | How | Status |
|---|---|---|---|
| **Logit agreement** | Max / mean abs difference of logits vs reference. | `tools/dump_logits.py` against llama.cpp | ✅ |
| **Greedy-token match** | Same generated token IDs as the reference with temperature 0. | `tools/compare_greedy.py --compare` | ✅ |
| **Perplexity** | Quality after changing quantization or kernels. | run a fixed text corpus, compare to llama.cpp | ⏳ |

Small float differences (~1e-3 on logits) are expected from summation order.

## 4. Serving metrics

Relevant once nanoinfer handles more than one request at a time. None of these are
measurable today.

| Metric | Definition | Status |
|---|---|---|
| **Requests per second** | Completed requests per second at a given load. | ⏳ |
| **Goodput** | Requests per second that meet a latency target (for example TTFT < 1 s and TPOT < 100 ms). | ⏳ |
| **Aggregate throughput** | Total output tokens per second across all concurrent streams. | ⏳ |
| **Batching efficiency** | Throughput gain from batching vs one stream at a time. | ⏳ |
| **Concurrent sessions** | How many `InferState`s fit in memory (bounded by KV-cache size). | ⏳ |
| **Model load / cold start** | Time from process start to first token ready (mmap, page faults). | 🔧 |

---

## Reporting checklist

For each benchmark run, record so results are comparable:

- git commit, compiler flags (`-O3`, threads, SIMD)
- machine: CPU/SoC, RAM, peak memory bandwidth used for MBU
- model file and quantization (for example TinyLlama 1.1B Q4_K_M)
- prompt length and number of generated tokens
- warm vs cold run (first run includes mmap page faults)
- probes on or off (probe overhead is small but non-zero)
- number of repetitions; report median and p99, not a single run

## Suggested next steps

1. Add a metrics block to `tools/plot_probe.py` for TTFT, TPOT p50/p95/p99, and tokens per second.
2. Write bytes-per-token and peak RSS into the output so MBU can be computed.
3. Record per-token prefill timings, not only the total.
4. Re-run after each optimization (single-row embedding dequantization,
   quantized dot products without dequantization, threading, batched prefill)
   and compare against this baseline.
5. Upgrade `tokenize()` from $O(N^2)$ to $O(N \log N)$ using a priority queue and doubly-linked list.
