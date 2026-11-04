# nanoinfer

> **Big models. Tiny footprint. Zero dependencies.**

`nanoinfer` is a bare-metal, pure C++ LLM inference engine and GGUF parser built from first principles. It loads multi-gigabyte models directly from disk via memory-mapping (`mmap`), parses raw GGUF binary formats, and executes custom quantization kernels with zero external libraries.

---

## Features

- **Zero External Dependencies**: Built entirely with standard C++17 and POSIX system calls (`mmap`, `munmap`). No PyTorch, no llama.cpp, no third-party runtimes.
- **Zero-Copy Memory-Mapped I/O**: Maps model weights directly into virtual memory. Instant startup time and minimal RAM overhead even for large models.
- **Full GGUF v2/v3 Parser**:
  - Binary header and magic number validation
  - Arbitrary metadata key-value parser (strings, arrays, integers, floats, booleans)
  - Tensor directory extraction with dimension and byte offset calculation
- **Custom Dequantization Kernels**:
  - **`Q8_0`**: 32-weight blocks with 16-bit float scales (~3.7× compression vs. F32)
  - **`Q4_K`**: 256-weight super-blocks with packed sub-block scales and 4-bit nibbles (~7× compression vs. F32)
  - **`F16` to `F32`**: Bitwise IEEE 754 half-precision float expansion

---

## Project Structure

```text
nanoinfer/
├── model_loader.h      # GGUF header, tensor structures, and model loading API
├── model_loader.cpp    # mmap file loading, metadata parsing, tensor indexer
├── dequant.h           # Block structures (BlockQ8_0, BlockQ4_K) and dequant signatures
├── dequant.cpp         # Bit-manipulation and scale expansion routines
├── model_test.cpp      # Verification CLI for tensor inspection & dequant testing
├── Makefile            # Build configuration (clang++ -O3)
└── .gitignore
```

---

## Getting Started

### Prerequisites

A modern C++ compiler supporting C++17 (e.g. `clang++` or `g++`) and `make`.

### Build

```bash
make
```

This compiles the `model_test` binary with `-O3` optimizations.

### Run

Point the binary to any GGUF model file:

```bash
./model_test path/to/model.gguf
```

Example output:
```text
=== GGUF Header ===
magic:               0x46554747 (GGUF)
version:             3
tensor_count:        291
metadata_kv_count:   24

--- F32 tensor: blk.0.attn_norm.weight ---
first 5 values (direct, no dequant needed): 0.402344 0.380859 0.412109 ...

--- Q4_K tensor: blk.0.attn_q.weight ---
dims: [4096, 4096]
first 10 dequantized weights: -0.012451 0.003906 -0.021484 ...
```

---

## Performance & Benchmarks

Measured on Apple Silicon (Apple M3, ARM NEON SDOT `vdotq_s32`, dynamic work-stealing thread pool, 6 threads) running **TinyLlama 1.1B (`Q4_K_M`)** across the full **2,048 context window** (649 prefill tokens in batches of $B=16$, followed by 1,399 decode tokens generated):

- **Batched Prefill Throughput:** **~6.0 ms/token** median (**~165.9 tokens/sec**, `4,312 ms` total for 649 prompt tokens) using 4-token register-blocked NEON GEMM (`matmul_q4_k` / `matmul_q6_k`), achieving up to **~458 GFLOP/s** and a **~2.5× speedup** over sequential single-token prefill.
- **Average Decode Latency:** **~17.3 ms/token** median (**~57.7 tokens/sec**, down from **18.9 ms/tok** before in-register NEON scale accumulation and **1041 ms/tok** naive baseline — **~60× speedup**).
- **Graceful Context Scaling:** Decode latency scales smoothly from **15.10 ms/token** (66.2 tok/s) at position 650 to **20.42 ms/token** (49.0 tok/s) at position 1,980.
- **Fused Multithreaded NEON Attention:** Multi-head attention (MHA) across all 22 layers takes only **6.58 ms** total at position 1,980 (down from **34.4 ms** unvectorized single-threaded baseline — **~5.2× speedup**).
- **DRAM Allocation per Token:** **0 bytes** (fused in-register quantized GEMV / GEMM).

### Profiling Analysis (649 Prefill Tokens + 1,399 Decode Tokens across 2,048 Context)

![Performance Profile across 2048 Context Length](probe.png)

### Per-Token Stage Breakdown: Prefill ($B=16$, 649 tokens) vs. Decode ($B=1$, 1,399 tokens)

| Stage (summed over 22 layers) | Prefill (`ms/tok`) | Prefill Share | Decode (`ms/tok`) | Decode Share | Prefill Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `layer:ffn` (SwiGLU `Q4_K` / `Q6_K`) | **4.06 ms** | 61.0% | **8.30 ms** | 47.8% | **2.04×** |
| `layer:mha` (Causal NEON GQA) | **0.97 ms** | 14.6% | **4.45 ms** | 25.6% | *4.59× (shorter avg pos)* |
| `layer:qkv_proj` (`Q4_K` / `Q6_K`) | **0.78 ms** | 11.8% | **2.29 ms** | 13.2% | **2.94×** |
| `layer:attn_out_proj` (`Q4_K`) | **0.51 ms** | 7.6% | **1.26 ms** | 7.3% | **2.47×** |
| `final_norm_and_lm_head` (`Q6_K`) | **0.003 ms** | 0.1% | **0.80 ms** | 4.6% | *Evaluated once on last tok* |
| `layer:rope` | **0.18 ms** | 2.7% | **0.21 ms** | 1.2% | 1.17× |
| `layer:attn_norm` | **0.04 ms** | 0.6% | **0.06 ms** | 0.3% | 1.50× |
| `embedding_lookup` | **0.11 ms** | 1.7% | **0.004 ms** | 0.0% | — |
| **Total (Median Step)** | **6.02 ms/tok** | **165.9 tok/s** | **17.32 ms/tok** | **57.7 tok/s** | **2.88×** |

### Raw Probe Summary (from `probe.csv`)

| Component / Stage | Phase | Samples | Mean | p50 | p99 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `prefill_total` | `prefill` | 1 | **4312.40 ms** | **4312.40 ms** | 4312.40 ms | Full 649-token prompt prefill (`6.02 ms/tok` med) |
| `prefill:forward_token` | `prefill` | 41 | **105.18 ms** | **96.35 ms** | 106.37 ms | Per 16-token prefill batch (~165.9 tok/s med) |
| `decode:forward_token` | `decode` | 1,399 | **17.38 ms** | **17.32 ms** | 21.79 ms | Full per-token decode step (~57.7 tok/s med) |
| `all_layers` | `decode` | 1,399 | **16.57 ms** | **16.52 ms** | 20.89 ms | 22 transformer layers combined |
| ├── `layer:ffn` | `decode` | 30,778 | **0.377 ms** | **0.367 ms** | 0.556 ms | SwiGLU FFN (`Q4_K` / `Q6_K` SDOT kernels) |
| ├── `layer:mha` | `decode` | 30,778 | **0.202 ms** | **0.199 ms** | 0.324 ms | Fused NEON multi-head attention across 32 heads |
| ├── `layer:qkv_proj` | `decode` | 30,778 | **0.104 ms** | **0.101 ms** | 0.155 ms | Q, K, V projections |
| ├── `layer:attn_out_proj` | `decode` | 30,778 | **0.057 ms** | **0.055 ms** | 0.095 ms | Attention output projection |
| ├── `layer:rope` | `decode` | 30,778 | **0.009 ms** | **0.009 ms** | 0.011 ms | Rotary Position Embeddings |
| └── `layer:attn_norm` | `decode` | 30,778 | **0.003 ms** | **0.003 ms** | 0.003 ms | RMSNorm |
| `final_norm_and_lm_head`| `decode` | 1,399 | **0.803 ms** | **0.789 ms** | 1.069 ms | RMSNorm + Vocab projection (`Q6_K` SDOT) |
| `embedding_lookup` | `decode` | 1,399 | **0.004 ms** | **0.001 ms** | 0.002 ms | Single-row dequantization |
| `decode:argmax` | `decode` | 1,400 | **0.021 ms** | **0.020 ms** | 0.066 ms | Greedy token sampling |

### Latency Scaling with Context Length

| Token Position | Phase | Latency per Token | Throughput | 22-Layer MHA Total |
| :--- | :--- | :--- | :--- | :--- |
| `pos = 111` | `prefill` ($B=16$) | **5.69 ms/tok** | **175.8 tok/s** | 0.64 ms/tok (0.029 ms/layer) |
| `pos = 511` | `prefill` ($B=16$) | **6.28 ms/tok** | **159.3 tok/s** | 1.27 ms/tok (0.058 ms/layer) |
| `pos = 650` | `decode` ($B=1$) | **15.10 ms/tok** | **66.2 tok/s** | 2.16 ms/tok (0.098 ms/layer) |
| `pos = 1000` | `decode` ($B=1$) | **15.37 ms/tok** | **65.1 tok/s** | 3.18 ms/tok (0.145 ms/layer) |
| `pos = 1500` | `decode` ($B=1$) | **17.47 ms/tok** | **57.3 tok/s** | 4.81 ms/tok (0.219 ms/layer) |
| `pos = 1980` | `decode` ($B=1$) | **20.42 ms/tok** | **49.0 tok/s** | 6.58 ms/tok (0.299 ms/layer) |

To reproduce these metrics and generate the plots on your machine:
```bash
# Run full 2,048-context generation with the 649-token prompt
NANO_IGNORE_EOS=1 NANO_PROMPT_FILE=prompts/prompt_500.txt NANO_N=2000 ./model_test ../llama.cpp/models/tinyllama.gguf

# Plot both prefill and decode probe results side-by-side
python3 tools/plot_probe.py probe.csv --phase both -o probe.png

# Benchmark batched prefill across B in {1, 2, 4, 8, 16, 32, 64}
python3 tools/bench_prefill.py -m ../llama.cpp/models/tinyllama.gguf
```

---

## Roadmap

- [x] Fast GGUF binary parser with `mmap`
- [x] Custom dequantization routines (`Q8_0`, `Q4_K`, `Q6_K`, `F16`)
- [x] Attention mechanism (GQA), RoPE, and KV Cache
- [x] Tokenizer (SentencePiece / BPE)
- [x] End-to-end autoregressive text generation
- [x] Single-row embedding lookup optimization (20.5 ms -> 0.09 ms)
- [x] ARM NEON SDOT (`vdotq_s32`) fused quantized GEMV for `Q4_K` and `Q6_K`
- [x] In-register NEON scale accumulation (1 horizontal `vaddvq_s32` reduction per 256-weight super-block)
- [x] Fused `gate` + `up` projections (saving 22 thread sync barriers per step)
- [x] Dynamic atomic work-stealing thread pool with QoS across heterogeneous cores
- [x] Fused multithreaded NEON Multi-Head Attention kernel (`mha_decode`)
- [x] 4-token activation register-blocked NEON GEMM (`matmul_q4_k`, `matmul_q6_k`) and batched `forward_batch` prefill
- [ ] Speculative decoding (N-Gram / Prompt-Lookup drafting with $B=4$ parallel verification)
- [ ] `Q8_0` quantized KV cache
- [ ] Pipeline parallelism over TCP sockets

---

## License

MIT
