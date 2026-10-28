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

Measured on Apple Silicon (Apple M3, ARM NEON SDOT `vdotq_s32`, persistent thread pool, 4 threads) running **TinyLlama 1.1B (`Q4_K_M`)** across **200 generated tokens**:

- **Decode Latency:** **~17.0 ms/token** (**~58.8 tokens/sec**) — down from **1041 ms/token** naive baseline (**~61× speedup**)
- **DRAM Traffic per Token:** **0 GB** (fused in-register quantized GEMV)

### Breakdown by Component (averaged over 200 tokens from `probe.csv`)

| Component / Stage | Phase | Samples | Avg Latency | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `decode:forward_token` | `decode` | 200 | **17.01 ms** | Total per-token decode step (~58.8 tok/s) |
| `all_layers` | `decode` | 200 | **16.07 ms** | 22 transformer layers combined |
| ├── `layer:ffn` | `decode` | 4400 | **0.46 ms** | SwiGLU FFN (`Q4_K` / `Q6_K` SDOT kernels) |
| ├── `layer:qkv_proj` | `decode` | 4400 | **0.10 ms** | Q, K, V projections |
| ├── `layer:attn_out_proj` | `decode` | 4400 | **0.06 ms** | Attention output projection |
| ├── `layer:mha` | `decode` | 4400 | **0.09 ms** | Multi-head attention (KV cache lookup & softmax) |
| ├── `layer:rope` | `decode` | 4400 | **0.009 ms** | Rotary Position Embeddings |
| └── `layer:attn_norm` | `decode` | 4400 | **0.002 ms** | RMSNorm |
| `embedding_lookup` | `decode` | 200 | **0.09 ms** | Single-row dequantization (zero full-weight alloc) |
| `final_norm_and_lm_head`| `decode` | 200 | **0.85 ms** | RMSNorm + Vocab projection (`Q6_K` SDOT) |
| `decode:argmax` | `decode` | 200 | **0.02 ms** | Greedy token sampling |

To reproduce these metrics on your machine:
```bash
python3 benchmarks/run_bench.py path/to/tinyllama.gguf 200
```

---

## Roadmap

- [x] Fast GGUF binary parser with `mmap`
- [x] Custom dequantization routines (`Q8_0`, `Q4_K`, `Q6_K`, `F16`)
- [x] Attention mechanism (GQA), RoPE, and KV Cache
- [x] Tokenizer (SentencePiece / BPE)
- [x] End-to-end autoregressive text generation
- [x] Single-row embedding lookup optimization (20.5 ms -> 0.09 ms)
- [x] Persistent zero-heap thread pool (4 threads, static row partition)
- [x] ARM NEON SDOT (`vdotq_s32`) fused quantized GEMV for `Q4_K` and `Q6_K`
- [ ] Fuse `gate` + `up` GEMVs (shared quantized activations)
- [ ] Pipeline parallelism over TCP sockets

---

## License

MIT
