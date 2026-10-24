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

Measured on Apple Silicon (single-threaded ARM NEON with multi-accumulators) running **TinyLlama 1.1B (`Q4_K_M`)**:

- **Decode Latency:** **~151 ms/token** (~6.60 tokens/sec) — down from **1041 ms/token** baseline (**~6.9× speedup**)
- **DRAM Traffic per Token:** **0 GB** (fused in-register quantized GEMV)

### Breakdown by Component (from `probe.csv`)

| Component / Stage | Phase | Samples | Avg Latency | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `decode:forward_token` | `decode` | 50 | **151.48 ms** | Total per-token decode step |
| `all_layers` | `decode` | 50 | **122.67 ms** | 22 transformer layers combined |
| ├── `layer:ffn` | `decode` | 1100 | **4.40 ms** | SwiGLU FFN (Gate + Up + Down projections) |
| ├── `layer:qkv_proj` | `decode` | 1100 | **0.64 ms** | Q, K, V projections |
| ├── `layer:attn_out_proj` | `decode` | 1100 | **0.51 ms** | Attention output projection |
| ├── `layer:mha` | `decode` | 1100 | **0.02 ms** | Multi-head attention (KV cache lookup & softmax) |
| ├── `layer:rope` | `decode` | 1100 | **0.01 ms** | Rotary Position Embeddings |
| └── `layer:attn_norm` | `decode` | 1100 | **0.002 ms** | RMSNorm |
| `embedding_lookup` | `decode` | 50 | **20.34 ms** | Token embedding dequantization |
| `final_norm_and_lm_head`| `decode` | 50 | **8.47 ms** | RMSNorm + Vocab projection (fused `Q6_K`) |
| `decode:argmax` | `decode` | 50 | **0.02 ms** | Greedy token sampling |

To reproduce these metrics on your machine:
```bash
python3 benchmarks/run_bench.py path/to/tinyllama.gguf 20
```

---

## Roadmap

- [x] Fast GGUF binary parser with `mmap`
- [x] Custom dequantization routines (`Q8_0`, `Q4_K`, `Q6_K`, `F16`)
- [x] Attention mechanism (GQA), RoPE, and KV Cache
- [x] Tokenizer (SentencePiece / BPE)
- [x] End-to-end autoregressive text generation
- [x] Fused in-register quantized GEMV (`Q8_0`, `Q4_K`, `Q6_K`)
- [x] ARM NEON SIMD vectorization & Multi-Accumulators (ILP)
- [ ] Multi-threaded GEMV (distributing outer row loop across CPU cores)
- [ ] Quantized embedding lookup optimization

---

## License

MIT
