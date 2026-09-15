#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include "model_loader.h"

// ── Hyperparameters (extracted from GGUF metadata) ──

struct LlamaHparams {
    uint32_t n_vocab;         // vocabulary size (32000 for TinyLlama)
    uint32_t n_embd;          // embedding dimension (2048)
    uint32_t n_layers;        // number of transformer blocks (22)
    uint32_t n_heads;         // number of attention heads (32)
    uint32_t n_kv_heads;      // number of KV heads for GQA (4)
    uint32_t n_ff;            // feed-forward hidden size (5632)
    uint32_t n_ctx;           // max context length (2048)
    float    rope_freq_base;  // RoPE frequency base (10000.0)
};

// ── One transformer block ──
// Each field points to a TensorInfo in the underlying Model.
// Use get_tensor_data() to get the raw weight pointer for inference.

struct LlamaLayer {
    // pre-attention norm (RMSNorm)
    TensorInfo *attn_norm;

    // self-attention projections
    TensorInfo *attn_q;
    TensorInfo *attn_k;
    TensorInfo *attn_v;
    TensorInfo *attn_output;

    // pre-FFN norm (RMSNorm)
    TensorInfo *ffn_norm;

    // feed-forward network (SwiGLU: gate * silu(up), then down)
    TensorInfo *ffn_gate;
    TensorInfo *ffn_up;
    TensorInfo *ffn_down;
};

// ── The complete LLaMA model ──

struct LlamaModel {
    // the underlying GGUF model (owns the mmap'd data)
    Model *model;

    // hyperparameters
    LlamaHparams hparams;

    // ── global tensors ──
    TensorInfo *token_embd;     // token embeddings  [n_vocab, n_embd]
    TensorInfo *output_norm;    // final RMSNorm     [n_embd]
    TensorInfo *output;         // LM head           [n_vocab, n_embd]

    // ── transformer layers ──
    LlamaLayer *layers;         // array of hparams.n_layers
};

// ── API ──

// Load a LLaMA model from a GGUF file.
// Parses GGUF, extracts hyperparams from metadata, maps tensors into the struct.
// Returns nullptr on failure.
LlamaModel *load_llama_model(const char *path);

// Free the LlamaModel and the underlying GGUF model.
void free_llama_model(LlamaModel *llama);

// Print the model architecture: hyperparams + per-layer tensor summary.
void display_llama_model(LlamaModel *llama);

#endif // LLAMA_MODEL_H
