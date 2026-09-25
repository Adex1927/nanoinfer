#ifndef INFER_STATE_H
#define INFER_STATE_H

#include "llama_model.h"

// ── InferState ──
// All mutable buffers needed during a forward pass.
//
// The model weights (LlamaModel) never change — they stay mmap'd and quantized.
// InferState holds everything that gets written to on every token:
//   - the activation flowing through layers  (x, xb)
//   - the attention projections              (q, k, v)
//   - the attention score scratch buffer     (att)
//   - the FFN hidden buffer                  (hb)
//   - the final output logits                (logits)
//   - the KV cache                           (k_cache, v_cache)
//
// One InferState = one independent inference session.
// Multiple sessions can share the same LlamaModel.

struct InferState {
    // ── per-token activation buffers ──

    float *x;       // [n_embd] — main residual stream (current token's state)
    float *xb;      // [n_embd] — scratch: holds the normed x before attention/FFN

    // ── attention buffers ──

    float *q;       // [n_heads    * head_dim] — query projection
    float *k;       // [n_kv_heads * head_dim] — key   projection
    float *v;       // [n_kv_heads * head_dim] — value projection
    float *att;     // [n_heads * n_ctx]        — attention scores (one row per head)

    // ── FFN buffers ──
    // SwiGLU: output = silu(gate) * up, then projected by down.
    // gate and up are computed simultaneously from the same xb input,
    // so both must exist at once before the element-wise multiply.

    float *hb;      // [n_ff] — gate projection (after silu)
    float *hb2;     // [n_ff] — up   projection

    // ── output ──

    float *logits;  // [n_vocab] — raw scores over vocabulary (last layer output)

    // ── KV cache ──
    // Stores the key and value projections for every layer and every past position.
    // Layout: k_cache[layer * n_ctx * kv_dim + pos * kv_dim + d]
    // where kv_dim = n_kv_heads * head_dim
    //
    // This is what enables autoregressive generation: we compute K and V once
    // per token and cache them, so we never recompute past context.

    float *k_cache; // [n_layers * n_ctx * kv_dim]
    float *v_cache; // [n_layers * n_ctx * kv_dim]

    // ── dimensions (stored for convenience) ──

    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;   // n_embd / n_heads
    int kv_dim;     // n_kv_heads * head_dim
    int n_embd;
    int n_ff;
    int n_ctx;
    int n_vocab;
};

// Allocate all buffers based on model hyperparameters.
// Returns nullptr on allocation failure.
InferState *alloc_infer_state(const LlamaHparams *hp);

// Free all buffers and the InferState itself.
void free_infer_state(InferState *s);

#endif // INFER_STATE_H
