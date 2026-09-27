#ifndef FORWARD_H
#define FORWARD_H

#include "infer_state.h"
#include "llama_model.h"

// ── forward.h ──
// Stateful forward operations that read/write InferState.
//
// Unlike ops.h (pure math on float*), functions here are model-aware:
// they consume and produce state rather than isolated buffers.
//
// This is also the natural home for alternative attention implementations
// (e.g. sliding window, flash attention, cross-attention, convolution layers)
// as the project grows.

// ── forward_layer ──
// Runs one complete transformer layer: attention block + FFN block.
// Reads quantized weights from llama->layers[layer], dequantizes on the fly.
//
// On entry:  s->x holds the residual stream for the current token.
// On exit:   s->x holds the updated residual after this layer.
//
void forward_layer(LlamaModel *llama, InferState *s, int layer, int pos);

// ── mha — Multi-Head Attention with GQA ──
// Computes scaled dot-product attention for one token at position `pos`.\
//
// Reads:   s->q, s->k, s->v        (set by the caller before this call)
// Writes:  s->xb                   (attention output, [n_embd])
//          s->k_cache, s->v_cache  (current K and V stored at layer/pos slot)
//
// GQA grouping: kv_head = h / (n_heads / n_kv_heads)
//   For TinyLlama: 32 Q heads / 4 KV heads = 8 Q heads per KV group.
//
void mha(InferState *s, int layer, int pos);

// ── forward_token ──
// Runs a full forward pass for a single token at sequence position `pos`:
//   1. Embedding lookup: token id -> s->x
//   2. Run through all N transformer layers (updates KV cache at `pos`)
//   3. If need_logits is true: Final RMSNorm + output projection -> s->logits
//
// If need_logits is false (e.g. during prompt prefill before the last token),
// step 3 is skipped to avoid dequantizing the giant output weight matrix.
void forward_token(LlamaModel *llama, InferState *s, int token, int pos, bool need_logits);

// ── prefill ──
// Feeds a sequence of prompt tokens into the model to populate the KV cache.
// Loops pos from 0 to n_tokens - 1.
// Only computes logits on the final token (pos == n_tokens - 1).
void prefill(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens);

#endif // FORWARD_H
