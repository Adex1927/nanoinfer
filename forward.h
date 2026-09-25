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

#endif // FORWARD_H
