#ifndef FORWARD_H
#define FORWARD_H

#include "infer_state.h"

// ── forward.h ──
// Stateful forward operations that read/write InferState.
//
// Unlike ops.h (pure math on float*), functions here are model-aware:
// they consume and produce state rather than isolated buffers.
//
// This is also the natural home for alternative attention implementations
// (e.g. sliding window, flash attention, cross-attention, convolution layers)
// as the project grows.

// ── Multi-Head Attention with GQA ──
// Computes scaled dot-product attention for one token at position `pos`.
//
// Reads:   s->q, s->k, s->v        (set by the caller before this call)
// Writes:  s->xb                   (attention output, [n_embd])
//          s->k_cache, s->v_cache  (current K and V stored at layer/pos slot)
//
// Steps per Q head h:
//   1. Store s->k, s->v into kv_cache[layer][pos]
//   2. score[t] = dot(q[h], k_cache[layer][t][h/group]) / sqrt(head_dim)
//   3. softmax(score[0..pos])
//   4. xb[h] = sum_t( score[t] * v_cache[layer][t][h/group] )
//
// GQA grouping: kv_head = h / (n_heads / n_kv_heads)
//   For TinyLlama: 32 Q heads / 4 KV heads = 8 Q heads per KV group.
//
void mha(InferState *s, int layer, int pos);

#endif // FORWARD_H
