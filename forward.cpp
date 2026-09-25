#include "forward.h"
#include "ops.h"   // softmax
#include <cmath>   // sqrtf

// ── mha — Multi-Head Attention with Grouped Query Attention ──
//
// This is the heart of the transformer. It lets every token look back at all
// previous tokens and decide what information to pull in, weighted by relevance.
//
// Called once per layer, once per token during the forward pass.

void mha(InferState *s, int layer, int pos) {
    int head_dim    = s->head_dim;
    int n_heads     = s->n_heads;
    int n_kv_heads  = s->n_kv_heads;
    int kv_dim      = s->kv_dim;    // n_kv_heads * head_dim
    int n_ctx       = s->n_ctx;

    // how many Q heads share one KV head (e.g. 32/4 = 8 for TinyLlama)
    int heads_per_group = n_heads / n_kv_heads;

    // scale applied before softmax to prevent dot products from growing too large.
    // Standard transformer formula: 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    // ── Step 1: write current K and V into the cache ──
    // Cache layout (flat array): [n_layers][n_ctx][kv_dim]
    // Slice for this layer + position:
    float *k_pos = s->k_cache + layer * n_ctx * kv_dim + pos * kv_dim;
    float *v_pos = s->v_cache + layer * n_ctx * kv_dim + pos * kv_dim;

    for (int i = 0; i < kv_dim; i++) k_pos[i] = s->k[i];
    for (int i = 0; i < kv_dim; i++) v_pos[i] = s->v[i];

    // ── Steps 2-4: per Q head ──
    for (int h = 0; h < n_heads; h++) {

        // which KV head this Q head reads from.
        // Integer division groups them: heads 0-7 → KV 0, heads 8-15 → KV 1, etc.
        int kv_head = h / heads_per_group;

        float *qh    = s->q   + h * head_dim;  // this head's query vector
        float *att_h = s->att + h * n_ctx;      // this head's score scratch row
        float *xbh   = s->xb  + h * head_dim;  // this head's output slot in xb

        // ── Step 2: dot-product scores ──
        // score[t] = dot(q[h], k_cache[layer][t][kv_head]) * scale
        for (int t = 0; t <= pos; t++) {
            float *kt = s->k_cache + layer * n_ctx * kv_dim
                                   + t     * kv_dim
                                   + kv_head * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += qh[d] * kt[d];
            }
            att_h[t] = score * scale;
        }

        // ── Step 3: softmax over scores[0..pos] ──
        // Positions pos+1..n_ctx-1 are future tokens — we never attend to them.
        softmax(att_h, att_h, pos + 1);

        // ── Step 4: weighted sum of value vectors ──
        // xb[h] = sum_t( att[t] * v_cache[layer][t][kv_head] )
        for (int d = 0; d < head_dim; d++) xbh[d] = 0.0f;

        for (int t = 0; t <= pos; t++) {
            float *vt = s->v_cache + layer * n_ctx * kv_dim
                                   + t     * kv_dim
                                   + kv_head * head_dim;
            float w = att_h[t];
            for (int d = 0; d < head_dim; d++) {
                xbh[d] += w * vt[d];
            }
        }
    }
    // s->xb now holds the full attention output: [n_heads * head_dim] = [n_embd]
}
