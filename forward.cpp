#include "forward.h"
#include "ops.h"
#include "dequant.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

// Helper: dequantize a named tensor into a freshly calloc'd float buffer.
// Caller must free() the result.
static float *dequant(LlamaModel *llama, TensorInfo *info) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < info->n_dims; d++) n *= info->dims[d];
    float *buf = (float *)calloc(n, sizeof(float));
    void  *raw = get_tensor_data(llama->model, info->name);
    dequantize(raw, buf, n, info->type);
    return buf;
}

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

// ── forward_layer ──
// One complete transformer block. Weights are dequantized one at a time
// and immediately freed — we never hold more than one weight matrix in
// float32 RAM at once, which keeps peak memory low.

void forward_layer(LlamaModel *llama, InferState *s, int layer, int pos) {
    const LlamaHparams &hp = llama->hparams;
    const LlamaLayer   &lw = llama->layers[layer];

    int n_embd  = s->n_embd;
    int kv_dim  = s->kv_dim;
    int n_ff    = s->n_ff;

    // ── Attention block ──

    // 1. Pre-attention RMSNorm: normalize s->x → s->xb using learned scale.
    //    s->x is the residual stream; s->xb is the scratch after normalization.
    {
        float *w = (float *)get_tensor_data(llama->model, lw.attn_norm->name);
        rmsnorm(s->xb, s->x, w, n_embd, hp.rms_norm_eps);
    }

    // 2. Q / K / V projections: s->xb → s->q, s->k, s->v
    {
        float *wq = dequant(llama, lw.attn_q);
        matvec(s->q, wq, s->xb, n_embd, n_embd);
        free(wq);

        float *wk = dequant(llama, lw.attn_k);
        matvec(s->k, wk, s->xb, kv_dim, n_embd);
        free(wk);

        float *wv = dequant(llama, lw.attn_v);
        matvec(s->v, wv, s->xb, kv_dim, n_embd);
        free(wv);
    }

    // 3. RoPE: rotate Q and K in-place to encode position.
    //    V is left unchanged — positions are only needed for dot-product scores.
    rope(s->q, s->k, pos,
         s->n_heads, s->n_kv_heads, s->head_dim,
         (int)hp.rope_dim_count, hp.rope_freq_base);

    // 4. Multi-head attention: reads Q/K/V, writes result to s->xb.
    //    Also writes current K and V into the KV cache at this layer+pos.
    mha(s, layer, pos);

    // 5. Attention output projection: s->xb → xb2, then add to residual s->x.
    //    This projects the attention output back to n_embd dimension.
    //    The residual add is the "skip connection" around the attention block.
    {
        float *wo  = dequant(llama, lw.attn_output);
        float *xb2 = (float *)calloc(n_embd, sizeof(float));
        matvec(xb2, wo, s->xb, n_embd, n_embd);
        free(wo);

        for (int i = 0; i < n_embd; i++) s->x[i] += xb2[i];  // residual add
        free(xb2);
    }

    // ── FFN block (SwiGLU) ──

    // 6. Pre-FFN RMSNorm: normalize updated s->x → s->xb.
    {
        float *w = (float *)get_tensor_data(llama->model, lw.ffn_norm->name);
        rmsnorm(s->xb, s->x, w, n_embd, hp.rms_norm_eps);
    }

    // 7. Gate and Up projections: both read the same s->xb, produce n_ff outputs.
    //    SwiGLU formula: FFN(x) = (silu(gate) ⊙ up) · W_down
    {
        float *wgate = dequant(llama, lw.ffn_gate);
        matvec(s->hb,  wgate, s->xb, n_ff, n_embd);  // gate → hb
        free(wgate);

        float *wup   = dequant(llama, lw.ffn_up);
        matvec(s->hb2, wup,   s->xb, n_ff, n_embd);  // up   → hb2
        free(wup);
    }

    // 8. SwiGLU activation: apply silu to gate, then element-wise multiply by up.
    //    silu(gate) keeps informative values, gates out uninformative ones.
    //    Multiplying by up selects which dimensions flow through.
    silu(s->hb, s->hb, n_ff);                          // hb  = silu(gate)
    for (int i = 0; i < n_ff; i++) s->hb[i] *= s->hb2[i];  // hb = silu(gate) * up

    // 9. Down projection: n_ff → n_embd, then residual add.
    {
        float *wdown = dequant(llama, lw.ffn_down);
        float *xb2   = (float *)calloc(n_embd, sizeof(float));
        matvec(xb2, wdown, s->hb, n_embd, n_ff);
        free(wdown);

        for (int i = 0; i < n_embd; i++) s->x[i] += xb2[i];  // residual add
        free(xb2);
    }
    // s->x now holds the residual after this complete layer.
}
