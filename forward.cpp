#include "forward.h"
#include "ops.h"
#include "dequant.h"
#include "probe.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

// Helper: dequantize a named tensor into a freshly calloc'd float buffer.
// Caller must free() the result.
static float *dequant(LlamaModel *llama, TensorInfo *info) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < info->n_dims; d++) n *= info->dims[d];
    float *buf = (float *)calloc(n, sizeof(float));
    void  *raw = get_tensor_data(llama->model, info->name);
    if (!dequantize(raw, buf, n, info->type)) {
        fprintf(stdout, "ERROR: unsupported type %s (%u) for tensor %s\n",
                ggml_type_name(info->type), (unsigned)info->type, info->name);
    }
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

void forward_layer(LlamaModel *llama, InferState *s, int layer, int pos,
                   ProbeAccum *pa_norm,
                   ProbeAccum *pa_qkv,
                   ProbeAccum *pa_rope,
                   ProbeAccum *pa_mha,
                   ProbeAccum *pa_out_proj,
                   ProbeAccum *pa_ffn) {
    const LlamaHparams &hp = llama->hparams;
    const LlamaLayer   &lw = llama->layers[layer];

    int n_embd  = s->n_embd;
    int kv_dim  = s->kv_dim;
    int n_ff    = s->n_ff;

    // ── Attention block ──

    // 1. Pre-attention RMSNorm: normalize s->x → s->xb using learned scale.
    //    s->x is the residual stream; s->xb is the scratch after normalization.
    {
        auto t = pa_norm->start();
        float *w = (float *)get_tensor_data(llama->model, lw.attn_norm->name);
        rmsnorm(s->xb, s->x, w, n_embd, hp.rms_norm_eps);
        pa_norm->stop(t);
    }

    // 2. Q / K / V projections: s->xb → s->q, s->k, s->v
    {
        auto t = pa_qkv->start();
        float *wq = dequant(llama, lw.attn_q);
        matvec(s->q, wq, s->xb, n_embd, n_embd);
        free(wq);

        float *wk = dequant(llama, lw.attn_k);
        matvec(s->k, wk, s->xb, kv_dim, n_embd);
        free(wk);

        float *wv = dequant(llama, lw.attn_v);
        matvec(s->v, wv, s->xb, kv_dim, n_embd);
        free(wv);
        pa_qkv->stop(t);
    }

    // 3. RoPE: rotate Q and K in-place to encode position.
    //    V is left unchanged — positions are only needed for dot-product scores.
    {
        auto t = pa_rope->start();
        rope(s->q, s->k, pos,
             s->n_heads, s->n_kv_heads, s->head_dim,
             (int)hp.rope_dim_count, hp.rope_freq_base);
        pa_rope->stop(t);
    }

    // 4. Multi-head attention: reads Q/K/V, writes result to s->xb.
    //    Also writes current K and V into the KV cache at this layer+pos.
    {
        auto t = pa_mha->start();
        mha(s, layer, pos);
        pa_mha->stop(t);
    }

    // 5. Attention output projection: s->xb → xb2, then add to residual s->x.
    //    This projects the attention output back to n_embd dimension.
    //    The residual add is the "skip connection" around the attention block.
    {
        auto t = pa_out_proj->start();
        float *wo  = dequant(llama, lw.attn_output);
        float *xb2 = (float *)calloc(n_embd, sizeof(float));
        matvec(xb2, wo, s->xb, n_embd, n_embd);
        free(wo);

        for (int i = 0; i < n_embd; i++) s->x[i] += xb2[i];  // residual add
        free(xb2);
        pa_out_proj->stop(t);
    }

    // ── FFN block (SwiGLU) ──
    {
        auto t = pa_ffn->start();

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

        pa_ffn->stop(t);
    }
    // s->x now holds the residual after this complete layer.
}

// ── forward_token ──
// Runs a full forward pass for a single token at sequence position `pos`.
void forward_token(LlamaModel *llama, InferState *s, int token, int pos, bool need_logits) {
    // Defensive guards: prevent out-of-bounds access / buffer overflow
    if (pos < 0 || pos >= s->n_ctx) {
        fprintf(stderr, "ERROR: position %d exceeds context window limit %d\n", pos, s->n_ctx);
        exit(1);
    }
    if (token < 0 || token >= s->n_vocab) {
        fprintf(stderr, "ERROR: token ID %d out of vocabulary bounds [0, %d)\n", token, s->n_vocab);
        exit(1);
    }

    int n_embd = s->n_embd;

    // 1. Embedding lookup: read token embedding into residual stream s->x
    {
        Probe p("embedding_lookup");
        TensorInfo *embd_info = llama->token_embd;
        uint64_t n = 1;
        for (uint32_t d = 0; d < embd_info->n_dims; d++) n *= embd_info->dims[d];
        float *embd = (float *)calloc(n, sizeof(float));
        dequantize(get_tensor_data(llama->model, embd_info->name), embd, n, embd_info->type);
        memcpy(s->x, embd + token * n_embd, n_embd * sizeof(float));
        free(embd);
    }

    // 2. Transformer layers (accumulates K & V into cache at position `pos`)
    //
    // ProbeAccum collects one sample per layer, then reports a single summary
    // line (total / avg / min / max) after the loop — avoids log spam.
    {
        ProbeAccum pa_norm    ("  layer:attn_norm");
        ProbeAccum pa_qkv     ("  layer:qkv_proj");
        ProbeAccum pa_rope    ("  layer:rope");
        ProbeAccum pa_mha     ("  layer:mha");
        ProbeAccum pa_out_proj("  layer:attn_out_proj");
        ProbeAccum pa_ffn     ("  layer:ffn");

        Probe p_layers("all_layers");
        for (int layer = 0; layer < s->n_layers; layer++) {
            forward_layer(llama, s, layer, pos,
                          &pa_norm, &pa_qkv, &pa_rope,
                          &pa_mha, &pa_out_proj, &pa_ffn);
        }
        // p_layers destructor fires here → prints total time for all layers
        // then the ProbeAccum reports print per-stage breakdowns

        pa_norm.report();
        pa_qkv.report();
        pa_rope.report();
        pa_mha.report();
        pa_out_proj.report();
        pa_ffn.report();
    }

    // 3. Final RMSNorm + Output Projection (only when logits are needed)
    if (need_logits) {
        Probe p("final_norm_and_lm_head");
        float *w = (float *)get_tensor_data(llama->model, llama->output_norm->name);
        rmsnorm(s->x, s->x, w, s->n_embd, llama->hparams.rms_norm_eps);

        TensorInfo *out_info = llama->output;
        uint64_t n = 1;
        for (uint32_t d = 0; d < out_info->n_dims; d++) n *= out_info->dims[d];
        float *wout = (float *)calloc(n, sizeof(float));
        dequantize(get_tensor_data(llama->model, out_info->name), wout, n, out_info->type);
        matvec(s->logits, wout, s->x, s->n_vocab, s->n_embd);
        free(wout);
    }
}

// ── prefill ──
// Feeds a sequence of prompt tokens into the model.
// Evaluates each token, storing keys and values in the KV cache.
// Only computes final output logits on the last token.
void prefill(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens) {
    if (n_tokens > s->n_ctx) {
        fprintf(stderr, "ERROR: prompt length %d exceeds context limit %d\n", n_tokens, s->n_ctx);
        exit(1);
    }

    Probe p_prefill("prefill_total");
    for (int pos = 0; pos < n_tokens; pos++) {
        fprintf(stderr, "\n[PROBE] ── token pos=%d ──\n", pos);
        bool need_logits = (pos == n_tokens - 1);
        forward_token(llama, s, tokens[pos], pos, need_logits);
    }
    // p_prefill destructor fires here → prints total prefill wall time
}
