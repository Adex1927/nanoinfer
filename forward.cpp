#include "forward.h"
#include "ops.h"
#include "dequant.h"
#include "probe.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>


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

// ── matvec_tensor ──
// Fused quantized matvec dispatcher:
// Evaluates quantized blocks in-place without heap allocation.
// Falls back to dequant + matvec if format is not yet fused.
static void matvec_tensor(float *out, LlamaModel *llama, TensorInfo *info, const float *x, int n_out, int n_in, BlockQ8_K *q8_buf = nullptr) {
    static const bool force_old = (getenv("NANO_OLD_MATVEC") != nullptr);
    static const DiagMode diag_mode = []() {
        const char *d = getenv("NANO_DIAG");
        if (!d) return DiagMode::NORMAL;
        if (strcmp(d, "mem") == 0) return DiagMode::MEM_ONLY;
        if (strcmp(d, "compute") == 0) return DiagMode::COMPUTE_2X;
        return DiagMode::NORMAL;
    }();

    if (!force_old) {
        const void *raw = get_tensor_data(llama->model, info->name);
        if (info->type == GGML_TYPE_Q4_K && (n_in % 256 == 0)) {
            matvec_q4_k_diag(out, raw, x, n_out, n_in, diag_mode, q8_buf);
            return;
        } else if (info->type == GGML_TYPE_Q8_0 && (n_in % 32 == 0)) {
            matvec_q8_0_diag(out, raw, x, n_out, n_in, diag_mode);
            return;
        } else if (info->type == GGML_TYPE_Q6_K && (n_in % 256 == 0)) {
            matvec_q6_k_diag(out, raw, x, n_out, n_in, diag_mode, q8_buf);
            return;
        }
    }
    float *w = dequant(llama, info);
    matvec(out, w, x, n_out, n_in);
    free(w);
}

// ── matmul_tensor ──
// Batched quantized GEMM dispatcher for n_tokens input vectors.
static void matmul_tensor(float *out, LlamaModel *llama, TensorInfo *info,
                          const float *x, int n_tokens, int n_out, int n_in, BlockQ8_K *q8_buf = nullptr) {
    if (n_tokens == 1) {
        matvec_tensor(out, llama, info, x, n_out, n_in, q8_buf);
        return;
    }

    static const bool force_old = (getenv("NANO_OLD_MATVEC") != nullptr);
    if (!force_old) {
        const void *raw = get_tensor_data(llama->model, info->name);
        if (info->type == GGML_TYPE_Q4_K && (n_in % 256 == 0)) {
            matmul_q4_k(out, raw, x, n_tokens, n_out, n_in, q8_buf);
            return;
        } else if (info->type == GGML_TYPE_Q6_K && (n_in % 256 == 0)) {
            matmul_q6_k(out, raw, x, n_tokens, n_out, n_in, q8_buf);
            return;
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        matvec_tensor(out + (size_t)t * n_out, llama, info, x + (size_t)t * n_in, n_out, n_in, q8_buf);
    }
}

// ── cache_kv ──
// Copies n_tokens K and V vectors from s->k and s->v into layer's KV cache starting at start_pos.
static void cache_kv(InferState *s, int layer, int start_pos, int n_tokens) {
    int kv_dim = s->kv_dim;
    int n_ctx  = s->n_ctx;
    float *k_dst = s->k_cache + (size_t)layer * n_ctx * kv_dim + (size_t)start_pos * kv_dim;
    float *v_dst = s->v_cache + (size_t)layer * n_ctx * kv_dim + (size_t)start_pos * kv_dim;
    memcpy(k_dst, s->k, (size_t)n_tokens * kv_dim * sizeof(float));
    memcpy(v_dst, s->v, (size_t)n_tokens * kv_dim * sizeof(float));
}

// ── mha — Multi-Head Attention with Grouped Query Attention ──
//
// Writes current token's K and V to the KV cache, then invokes the fused
// mha_decode kernel in ops.cpp.
void mha(InferState *s, int layer, int pos) {
    int kv_dim = s->kv_dim;
    int n_ctx  = s->n_ctx;

    // 1. Write current K and V into the cache
    cache_kv(s, layer, pos, 1);

    // 2. Compute multi-head attention (fused ops kernel)
    const float *k_layer = s->k_cache + (size_t)layer * n_ctx * kv_dim;
    const float *v_layer = s->v_cache + (size_t)layer * n_ctx * kv_dim;

    mha_decode(s->xb, s->q, k_layer, v_layer, s->att, pos,
               s->n_heads, s->n_kv_heads, s->head_dim, s->kv_dim, s->n_ctx);
}

void mha_batch(InferState *s, int layer, int start_pos, int n_tokens) {
    int n_embd = s->n_embd;
    int kv_dim = s->kv_dim;
    int n_ctx  = s->n_ctx;

    // 1. Write all n_tokens K and V vectors into the KV cache
    cache_kv(s, layer, start_pos, n_tokens);

    // 2. Causal multi-head attention for each token t at sequence position (start_pos + t)
    const float *k_layer = s->k_cache + (size_t)layer * n_ctx * kv_dim;
    const float *v_layer = s->v_cache + (size_t)layer * n_ctx * kv_dim;

    for (int t = 0; t < n_tokens; t++) {
        mha_decode(s->xb + (size_t)t * n_embd,
                   s->q  + (size_t)t * n_embd,
                   k_layer, v_layer, s->att, start_pos + t,
                   s->n_heads, s->n_kv_heads, s->head_dim, kv_dim, n_ctx);
    }
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
    probe_set_layer(layer);

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
        matvec_tensor(s->q, llama, lw.attn_q, s->xb, n_embd, n_embd, s->q8_buf);
        matvec_tensor(s->k, llama, lw.attn_k, s->xb, kv_dim, n_embd, s->q8_buf);
        matvec_tensor(s->v, llama, lw.attn_v, s->xb, kv_dim, n_embd, s->q8_buf);
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

    // 5. Attention output projection: s->xb → s->xb2, then add to residual s->x.
    //    This projects the attention output back to n_embd dimension.
    //    The residual add is the "skip connection" around the attention block.
    {
        auto t = pa_out_proj->start();
        matvec_tensor(s->xb2, llama, lw.attn_output, s->xb, n_embd, n_embd, s->q8_buf);

        for (int i = 0; i < n_embd; i++) s->x[i] += s->xb2[i];  // residual add
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
            if (lw.ffn_gate->type == GGML_TYPE_Q4_K && lw.ffn_up->type == GGML_TYPE_Q4_K && (n_embd % 256 == 0)) {
                const void *raw_gate = get_tensor_data(llama->model, lw.ffn_gate->name);
                const void *raw_up   = get_tensor_data(llama->model, lw.ffn_up->name);
                matvec_q4_k_gate_up(s->hb, s->hb2, raw_gate, raw_up, s->xb, n_ff, n_embd, s->q8_buf);
            } else {
                matvec_tensor(s->hb,  llama, lw.ffn_gate, s->xb, n_ff, n_embd, s->q8_buf);  // gate → hb
                matvec_tensor(s->hb2, llama, lw.ffn_up,   s->xb, n_ff, n_embd, s->q8_buf);  // up   → hb2
            }
        }

        // 8. SwiGLU activation: apply silu to gate, then element-wise multiply by up.
        silu(s->hb, s->hb, n_ff);                          // hb  = silu(gate)
        for (int i = 0; i < n_ff; i++) s->hb[i] *= s->hb2[i];  // hb = silu(gate) * up

        // 9. Down projection: n_ff → n_embd, then residual add.
        {
            matvec_tensor(s->xb2, llama, lw.ffn_down, s->hb, n_embd, n_ff, s->q8_buf);

            for (int i = 0; i < n_embd; i++) s->x[i] += s->xb2[i];  // residual add
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
    probe_set_pos(pos);

    // 1. Embedding lookup: dequantize only row `token` straight into s->x
    {
        Probe p("embedding_lookup");
        TensorInfo *embd_info = llama->token_embd;
        size_t stride = row_bytes(embd_info->type, n_embd);
        if (stride == 0) {
            fprintf(stderr, "ERROR: unsupported token_embd type %u\n", (unsigned)embd_info->type);
            exit(1);
        }
        const uint8_t *base = (const uint8_t *)get_tensor_data(llama->model, embd_info->name);
        dequantize(base + (size_t)token * stride, s->x, n_embd, embd_info->type);
    }

    // 2. Transformer layers (accumulates K & V into cache at position `pos`)
    //
    // ProbeAccum collects one sample per layer, then reports a single summary
    // line (total / avg / min / max) after the loop — avoids log spam.
    {
        ProbeAccum pa_norm    ("layer:attn_norm");
        ProbeAccum pa_qkv     ("layer:qkv_proj");
        ProbeAccum pa_rope    ("layer:rope");
        ProbeAccum pa_mha     ("layer:mha");
        ProbeAccum pa_out_proj("layer:attn_out_proj");
        ProbeAccum pa_ffn     ("layer:ffn");

        Probe p_layers("all_layers");
        for (int layer = 0; layer < s->n_layers; layer++) {
            forward_layer(llama, s, layer, pos,
                          &pa_norm, &pa_qkv, &pa_rope,
                          &pa_mha, &pa_out_proj, &pa_ffn);
        }
        probe_set_layer(-1);
        // samples are stored in the probe table; dump with probe_dump_csv()
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
        matvec_tensor(s->logits, llama, llama->output, s->x, s->n_vocab, s->n_embd, s->q8_buf);
    }
}

// ── forward_layer_batch ──
// Evaluates one transformer layer across a batch of `n_tokens` tokens starting at `start_pos`.
static void forward_layer_batch(LlamaModel *llama, InferState *s, int layer, int start_pos, int n_tokens,
                                ProbeAccum *pa_norm,
                                ProbeAccum *pa_qkv,
                                ProbeAccum *pa_rope,
                                ProbeAccum *pa_mha,
                                ProbeAccum *pa_out_proj,
                                ProbeAccum *pa_ffn) {
    const LlamaHparams &hp = llama->hparams;
    const LlamaLayer   &lw = llama->layers[layer];
    probe_set_layer(layer);

    int n_embd = s->n_embd;
    int kv_dim = s->kv_dim;
    int n_ff   = s->n_ff;

    // 1. Pre-attention RMSNorm across batch
    {
        auto t0 = pa_norm->start();
        const float *w = (const float *)get_tensor_data(llama->model, lw.attn_norm->name);
        for (int t = 0; t < n_tokens; t++) {
            rmsnorm(s->xb + (size_t)t * n_embd, s->x + (size_t)t * n_embd, w, n_embd, hp.rms_norm_eps);
        }
        pa_norm->stop(t0);
    }

    // 2. Batched Q / K / V projections (4x register-blocked weight reuse)
    {
        auto t0 = pa_qkv->start();
        matmul_tensor(s->q, llama, lw.attn_q, s->xb, n_tokens, n_embd, n_embd, s->q8_buf);
        matmul_tensor(s->k, llama, lw.attn_k, s->xb, n_tokens, kv_dim, n_embd, s->q8_buf);
        matmul_tensor(s->v, llama, lw.attn_v, s->xb, n_tokens, kv_dim, n_embd, s->q8_buf);
        pa_qkv->stop(t0);
    }

    // 3. RoPE per token at position (start_pos + t)
    {
        auto t0 = pa_rope->start();
        for (int t = 0; t < n_tokens; t++) {
            rope(s->q + (size_t)t * n_embd,
                 s->k + (size_t)t * kv_dim,
                 start_pos + t,
                 s->n_heads, s->n_kv_heads, s->head_dim,
                 (int)hp.rope_dim_count, hp.rope_freq_base);
        }
        pa_rope->stop(t0);
    }

    // 4. Batched KV cache update + Causal Multi-Head Attention
    {
        auto t0 = pa_mha->start();
        mha_batch(s, layer, start_pos, n_tokens);
        pa_mha->stop(t0);
    }

    // 5. Batched Attention Output Projection + Residual Add
    {
        auto t0 = pa_out_proj->start();
        matmul_tensor(s->xb2, llama, lw.attn_output, s->xb, n_tokens, n_embd, n_embd, s->q8_buf);
        size_t total_embd = (size_t)n_tokens * n_embd;
        for (size_t i = 0; i < total_embd; i++) s->x[i] += s->xb2[i];
        pa_out_proj->stop(t0);
    }

    // 6..9. Batched FFN Block (SwiGLU)
    {
        auto t0 = pa_ffn->start();

        const float *w = (const float *)get_tensor_data(llama->model, lw.ffn_norm->name);
        for (int t = 0; t < n_tokens; t++) {
            rmsnorm(s->xb + (size_t)t * n_embd, s->x + (size_t)t * n_embd, w, n_embd, hp.rms_norm_eps);
        }

        if (lw.ffn_gate->type == GGML_TYPE_Q4_K && lw.ffn_up->type == GGML_TYPE_Q4_K && (n_embd % 256 == 0)) {
            const void *raw_gate = get_tensor_data(llama->model, lw.ffn_gate->name);
            const void *raw_up   = get_tensor_data(llama->model, lw.ffn_up->name);
            matmul_q4_k_gate_up(s->hb, s->hb2, raw_gate, raw_up, s->xb, n_tokens, n_ff, n_embd, s->q8_buf);
        } else {
            matmul_tensor(s->hb,  llama, lw.ffn_gate, s->xb, n_tokens, n_ff, n_embd, s->q8_buf);
            matmul_tensor(s->hb2, llama, lw.ffn_up,   s->xb, n_tokens, n_ff, n_embd, s->q8_buf);
        }

        size_t total_ff = (size_t)n_tokens * n_ff;
        silu(s->hb, s->hb, (int)total_ff);
        for (size_t i = 0; i < total_ff; i++) s->hb[i] *= s->hb2[i];

        matmul_tensor(s->xb2, llama, lw.ffn_down, s->hb, n_tokens, n_embd, n_ff, s->q8_buf);
        size_t total_embd = (size_t)n_tokens * n_embd;
        for (size_t i = 0; i < total_embd; i++) s->x[i] += s->xb2[i];

        pa_ffn->stop(t0);
    }
}

// ── forward_batch ──
// Runs a full forward pass for a batch of `n_tokens` tokens starting at `start_pos`.
void forward_batch(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens, int start_pos,
                   bool need_logits, bool all_logits) {
    if (n_tokens <= 0) return;
    if (n_tokens == 1) {
        forward_token(llama, s, tokens[0], start_pos, need_logits);
        return;
    }
    if (start_pos < 0 || start_pos + n_tokens > s->n_ctx) {
        fprintf(stderr, "ERROR: batch [%d, %d) exceeds context window limit %d\n",
                start_pos, start_pos + n_tokens, s->n_ctx);
        exit(1);
    }
    if (n_tokens > s->max_batch) {
        fprintf(stderr, "ERROR: batch size %d exceeds max_batch %d\n", n_tokens, s->max_batch);
        exit(1);
    }

    int n_embd = s->n_embd;
    // Record the token position of this batch in the probe context:
    // - prefill (!all_logits): record last token of chunk (start_pos + n_tokens - 1)
    // - speculative decode (all_logits): record start_pos (strictly monotonic with accepted pos)
    probe_set_pos(all_logits ? start_pos : (start_pos + n_tokens - 1));

    // 1. Embedding lookup for all n_tokens
    {
        Probe p("embedding_lookup");
        TensorInfo *embd_info = llama->token_embd;
        size_t stride = row_bytes(embd_info->type, n_embd);
        if (stride == 0) {
            fprintf(stderr, "ERROR: unsupported token_embd type %u\n", (unsigned)embd_info->type);
            exit(1);
        }
        const uint8_t *base = (const uint8_t *)get_tensor_data(llama->model, embd_info->name);
        for (int t = 0; t < n_tokens; t++) {
            int tok = tokens[t];
            if (tok < 0 || tok >= s->n_vocab) {
                fprintf(stderr, "ERROR: token ID %d out of vocabulary bounds [0, %d)\n", tok, s->n_vocab);
                exit(1);
            }
            dequantize(base + (size_t)tok * stride, s->x + (size_t)t * n_embd, n_embd, embd_info->type);
        }
    }

    // 2. Transformer layers across the batch
    {
        ProbeAccum pa_norm    ("layer:attn_norm");
        ProbeAccum pa_qkv     ("layer:qkv_proj");
        ProbeAccum pa_rope    ("layer:rope");
        ProbeAccum pa_mha     ("layer:mha");
        ProbeAccum pa_out_proj("layer:attn_out_proj");
        ProbeAccum pa_ffn     ("layer:ffn");

        Probe p_layers("all_layers");
        for (int layer = 0; layer < s->n_layers; layer++) {
            forward_layer_batch(llama, s, layer, start_pos, n_tokens,
                                &pa_norm, &pa_qkv, &pa_rope,
                                &pa_mha, &pa_out_proj, &pa_ffn);
        }
        probe_set_layer(-1);
    }

    // 3. Final RMSNorm + Output Projection (on all n_tokens if all_logits, else only the last token)
    if (need_logits) {
        Probe p("final_norm_and_lm_head");
        const float *w = (const float *)get_tensor_data(llama->model, llama->output_norm->name);
        if (all_logits) {
            for (int t = 0; t < n_tokens; t++) {
                rmsnorm(s->x + (size_t)t * n_embd, s->x + (size_t)t * n_embd, w, n_embd, llama->hparams.rms_norm_eps);
            }
            matmul_tensor(s->logits, llama, llama->output, s->x, n_tokens, s->n_vocab, n_embd, s->q8_buf);
        } else {
            float *x_last = s->x + (size_t)(n_tokens - 1) * n_embd;
            rmsnorm(x_last, x_last, w, n_embd, llama->hparams.rms_norm_eps);
            matvec_tensor(s->logits, llama, llama->output, x_last, s->n_vocab, n_embd, s->q8_buf);
        }
    }
}

// ── prefill ──
// Feeds a sequence of prompt tokens into the model in batches of up to max_batch.
// Evaluates each batch, storing keys and values in the KV cache.
// Only computes final output logits on the last token.
void prefill(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens) {
    if (n_tokens > s->n_ctx) {
        fprintf(stderr, "ERROR: prompt length %d exceeds context limit %d\n", n_tokens, s->n_ctx);
        exit(1);
    }

    int batch_size = 16;
    if (const char *env_b = getenv("NANO_BATCH")) {
        int b = atoi(env_b);
        if (b >= 1 && b <= s->max_batch) batch_size = b;
    }

    probe_set_pos(-1);
    Probe p_prefill("prefill_total");
    ProbeAccum pa_forward("prefill:forward_token");
    for (int pos = 0; pos < n_tokens; pos += batch_size) {
        int chunk = (pos + batch_size <= n_tokens) ? batch_size : (n_tokens - pos);
        bool need_logits = (pos + chunk == n_tokens);
        auto t = pa_forward.start();
        forward_batch(llama, s, tokens + pos, chunk, pos, need_logits);
        pa_forward.stop(t);
    }
    probe_set_pos(-1);
    // p_prefill destructor fires here → prints total prefill wall time
}

// ── argmax_token ──
int argmax_token(const InferState *s, int token_idx) {
    const float *row = s->logits + (size_t)token_idx * s->n_vocab;
    int best = 0;
    for (int i = 1; i < s->n_vocab; i++) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

// ── draft_ngram ──
// Zero-weight Prompt-Lookup / N-Gram speculative drafter.
// Searches `history[0..len-1]` from most recent to oldest for a matching 3-gram
// ending at `history[len - 1]`. Using a 3-gram minimum avoids false-positive
// B=4 verification steps on generic 2-grams (like ", the" or ", and").
// Copies up to `max_draft` continuation tokens into `draft_out` and returns count in [0, max_draft].
static int draft_ngram(const int *history, int len, int max_draft, int *draft_out) {
    constexpr int ngram = 3;
    if (max_draft <= 0 || len <= ngram) return 0;
    const int *suffix = history + (len - ngram);
    for (int i = len - ngram - 1; i >= 0; i--) {
        if (history[i] == suffix[0] && history[i + 1] == suffix[1] && history[i + 2] == suffix[2]) {
            int match_next = i + ngram;
            int avail = len - match_next;
            int k = (avail < max_draft) ? avail : max_draft;
            for (int j = 0; j < k; j++) {
                draft_out[j] = history[match_next + j];
            }
            return k;
        }
    }
    return 0;
}

// ── generate ──
// Greedy autoregressive generation with N-Gram Speculative Decoding (B=4 parallel verification).
//   1. prefill(prompt) -> s->logits[0] holds the distribution for the first generated token
//   2. Emit the first token `cur_tok`, then in each decode step:
//      - Draft up to `spec_k` (default 3) continuation tokens from `history` via N-gram lookup
//      - Feed `[cur_tok, draft_1, ..., draft_k]` (`B = 1 + k <= 4`) into `forward_batch(..., all_logits=true)`
//      - Greedily verify `argmax_token(s, t)` for t = 0..k, accepting up to 4 tokens in a single weight pass!
int generate(LlamaModel *llama, InferState *s,
             const int *prompt, int n_prompt,
             int max_new, int eos_id,
             TokenCallback on_token, void *user) {
    {
        probe_set_phase("prefill");
        probe_set_pos(-1);
        Probe p("prefill");
        prefill(llama, s, prompt, n_prompt);
    }

    if (max_new <= 0) {
        probe_set_pos(-1);
        probe_set_phase("");
        return 0;
    }

    int spec_k = 3; // Default: draft up to 3 tokens -> B=4 verification batch
    if (const char *env_k = getenv("NANO_SPEC_K")) {
        int k = atoi(env_k);
        if (k >= 0 && k < s->max_batch) spec_k = k;
    }

    std::vector<int> history(prompt, prompt + n_prompt);
    int pos = n_prompt;  // sequence position of `cur_tok` (to be forwarded next)
    int n_generated = 0;
    int n_steps = 0;
    int n_drafted = 0;
    int n_accepted = 0;

    probe_set_phase("decode");
    ProbeAccum pa_argmax ("decode:argmax");
    ProbeAccum pa_forward("decode:forward_token");

    // Sample the very first generated token from prefill's final logits
    probe_set_pos(pos);
    int cur_tok;
    {
        auto t = pa_argmax.start();
        cur_tok = argmax_token(s, 0);
        pa_argmax.stop(t);
    }
    if (cur_tok == eos_id) {
        probe_set_pos(-1);
        probe_set_phase("");
        return 0;
    }
    if (on_token) on_token(cur_tok, user);
    n_generated++;
    history.push_back(cur_tok);

    int batch[InferState::MAX_BATCH];
    bool hit_eos = false;

    while (n_generated < max_new && pos < s->n_ctx && !hit_eos) {
        probe_set_pos(pos);

        int max_k = std::min({spec_k, s->n_ctx - pos - 1, max_new - n_generated - 1});
        batch[0] = cur_tok;
        int k = draft_ngram(history.data(), (int)history.size(), max_k, batch + 1);
        int B = 1 + k;

        n_steps++;
        n_drafted += k;

        {
            auto t = pa_forward.start();
            forward_batch(llama, s, batch, B, pos, true, /*all_logits=*/true);
            pa_forward.stop(t);
        }

        // Verify drafted tokens against greedy target argmax at each position t = 0..k
        for (int t = 0; t <= k; t++) {
            probe_set_pos(pos + t);
            int next;
            {
                auto ta = pa_argmax.start();
                next = argmax_token(s, t);
                pa_argmax.stop(ta);
            }
            if (t > 0) {
                memcpy(s->logits, s->logits + (size_t)t * s->n_vocab, (size_t)s->n_vocab * sizeof(float));
            }
            if (next == eos_id) {
                hit_eos = true;
                pos += t + 1;
                break;
            }

            if (on_token) on_token(next, user);
            n_generated++;
            history.push_back(next);
            cur_tok = next;

            if (t < k && next == batch[t + 1]) {
                n_accepted++;
                if (n_generated >= max_new) {
                    pos += t + 1;
                    break;
                }
            } else {
                pos += t + 1;
                break;
            }
        }
    }

    if (n_steps > 0) {
        double accept_pct = (n_drafted > 0) ? (100.0 * n_accepted / n_drafted) : 0.0;
        double tok_per_step = (double)(n_steps + n_accepted) / n_steps;
        fprintf(stderr, "\n[SPEC] steps=%d, forward_tokens_emitted=%d (%.2f tok/step), drafted=%d, accepted=%d (%.1f%%)\n",
                n_steps, n_steps + n_accepted, tok_per_step, n_drafted, n_accepted, accept_pct);
    }

    probe_set_pos(-1);
    probe_set_phase("");
    return n_generated;
}
