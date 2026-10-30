#include "forward.h"
#include "ops.h"
#include "dequant.h"
#include "probe.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>


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

// ── mha — Multi-Head Attention with Grouped Query Attention ──
//
// Writes current token's K and V to the KV cache, then invokes the fused
// mha_decode kernel in ops.cpp.
void mha(InferState *s, int layer, int pos) {
    int kv_dim = s->kv_dim;
    int n_ctx  = s->n_ctx;

    // 1. Write current K and V into the cache
    float *k_pos = s->k_cache + layer * n_ctx * kv_dim + pos * kv_dim;
    float *v_pos = s->v_cache + layer * n_ctx * kv_dim + pos * kv_dim;

    for (int i = 0; i < kv_dim; i++) k_pos[i] = s->k[i];
    for (int i = 0; i < kv_dim; i++) v_pos[i] = s->v[i];

    // 2. Compute multi-head attention (fused ops kernel)
    const float *k_layer = s->k_cache + layer * n_ctx * kv_dim;
    const float *v_layer = s->v_cache + layer * n_ctx * kv_dim;

    mha_decode(s->xb, s->q, k_layer, v_layer, s->att, pos,
               s->n_heads, s->n_kv_heads, s->head_dim, s->kv_dim, s->n_ctx);
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

// ── prefill ──
// Feeds a sequence of prompt tokens into the model.
// Evaluates each token, storing keys and values in the KV cache.
// Only computes final output logits on the last token.
void prefill(LlamaModel *llama, InferState *s, const int *tokens, int n_tokens) {
    if (n_tokens > s->n_ctx) {
        fprintf(stderr, "ERROR: prompt length %d exceeds context limit %d\n", n_tokens, s->n_ctx);
        exit(1);
    }

    probe_set_pos(-1);
    Probe p_prefill("prefill_total");
    for (int pos = 0; pos < n_tokens; pos++) {
        bool need_logits = (pos == n_tokens - 1);
        forward_token(llama, s, tokens[pos], pos, need_logits);
    }
    // p_prefill destructor fires here → prints total prefill wall time
}

// ── argmax_token ──
int argmax_token(const InferState *s) {
    int best = 0;
    for (int i = 1; i < s->n_vocab; i++) {
        if (s->logits[i] > s->logits[best]) best = i;
    }
    return best;
}

// ── generate ──
// After prefill, s->logits holds the distribution for the token that follows
// the prompt. Each iteration: sample it, then forward it at the next position
// (which also appends its K/V to the cache) to get logits for the one after.
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

    int pos = n_prompt;  // position the next generated token will occupy
    int n_generated = 0;

    // One sample per generated token, stored with its token position.
    probe_set_phase("decode");
    ProbeAccum pa_argmax ("decode:argmax");
    ProbeAccum pa_forward("decode:forward_token");

    while (n_generated < max_new) {
        probe_set_pos(pos);
        int next;
        {
            auto t = pa_argmax.start();
            next = argmax_token(s);
            pa_argmax.stop(t);
        }
        if (next == eos_id) break;

        if (on_token) on_token(next, user);
        n_generated++;

        if (pos >= s->n_ctx) break;  // context window full
        {
            auto t = pa_forward.start();
            forward_token(llama, s, next, pos, true);
            pa_forward.stop(t);
        }
        pos++;
    }

    probe_set_pos(-1);
    probe_set_phase("");
    return n_generated;
}
