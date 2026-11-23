#include "infer_state.h"

#include <cstdlib>
#include <cstdio>

// ── alloc_infer_state ──
// Allocates every buffer the forward pass needs, all at once.
// Sizes are derived purely from hyperparameters — no file I/O here.

InferState *alloc_infer_state(const LlamaHparams *hp) {
    InferState *s = (InferState *)calloc(1, sizeof(InferState));
    if (!s) return nullptr;

    // store dimensions for easy access during the forward pass
    s->n_layers   = (int)hp->n_layers;
    s->n_heads    = (int)hp->n_heads;
    s->n_kv_heads = (int)hp->n_kv_heads;
    s->head_dim   = (int)(hp->n_embd / hp->n_heads);  // e.g. 2048/32 = 64
    s->kv_dim     = (int)(hp->n_kv_heads * (hp->n_embd / hp->n_heads));  // 4*64 = 256
    s->n_embd     = (int)hp->n_embd;
    s->n_ff       = (int)hp->n_ff;
    s->n_ctx      = (int)hp->n_ctx;
    s->n_vocab    = (int)hp->n_vocab;
    s->max_batch  = InferState::MAX_BATCH;

    size_t B = (size_t)s->max_batch;

    // ── activation buffers (sized for up to max_batch tokens) ──

    s->x      = (float *)calloc(B * s->n_embd, sizeof(float));
    s->xb     = (float *)calloc(B * s->n_embd, sizeof(float));
    s->xb2    = (float *)calloc(B * s->n_embd, sizeof(float));

    // ── attention buffers ──

    s->q   = (float *)calloc(B * s->n_heads    * s->head_dim, sizeof(float));
    s->k   = (float *)calloc(B * s->n_kv_heads * s->head_dim, sizeof(float));
    s->v   = (float *)calloc(B * s->n_kv_heads * s->head_dim, sizeof(float));

    // att scores: n_heads rows, each row is n_ctx scores (one per past position).
    // During attention for head h at position pos, we fill att[h * n_ctx .. pos].
    s->att = (float *)calloc((size_t)s->n_heads * s->n_ctx, sizeof(float));

    // ── FFN buffer ──

    s->hb  = (float *)calloc(B * s->n_ff, sizeof(float));
    s->hb2 = (float *)calloc(B * s->n_ff, sizeof(float));

    // ── output logits ──

    s->logits = (float *)calloc(B * s->n_vocab, sizeof(float));

    // ── KV cache ──
    // For TinyLlama (n_layers=22, n_ctx=2048, kv_dim=256):
    //   - float32: 22 * 2048 * 256 * 4 bytes = ~46.1 MB per cache (92.2 MB total)
    //   - Q8_0 KV: 22 * 2048 * (256/32) * 34 bytes = ~12.25 MB per cache (24.5 MB total, 3.76x smaller)
    // Controlled by NANO_KV_Q8 (default: 1 = Q8_0 k_cache & v_cache, 0 = float32 k_cache & v_cache).
    s->kv_q8 = true;
    if (const char *env_kv = getenv("NANO_KV_Q8")) {
        s->kv_q8 = (atoi(env_kv) != 0);
    }

    size_t kv_cache_size = (size_t)s->n_layers * s->n_ctx * s->kv_dim;
    if (s->kv_q8) {
        size_t kv_blocks = kv_cache_size / 32;
        s->k_cache_q8 = (BlockQ8_0 *)calloc(kv_blocks, sizeof(BlockQ8_0));
        s->v_cache_q8 = (BlockQ8_0 *)calloc(kv_blocks, sizeof(BlockQ8_0));
    } else {
        s->k_cache    = (float *)calloc(kv_cache_size, sizeof(float));
        s->v_cache    = (float *)calloc(kv_cache_size, sizeof(float));
    }

    // ── Quantized activation scratch buffer ──
    size_t max_n_in = (size_t)((s->n_embd > s->n_ff) ? s->n_embd : s->n_ff);
    size_t q8_blocks = (max_n_in + 255) / 256;
    s->q8_buf = (BlockQ8_K *)calloc(B * q8_blocks, sizeof(BlockQ8_K));

    // basic allocation check
    bool kv_ok = s->kv_q8 ? (s->k_cache_q8 != nullptr && s->v_cache_q8 != nullptr)
                          : (s->k_cache != nullptr && s->v_cache != nullptr);
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->hb || !s->hb2 || !s->logits || !kv_ok || !s->q8_buf) {
        fprintf(stderr, "alloc_infer_state: out of memory\n");
        free_infer_state(s);
        return nullptr;
    }

    return s;
}

// ── free_infer_state ──

void free_infer_state(InferState *s) {
    if (!s) return;
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->q8_buf);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->hb);
    free(s->hb2);
    free(s->logits);
    free(s->k_cache);
    free(s->v_cache);
    free(s->k_cache_q8);
    free(s->v_cache_q8);
    free(s);
}
