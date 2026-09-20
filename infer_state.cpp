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

    // ── activation buffers (small — one vector per slot) ──

    s->x      = (float *)calloc(s->n_embd, sizeof(float));
    s->xb     = (float *)calloc(s->n_embd, sizeof(float));

    // ── attention buffers ──

    s->q   = (float *)calloc(s->n_heads    * s->head_dim, sizeof(float));
    s->k   = (float *)calloc(s->n_kv_heads * s->head_dim, sizeof(float));
    s->v   = (float *)calloc(s->n_kv_heads * s->head_dim, sizeof(float));

    // att scores: n_heads rows, each row is n_ctx scores (one per past position).
    // During attention for head h at position pos, we fill att[h * n_ctx .. pos].
    s->att = (float *)calloc(s->n_heads * s->n_ctx, sizeof(float));

    // ── FFN buffer ──

    s->hb = (float *)calloc(s->n_ff, sizeof(float));

    // ── output logits ──

    s->logits = (float *)calloc(s->n_vocab, sizeof(float));

    // ── KV cache ──
    // This is by far the largest allocation.
    // For TinyLlama (n_layers=22, n_ctx=2048, kv_dim=256):
    //   22 * 2048 * 256 * 4 bytes = ~46 MB per cache (K and V)
    //   Total: ~92 MB — affordable and fixed regardless of model quantization.
    //
    // We keep the KV cache in float32 because:
    //   1. It's written once per token and read many times — no quantization savings.
    //   2. Quantizing the KV cache would add error that accumulates over the context.

    size_t kv_cache_size = (size_t)s->n_layers * s->n_ctx * s->kv_dim;
    s->k_cache = (float *)calloc(kv_cache_size, sizeof(float));
    s->v_cache = (float *)calloc(kv_cache_size, sizeof(float));

    // basic allocation check
    if (!s->x || !s->xb || !s->q || !s->k || !s->v ||
        !s->att || !s->hb || !s->logits || !s->k_cache || !s->v_cache) {
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
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->hb);
    free(s->logits);
    free(s->k_cache);
    free(s->v_cache);
    free(s);
}
