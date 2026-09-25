#include "llama_model.h"
#include "infer_state.h"
#include "dequant.h"
#include "ops.h"
#include "forward.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Helper: dequantize a tensor's weights into a freshly malloc'd float buffer.
// Caller must free() the result.
static float *dequant_tensor(LlamaModel *llama, TensorInfo *info) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < info->n_dims; d++) n *= info->dims[d];

    printf("  dequant: %-30s  type=%-6s  elements=%llu\n",
           info->name, ggml_type_name(info->type), (unsigned long long)n);

    float *buf = (float *)calloc(n, sizeof(float));   // calloc so zeros are explicit
    void  *raw = get_tensor_data(llama->model, info->name);
    if (!dequantize(raw, buf, n, info->type)) {
        fprintf(stderr, "  ERROR: dequantize failed for %s (type %s)\n",
                info->name, ggml_type_name(info->type));
    }
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) return 1;

    InferState *s = alloc_infer_state(&llama->hparams);
    if (!s) { free_llama_model(llama); return 1; }

    const LlamaHparams &hp = llama->hparams;
    int n_embd   = s->n_embd;
    int kv_dim   = s->kv_dim;
    int test_token = 1;   // <s> (BOS)
    int pos        = 0;   // first position in the sequence
    int layer      = 0;

    printf("n_embd=%d  kv_dim=%d  n_heads=%d  n_kv_heads=%d  head_dim=%d\n\n",
           n_embd, kv_dim, s->n_heads, s->n_kv_heads, s->head_dim);

    // ── 1. Token embedding lookup ──
    // Dequantize the full embedding table, copy row `test_token` into s->x.
    {
        TensorInfo *embd_info = llama->token_embd;
        float *embd = dequant_tensor(llama, embd_info);
        memcpy(s->x, embd + test_token * n_embd, n_embd * sizeof(float));
        free(embd);
    }
    printf("x (embed) first 5:  ");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->x[i]);
    printf("\n");

    // ── 2. RMSNorm (pre-attention) ──
    // Normalizes s->x using layer 0's attn_norm weight → result in s->xb.
    {
        float *w = (float *)get_tensor_data(llama->model,
                                            llama->layers[layer].attn_norm->name);
        rmsnorm(s->xb, s->x, w, n_embd, hp.rms_norm_eps);
    }
    printf("xb (normed) first 5:");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->xb[i]);
    printf("\n");

    // ── 3. Q / K / V projections ──
    // Each is a mat-vec: project s->xb into query/key/value space.
    // Q: [n_embd → n_embd],  K/V: [n_embd → kv_dim]
    {
        float *wq = dequant_tensor(llama, llama->layers[layer].attn_q);
        float *wk = dequant_tensor(llama, llama->layers[layer].attn_k);
        float *wv = dequant_tensor(llama, llama->layers[layer].attn_v);

        // ── diagnostic: is the V weight matrix non-zero after dequant? ──
        printf("wv first 5 (raw dequant): ");
        for (int i = 0; i < 5; i++) printf("%e ", wv[i]);
        printf("\n");

        matvec(s->q, wq, s->xb, n_embd,  n_embd);
        matvec(s->k, wk, s->xb, kv_dim,  n_embd);
        matvec(s->v, wv, s->xb, kv_dim,  n_embd);

        free(wq); free(wk); free(wv);
    }
    printf("q first 5:          ");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->q[i]);
    printf("\n");

    // ── 4. RoPE ──
    rope(s->q, s->k, pos,
         s->n_heads, s->n_kv_heads, s->head_dim,
         (int)hp.rope_dim_count, hp.rope_freq_base);

    printf("q  (after RoPE) first 5: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->q[i]);
    printf("\n");

    // ── diagnostic: are k and v non-zero before mha? ──
    printf("k  (before mha) first 5: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->k[i]);
    printf("\n");
    printf("v  (before mha) first 5: ");
    for (int i = 0; i < 5; i++) printf("%e ", s->v[i]);
    printf("\n");

    // ── 5. Multi-Head Attention ──
    mha(s, layer, pos);

    // att[0] is the only score at pos=0 — softmax makes it 1.0.
    // If it's not 1.0, the score computation or softmax is broken.
    printf("\natt[0] (should be 1.0):  %.5f\n", s->att[0]);

    printf("xb (attn out) first 5:   ");
    for (int i = 0; i < 5; i++) printf("%.5f ", s->xb[i]);
    printf("\n");

    printf("\n(sanity: xb head 0 should equal v head 0)\n");
    printf("xb[0..4]: "); for (int i=0;i<5;i++) printf("%.5f ", s->xb[i]); printf("\n");
    printf("v[0..4]:  "); for (int i=0;i<5;i++) printf("%.5f ", s->v[i]); printf("\n");

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
