#include "llama_model.h"
#include "dequant.h"
#include "ops.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // ── load the model ──
    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    display_llama_model(llama);

    int n_embd = llama->hparams.n_embd;  // 2048 for TinyLlama

    // ── Step 1: Token embedding lookup ──
    // This is the very first thing in inference: given a token ID,
    // look up its embedding vector from the token_embd matrix.
    //
    // token_embd is [n_embd, n_vocab] — each row is one token's embedding.
    // To get token i's embedding, we index: &embd_data[i * n_embd]
    //
    // IMPORTANT: token_embd is quantized (e.g. Q4_K), so we can't just
    // cast the raw bytes to float*. We must dequantize first.

    int test_token = 1;  // token 1 is typically <s> (BOS)

    TensorInfo *embd_info = llama->token_embd;
    uint64_t embd_total = 1;
    for (uint32_t d = 0; d < embd_info->n_dims; d++) {
        embd_total *= embd_info->dims[d];
    }

    printf("\ntoken_embd type: %s, total elements: %llu\n",
           ggml_type_name(embd_info->type), (unsigned long long)embd_total);

    float *embd_table = (float *)malloc(embd_total * sizeof(float));
    void *embd_raw = get_tensor_data(llama->model, embd_info->name);
    dequantize(embd_raw, embd_table, embd_total, embd_info->type);

    float *token_vec = &embd_table[test_token * n_embd];

    printf("\n── Token Embedding (token %d) ──\n", test_token);
    printf("first 10: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", token_vec[i]);
    printf("\n");

    // ── Step 2: RMSNorm ──
    // Before attention in layer 0, we normalize the embedding.
    // The norm weight is an F32 tensor of shape [n_embd].

    float *norm_weight = (float *)get_tensor_data(llama->model,
                                                   llama->layers[0].attn_norm->name);

    float *normed = (float *)malloc(n_embd * sizeof(float));
    rmsnorm(normed, token_vec, norm_weight, n_embd, llama->hparams.rms_norm_eps);

    printf("\n── After RMSNorm (layer 0 attn_norm) ──\n");
    printf("first 10: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", normed[i]);
    printf("\n");

    // sanity check: the RMS of the normalized output (before weight scaling)
    // should be close to 1.0 if the weight were all 1s.
    // With actual weights it will differ, but let's check the raw normalized RMS:
    float ss = 0.0f;
    for (int i = 0; i < n_embd; i++) {
        // divide out the weight to see the raw normalized value
        float raw = (norm_weight[i] != 0.0f) ? normed[i] / norm_weight[i] : 0.0f;
        ss += raw * raw;
    }
    ss = sqrtf(ss / n_embd);
    printf("RMS of normalized values (should be ~1.0): %.6f\n", ss);

    // ── Step 3: Mat-vec multiply — Q projection ──
    // After RMSNorm, the next step is to project the normalized vector
    // into Q (query), K (key), and V (value) spaces.
    //
    // Q = attn_q_weight × normed
    //
    // attn_q weight is [n_in, n_out] = [2048, 2048] but quantized (Q4_K).
    // We dequantize the whole thing to float, then do the multiply.
    // (Naive but clear — we'll optimize later.)

    TensorInfo *q_info = llama->layers[0].attn_q;
    int n_in  = (int)q_info->dims[0];   // input dim  (n_embd = 2048)
    int n_out = (int)q_info->dims[1];   // output dim (n_embd = 2048)
    uint64_t q_total = (uint64_t)n_out * n_in;

    printf("\n── Q Projection (layer 0) ──\n");
    printf("weight: %s, shape [%d, %d], dequantizing %llu values...\n",
           ggml_type_name(q_info->type), n_in, n_out, (unsigned long long)q_total);

    // dequantize the full weight matrix
    float *q_weight = (float *)malloc(q_total * sizeof(float));
    void *q_raw = get_tensor_data(llama->model, q_info->name);
    dequantize(q_raw, q_weight, q_total, q_info->type);

    // multiply: q_vec = q_weight × normed
    float *q_vec = (float *)malloc(n_out * sizeof(float));
    matvec(q_vec, q_weight, normed, n_out, n_in);

    printf("first 10 of Q: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", q_vec[i]);
    printf("\n");

    free(q_weight);
    free(q_vec);
    free(normed);
    free(embd_table);
    free_llama_model(llama);
    printf("\nmodel freed.\n");

    return 0;
}
