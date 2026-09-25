#include "llama_model.h"
#include "infer_state.h"
#include "dequant.h"
#include "ops.h"
#include "forward.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) return 1;

    display_llama_model(llama);

    InferState *s = alloc_infer_state(&llama->hparams);
    if (!s) { free_llama_model(llama); return 1; }

    int test_token = 1;   // <s> (BOS)
    int pos        = 0;

    // ── Token embedding lookup ──
    {
        int n_embd = s->n_embd;
        TensorInfo *embd_info = llama->token_embd;
        uint64_t n = 1;
        for (uint32_t d = 0; d < embd_info->n_dims; d++) n *= embd_info->dims[d];
        float *embd = (float *)calloc(n, sizeof(float));
        dequantize(get_tensor_data(llama->model, embd_info->name), embd, n, embd_info->type);
        memcpy(s->x, embd + test_token * n_embd, n_embd * sizeof(float));
        free(embd);
    }

    printf("x before layer 0, first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", s->x[i]);
    printf("\n");

    // ── Run all transformer layers ──
    // Each layer reads s->x, runs attention + FFN, writes back to s->x.
    // After all layers, s->x holds the final hidden state for this token.
    for (int layer = 0; layer < s->n_layers; layer++) {
        printf("  layer %d / %d ...\n", layer, s->n_layers);
        forward_layer(llama, s, layer, pos);
    }

    printf("x after all layers, first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", s->x[i]);
    printf("\n");

    // ── Final RMSNorm ──
    // The last layer's output gets one more normalization before the logit projection.
    // This uses the output_norm weight (not any layer's norm — it's a global tensor).
    {
        float *w = (float *)get_tensor_data(llama->model, llama->output_norm->name);
        rmsnorm(s->x, s->x, w, s->n_embd, llama->hparams.rms_norm_eps);
    }

    // ── Output projection: hidden state → logits ──
    // This is the final linear layer: [n_embd] → [n_vocab].
    // Each row of the output weight matrix represents one token in the vocabulary.
    // The dot product of s->x with row i gives the model's "score" for token i.
    // Higher score = model thinks token i is more likely to come next.
    {
        TensorInfo *out_info = llama->output;
        uint64_t n = 1;
        for (uint32_t d = 0; d < out_info->n_dims; d++) n *= out_info->dims[d];
        float *wout = (float *)calloc(n, sizeof(float));
        dequantize(get_tensor_data(llama->model, out_info->name), wout, n, out_info->type);
        matvec(s->logits, wout, s->x, s->n_vocab, s->n_embd);
        free(wout);
    }

    // ── Argmax: find the highest-scoring token ──
    // This is greedy decoding: always pick the single most likely next token.
    // (Real inference might use temperature sampling, top-k, top-p, etc.)
    int best_token = 0;
    for (int i = 1; i < s->n_vocab; i++) {
        if (s->logits[i] > s->logits[best_token]) {
            best_token = i;
        }
    }

    printf("\n════════════════════════════════════════\n");
    printf("  Input token:     %d (BOS)\n", test_token);
    printf("  Predicted next:  %d\n", best_token);
    printf("  Logit score:     %.4f\n", s->logits[best_token]);
    printf("════════════════════════════════════════\n");

    // ── Debug: top 10 logits ──
    printf("\nTop 10 tokens by logit score:\n");
    for (int rank = 0; rank < 10; rank++) {
        int top = 0;
        for (int i = 1; i < s->n_vocab; i++) {
            if (s->logits[i] > s->logits[top]) top = i;
        }
        printf("  #%d  token=%5d  logit=%.4f\n", rank + 1, top, s->logits[top]);
        s->logits[top] = -1e30f;  // mask it out for next iteration
    }
    printf("\n(Expected: token 450 = 'The')\n");

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
