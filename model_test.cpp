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

    // ── Run one full transformer layer ──
    forward_layer(llama, s, 0, pos);

    printf("x after  layer 0, first 5: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", s->x[i]);
    printf("\n");

    // Sanity: x should be meaningfully different from the embedding.
    // The layer applies attention + FFN, so values should shift noticeably.
    // They should NOT be NaN, Inf, or identical to the input.

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
