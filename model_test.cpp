#include "llama_model.h"
#include "dequant.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // ── load via the LlamaModel loader ──
    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    // display the full architecture
    display_llama_model(llama);

    // ── quick sanity check: dequantize one tensor from layer 0 ──
    auto test_tensor = [&](const char *label, TensorInfo *info) {
        if (!info) {
            printf("\n%s: (missing)\n", label);
            return;
        }

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < info->n_dims; d++) {
            n_elements *= info->dims[d];
        }

        float *out = (float *)malloc(n_elements * sizeof(float));
        void *raw = get_tensor_data(llama->model, info->name);

        if (!dequantize(raw, out, n_elements, info->type)) {
            printf("\n%s: unsupported type %s\n", label, ggml_type_name(info->type));
            free(out);
            return;
        }

        printf("\n--- %s (%s) ---\n", label, ggml_type_name(info->type));
        printf("first 5: ");
        for (int i = 0; i < 5 && (uint64_t)i < n_elements; i++) printf("%.6f ", out[i]);
        printf("\n");

        free(out);
    };

    // test one tensor of each type from layer 0
    test_tensor("layer0.attn_norm (F32)",  llama->layers[0].attn_norm);
    test_tensor("layer0.attn_q   (Q4_K)", llama->layers[0].attn_q);
    test_tensor("layer0.ffn_down (Q6_K)", llama->layers[0].ffn_down);

    free_llama_model(llama);
    printf("\nmodel freed.\n");

    return 0;
}
