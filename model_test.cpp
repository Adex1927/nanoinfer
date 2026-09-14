#include "model_loader.h"
#include "dequant.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    // load the model — parses everything, mmaps the file
    Model *model = load_model(argv[1]);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    // ── Test 1: read an F32 tensor directly (same as before) ──
    const char *f32_name = "blk.0.attn_norm.weight";
    TensorInfo *f32_info = get_tensor_info(model, f32_name);
    if (f32_info && f32_info->type == 0) {
        float *data = (float *)get_tensor_data(model, f32_name);
        printf("\n--- F32 tensor: %s ---\n", f32_name);
        printf("first 5 values (direct, no dequant needed): ");
        for (int i = 0; i < 5; i++) printf("%.6f ", data[i]);
        printf("\n");
    }

    // ── Test 2: dequantize a Q4_K tensor ──
    const char *q4k_name = "blk.0.attn_q.weight";
    TensorInfo *q4k_info = get_tensor_info(model, q4k_name);
    if (q4k_info && q4k_info->type == 12) {  // 12 = Q4_K
        printf("\n--- Q4_K tensor: %s ---\n", q4k_name);
        printf("dims: [");
        for (uint32_t d = 0; d < q4k_info->n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%llu", (unsigned long long)q4k_info->dims[d]);
        }
        printf("]\n");

        // compute total number of elements
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < q4k_info->n_dims; d++) {
            n_elements *= q4k_info->dims[d];
        }

        // allocate output buffer and dequantize
        float *dequantized = (float *)malloc(n_elements * sizeof(float));
        void *raw_data = get_tensor_data(model, q4k_name);

        dequantize_q4_k(raw_data, dequantized, n_elements);

        printf("dequantized %llu values\n", (unsigned long long)n_elements);
        printf("first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", dequantized[i]);
        printf("\nlast 5:   ");
        for (uint64_t i = n_elements - 5; i < n_elements; i++) {
            printf("%.6f ", dequantized[i]);
        }
        printf("\n");

        free(dequantized);
    } else {
        printf("tensor '%s' not found or not Q4_K\n", q4k_name);
    }

    free_model(model);
    printf("\nmodel freed.\n");

    return 0;
}
