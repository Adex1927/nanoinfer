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
    // display all tensors with types and dimensions
    display_model(model);

    // ── Test 1: read an F32 tensor via the unified dispatcher ──
    const char *f32_name = "blk.0.attn_norm.weight";
    TensorInfo *f32_info = get_tensor_info(model, f32_name);
    if (f32_info && f32_info->type == GGML_TYPE_F32) {
        uint64_t n = 1;
        for (uint32_t d = 0; d < f32_info->n_dims; d++) n *= f32_info->dims[d];

        float *out = (float *)malloc(n * sizeof(float));
        void *raw = get_tensor_data(model, f32_name);
        dequantize(raw, out, n, f32_info->type);

        printf("\n--- F32 tensor: %s ---\n", f32_name);
        printf("first 5 values: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", out[i]);
        printf("\n");
        free(out);
    }

    // ── helper lambda: test any tensor through the unified dispatcher ──
    auto test_tensor = [&](const char *name) {
        TensorInfo *info = get_tensor_info(model, name);
        if (!info) {
            printf("\ntensor '%s' not found\n", name);
            return;
        }

        printf("\n--- tensor: %s (type %u) ---\n", name, (unsigned)info->type);
        printf("dims: [");
        for (uint32_t d = 0; d < info->n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%llu", (unsigned long long)info->dims[d]);
        }
        printf("]\n");

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < info->n_dims; d++) {
            n_elements *= info->dims[d];
        }

        float *out = (float *)malloc(n_elements * sizeof(float));
        void *raw = get_tensor_data(model, name);

        if (!dequantize(raw, out, n_elements, info->type)) {
            printf("unsupported type, skipping\n");
            free(out);
            return;
        }

        printf("dequantized %llu values\n", (unsigned long long)n_elements);
        printf("first 10: ");
        for (int i = 0; i < 10 && (uint64_t)i < n_elements; i++) printf("%.6f ", out[i]);
        printf("\nlast 5:   ");
        for (uint64_t i = (n_elements > 5 ? n_elements - 5 : 0); i < n_elements; i++) {
            printf("%.6f ", out[i]);
        }
        printf("\n");

        free(out);
    };

    // ── Test 2: Q4_K tensor ──
    test_tensor("blk.0.attn_q.weight");

    // ── Test 3: Q6_K tensor ──
    test_tensor("blk.0.ffn_down.weight");

    free_model(model);
    printf("\nmodel freed.\n");

    return 0;
}

