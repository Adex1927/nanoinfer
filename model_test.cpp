#include "llama_model.h"
#include "infer_state.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) return 1;

    // allocate all inference buffers
    InferState *state = alloc_infer_state(&llama->hparams);
    if (!state) {
        fprintf(stderr, "failed to allocate infer state\n");
        free_llama_model(llama);
        return 1;
    }

    display_llama_model(llama);

    // ── print buffer layout ──
    // Show exactly what got allocated and how large each buffer is.
    // This makes the memory cost of inference concrete.

    printf("\n── InferState buffer layout ──\n");
    printf("  %-20s  %d floats  = %zu KB\n", "x",
           state->n_embd, state->n_embd * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "xb",
           state->n_embd, state->n_embd * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "q  (n_heads*head_dim)",
           state->n_heads * state->head_dim,
           state->n_heads * state->head_dim * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "k  (n_kv_heads*head_dim)",
           state->n_kv_heads * state->head_dim,
           state->n_kv_heads * state->head_dim * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "v  (n_kv_heads*head_dim)",
           state->n_kv_heads * state->head_dim,
           state->n_kv_heads * state->head_dim * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "att (n_heads*n_ctx)",
           state->n_heads * state->n_ctx,
           state->n_heads * state->n_ctx * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "hb (n_ff)",
           state->n_ff, state->n_ff * sizeof(float) / 1024);
    printf("  %-20s  %d floats  = %zu KB\n", "logits (n_vocab)",
           state->n_vocab, state->n_vocab * sizeof(float) / 1024);

    size_t kv_size = (size_t)state->n_layers * state->n_ctx * state->kv_dim;
    printf("  %-20s  %zu floats = %zu MB\n", "k_cache",
           kv_size, kv_size * sizeof(float) / (1024 * 1024));
    printf("  %-20s  %zu floats = %zu MB\n", "v_cache",
           kv_size, kv_size * sizeof(float) / (1024 * 1024));

    size_t total_bytes =
        2 * state->n_embd * sizeof(float) +
        (state->n_heads + state->n_kv_heads * 2) * state->head_dim * sizeof(float) +
        state->n_heads * state->n_ctx * sizeof(float) +
        state->n_ff * sizeof(float) +
        state->n_vocab * sizeof(float) +
        2 * kv_size * sizeof(float);

    printf("\n  total: ~%zu MB\n", total_bytes / (1024 * 1024));

    free_infer_state(state);
    free_llama_model(llama);
    printf("\nall freed.\n");
    return 0;
}
