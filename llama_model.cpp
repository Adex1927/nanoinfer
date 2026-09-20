#include "llama_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── load_llama_model ──
// 1. Load the raw GGUF via load_model()
// 2. Extract hyperparams from stored metadata
// 3. Map tensor names to the LlamaModel struct

LlamaModel *load_llama_model(const char *path) {
    Model *model = load_model(path);
    if (!model) return nullptr;

    LlamaModel *llama = (LlamaModel *)calloc(1, sizeof(LlamaModel));
    llama->model = model;

    // ── extract hyperparams from GGUF metadata ──
    LlamaHparams *hp = &llama->hparams;

    hp->n_embd        = get_metadata_u32(model, "llama.embedding_length", 0);
    hp->n_layers      = get_metadata_u32(model, "llama.block_count", 0);
    hp->n_heads       = get_metadata_u32(model, "llama.attention.head_count", 0);
    hp->n_kv_heads    = get_metadata_u32(model, "llama.attention.head_count_kv", 0);
    hp->n_ff          = get_metadata_u32(model, "llama.feed_forward_length", 0);
    hp->n_ctx         = get_metadata_u32(model, "llama.context_length", 2048);
    hp->rope_freq_base = get_metadata_f32(model, "llama.rope.freq_base", 10000.0f);
    hp->rope_dim_count = get_metadata_u32(model, "llama.rope.dimension_count",
                                          hp->n_embd / hp->n_heads);  // default: full head_dim
    hp->rms_norm_eps   = get_metadata_f32(model, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    // n_vocab: try metadata first, fall back to token_embd tensor shape
    hp->n_vocab = get_metadata_u32(model, "llama.vocab_size", 0);

    if (hp->n_embd == 0 || hp->n_layers == 0) {
        fprintf(stderr, "load_llama_model: missing critical hyperparameters "
                "(n_embd=%u, n_layers=%u)\n", hp->n_embd, hp->n_layers);
        free_model(model);
        free(llama);
        return nullptr;
    }

    // ── map global tensors ──
    llama->token_embd  = get_tensor_info(model, "token_embd.weight");
    llama->output_norm = get_tensor_info(model, "output_norm.weight");
    llama->output      = get_tensor_info(model, "output.weight");

    // if output.weight is absent, it's weight-tied to token_embd
    if (!llama->output) {
        llama->output = llama->token_embd;
    }

    // infer n_vocab from token_embd shape if not in metadata
    if (hp->n_vocab == 0 && llama->token_embd) {
        hp->n_vocab = (uint32_t)llama->token_embd->dims[1];
    }

    // ── map per-layer tensors ──
    uint32_t n = hp->n_layers;
    llama->layers = (LlamaLayer *)calloc(n, sizeof(LlamaLayer));

    for (uint32_t i = 0; i < n; i++) {
        char name[128];
        LlamaLayer *L = &llama->layers[i];

        // macro to reduce boilerplate: builds "blk.{i}.{suffix}" and looks it up
        #define LOAD(field, suffix) \
            snprintf(name, sizeof(name), "blk.%u." suffix, i); \
            L->field = get_tensor_info(model, name)

        LOAD(attn_norm,    "attn_norm.weight");
        LOAD(attn_q,       "attn_q.weight");
        LOAD(attn_k,       "attn_k.weight");
        LOAD(attn_v,       "attn_v.weight");
        LOAD(attn_output,  "attn_output.weight");
        LOAD(ffn_norm,     "ffn_norm.weight");
        LOAD(ffn_gate,     "ffn_gate.weight");
        LOAD(ffn_up,       "ffn_up.weight");
        LOAD(ffn_down,     "ffn_down.weight");

        #undef LOAD
    }

    return llama;
}

// ── free_llama_model ──

void free_llama_model(LlamaModel *llama) {
    if (!llama) return;
    free(llama->layers);
    free_model(llama->model);  // frees mmap, tensors, metadata
    free(llama);
}

// ── display_llama_model ──

static void print_tensor_line(const char *label, TensorInfo *t) {
    if (!t) {
        printf("  %-22s  (missing)\n", label);
        return;
    }
    // build dims string
    char dims[64];
    int pos = 0;
    pos += snprintf(dims + pos, sizeof(dims) - pos, "[");
    for (uint32_t d = 0; d < t->n_dims; d++) {
        if (d > 0) pos += snprintf(dims + pos, sizeof(dims) - pos, ", ");
        pos += snprintf(dims + pos, sizeof(dims) - pos, "%llu", (unsigned long long)t->dims[d]);
    }
    snprintf(dims + pos, sizeof(dims) - pos, "]");

    printf("  %-22s  %-6s  %s\n", label, ggml_type_name(t->type), dims);
}

void display_llama_model(LlamaModel *llama) {
    if (!llama) { printf("(null model)\n"); return; }

    LlamaHparams *hp = &llama->hparams;

    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║          LLaMA Model Architecture            ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  n_vocab       = %-10u                   ║\n", hp->n_vocab);
    printf("║  n_embd        = %-10u                   ║\n", hp->n_embd);
    printf("║  n_layers      = %-10u                   ║\n", hp->n_layers);
    printf("║  n_heads       = %-10u                   ║\n", hp->n_heads);
    printf("║  n_kv_heads    = %-10u                   ║\n", hp->n_kv_heads);
    printf("║  n_ff          = %-10u                   ║\n", hp->n_ff);
    printf("║  n_ctx         = %-10u                   ║\n", hp->n_ctx);
    printf("║  rope_freq_base= %-10.1f                   ║\n", hp->rope_freq_base);
    printf("╚══════════════════════════════════════════════╝\n");

    printf("\n── Global Tensors ──\n");
    print_tensor_line("token_embd", llama->token_embd);
    print_tensor_line("output_norm", llama->output_norm);
    print_tensor_line("output", llama->output);

    printf("\n── Transformer Layers (%u) ──\n", hp->n_layers);

    for (uint32_t i = 0; i < hp->n_layers; i++) {
        LlamaLayer *L = &llama->layers[i];
        printf("\n  [layer %u]\n", i);
        print_tensor_line("attn_norm", L->attn_norm);
        print_tensor_line("attn_q", L->attn_q);
        print_tensor_line("attn_k", L->attn_k);
        print_tensor_line("attn_v", L->attn_v);
        print_tensor_line("attn_output", L->attn_output);
        print_tensor_line("ffn_norm", L->ffn_norm);
        print_tensor_line("ffn_gate", L->ffn_gate);
        print_tensor_line("ffn_up", L->ffn_up);
        print_tensor_line("ffn_down", L->ffn_down);
    }

    printf("\n");
}
