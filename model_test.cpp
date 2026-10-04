#include "llama_model.h"
#include "infer_state.h"
#include "forward.h"
#include "probe.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaModel *llama = load_llama_model(argv[1]);
    if (!llama) return 1;

    // display_llama_model(llama);

    InferState *s = alloc_infer_state(&llama->hparams);
    if (!s) { free_llama_model(llama); return 1; }

    // ── Prompt definition ──
    // If token IDs are passed on CLI: ./model_test model.gguf 1 2787
    // Otherwise default to prompt "World" = {1, 2787} (<s>, ' World')
    std::vector<int> prompt;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            prompt.push_back(atoi(argv[i]));
        }
    } else {
        prompt = {1, 2787}; // Default: <s>, ' World'
    }

    printf("Prompt tokens (%zu):", prompt.size());
    for (size_t i = 0; i < prompt.size(); i++) {
        printf(" %d", prompt[i]);
    }
    printf("\n");

    // ── Generation: greedy decode, streaming token IDs ──
    // Usage: ./model_test model.gguf [token ids...] [-n N]  (N via env NANO_N, default 20)
    const int eos_id = 2;  // </s> for TinyLlama (tokenizer.ggml.eos_token_id)
    int max_new = getenv("NANO_N") ? atoi(getenv("NANO_N")) : 20;

    printf("Generated token IDs:");
    fflush(stdout);
    int n_gen = generate(llama, s, prompt.data(), (int)prompt.size(), max_new, eos_id,
                         [](int tok, void *) { printf(" %d", tok); fflush(stdout); },
                         nullptr);
    printf("\n(%d tokens)\n", n_gen);

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
