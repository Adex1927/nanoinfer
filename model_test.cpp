#include "llama_model.h"
#include "infer_state.h"
#include "dequant.h"
#include "ops.h"
#include "forward.h"
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

    // ── Prefill: feed all prompt tokens through the model ──
    // Each token is forwarded into the KV cache at position pos = 0, 1, ...
    // Logits are only computed on the final prompt token.
    prefill(llama, s, prompt.data(), (int)prompt.size());

    // ── Argmax: find the highest-scoring next token ──
    int best_token = 0;
    for (int i = 1; i < s->n_vocab; i++) {
        if (s->logits[i] > s->logits[best_token]) {
            best_token = i;
        }
    }

    printf("\n════════════════════════════════════════\n");
    printf("  Last prompt token:  %d (at pos %zu)\n", prompt.back(), prompt.size() - 1);
    printf("  Predicted next:     %d\n", best_token);
    printf("  Logit score:        %.4f\n", s->logits[best_token]);
    printf("════════════════════════════════════════\n");

    // ── Top 10 logits ──
    printf("\nTop 10 tokens by logit score:\n");
    for (int rank = 0; rank < 10; rank++) {
        int top = 0;
        for (int i = 1; i < s->n_vocab; i++) {
            if (s->logits[i] > s->logits[top]) top = i;
        }
        printf("  #%-2d token=%5d  logit=%.4f\n", rank + 1, top, s->logits[top]);
        s->logits[top] = -1e30f;  // mask it out for next iteration
    }

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
