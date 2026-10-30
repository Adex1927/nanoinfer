#include "llama_model.h"
#include "infer_state.h"
#include "forward.h"
#include "probe.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// Reads a whole file; strips one trailing newline (like llama.cpp -f). Returns "" on failure.
static std::string read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return "";
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    fclose(f);
    // Same as llama.cpp's -f handler (common/arg.cpp): drop exactly one trailing '\n'.
    if (!text.empty() && text.back() == '\n') text.pop_back();
    return text;
}

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
    // Otherwise the text in prompt.txt (or $NANO_PROMPT_FILE) is tokenized.
    std::vector<int> prompt;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            prompt.push_back(atoi(argv[i]));
        }
    } else {
        const char *path = getenv("NANO_PROMPT_FILE") ? getenv("NANO_PROMPT_FILE") : "prompt.txt";
        std::string text = read_text_file(path);
        if (text.empty()) {
            fprintf(stderr, "ERROR: no token IDs given and could not read a prompt from '%s'\n", path);
            free_infer_state(s);
            free_llama_model(llama);
            return 1;
        }
        printf("Prompt file: %s\n", path);
        prompt = tokenize(llama->model, text);
    }

    printf("Prompt tokens (%zu):", prompt.size());
    for (size_t i = 0; i < prompt.size(); i++) {
        printf(" %d", prompt[i]);
    }
    printf("\n");

    printf("Prompt text: \"%s\"\n", detokenize(llama->model, prompt.data(), (int)prompt.size()).c_str());

    // NANO_TOKENIZE_ONLY=1: stop here (used by tools/test_nanoinfer.py)
    if (getenv("NANO_TOKENIZE_ONLY")) {
        free_infer_state(s);
        free_llama_model(llama);
        return 0;
    }

    // ── Generation: greedy decode, streaming text ──
    // N via env NANO_N (default 20)
    const int eos_id = getenv("NANO_IGNORE_EOS") ? -1 : 2;  // </s> for TinyLlama
    int max_new = getenv("NANO_N") ? atoi(getenv("NANO_N")) : 50;

    struct Out {
        Model *model;
        InferState *s;
        int topk;                          // 0 = off
        std::vector<int> ids;
        std::vector<std::string> topk_lines;
    } out = { llama->model, s, getenv("NANO_TOPK") ? atoi(getenv("NANO_TOPK")) : 0, {}, {} };

    printf("Generated text: ");
    fflush(stdout);
    int n_gen = generate(llama, s, prompt.data(), (int)prompt.size(), max_new, eos_id,
                         [](int tok, void *user) {
                             Out *o = (Out *)user;

                             // s->logits still holds the distribution that produced `tok`
                             if (o->topk > 0) {
                                 int n = o->s->n_vocab;
                                 std::vector<int> idx(n);
                                 for (int i = 0; i < n; i++) idx[i] = i;
                                 int k = o->topk < n ? o->topk : n;
                                 std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                                   [&](int a, int b) { return o->s->logits[a] > o->s->logits[b]; });
                                 char buf[64];
                                 snprintf(buf, sizeof(buf), "step %zu:", o->ids.size());
                                 std::string line = buf;
                                 for (int r = 0; r < k; r++) {
                                     snprintf(buf, sizeof(buf), " %d:%.4f", idx[r], o->s->logits[idx[r]]);
                                     line += buf;
                                 }
                                 o->topk_lines.push_back(line);
                             }

                             o->ids.push_back(tok);
                             std::string piece = token_to_piece(o->model, tok);
                             fwrite(piece.data(), 1, piece.size(), stdout);
                             fflush(stdout);
                         },
                         &out);
    printf("\n(%d tokens)\nGenerated token IDs:", n_gen);
    for (int id : out.ids) printf(" %d", id);
    printf("\n");

    if (!out.topk_lines.empty()) {
        printf("\nTop-%d logits per step (token:logit), step i = distribution that produced generated token i:\n", out.topk);
        for (const auto &l : out.topk_lines) printf("%s\n", l.c_str());
    }

    probe_print_summary();
    probe_dump_csv(getenv("NANO_PROBE_CSV") ? getenv("NANO_PROBE_CSV") : "probe.csv");

    free_infer_state(s);
    free_llama_model(llama);
    return 0;
}
