#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

// Token types as stored in tokenizer.ggml.token_type
enum {
    TOKEN_TYPE_NORMAL       = 1,
    TOKEN_TYPE_UNKNOWN      = 2,
    TOKEN_TYPE_CONTROL      = 3,
    TOKEN_TYPE_USER_DEFINED = 4,
    TOKEN_TYPE_UNUSED       = 5,
    TOKEN_TYPE_BYTE         = 6,
};

// Parses "<0xXX>" -> byte value, or -1 if the string isn't a byte token.
static int parse_byte_token(const char *s) {
    size_t n = strlen(s);
    if (n != 6 || strncmp(s, "<0x", 3) != 0 || s[5] != '>') return -1;
    char hex[3] = { s[3], s[4], '\0' };
    char *end = nullptr;
    long v = strtol(hex, &end, 16);
    if (*end != '\0') return -1;
    return (int)v;
}

std::string token_to_piece(const Model *model, int token) {
    if (!model->vocab || token < 0 || (uint64_t)token >= model->vocab_size) return "";

    int type = model->token_types ? model->token_types[token] : TOKEN_TYPE_NORMAL;
    if (type == TOKEN_TYPE_CONTROL || type == TOKEN_TYPE_UNKNOWN || type == TOKEN_TYPE_UNUSED) {
        return "";
    }

    const char *s = model->vocab[token];

    // byte-fallback token: emit the raw byte
    if (type == TOKEN_TYPE_BYTE) {
        int b = parse_byte_token(s);
        if (b >= 0) return std::string(1, (char)b);
    }

    // normal token: replace U+2581 "▁" (UTF-8: E2 96 81) with a space
    std::string out;
    for (size_t i = 0; s[i] != '\0'; ) {
        if ((unsigned char)s[i]   == 0xE2 &&
            (unsigned char)s[i+1] == 0x96 &&
            (unsigned char)s[i+2] == 0x81) {
            out += ' ';
            i += 3;
        } else {
            out += s[i++];
        }
    }
    return out;
}

std::string detokenize(const Model *model, const int *tokens, int n_tokens) {
    std::string out;
    for (int i = 0; i < n_tokens; i++) out += token_to_piece(model, tokens[i]);
    return out;
}

// ── tokenize ──
std::vector<int> tokenize(const Model *model, const std::string &text, bool add_bos) {
    std::vector<int> ids;
    if (!model->vocab || model->vocab_size == 0) return ids;

    // piece string -> id, only for tokens that may result from merging plain text
    // (not control / unused / byte tokens).
    std::unordered_map<std::string, int> lookup;
    lookup.reserve(model->vocab_size * 2);
    for (uint64_t i = 0; i < model->vocab_size; i++) {
        int type = model->token_types ? model->token_types[i] : TOKEN_TYPE_NORMAL;
        if (type == TOKEN_TYPE_NORMAL || type == TOKEN_TYPE_USER_DEFINED) {
            lookup.emplace(model->vocab[i], (int)i);
        }
    }
    auto score_of = [&](int id) { return model->token_scores ? model->token_scores[id] : 0.0f; };

    if (add_bos) ids.push_back((int)get_metadata_u32(const_cast<Model *>(model),
                                                     "tokenizer.ggml.bos_token_id", 1));

    // 1. normalize: spaces -> "▁", plus a leading "▁"
    std::string norm = "\xE2\x96\x81";
    for (char c : text) {
        if (c == ' ') norm += "\xE2\x96\x81";
        else          norm += c;
    }

    // 2. split into UTF-8 characters
    std::vector<std::string> sym;
    for (size_t i = 0; i < norm.size(); ) {
        unsigned char c = (unsigned char)norm[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        if (i + len > norm.size()) len = norm.size() - i;
        sym.push_back(norm.substr(i, len));
        i += len;
    }

    // 3. greedy merges: best-scoring adjacent pair first, leftmost on ties.
    //    O(n^2) rescans are fine for prompt-sized inputs.
    while (sym.size() > 1) {
        int best_i = -1;
        float best_score = 0.0f;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = lookup.find(sym[i] + sym[i + 1]);
            if (it == lookup.end()) continue;
            float sc = score_of(it->second);
            if (best_i < 0 || sc > best_score) { best_i = (int)i; best_score = sc; }
        }
        if (best_i < 0) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + best_i + 1);
    }

    // 4. emit ids; unknown pieces fall back to one byte token per byte
    for (const std::string &s : sym) {
        auto it = lookup.find(s);
        if (it != lookup.end()) { ids.push_back(it->second); continue; }
        for (unsigned char b : s) {
            char name[8];
            snprintf(name, sizeof(name), "<0x%02X>", b);
            int id = 0;  // <unk> if the vocab has no byte tokens
            for (uint64_t i = 0; i < model->vocab_size; i++) {
                if (strcmp(model->vocab[i], name) == 0) { id = (int)i; break; }
            }
            ids.push_back(id);
        }
    }
    return ids;
}
