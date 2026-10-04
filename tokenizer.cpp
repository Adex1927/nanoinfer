#include "tokenizer.h"
#include <cstdlib>
#include <cstring>

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
