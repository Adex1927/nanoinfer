#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model_loader.h"
#include <string>

// ── tokenizer.h ──
// Text <-> token ID conversion for SentencePiece-style ("llama") vocabularies.
// The vocab itself lives in Model (loaded from GGUF metadata).

// ── token_to_piece ──
// Converts one token ID to the raw bytes it represents:
//   - "▁" (U+2581, SentencePiece's space marker) becomes ' '
//   - byte-fallback tokens "<0xXX>" become the single byte 0xXX
//     (multi-byte UTF-8 characters arrive as several consecutive byte tokens,
//      so print/append bytes as-is and the terminal reassembles them)
//   - control / unknown / unused tokens (<s>, </s>, <unk>) become ""
//   - out-of-range IDs become ""
std::string token_to_piece(const Model *model, int token);

// ── detokenize ──
// Concatenates token_to_piece() over a whole sequence.
std::string detokenize(const Model *model, const int *tokens, int n_tokens);

#endif // TOKENIZER_H
