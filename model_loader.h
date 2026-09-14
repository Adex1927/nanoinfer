#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <cstdint>
#include <cstddef>

// ── Structs we already built ──

struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
};

// Info about a single tensor — what we read from the tensor info table
struct TensorInfo {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    GGMLType  type;       // ggml_type enum value
    uint64_t  offset;     // relative to tensor data section start
};

// The loaded model — holds the mmap'd file and parsed tensor directory
struct Model {
    // mmap state
    uint8_t  *mapped;           // pointer to mmap'd file
    size_t    file_size;        // total file size in bytes

    // parsed header
    GGUFHeader header;

    // tensor directory (array of tensor_count entries)
    TensorInfo *tensors;
    uint64_t    tensor_count;

    // where tensor data starts in the file
    uint64_t tensor_data_start;
};

// ── API ──

// Load a GGUF file: parse header, metadata, tensor info, mmap the file.
// Returns nullptr on failure. Prints metadata during loading.
Model *load_model(const char *path);

// Get a pointer to a tensor's raw data by name.
// Returns nullptr if not found.
void *get_tensor_data(Model *model, const char *name);

// Get a TensorInfo by name. Returns nullptr if not found.
TensorInfo *get_tensor_info(Model *model, const char *name);

// Free everything: munmap + free tensor names + free model.
void free_model(Model *model);

#endif // MODEL_LOADER_H
