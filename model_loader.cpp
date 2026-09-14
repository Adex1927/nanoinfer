#include "model_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

// ── Same helpers from model_test.cpp ──

static char *read_gguf_string(FILE *f, uint64_t *out_len) {
    uint64_t len;
    fread(&len, sizeof(len), 1, f);

    char *buf = (char *)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';

    if (out_len) *out_len = len;
    return buf;
}

static size_t value_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:  case GGUF_TYPE_INT16:  return 2;
        case GGUF_TYPE_UINT32:  case GGUF_TYPE_INT32:  case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:  case GGUF_TYPE_INT64:  case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void skip_value(FILE *f, uint32_t type) {
    size_t sz = value_type_size(type);
    if (sz > 0) {
        fseek(f, sz, SEEK_CUR);
        return;
    }
    if (type == GGUF_TYPE_STRING) {
        uint64_t len;
        fread(&len, sizeof(len), 1, f);
        fseek(f, len, SEEK_CUR);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type;
        uint64_t count;
        fread(&elem_type, sizeof(elem_type), 1, f);
        fread(&count, sizeof(count), 1, f);
        for (uint64_t i = 0; i < count; i++) {
            skip_value(f, elem_type);
        }
    }
}

// ── load_model ──
// Same flow as model_test.cpp:
//   1. fopen + fread to parse header, metadata, tensor info
//   2. close FILE*
//   3. mmap the file
//   4. return persistent Model*

Model *load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return nullptr;
    }

    // allocate the model struct
    Model *model = (Model *)calloc(1, sizeof(Model));

    // ── read header ──
    if (fread(&model->header, sizeof(GGUFHeader), 1, f) != 1) {
        fprintf(stderr, "failed to read header\n");
        fclose(f);
        free(model);
        return nullptr;
    }

    if (model->header.magic != 0x46554747) {
        fprintf(stderr, "not a GGUF file (magic: 0x%08X)\n", model->header.magic);
        fclose(f);
        free(model);
        return nullptr;
    }

    printf("=== GGUF Header ===\n");
    printf("magic:            0x%08X (GGUF ✓)\n", model->header.magic);
    printf("version:          %u\n", model->header.version);
    printf("tensor count:     %llu\n", (unsigned long long)model->header.tensor_count);
    printf("metadata kv count:%llu\n", (unsigned long long)model->header.metadata_kv_count);

    // ── read metadata (print + skip) ──
    printf("\n=== Metadata ===\n");

    for (uint64_t i = 0; i < model->header.metadata_kv_count; i++) {
        char *key = read_gguf_string(f, nullptr);
        uint32_t vtype;
        fread(&vtype, sizeof(vtype), 1, f);

        printf("[%3llu] %-45s ", (unsigned long long)i, key);

        switch (vtype) {
            case GGUF_TYPE_UINT8:   { uint8_t v;  fread(&v, 1, 1, f); printf("(uint8)   %u\n", v); break; }
            case GGUF_TYPE_INT8:    { int8_t v;   fread(&v, 1, 1, f); printf("(int8)    %d\n", v); break; }
            case GGUF_TYPE_UINT16:  { uint16_t v; fread(&v, 2, 1, f); printf("(uint16)  %u\n", v); break; }
            case GGUF_TYPE_INT16:   { int16_t v;  fread(&v, 2, 1, f); printf("(int16)   %d\n", v); break; }
            case GGUF_TYPE_UINT32:  { uint32_t v; fread(&v, 4, 1, f); printf("(uint32)  %u\n", v); break; }
            case GGUF_TYPE_INT32:   { int32_t v;  fread(&v, 4, 1, f); printf("(int32)   %d\n", v); break; }
            case GGUF_TYPE_FLOAT32: { float v;    fread(&v, 4, 1, f); printf("(float32) %f\n", v); break; }
            case GGUF_TYPE_BOOL:    { uint8_t v;  fread(&v, 1, 1, f); printf("(bool)    %s\n", v ? "true" : "false"); break; }
            case GGUF_TYPE_UINT64:  { uint64_t v; fread(&v, 8, 1, f); printf("(uint64)  %llu\n", (unsigned long long)v); break; }
            case GGUF_TYPE_INT64:   { int64_t v;  fread(&v, 8, 1, f); printf("(int64)   %lld\n", (long long)v); break; }
            case GGUF_TYPE_FLOAT64: { double v;   fread(&v, 8, 1, f); printf("(float64) %f\n", v); break; }
            case GGUF_TYPE_STRING: {
                char *val = read_gguf_string(f, nullptr);
                if (strlen(val) > 80) val[80] = '\0';
                printf("(string)  \"%s\"\n", val);
                free(val);
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t elem_type;
                uint64_t count;
                fread(&elem_type, sizeof(elem_type), 1, f);
                fread(&count, sizeof(count), 1, f);
                printf("(array)   %llu elements of type %u\n", (unsigned long long)count, elem_type);
                for (uint64_t j = 0; j < count; j++) {
                    skip_value(f, elem_type);
                }
                break;
            }
            default:
                printf("(unknown type %u)\n", vtype);
                break;
        }

        free(key);
    }

    // ── read tensor info table ──
    model->tensor_count = model->header.tensor_count;
    model->tensors = (TensorInfo *)calloc(model->tensor_count, sizeof(TensorInfo));

    printf("\n=== Tensor Info (%llu tensors) ===\n", (unsigned long long)model->tensor_count);

    for (uint64_t i = 0; i < model->tensor_count; i++) {
        TensorInfo *t = &model->tensors[i];

        t->name = read_gguf_string(f, nullptr);
        fread(&t->n_dims, sizeof(t->n_dims), 1, f);

        for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
            fread(&t->dims[d], sizeof(uint64_t), 1, f);
        }

        fread(&t->type, sizeof(t->type), 1, f);
        fread(&t->offset, sizeof(t->offset), 1, f);
    }

    // ── compute tensor data start ──
    long pos = ftell(f);
    uint64_t alignment = 32;
    model->tensor_data_start = ((pos + alignment - 1) / alignment) * alignment;

    // done with fread-based parsing
    fclose(f);

    // ── mmap the file ──
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        free(model->tensors);
        free(model);
        return nullptr;
    }

    struct stat sb;
    fstat(fd, &sb);
    model->file_size = sb.st_size;

    model->mapped = (uint8_t *)mmap(nullptr, model->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (model->mapped == MAP_FAILED) {
        perror("mmap");
        free(model->tensors);
        free(model);
        return nullptr;
    }

    printf("\nModel loaded: %.1f MB, %llu tensors, mmap'd at %p\n",
           model->file_size / (1024.0 * 1024.0),
           (unsigned long long)model->tensor_count,
           (void *)model->mapped);

    return model;
}

// ── get_tensor_info ──
// Linear scan by name. Simple and fine for now.

TensorInfo *get_tensor_info(Model *model, const char *name) {
    for (uint64_t i = 0; i < model->tensor_count; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return nullptr;
}

// ── get_tensor_data ──
// Returns a pointer directly into the mmap'd file.

void *get_tensor_data(Model *model, const char *name) {
    TensorInfo *t = get_tensor_info(model, name);
    if (!t) return nullptr;
    return model->mapped + model->tensor_data_start + t->offset;
}

// ── free_model ──

void free_model(Model *model) {
    if (!model) return;

    if (model->mapped && model->mapped != MAP_FAILED) {
        munmap(model->mapped, model->file_size);
    }

    // free all tensor names (we strdup'd/malloc'd them during parsing)
    for (uint64_t i = 0; i < model->tensor_count; i++) {
        free(model->tensors[i].name);
    }
    free(model->tensors);
    free(model);
}

// ── ggml_type_name ──

const char *ggml_type_name(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        default:             return "???";
    }
}

// ── display_model ──

void display_model(Model *model) {
    if (!model) {
        printf("(null model)\n");
        return;
    }

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Model Summary: %llu tensors, %.1f MB                              \n",
           (unsigned long long)model->tensor_count,
           model->file_size / (1024.0 * 1024.0));
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  %-4s  %-6s  %-20s  %-35s ║\n", "#", "Type", "Dimensions", "Name");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");

    for (uint64_t i = 0; i < model->tensor_count; i++) {
        TensorInfo *t = &model->tensors[i];

        // build dims string like "[2048, 2048]"
        char dims_str[64];
        int pos = 0;
        pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "[");
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d > 0) pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, ", ");
            pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "%llu",
                           (unsigned long long)t->dims[d]);
        }
        snprintf(dims_str + pos, sizeof(dims_str) - pos, "]");

        printf("║  %-4llu  %-6s  %-20s  %-35s ║\n",
               (unsigned long long)i,
               ggml_type_name(t->type),
               dims_str,
               t->name);
    }

    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
}
