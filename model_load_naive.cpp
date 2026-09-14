#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The GGUF header is the first 24 bytes of the file.
// It tells us: is this really a GGUF file, what version,
// how many tensors, and how many metadata key-value pairs to expect.
struct GGUFHeader {
    uint32_t magic;          // must be 0x46554747 ("GGUF" in little-endian)
    uint32_t version;        // format version (currently 3)
    uint64_t tensor_count;   // number of tensors in the file
    uint64_t metadata_kv_count; // number of metadata key-value pairs
};

// GGUF metadata value types — this enum tells us how to interpret each value
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

// Helper: read a GGUF string from the file stream.
// GGUF strings are: [uint64_t length][length bytes, NOT null-terminated]
// We malloc a buffer, read into it, and null-terminate it ourselves.
// Caller must free() the returned pointer.
char *read_gguf_string(FILE *f, uint64_t *out_len) {
    uint64_t len;
    fread(&len, sizeof(len),1 , f);

    char *buf = (char *)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';

    if (out_len) *out_len = len;
    return buf;
}

// Helper: how many bytes does a single value of this type occupy?
// Returns 0 for string/array since those are variable-length.
size_t value_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:  case GGUF_TYPE_INT16:  return 2;
        case GGUF_TYPE_UINT32:  case GGUF_TYPE_INT32:  case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:  case GGUF_TYPE_INT64:  case GGUF_TYPE_FLOAT64: return 8;
        default: return 0; // string, array — handle separately
    }
}

// Helper: skip over a single metadata value in the stream.
// For fixed-size types we just fseek. For strings we read & discard.
// For arrays we recurse over each element.
void skip_value(FILE *f, uint32_t type) {
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    // read the header in one shot — it's a flat struct, no padding issues
    GGUFHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "failed to read header\n");
        fclose(f);
        return 1;
    }

    // "GGUF" as bytes: G=0x47 G=0x47 U=0x55 F=0x46
    // stored little-endian → 0x46554747
    if (header.magic != 0x46554747) {
        fprintf(stderr, "not a GGUF file (magic: 0x%08X)\n", header.magic);
        fclose(f);
        return 1;
    }

    printf("=== GGUF Header ===\n");
    printf("magic:            0x%08X (GGUF ✓)\n", header.magic);
    printf("version:          %u\n", header.version);
    printf("tensor count:     %llu\n", (unsigned long long)header.tensor_count);
    printf("metadata kv count:%llu\n", (unsigned long long)header.metadata_kv_count);

    // ── Step 2: Read metadata key-value pairs ──
    // They come right after the header, one after another.
    // Each one is: [string key][uint32 type][value]
    printf("\n=== Metadata ===\n");

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        // 1) read the key (a GGUF string)
        char *key = read_gguf_string(f, nullptr);

        // 2) read the value type
        uint32_t vtype;
        fread(&vtype, sizeof(vtype), 1, f);

        // 3) read or skip the value depending on type
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
                // truncate long strings (like chat templates) for display
                if (strlen(val) > 80) val[80] = '\0';
                printf("(string)  \"%s\"\n", val);
                free(val);
                break;
            }

            case GGUF_TYPE_ARRAY: {
                // arrays are: [uint32 element_type][uint64 count][count × element]
                // we just print the count and skip the data
                uint32_t elem_type;
                uint64_t count;
                fread(&elem_type, sizeof(elem_type), 1, f);
                fread(&count, sizeof(count), 1, f);
                printf("(array)   %llu elements of type %u\n", (unsigned long long)count, elem_type);
                // skip all elements
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

    // ── Step 3: Read tensor info entries ──
    // Right after metadata, we get tensor_count entries.
    // This is NOT the weight data itself — just a directory listing.
    //
    // Each entry is read field by field:
    //   1. name       → a GGUF string (same format as metadata keys)
    //   2. n_dims     → uint32: how many dimensions (1D, 2D, etc.)
    //   3. dims[]     → n_dims × uint64: size along each axis
    //   4. type       → uint32: the ggml_type enum (F32, F16, Q4_K, etc.)
    //   5. offset     → uint64: where this tensor's data starts,
    //                    relative to the START of the tensor data section
    //                    (not the start of the file!)

    // human-readable names for common ggml types
    const char *type_name[] = {
        "F32", "F16", "Q4_0", "Q4_1", "???", "???",
        "Q5_0", "Q5_1", "Q8_0", "Q8_1",
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
        "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
        "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
    };
    const int type_name_count = sizeof(type_name) / sizeof(type_name[0]);

    printf("\n=== Tensor Info (%llu tensors) ===\n", (unsigned long long)header.tensor_count);

    // we'll save info about the first F32 tensor we find, to read it later
    uint64_t first_f32_offset = (uint64_t)-1;
    char    *first_f32_name   = nullptr;
    uint64_t first_f32_numel  = 0;


    for (uint64_t i = 0; i < header.tensor_count; i++) {
        // 1. tensor name — same GGUF string format: [uint64 len][bytes]
        char *name = read_gguf_string(f, nullptr);

        // 2. number of dimensions — e.g. 1 for bias vectors, 2 for weight matrices
        uint32_t n_dims;
        fread(&n_dims, sizeof(n_dims), 1, f);

        // 3. dimension sizes — read n_dims uint64 values
        //    e.g. for a [4096 x 4096] matrix: dims[0]=4096, dims[1]=4096
        uint64_t dims[4] = {0};  // most tensors are ≤ 4D
        for (uint32_t d = 0; d < n_dims && d < 4; d++) {
            fread(&dims[d], sizeof(uint64_t), 1, f);
        }

        // 4. data type — which ggml_type (F16, Q4_K, etc.)
        uint32_t type;
        fread(&type, sizeof(type), 1, f);

        // 5. offset — byte offset within the tensor data section
        uint64_t offset;
        fread(&offset, sizeof(offset), 1, f);

        // print it: index, name, shape, type, offset
        const char *tname = (type < type_name_count) ? type_name[type] : "???";
        printf("[%3llu] %-50s  ", (unsigned long long)i, name);
        // print shape like [4096, 4096]
        printf("[");
        for (uint32_t d = 0; d < n_dims; d++) {
            if (d > 0) printf(", ");
            printf("%llu", (unsigned long long)dims[d]);
        }
        printf("]");
        printf("  type=%-20s  offset=%llu\n", tname, (unsigned long long)offset);

        // save the first F32 tensor we find so we can read it later
        if (type == 0 && first_f32_offset == (uint64_t)-1) {
            first_f32_offset = offset;
            first_f32_name = strdup(name);
            first_f32_numel = 1;
            for (uint32_t d = 0; d < n_dims; d++) first_f32_numel *= dims[d];
        }

        free(name);
    }

    // ── Step 4: Locate the tensor data section ──
    // After all tensor info entries, the file is padded to a 32-byte boundary.
    // That's where the actual weight data starts.
    long pos = ftell(f);
    uint64_t alignment = 32;  // default GGUF alignment
    // round up: (pos + alignment - 1) / alignment * alignment
    uint64_t tensor_data_start = ((pos + alignment - 1) / alignment) * alignment;

    printf("\n=== Tensor Data ===\n");
    printf("tensor info ends at:   byte %ld\n", pos);
    printf("aligned data starts at:byte %llu\n", (unsigned long long)tensor_data_start);

    // ── Step 5: Read one actual tensor ──
    // Seek to tensor_data_start + offset, read the floats, print a few.
    if (first_f32_name) {
        uint64_t abs_offset = tensor_data_start + first_f32_offset;
        printf("\nReading tensor: %s (%llu floats)\n", first_f32_name,
               (unsigned long long)first_f32_numel);
        printf("file position:  byte %llu\n", (unsigned long long)abs_offset);

        fseek(f, abs_offset, SEEK_SET);

        // read ALL the floats for this tensor (it's small — a few KB)
        float *data = (float *)malloc(first_f32_numel * sizeof(float));
        fread(data, sizeof(float), first_f32_numel, f);

        // print first 10 and last 5 values
        printf("first 10 values:\n  ");
        for (uint64_t j = 0; j < 10 && j < first_f32_numel; j++) {
            printf("%.6f ", data[j]);
        }
        printf("\nlast 5 values:\n  ");
        for (uint64_t j = (first_f32_numel > 5 ? first_f32_numel - 5 : 0);
             j < first_f32_numel; j++) {
            printf("%.6f ", data[j]);
        }
        printf("\n");

        free(data);
        free(first_f32_name);
    }

    fclose(f);
    return 0;
}
