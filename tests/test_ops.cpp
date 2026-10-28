#include "ops.h"
#include "dequant.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void test_matvec_q8_0() {
    printf("Testing matvec_q8_0... ");

    const int n_out = 4;
    const int n_in = 64; // 2 blocks per row
    const int n_blocks = (n_out * n_in) / 32;

    std::vector<BlockQ8_0> blocks(n_blocks);
    // float16: 0x3C00 is 1.0f, 0x3800 is 0.5f, 0x3400 is 0.25f, 0x3E00 is 1.5f
    const uint16_t test_scales[4] = {0x3C00, 0x3800, 0x3400, 0x3E00};

    for (int b = 0; b < n_blocks; b++) {
        blocks[b].scale = test_scales[b % 4];
        for (int j = 0; j < 32; j++) {
            blocks[b].quants[j] = (int8_t)((b * 31 + j * 7) % 250 - 125);
        }
    }

    std::vector<float> x(n_in);
    for (int j = 0; j < n_in; j++) {
        x[j] = sinf((float)j + 0.5f);
    }

    // 1. Reference: dequantize then matvec
    std::vector<float> W_dequant(n_out * n_in);
    dequantize_q8_0(blocks.data(), W_dequant.data(), n_out * n_in);

    std::vector<float> out_ref(n_out);
    matvec(out_ref.data(), W_dequant.data(), x.data(), n_out, n_in);

    // 2. Fused quantized matvec
    std::vector<float> out_test(n_out);
    matvec_q8_0(out_test.data(), blocks.data(), x.data(), n_out, n_in);

    // 3. Compare
    float max_diff = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float diff = fabsf(out_ref[i] - out_test[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("max diff = %e -> %s\n", max_diff, (max_diff < 1e-4f) ? "PASS" : "FAIL");
    if (max_diff >= 1e-4f) {
        exit(1);
    }
}

static void test_matvec_q4_k() {
    printf("Testing matvec_q4_k... ");

    const int n_out = 4;
    const int n_in = 512; // 2 super-blocks of 256 per row
    const int n_blocks = (n_out * n_in) / 256;

    std::vector<BlockQ4_K> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        blocks[b].d = 0x3800;    // 0.5f in f16
        blocks[b].dmin = 0x3400; // 0.25f in f16
        for (int i = 0; i < 12; i++) {
            blocks[b].scales[i] = (uint8_t)((b * 7 + i * 13) & 0x3F);
        }
        for (int i = 0; i < 128; i++) {
            blocks[b].quants[i] = (uint8_t)((b * 17 + i * 29) & 0xFF);
        }
    }

    std::vector<float> x(n_in);
    for (int j = 0; j < n_in; j++) {
        x[j] = sinf((float)j + 0.5f);
    }

    // 1. Reference: dequantize then matvec
    std::vector<float> W_dequant(n_out * n_in);
    dequantize_q4_k(blocks.data(), W_dequant.data(), n_out * n_in);

    std::vector<float> out_ref(n_out);
    matvec(out_ref.data(), W_dequant.data(), x.data(), n_out, n_in);

    // 2. Fused quantized matvec
    std::vector<float> out_test(n_out);
    matvec_q4_k(out_test.data(), blocks.data(), x.data(), n_out, n_in);

    // 2b. Reconstructed x reference (dequantized weights * quantized activations)
    std::vector<BlockQ8_K> q8_test(n_in / 256);
    quantize_q8_k(x.data(), q8_test.data(), n_in);
    std::vector<float> x_recon(n_in);
    for (int j = 0; j < n_in; j++) {
        x_recon[j] = q8_test[j / 256].d * (float)q8_test[j / 256].qs[j % 256];
    }
    std::vector<float> out_recon(n_out);
    matvec(out_recon.data(), W_dequant.data(), x_recon.data(), n_out, n_in);

    // 3. Verify arithmetic exactness against quantized reference (float rounding tolerance)
    float max_recon_diff = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float diff = fabsf(out_recon[i] - out_test[i]);
        if (diff > max_recon_diff) max_recon_diff = diff;
    }

    // 4. Verify quantization error against unquantized float reference (relative < 5%)
    float max_ref_diff = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float diff = fabsf(out_ref[i] - out_test[i]);
        if (diff > max_ref_diff) max_ref_diff = diff;
    }

    bool pass = (max_recon_diff < 5e-4f) && (max_ref_diff / fabsf(out_ref[0]) < 0.06f);
    printf("recon_diff = %e, ref_diff = %e (rel %.2f%%) -> %s\n",
           max_recon_diff, max_ref_diff, (max_ref_diff / fabsf(out_ref[0])) * 100.0f,
           pass ? "PASS" : "FAIL");
    if (!pass) {
        exit(1);
    }
}

static void test_matvec_q6_k() {
    printf("Testing matvec_q6_k... ");

    const int n_out = 4;
    const int n_in = 512; // 2 super-blocks of 256 per row
    const int n_blocks = (n_out * n_in) / 256;

    std::vector<BlockQ6_K> blocks(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        blocks[b].d = 0x3800; // 0.5f in f16
        for (int i = 0; i < 128; i++) {
            blocks[b].ql[i] = (uint8_t)((b * 13 + i * 17) & 0xFF);
        }
        for (int i = 0; i < 64; i++) {
            blocks[b].qh[i] = (uint8_t)((b * 7 + i * 23) & 0xFF);
        }
        for (int i = 0; i < 16; i++) {
            blocks[b].scales[i] = (int8_t)((b * 5 + i * 11) % 64 - 32);
        }
    }

    std::vector<float> x(n_in);
    for (int j = 0; j < n_in; j++) {
        x[j] = sinf((float)j + 0.5f);
    }

    // 1. Reference: dequantize then matvec
    std::vector<float> W_dequant(n_out * n_in);
    dequantize_q6_k(blocks.data(), W_dequant.data(), n_out * n_in);

    std::vector<float> out_ref(n_out);
    matvec(out_ref.data(), W_dequant.data(), x.data(), n_out, n_in);

    // 2. Fused quantized matvec
    std::vector<float> out_test(n_out);
    matvec_q6_k(out_test.data(), blocks.data(), x.data(), n_out, n_in);

    // 3. Compare
    float max_diff = 0.0f;
    for (int i = 0; i < n_out; i++) {
        float diff = fabsf(out_ref[i] - out_test[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("max diff = %e (ref[0]=%f, test[0]=%f) -> %s\n",
           max_diff, out_ref[0], out_test[0], (max_diff < 5e-3f) ? "PASS" : "FAIL");
    if (max_diff >= 5e-3f) {
        exit(1);
    }
}

static void test_quantize_q8_k() {
    printf("Testing quantize_q8_k... ");

    const int n = 512; // 2 blocks of 256
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = sinf((float)i * 0.1f) * ((i % 2 == 0) ? 5.0f : -3.5f);
    }

    std::vector<BlockQ8_K> blocks(n / 256);
    quantize_q8_k(x.data(), blocks.data(), n);

    for (int b = 0; b < n / 256; b++) {
        const BlockQ8_K &blk = blocks[b];
        const float *xb = x.data() + b * 256;

        // 1. Verify round-trip error <= d / 2
        float max_error = 0.0f;
        for (int i = 0; i < 256; i++) {
            float recon = (float)blk.qs[i] * blk.d;
            float err = fabsf(xb[i] - recon);
            if (err > max_error) max_error = err;
            if (err > blk.d * 0.5f + 1e-5f) {
                printf("FAIL: block %d idx %d error %f exceeds tolerance %f (d=%f)\n",
                       b, i, err, blk.d * 0.5f, blk.d);
                exit(1);
            }
        }

        // 2. Verify bsums
        for (int c = 0; c < 16; c++) {
            int16_t expected_sum = 0;
            for (int l = 0; l < 16; l++) {
                expected_sum += blk.qs[c * 16 + l];
            }
            if (blk.bsums[c] != expected_sum) {
                printf("FAIL: block %d bsum[%d] = %d != expected %d\n",
                       b, c, blk.bsums[c], expected_sum);
                exit(1);
            }
        }
    }

    printf("PASS\n");
}

int main() {
    test_quantize_q8_k();
    test_matvec_q8_0();
    test_matvec_q4_k();
    test_matvec_q6_k();
    return 0;
}
