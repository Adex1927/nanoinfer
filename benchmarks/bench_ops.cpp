#include "ops.h"
#include "dequant.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct BenchResult {
    double normal_ms;
    double mem_only_ms;
    double compute_2x_ms;
    double bandwidth_gb_s;
    double gflops;
};

static BenchResult benchmark_layer(const char *name, int n_out, int n_in, int iters = 50) {
    printf("Benchmarking %s (%d x %d)...\n", name, n_out, n_in);

    int n_blocks = (n_out * n_in) / 256;
    size_t weight_bytes = n_blocks * sizeof(BlockQ4_K);

    std::vector<BlockQ4_K> W(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        W[b].d = 0x3800;
        W[b].dmin = 0x3400;
        for (int i = 0; i < 12; i++) W[b].scales[i] = (uint8_t)(i * 7);
        for (int i = 0; i < 128; i++) W[b].quants[i] = (uint8_t)(i * 13);
    }

    std::vector<float> x(n_in, 1.0f);
    std::vector<float> out(n_out);

    // Warm-up
    for (int i = 0; i < 5; i++) {
        matvec_q4_k_diag(out.data(), W.data(), x.data(), n_out, n_in, DiagMode::NORMAL);
    }

    auto time_mode = [&](DiagMode mode) -> double {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            matvec_q4_k_diag(out.data(), W.data(), x.data(), n_out, n_in, mode);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t1 - t0;
        return ms.count() / iters;
    };

    double t_normal  = time_mode(DiagMode::NORMAL);
    double t_mem     = time_mode(DiagMode::MEM_ONLY);
    double t_compute = time_mode(DiagMode::COMPUTE_2X);

    // Bandwidth: weight_bytes / t_normal
    double bw = (double)weight_bytes / (t_normal * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    // 2 operations per weight (1 multiply + 1 add)
    double flops = (2.0 * n_out * n_in) / (t_normal * 1e-3) / 1e9;

    printf("  Normal:     %6.3f ms  (%5.2f GB/s, %5.2f GFLOP/s)\n", t_normal, bw, flops);
    printf("  Mem-Only:   %6.3f ms  (Hardware memory bandwidth floor)\n", t_mem);
    printf("  Compute-2x: %6.3f ms  (Sensitivity ratio: %.2fx)\n", t_compute, t_compute / t_normal);

    double compute_share = 100.0 * (t_normal - t_mem) / t_normal;
    printf("  Diagnostic: %.1f%% Compute Bound / %.1f%% Memory Floor\n\n",
           compute_share, 100.0 - compute_share);

    return {t_normal, t_mem, t_compute, bw, flops};
}

int main() {
    printf("=== nanoinfer Quantized GEMV (Q4_K) Diagnostic Benchmark ===\n\n");

    // 1. Attention Projection: 2048 x 2048
    benchmark_layer("Attention Output (attn_output)", 2048, 2048);

    // 2. FFN Gate / Up Projection: 5632 x 2048
    benchmark_layer("FFN Gate/Up (ffn_gate / ffn_up)", 5632, 2048);

    // 3. FFN Down Projection: 2048 x 5632
    benchmark_layer("FFN Down (ffn_down)", 2048, 5632);

    return 0;
}
