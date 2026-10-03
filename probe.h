#ifndef PROBE_H
#define PROBE_H

// ── probe.h ──
// Lightweight, zero-dependency latency measurement.
//
// Two tools:
//
//   Probe    — RAII scoped timer. Prints one line to stderr when it goes out
//              of scope. Use for one-shot measurements.
//
//   ProbeAccum — Accumulates multiple samples into a named bucket, then
//                prints a summary (total, count, avg, min, max).
//                Use inside loops (e.g. per-layer) to avoid log spam.
//
// Usage (Probe):
//   { Probe p("embedding_lookup"); /* ... work ... */ }
//   // → "[PROBE] embedding_lookup              12.345 ms"
//
// Usage (ProbeAccum):
//   ProbeAccum pa("layer_qkv_proj");
//   for (int layer = 0; layer < n; layer++) {
//       auto t = pa.start();
//       /* ... work ... */
//       pa.stop(t);
//   }
//   pa.report();
//   // → "[PROBE] layer_qkv_proj  n=22  total=270.1ms  avg=12.3ms  min=11.9ms  max=13.1ms"
//
// Compile-time on/off:
//   Define PROBE_ENABLED=0 before including this header to compile out all
//   probes with zero overhead. Default is enabled.

#ifndef PROBE_ENABLED
#define PROBE_ENABLED 1
#endif

#include <chrono>
#include <cstdio>
#include <cfloat>   // FLT_MAX
#include <cstring>  // strncpy

// ─────────────────────────────────────────────────────────────────────────────
// Internal clock alias
// ─────────────────────────────────────────────────────────────────────────────

namespace probe_detail {
using Clock = std::chrono::steady_clock;
using TP    = Clock::time_point;

inline double ms_since(TP t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
} // namespace probe_detail

// ─────────────────────────────────────────────────────────────────────────────
// Probe — RAII scoped timer
// ─────────────────────────────────────────────────────────────────────────────

#if PROBE_ENABLED

struct Probe {
    const char               *name;
    probe_detail::TP          t0;

    explicit Probe(const char *n)
        : name(n), t0(probe_detail::Clock::now()) {}

    // non-copyable, non-movable
    Probe(const Probe &)            = delete;
    Probe &operator=(const Probe &) = delete;

    double elapsed_ms() const { return probe_detail::ms_since(t0); }

    ~Probe() {
        fprintf(stderr, "[PROBE] %-35s  %8.3f ms\n", name, elapsed_ms());
    }
};

#else // PROBE_ENABLED == 0

struct Probe {
    explicit Probe(const char *) {}
    double elapsed_ms() const { return 0.0; }
};

#endif // PROBE_ENABLED

// ─────────────────────────────────────────────────────────────────────────────
// ProbeAccum — multi-sample accumulator (for loops)
// ─────────────────────────────────────────────────────────────────────────────

#if PROBE_ENABLED

struct ProbeAccum {
    char   name[64];
    double total_ms = 0.0;
    double min_ms   = DBL_MAX;
    double max_ms   = 0.0;
    int    count    = 0;

    explicit ProbeAccum(const char *n) {
        strncpy(name, n, sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }

    // Call start() before the work, pass result to stop().
    probe_detail::TP start() const { return probe_detail::Clock::now(); }

    void stop(probe_detail::TP t0) {
        double ms = probe_detail::ms_since(t0);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
        count++;
    }

    void report() const {
        if (count == 0) {
            fprintf(stderr, "[PROBE] %-35s  (no samples)\n", name);
            return;
        }
        double avg = total_ms / count;
        fprintf(stderr,
            "[PROBE] %-35s  n=%-4d  total=%8.2f ms  avg=%7.3f ms"
            "  min=%7.3f ms  max=%7.3f ms\n",
            name, count, total_ms, avg, min_ms, max_ms);
    }
};

#else // PROBE_ENABLED == 0

struct ProbeAccum {
    explicit ProbeAccum(const char *) {}
    probe_detail::TP start() const { return {}; }
    void stop(probe_detail::TP)    {}
    void report() const            {}
};

#endif // PROBE_ENABLED

#endif // PROBE_H
