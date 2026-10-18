#ifndef PROBE_H
#define PROBE_H

// ── probe.h ──
// Lightweight, zero-dependency latency measurement.
//
// Probes do NOT print as they run. Every sample is appended to an in-memory
// table (name, phase, token_pos, layer, ms) so you can dump it to CSV once at
// the end and analyse / plot it however you like.
//
// Two tools:
//
//   Probe      — RAII scoped timer. Records one sample when it goes out of
//                scope. Use for one-shot measurements.
//
//   ProbeAccum — Records one sample per start()/stop() pair. Use inside loops
//                (e.g. per-layer). report() is kept for source compatibility
//                but prints nothing.
//
// Context columns (set by the code being measured, captured automatically):
//   probe_set_phase("prefill" | "decode")  — which stage of generation
//   probe_set_pos(pos)                      — token position (-1 = n/a)
//   probe_set_layer(layer)                  — transformer layer (-1 = n/a)
// A Probe captures the context when constructed; ProbeAccum when stopped.
//
// Usage:
//   { Probe p("embedding_lookup"); /* ... work ... */ }
//
//   ProbeAccum pa("layer:qkv_proj");
//   for (...) { auto t = pa.start(); /* ... */ pa.stop(t); }
//
//   // at the end of the program:
//   probe_print_summary();                 // aggregated table to stderr
//   probe_dump_csv("probe.csv");           // raw samples, one row each
//
// Compile-time on/off:
//   Define PROBE_ENABLED=0 before including this header to compile out all
//   probes with zero overhead. Default is enabled.

#ifndef PROBE_ENABLED
#define PROBE_ENABLED 1
#endif

#include <chrono>
#include <cstdio>
#include <cfloat>   // DBL_MAX
#include <cstring>  // strncpy
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

#if PROBE_ENABLED

// ─────────────────────────────────────────────────────────────────────────────
// Sample store
// ─────────────────────────────────────────────────────────────────────────────

namespace probe_detail {

struct Record {
    int         name_id;
    const char *phase;      // string literal ("" if unset)
    int         token_pos;  // -1 = n/a
    int         layer;      // -1 = n/a
    double      ms;
};

struct Context {
    const char *phase = "";
    int         pos   = -1;
    int         layer = -1;
};

struct Store {
    std::vector<std::string>              names;   // name_id -> name
    std::unordered_map<std::string, int>  ids;     // name -> name_id
    std::vector<Record>                   records;
    Context                               ctx;
};

inline Store &store() {
    static Store s;
    return s;
}

inline int intern(const char *name) {
    Store &s = store();
    auto it = s.ids.find(name);
    if (it != s.ids.end()) return it->second;
    int id = (int)s.names.size();
    s.names.push_back(name);
    s.ids.emplace(name, id);
    return id;
}

inline void push(int name_id, const Context &c, double ms) {
    store().records.push_back({name_id, c.phase, c.pos, c.layer, ms});
}

} // namespace probe_detail

// ── context setters ──
inline void probe_set_phase(const char *phase) { probe_detail::store().ctx.phase = phase; }
inline void probe_set_pos(int pos)             { probe_detail::store().ctx.pos   = pos;   }
inline void probe_set_layer(int layer)         { probe_detail::store().ctx.layer = layer; }

// ── probe_dump_csv ──
// Writes every raw sample: name,phase,token_pos,layer,ms
inline bool probe_dump_csv(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("probe_dump_csv"); return false; }
    const auto &s = probe_detail::store();
    fprintf(f, "name,phase,token_pos,layer,ms\n");
    for (const auto &r : s.records) {
        fprintf(f, "%s,%s,%d,%d,%.6f\n",
                s.names[r.name_id].c_str(), r.phase, r.token_pos, r.layer, r.ms);
    }
    fclose(f);
    fprintf(stderr, "[PROBE] wrote %zu samples to %s\n", s.records.size(), path);
    return true;
}

// ── probe_print_summary ──
// One line per (name, phase): count, total, avg, min, max.
inline void probe_print_summary() {
    struct Agg { int n = 0; double total = 0, mn = DBL_MAX, mx = 0; };
    const auto &s = probe_detail::store();
    std::map<std::pair<int, std::string>, Agg> agg;   // ordered by first-seen name
    for (const auto &r : s.records) {
        Agg &a = agg[{r.name_id, r.phase}];
        a.n++; a.total += r.ms;
        if (r.ms < a.mn) a.mn = r.ms;
        if (r.ms > a.mx) a.mx = r.ms;
    }
    fprintf(stderr, "\n[PROBE] %-28s %-8s %6s %12s %10s %10s %10s\n",
            "name", "phase", "n", "total ms", "avg", "min", "max");
    for (const auto &kv : agg) {
        const Agg &a = kv.second;
        fprintf(stderr, "[PROBE] %-28s %-8s %6d %12.2f %10.3f %10.3f %10.3f\n",
                s.names[kv.first.first].c_str(), kv.first.second.c_str(),
                a.n, a.total, a.total / a.n, a.mn, a.mx);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe — RAII scoped timer
// ─────────────────────────────────────────────────────────────────────────────

struct Probe {
    int                       name_id;
    probe_detail::Context     ctx;     // captured at construction
    probe_detail::TP          t0;

    explicit Probe(const char *n)
        : name_id(probe_detail::intern(n)),
          ctx(probe_detail::store().ctx),
          t0(probe_detail::Clock::now()) {}

    // non-copyable, non-movable
    Probe(const Probe &)            = delete;
    Probe &operator=(const Probe &) = delete;

    double elapsed_ms() const { return probe_detail::ms_since(t0); }

    ~Probe() { probe_detail::push(name_id, ctx, elapsed_ms()); }
};

// ─────────────────────────────────────────────────────────────────────────────
// ProbeAccum — one sample per start()/stop() pair (for loops)
// ─────────────────────────────────────────────────────────────────────────────

struct ProbeAccum {
    int name_id;

    explicit ProbeAccum(const char *n) : name_id(probe_detail::intern(n)) {}

    // Call start() before the work, pass result to stop().
    probe_detail::TP start() const { return probe_detail::Clock::now(); }

    // Records the sample with the context current at stop() time.
    void stop(probe_detail::TP t0) {
        probe_detail::push(name_id, probe_detail::store().ctx, probe_detail::ms_since(t0));
    }

    // Kept for source compatibility; samples are stored, not printed.
    void report() const {}
};

#else // PROBE_ENABLED == 0

inline void probe_set_phase(const char *) {}
inline void probe_set_pos(int)            {}
inline void probe_set_layer(int)          {}
inline bool probe_dump_csv(const char *)  { return true; }
inline void probe_print_summary()         {}

struct Probe {
    explicit Probe(const char *) {}
    double elapsed_ms() const { return 0.0; }
};

struct ProbeAccum {
    explicit ProbeAccum(const char *) {}
    probe_detail::TP start() const { return {}; }
    void stop(probe_detail::TP)    {}
    void report() const            {}
};

#endif // PROBE_ENABLED

#endif // PROBE_H
