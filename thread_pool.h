#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

// ── ThreadPool ──
//
// A fixed set of worker threads, created once and reused, instead of
// spawning fresh threads on every matvec call.
//
// Dynamic Work-Stealing: workers dynamically claim small row chunks
// via atomic increment, balancing heterogeneous P/E-core speeds.
class ThreadPool {
public:
    // A job is a plain function pointer plus an opaque context pointer.
    // This hides the lambda's real type ("type erasure") without heap allocation.
    using Job = void (*)(void *ctx, int r_start, int r_end);

    explicit ThreadPool(int n_workers);
    ~ThreadPool();

    // Non-copyable: a copy would share the same running threads.
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    // Distribute rows [0, n_out) across workers dynamically, and block
    // until all rows have finished. Must be called from one thread at a time.
    void run(Job fn, void *ctx, int n_out);

private:
    void worker_loop(int id);

    const int                n_workers;
    std::vector<std::thread> threads;
    std::mutex               mu;       // guards cv and state transitions
    std::condition_variable  work_cv;  // workers sleep here until a new job (or stop)
    std::condition_variable  done_cv;  // run() sleeps here until pending == 0

    // The current job
    Job                      job_fn  = nullptr;
    void                    *job_ctx = nullptr;
    int                      job_n   = 0;

    alignas(64) std::atomic<int> current_row{0}; // dynamic chunk counter (cache-aligned)

    uint64_t generation = 0;      // bumped once per run(); tells workers "this job is new"
    int      pending    = 0;      // workers still busy with the current job
    bool     stop       = false;
};

// Singleton pool for matvec operations
ThreadPool &global_pool();

// ── parallel_rows ──
//
// Runs worker(r_start, r_end) over rows [0, n_out), distributed via dynamic
// work-stealing across worker threads in global_pool().
//
// Zero-heap type erasure:
// The stateless lambda `trampoline` converts to a C function pointer (Job),
// and casts ctx back to the original worker reference.
template <typename F>
void parallel_rows(int n_out, F &&worker) {
    if (n_out < 64) {
        worker(0, n_out);
        return;
    }

    auto trampoline = [](void *ctx, int r_start, int r_end) {
        (*reinterpret_cast<typename std::remove_reference<F>::type *>(ctx))(r_start, r_end);
    };

    global_pool().run(trampoline, &worker, n_out);
}

#endif // THREAD_POOL_H
