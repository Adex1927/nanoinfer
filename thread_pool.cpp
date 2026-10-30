#include "thread_pool.h"
#include <algorithm>
#include <cstdlib>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

// Start n_workers threads. Each one runs worker_loop(i) on this pool.
// n_workers is stored first (initializer list) so workers can read it safely;
// threads.size() is still changing while we spawn them.
ThreadPool::ThreadPool(int n_workers) : n_workers(n_workers) {
    threads.reserve(n_workers);
    for (int i = 0; i < n_workers; i++) {
        threads.emplace_back(&ThreadPool::worker_loop, this, i);
    }
}

// Tell every worker to exit, wake them all, then wait for them to finish.
ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mu);
        stop = true;
    }
    work_cv.notify_all();
    for (auto &t : threads) {
        t.join();
    }
}

// Publish a job, wake the workers, wait until they're all done.
void ThreadPool::run(Job fn, void *ctx, int n_out) {
    {
        std::lock_guard<std::mutex> lock(mu);
        job_fn      = fn;
        job_ctx     = ctx;
        job_n       = n_out;
        current_row.store(0, std::memory_order_relaxed);
        pending     = n_workers;
        generation++;
    }
    work_cv.notify_all();

    std::unique_lock<std::mutex> lock(mu);
    done_cv.wait(lock, [this] { return pending == 0; });
}

// What each worker thread does for its whole lifetime:
// wait for a new job -> dynamically steal row chunks -> report done -> repeat.
void ThreadPool::worker_loop(int id) {
    (void)id;
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    uint64_t seen = 0;

    while (true) {
        std::unique_lock<std::mutex> lock(mu);
        work_cv.wait(lock, [&] { return stop || generation != seen; });
        if (stop) return;

        // Copy the job out while holding the lock, then release it so the
        // workers compute in parallel without holding the lock.
        seen     = generation;
        Job   fn  = job_fn;
        void *ctx = job_ctx;
        int   n   = job_n;
        lock.unlock();

        // Dynamic work stealing: claim chunks atomically.
        // Adaptive chunk sizing balances dispatch overhead and load balancing.
        const int chunk_size = (n <= 32) ? 4 : ((n < 256) ? 8 : (n < 2048 ? 16 : 32));
        while (true) {
            int r_start = current_row.fetch_add(chunk_size, std::memory_order_relaxed);
            if (r_start >= n) break;
            int r_end = std::min(r_start + chunk_size, n);
            fn(ctx, r_start, r_end);
        }

        // Report done. Whoever finishes last wakes run().
        lock.lock();
        if (--pending == 0) done_cv.notify_one();
    }
}

ThreadPool &global_pool() {
    static int n_workers = []() {
        const char *env = getenv("NANO_THREADS");
        if (env) {
            int n = atoi(env);
            if (n > 0) return n;
        }
        return 6; // Default to all 8 cores on Apple M3 (4 P-cores + 4 E-cores)
    }();
    static ThreadPool pool(n_workers);
    return pool;
}
