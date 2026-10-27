#include "thread_pool.h"

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
        job_fn  = fn;
        job_ctx = ctx;
        job_n   = n_out;
        pending = n_workers;
        generation++;
    }
    work_cv.notify_all();

    std::unique_lock<std::mutex> lock(mu);
    done_cv.wait(lock, [this] { return pending == 0; });
}

// What each worker thread does for its whole lifetime:
// wait for a new job -> run my chunk -> report done -> repeat.
void ThreadPool::worker_loop(int id) {
    // Start at 0, NOT at the current `generation`. If run() is called before
    // this thread gets here, generation is already 1 and we still see the job.
    uint64_t seen = 0;

    while (true) {
        std::unique_lock<std::mutex> lock(mu);
        work_cv.wait(lock, [&] { return stop || generation != seen; });
        if (stop) return;

        // Copy the job out while holding the lock, then release it so the
        // workers compute in parallel instead of one at a time.
        seen     = generation;
        Job   fn  = job_fn;
        void *ctx = job_ctx;
        int   n   = job_n;
        lock.unlock();

        // My chunk of rows; the last worker takes the remainder.
        int chunk   = n / n_workers;
        int r_start = id * chunk;
        int r_end   = (id == n_workers - 1) ? n : r_start + chunk;
        if (r_start < r_end) fn(ctx, r_start, r_end);

        // Report done. Whoever finishes last wakes run().
        lock.lock();
        if (--pending == 0) done_cv.notify_one();
    }
}

ThreadPool &global_pool() {
    static ThreadPool pool(4);
    return pool;
}
