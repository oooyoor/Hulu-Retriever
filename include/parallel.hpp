#pragma once

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

// ==============================
// 优化的并行执行器 - 减少Cache竞争和False Sharing
// ==============================
template <class Function>
inline void OptimizedParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads <= 1) {
        for (size_t id = start; id < end; id++) fn(id, 0);
        return;
    }

    // 使用chunk-based分配而不是atomic计数器，减少false sharing
    size_t total = end - start;
    size_t chunk_size = std::max<size_t>(1, total / (numThreads * 4)); // 每个chunk足够大，减少竞争
    if (chunk_size < 1) chunk_size = 1;
    
    // 使用cache-aligned的本地计数器，避免false sharing
    struct alignas(64) ThreadLocalCounter {  // 64字节对齐，避免false sharing
        size_t local_start;
        size_t local_end;
        size_t thread_id;
    };
    
    std::vector<ThreadLocalCounter> thread_data(numThreads);
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    // 预先分配chunk给每个线程，减少运行时的竞争
    size_t chunks_per_thread = (total + chunk_size - 1) / chunk_size / numThreads;
    if (chunks_per_thread < 1) chunks_per_thread = 1;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        // 计算该线程负责的chunk范围
        size_t thread_start_chunk = threadId * chunks_per_thread;
        size_t thread_start = start + thread_start_chunk * chunk_size;
        size_t thread_end = std::min(end, thread_start + chunks_per_thread * chunk_size);
        
        thread_data[threadId].local_start = thread_start;
        thread_data[threadId].local_end = thread_end;
        thread_data[threadId].thread_id = threadId;

        threads.emplace_back([&, threadId] {
            try {
                // 每个线程处理自己的chunk范围
                size_t my_start = thread_data[threadId].local_start;
                size_t my_end = thread_data[threadId].local_end;
                
                for (size_t id = my_start; id < my_end; id++) {
                    fn(id, threadId);
                }
                
                // 如果还有剩余工作，使用work-stealing但避免频繁的atomic操作
                // 剩余工作分配给空闲线程
                size_t remaining_start = start + numThreads * chunks_per_thread * chunk_size;
                if (remaining_start < end && threadId == 0) {
                    // 只有第一个线程处理剩余工作，避免竞争
                    for (size_t id = remaining_start; id < end; id++) {
                        fn(id, threadId);
                    }
                }
            } catch (...) {
                std::unique_lock<std::mutex> lock(lastExceptMutex);
                lastException = std::current_exception();
            }
        });
    }

    for (auto &t : threads) t.join();
    if (lastException) std::rethrow_exception(lastException);
}

// ==============================
// 更好的工作窃取版本 - 减少atomic访问频率
// ==============================
template <class Function>
inline void WorkStealingParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads <= 1) {
        for (size_t id = start; id < end; id++) fn(id, 0);
        return;
    }

    // 使用更大的chunk size，减少atomic操作频率
    size_t total = end - start;
    size_t chunk_size = std::max<size_t>(16, total / (numThreads * 8)); // 更大的chunk
    if (chunk_size < 1) chunk_size = 1;
    
    // 使用cache-aligned的atomic计数器
    struct alignas(64) CacheAlignedCounter {
        std::atomic<size_t> value;
    };
    CacheAlignedCounter counter;
    counter.value.store(start, std::memory_order_relaxed);
    
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        threads.emplace_back([&, threadId] {
            try {
                while (true) {
                    // 一次性获取一个chunk，减少atomic操作
                    size_t chunk_start = counter.value.fetch_add(chunk_size, std::memory_order_relaxed);
                    if (chunk_start >= end) break;
                    
                    size_t chunk_end = std::min(end, chunk_start + chunk_size);
                    for (size_t id = chunk_start; id < chunk_end; id++) {
                        fn(id, threadId);
                    }
                }
            } catch (...) {
                std::unique_lock<std::mutex> lock(lastExceptMutex);
                lastException = std::current_exception();
                counter.value.store(end, std::memory_order_relaxed); // 停止其他线程
            }
        });
    }

    for (auto &t : threads) t.join();
    if (lastException) std::rethrow_exception(lastException);
}

// ==============================
// 静态分区版本 - 完全避免atomic操作
// ==============================
template <class Function>
inline void StaticPartitionParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    if (numThreads <= 1) {
        for (size_t id = start; id < end; id++) fn(id, 0);
        return;
    }

    size_t total = end - start;
    size_t items_per_thread = total / numThreads;
    size_t remainder = total % numThreads;
    
    std::vector<std::thread> threads;
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    size_t current = start;
    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
        size_t thread_start = current;
        size_t thread_count = items_per_thread + (threadId < remainder ? 1 : 0);
        size_t thread_end = thread_start + thread_count;
        current = thread_end;

        threads.emplace_back([&, threadId, thread_start, thread_end] {
            try {
                for (size_t id = thread_start; id < thread_end; id++) {
                    fn(id, threadId);
                }
            } catch (...) {
                std::unique_lock<std::mutex> lock(lastExceptMutex);
                lastException = std::current_exception();
            }
        });
    }

    for (auto &t : threads) t.join();
    if (lastException) std::rethrow_exception(lastException);
}
