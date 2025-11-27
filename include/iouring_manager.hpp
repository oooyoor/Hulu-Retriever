#pragma once

#include <liburing.h>
#include <vector>
#include <string>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <cstring>
#include <stdexcept>
#include <system_error>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

#include "global_config.hpp"
#include "types.hpp"        // using ull = unsigned long long
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"

class IOuringManager
{
public:
    // ================================================================================================
    // 新增：用于 ANN 搜索过程中 overlap 的“纯 submit”预取接口（不等待完成）
    // ================================================================================================
    void prefetch_offsets_overlap(const std::vector<off_t>& offset_list)
    {
        if (offset_list.empty()) return;
        if (fds_.empty()) {
            throw std::runtime_error("IOuringManager: no device registered (fds_ empty).");
        }

        int fd = fds_[0];  // 这里假设 fds_[0] 是你的读 NVMe 设备 / 文件

        unsigned submitted = 0;

        for (off_t off : offset_list)
        {
            int slot_id = acquire_free_slot();
            if (slot_id < 0) {
                // 没有空闲 slot 了，先回收一部分完成事件再试一次
                reclaim_completions(32);
                slot_id = acquire_free_slot();
                if (slot_id < 0) {
                    // 实在拿不到，就提前结束这批预取
                    break;
                }
            }

            Slot& slot = slots_[slot_id];
            slot.in_use = true;
            slot.offset = off;

            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                // SQ ring 满了，先 submit 一波，再拿 sqe
                if (submitted > 0) {
                    io_uring_submit(&ring_);
                    submitted = 0;
                }
                sqe = io_uring_get_sqe(&ring_);
                if (!sqe) {
                    // 还是拿不到，只能放弃本 slot
                    slot.in_use = false;
                    break;
                }
            }

            // 使用已注册 buffer，对应 slot_id
            // 每个 iovec 对应一个注册的 buffer，索引就是 slot_id
            void* buf = iovecs_[slot_id].iov_base;
            size_t len = iovecs_[slot_id].iov_len;

            // 用 read_fixed，buffer index = slot_id
            io_uring_prep_read_fixed(
                sqe,
                fd,
                buf,
                len,
                off,
                static_cast<unsigned>(slot_id)   // buf_index
            );

            // 把 slot_id 塞到 user_data 里，完成时用来找回
            io_uring_sqe_set_data64(sqe, static_cast<std::uint64_t>(slot_id));

            ++submitted;
        }

        if (submitted > 0) {
            io_uring_submit(&ring_);
        }
    }
    // 搜索循环里偶尔调一下，非阻塞回收完成的 IO
    void poll_completions_nonblock(int max_to_reclaim = 32)
    {
        reclaim_completions(max_to_reclaim);
    }

    // 某些场景（比如 query 完成、统计）想收干净所有 IO
    void drain_all_completions()
    {
        reclaim_completions(-1);  // -1 表示不限
    }

    // ================================================================================================
    // 单次批读测试（不依赖 iovec 注册）
    // ================================================================================================
    bool batch_read_file(const std::vector<std::string>& file_paths)
    {

        int total = file_paths.size();
        if (iovecs_number_ < total) {
            throw std::runtime_error("Not enough registered buffers for batch read");
        }
        for (int i = 0; i < total; ++i) 
        {
            int fd = open(file_paths[i].c_str(),O_RDONLY | O_DIRECT);
            // int fd = open(file_paths[i].c_str(),O_RDONLY);
            if (fd < 0) {
                perror("open");
                return false;
            }

            auto sqe = io_uring_get_sqe(&ring_);
            io_uring_prep_read(sqe, fd, iovecs_[i].iov_base, iovecs_[i].iov_len, 0);
            io_uring_sqe_set_data(sqe, (void*)(intptr_t)fd);
        }

        io_uring_submit(&ring_);

        int completed = 0;

        while (completed < total)
        {
            struct io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(&ring_, &cqe);
            if (ret < 0)
                throw std::runtime_error("wait_cqe error");

            int fd = (int)(intptr_t)io_uring_cqe_get_data(cqe);
            if (cqe->res < 0)
                printf("IO error %d\n", cqe->res);

            close(fd);
            completed++;

            io_uring_cqe_seen(&ring_, cqe);
        }
        return true;
    }

    bool batch_read_offset(const std::vector<off_t>& offset_list)
    {
        int total = offset_list.size();
        if (iovecs_number_ < total) {
            throw std::runtime_error("Not enough registered buffers for batch read");
        }
        for (int i = 0; i < total; ++i) 
        {
            auto sqe = io_uring_get_sqe(&ring_);
            io_uring_prep_read(sqe, fds_[0], iovecs_[i].iov_base, iovecs_[i].iov_len, offset_list[i]);
        }

        io_uring_submit(&ring_);

        int completed = 0;

        while (completed < total)
        {
            struct io_uring_cqe* cqe;
            int ret = io_uring_wait_cqe(&ring_, &cqe);
            if (ret < 0)
                throw std::runtime_error("wait_cqe error");  
            if (cqe->res < 0)
                printf("IO error %d\n", cqe->res);

            completed++;
            io_uring_cqe_seen(&ring_, cqe);
        }
        return true;
    }

    // ================================================================================================
    // 构造函数
    // ================================================================================================
    IOuringManager(int iodepth,
                   const std::vector<std::string>& devpaths,
                   ull disk_size,
                   bool register_buffer = true,
                   int numa_node_for_iovecs = -1)
    {
        auto logger = GlobalConfig::instance().logger;

        io_depth_ = iodepth;
        disk_size_ = disk_size;
        numa_node_for_iovecs_ = numa_node_for_iovecs;

        // 初始化 io_uring
        int ret = io_uring_queue_init(io_depth_, &ring_, 0);
        if (ret != 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_queue_init failed");
        }

        // 注册缓冲区
        auto& cfg = GlobalConfig::instance();
        block_size_ = cfg.LOGIC_BLOCK_SIZE;

        if (register_buffer)
            initRegisterSizeExt(cfg.IOVEC_NUMBER,
                                cfg.LOGIC_BLOCK_SIZE,
                                cfg.IOVEC_EXTNUMBER);

        if (!devpaths.empty())
            registerDevpath(devpaths);

        // 初始化 slot 状态数组
        if (iovecs_number_ > 0) {
            slots_.resize(iovecs_number_);
        }
    }

    ~IOuringManager()
    {
        io_uring_queue_exit(&ring_);
        // 注意：iovecs_ 里的内存是我们自己 posix_memalign 的，需要手动 free
        if (iovecs_) {
            for (int i = 0; i < iovecs_number_; ++i) {
                if (iovecs_[i].iov_base) {
                    free(iovecs_[i].iov_base);
                    iovecs_[i].iov_base = nullptr;
                }
            }
        }
    }

private:
    // ================================================================================================
    // 成员变量
    // ================================================================================================
    int io_depth_ = 0;

    struct io_uring ring_{};
    struct io_uring_params params_{};

    std::vector<int> fds_;

    std::unique_ptr<iovec[]> iovecs_;
    int iovecs_number_ = 0;

    ull block_size_ = 0;
    ull iovec_len_ = 0;
    ull disk_size_ = 0;

    int numa_node_for_iovecs_ = -1;

    // 每个注册 buffer（iovec）对应一个 slot
    struct Slot {
        bool in_use = false;
        off_t offset = 0;
    };
    std::vector<Slot> slots_;

    // ================================================================================================
    // NUMA 对齐分配
    // ================================================================================================
    int allocate_aligned_on_node(void*& buf, size_t alignment, size_t size, int numa_node = -1)
    {
        if (numa_node < 0 || numa_available() < 0)
            return posix_memalign(&buf, alignment, size);

        if (posix_memalign(&buf, alignment, size) != 0)
            return -1;

        size_t page_size = getpagesize();
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;

        unsigned long nodemask = 1ULL << numa_node;
        unsigned long maxnode  = sizeof(nodemask) * 8;

        if (mbind(buf, aligned_size, MPOL_BIND, &nodemask, maxnode,
                  MPOL_MF_MOVE | MPOL_MF_STRICT) != 0)
        {
            free(buf);
            return -1;
        }

        return 0;
    }

    // ================================================================================================
    // 注册 IOVEC 缓冲区
    // ================================================================================================
    void initRegisterSizeExt(int iovecsnumber, ull block_size, ull number)
    {
        auto logger = GlobalConfig::instance().logger;

        iovecs_number_ = iovecsnumber;
        iovec_len_     = block_size * number;
        block_size_    = block_size;

        iovecs_ = std::make_unique<iovec[]>(iovecs_number_);

        for (int i = 0; i < iovecs_number_; i++)
        {
            void* buf = nullptr;
            if (allocate_aligned_on_node(buf, block_size_, iovec_len_, numa_node_for_iovecs_) != 0)
                throw std::runtime_error("Memory alignment failed");

            memset(buf, 0, iovec_len_);

            iovecs_[i].iov_base = buf;
            iovecs_[i].iov_len  = iovec_len_;
        }

        int ret = io_uring_register_buffers(&ring_, iovecs_.get(), iovecs_number_);
        if (ret != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_register_buffers failed");
    }

    // ================================================================================================
    // 注册文件路径
    // ================================================================================================
    void registerDevpath(const std::vector<std::string>& devpaths)
    {
        auto& cfg = GlobalConfig::instance();
        int n = devpaths.size();

        fds_.resize(n * 2);

        for (int i = 0; i < n; i++)
        {
            int fdr = open(devpaths[i].c_str(), O_RDONLY | O_DIRECT);
            int fdw = open(devpaths[i].c_str(), O_WRONLY | O_DIRECT);

            if (fdr < 0 || fdw < 0)
                throw std::system_error(errno, std::generic_category(),
                                        "open device failed");

            fds_[2*i] = fdr;
            fds_[2*i + 1] = fdw;
        }

        if (io_uring_register_files(&ring_, fds_.data(), fds_.size()) < 0)
            throw std::system_error(errno, std::generic_category(),
                                    "io_uring_register_files failed");
    }

    // ================================================================================================
    // slot / 完成事件管理
    // ================================================================================================
    int acquire_free_slot()
    {
        for (int i = 0; i < iovecs_number_; ++i) {
            if (!slots_[i].in_use) {
                return i;
            }
        }
        return -1;
    }

    void release_slot(int idx)
    {
        if (idx >= 0 && idx < iovecs_number_) {
            slots_[idx].in_use = false;
        }
    }

    // max_to_reclaim < 0 表示不限
    void reclaim_completions(int max_to_reclaim)
    {
        io_uring_cqe* cqe = nullptr;
        int reclaimed = 0;

        while (true) {
            int ret = io_uring_peek_cqe(&ring_, &cqe);
            if (ret < 0 || !cqe) {
                break;  // 没有更多完成事件
            }

            int slot_id = static_cast<int>(io_uring_cqe_get_data64(cqe));
            if (slot_id >= 0 && slot_id < iovecs_number_) {
                release_slot(slot_id);
            }

            if (cqe->res < 0) {
                // 这里可以用 logger 记录 IO error
                // printf("IO error %d\n", cqe->res);
            }

            io_uring_cqe_seen(&ring_, cqe);
            ++reclaimed;

            if (max_to_reclaim > 0 && reclaimed >= max_to_reclaim) {
                break;
            }
        }
    }
};