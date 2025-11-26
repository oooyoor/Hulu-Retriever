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
    bool base_read(const std::vector<off_t>& offset_list)
    {
        int total = offset_list.size();
        if (iovecs_number_ < total) {
            throw std::runtime_error("Not enough registered buffers for batch read");
        }
        for (int i = 0; i < total; ++i) 
        {
            auto sqe = io_uring_get_sqe(&ring_);
            // io_uring_prep_read(sqe, fds_[0], iovecs_[i].iov_base, iovecs_[i].iov_len, offset_list[i]);
            io_uring_prep_readv(sqe, fds_[0], &iovecs_[iovec_id_], 1, offset_list[i]<<12);
            iovec_id_ = (iovec_id_ + 1) % iovecs_number_;
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
    }

    ~IOuringManager()
    {
        io_uring_queue_exit(&ring_);
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

    int iovec_id_ = 0;

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
};
