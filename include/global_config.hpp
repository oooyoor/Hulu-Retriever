#pragma once
#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>

struct GlobalConfig {

    // ---------- Disk / IO 配置 ----------
    uint32_t LOGIC_BLOCK_SIZE = 0;
    uint64_t DISK_SIZE = 0;
    int IOVEC_NUMBER= 1024;
    int IOVEC_EXTNUMBER = 1;
    uint64_t IOVEC_LEN = 0;
    uint64_t IOVEC_OFF_MASK = 0;
    uint64_t IOVEC_ID_MASK = 0;
    int IOVEC_OFF_BITS_LEN = 0;

    uint64_t BLOCK_OFF_MASK = 0;
    uint64_t BLOCK_ID_MASK = 0;
    int OFF_BITS_LEN = 0;

    // ---------- 全局 Logger ----------
    std::shared_ptr<spdlog::logger> logger;

public:
    static GlobalConfig& instance() {
        static GlobalConfig inst;
        return inst;
    }

    GlobalConfig(const GlobalConfig&) = delete;
    GlobalConfig& operator=(const GlobalConfig&) = delete;

private:
    GlobalConfig() = default;
};
