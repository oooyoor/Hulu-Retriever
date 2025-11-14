#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <cmath>
#include <system_error>
#include <stdexcept>
#include <tuple>
#include <string>
#include <spdlog/spdlog.h>

#include "global_config.hpp"

std::tuple<uint32_t,uint64_t,uint64_t>
get_block_size(const std::string &device, int iovec_ext_number = 1)
{
    auto& cfg = GlobalConfig::instance();

    int fd = open(device.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::system_error(errno, std::generic_category(), "open device failed");

    uint32_t block_size = 0;
    if (ioctl(fd, BLKBSZGET, &block_size) == -1) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "ioctl BLKBSZGET failed");
    }

    cfg.LOGIC_BLOCK_SIZE = block_size;
    cfg.IOVEC_EXTNUMBER = iovec_ext_number;
    cfg.IOVEC_LEN = block_size * iovec_ext_number;
    cfg.IOVEC_OFF_MASK = cfg.IOVEC_LEN - 1;
    cfg.IOVEC_ID_MASK = ~cfg.IOVEC_OFF_MASK;
    cfg.IOVEC_OFF_BITS_LEN = static_cast<int>(log2(cfg.IOVEC_LEN));

    cfg.BLOCK_OFF_MASK = block_size - 1;
    cfg.BLOCK_ID_MASK = ~cfg.BLOCK_OFF_MASK;
    cfg.OFF_BITS_LEN = static_cast<int>(log2(block_size));

    uint64_t disk_size = 0;
    if (ioctl(fd, BLKGETSIZE64, &disk_size) == -1) {
        close(fd);
        throw std::system_error(errno, std::generic_category(), "ioctl BLKGETSIZE64 failed");
    }
    close(fd);

    disk_size &= cfg.BLOCK_ID_MASK;
    cfg.DISK_SIZE = disk_size;

    if (cfg.logger)
        cfg.logger->debug("block={}, disk={}", block_size, disk_size);

    return {block_size, disk_size, disk_size >> cfg.OFF_BITS_LEN};
}
