#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include "param_config.hpp"
#include "iouring_manager.hpp"
#include "logger_init.hpp"
#include "process_json.hpp"
#include "disk_info.hpp"
void print_usage() {
    std::cerr << "Usage: ./warmup --config <path_to_config_json>\n";
}

int warmup_mode(const std::string& config_json_path) {
    auto& pc = ParamConfig::instance();
    auto& cfg = GlobalConfig::instance();

    cfg.logger->info("Warmup Mode: starting...");

    // Step 1: 载入 config 文件
    auto js = read_json(config_json_path);
    ParamConfig::instance().load_from_json_global(js);  // Only load global parameters like dev_path

    // Step 2: 获取设备的 block size 和 disk size
    auto block_res = get_block_size(pc.dev_path, pc.iovec_ext_number);

    // Step 3: 初始化 io_uring
    IOuringManager ioer(
        pc.io_depths,
        {pc.dev_path},
        std::get<1>(block_res)
    );

    // Step 4: 构造固定偏移地址序列进行读操作
    std::vector<off_t> offsets;
    int warm_blocks = 256;  // Example: warmup 1MB (256 * 4KB blocks)
    for (int i = 0; i < warm_blocks; i++) {
        offsets.push_back(i * cfg.LOGIC_BLOCK_SIZE);
    }

    // Step 5: 执行 warmup（256次 offset 读）
    for (int round = 0; round < 32; round++) {
        ioer.batch_read_offset(offsets); // 调用批量读操作
    }

    cfg.logger->info("Warmup finished.");
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 3 || std::string(argv[1]) != "--config") {
        print_usage();
        return 1;
    }

    std::string config_json_path = argv[2];

    // 初始化日志
    init_logger(true);

    return warmup_mode(config_json_path);
}
