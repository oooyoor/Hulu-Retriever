#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Config {
    std::string mount_point = "/mnt/nvme0n1";
    std::uint64_t num_files = 1000001;
    std::size_t file_size = 4096;
    std::size_t threads = 64;
    bool create_tree = true;
    bool skip_lab1 = false;
    int fanout = 100; // 默认三层每层 fanout
};

void log_info(const std::string &msg) {
    std::cout << "[INFO] " << msg << std::endl;
}
void log_warn(const std::string &msg) {
    std::cout << "[WARN] " << msg << std::endl;
}
void log_error(const std::string &msg) {
    std::cerr << "[ERROR] " << msg << std::endl;
}

// 解析简单命令行参数
Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for argument: " + arg);
            }
            return argv[++i];
        };
        if (arg == "--mount-point") {
            cfg.mount_point = next();
        } else if (arg == "--num-files") {
            cfg.num_files = std::stoull(next());
        } else if (arg == "--file-size") {
            cfg.file_size = std::stoull(next());
        } else if (arg == "--threads") {
            cfg.threads = std::stoull(next());
        } else if (arg == "--no-tree") {
            cfg.create_tree = false;
        } else if (arg == "--skip-lab1") {
            cfg.skip_lab1 = true;
        } else if (arg == "--fanout") {
            cfg.fanout = std::stoi(next());
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return cfg;
}

// 写一个文件：如果已存在则直接返回
void write_one_file(const std::string &path,
                    const std::vector<char> &buffer,
                    std::size_t file_size) {
    if (fs::exists(path)) {
        return;
    }

    int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        // 打不开就算了，报个错
        std::perror(("open " + path).c_str());
        return;
    }

    std::size_t remaining = file_size;
    const char *data = buffer.data();
    while (remaining > 0) {
        std::size_t chunk = remaining; // 一次写剩下的全部
        ssize_t n = ::write(fd, data, chunk);
        if (n < 0) {
            std::perror(("write " + path).c_str());
            break;
        }
        remaining -= static_cast<std::size_t>(n);
    }
    ::close(fd);
}

// 多线程创建 lab1 文件
void create_lab1_all_files(const Config &cfg) {
    std::string base = cfg.mount_point + "/lab1";
    fs::create_directories(base);

    log_info("开始创建 lab1 文件, 基础路径: " + base);
    auto start = std::chrono::steady_clock::now();

    // 共享一个全零 buffer（只读，多线程安全）
    std::vector<char> buffer(cfg.file_size, 0);

    std::atomic<std::uint64_t> index{0};
    std::atomic<std::uint64_t> done{0};

    auto worker = [&]() {
        while (true) {
            std::uint64_t i = index.fetch_add(1, std::memory_order_relaxed);
            if (i >= cfg.num_files) break;

            std::string path = base + "/" + std::to_string(i) + ".bin";
            write_one_file(path, buffer, cfg.file_size);

            std::uint64_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (d % 10000 == 0) {
                std::cout << "\r[lab1] 已创建: " << d << " / " << cfg.num_files << std::flush;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(cfg.threads);
    for (std::size_t t = 0; t < cfg.threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) th.join();

    std::cout << "\r[lab1] 已创建: " << cfg.num_files << " / " << cfg.num_files << "\n";

    auto end = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    log_info("lab1 完成, 耗时 " + std::to_string(secs) + " 秒");
}

// 多线程创建 tree 目录三层结构中的文件
void create_uniform_tree_and_distribute(const Config &cfg) {
    int FANOUT = cfg.fanout;
    std::string base = cfg.mount_point + "/tree";
    fs::create_directories(base);

    std::uint64_t total_leaves = static_cast<std::uint64_t>(FANOUT) *
                                 static_cast<std::uint64_t>(FANOUT) *
                                 static_cast<std::uint64_t>(FANOUT);
    std::uint64_t per_leaf = cfg.num_files / total_leaves;
    std::uint64_t remainder = cfg.num_files % total_leaves;

    log_info("开始构建 tree 目录, 基础路径: " + base);
    log_info("叶子目录数: " + std::to_string(total_leaves) +
             ", 基本配额: " + std::to_string(per_leaf) +
             "/目录, 余数: " + std::to_string(remainder));

    auto start = std::chrono::steady_clock::now();

    std::vector<char> buffer(cfg.file_size, 0);

    std::atomic<std::uint64_t> leaf_index{0};
    std::atomic<std::uint64_t> files_done{0};

    auto worker = [&]() {
        while (true) {
            std::uint64_t leaf = leaf_index.fetch_add(1, std::memory_order_relaxed);
            if (leaf >= total_leaves) break;

            // 计算 i,j,k
            std::uint64_t tmp = leaf;
            std::uint64_t plane = FANOUT * FANOUT;
            int i = static_cast<int>(tmp / plane);
            tmp %= plane;
            int j = static_cast<int>(tmp / FANOUT);
            int k = static_cast<int>(tmp % FANOUT);

            std::uint64_t n = per_leaf + (leaf < remainder ? 1 : 0);
            if (n == 0) continue; // 这个叶子没有文件

            std::string dir = base + "/l" + std::to_string(i) +
                              "/l" + std::to_string(j) +
                              "/l" + std::to_string(k);
            fs::create_directories(dir);

            for (std::uint64_t x = 0; x < n; ++x) {
                std::string path = dir + "/f_" + std::to_string(x) + ".bin";
                write_one_file(path, buffer, cfg.file_size);

                std::uint64_t d = files_done.fetch_add(1, std::memory_order_relaxed) + 1;
                if (d % 10000 == 0) {
                    std::cout << "\r[tree] 已创建文件数: " << d << " / " << cfg.num_files << std::flush;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(cfg.threads);
    for (std::size_t t = 0; t < cfg.threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) th.join();

    std::cout << "\r[tree] 已创建文件数: " << cfg.num_files << " / " << cfg.num_files << "\n";

    auto end = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    log_info("tree 完成, 耗时 " + std::to_string(secs) + " 秒");
}

int main(int argc, char **argv) {
    /**
        功能：
        参数：

        --mount-point 挂载点，比如 /mnt/nvme0n1

        --num-files 总文件数（对 lab1、tree 各自使用）

        --file-size 每个文件大小（字节）

        --threads 并发线程数

        --no-tree 可选：只建 lab1，不建 tree（你实验不需要 tree 时可以加这个，加快很多）

        行为：

        lab1：<mount>/lab1/0.bin ... NUM_FILES-1.bin，不存在才写。

        tree：<mount>/tree/l<i>/l<j>/l<k>/f_x.bin，3 层 100×100×100，均匀分配 NUM_FILES 个文件，多线程创建。
    */
    try {
        Config cfg = parse_args(argc, argv);

        log_info("mount_point = " + cfg.mount_point);
        log_info("num_files   = " + std::to_string(cfg.num_files));
        log_info("file_size   = " + std::to_string(cfg.file_size));
        log_info("threads     = " + std::to_string(cfg.threads));
        log_info(std::string("create_tree = ") + (cfg.create_tree ? "true" : "false"));

        if (!fs::exists(cfg.mount_point) || !fs::is_directory(cfg.mount_point)) {
            log_error("挂载点不存在或不是目录: " + cfg.mount_point);
            return 1;
        }

        if (!cfg.skip_lab1) {
            create_lab1_all_files(cfg);
        } else {
            log_info("跳过 lab1 文件生成 (--skip-lab1)");
        }
        if (cfg.create_tree) {
            log_info("开始 tree 生成 (fanout=" + std::to_string(cfg.fanout) + ")...");
            create_uniform_tree_and_distribute(cfg);
        }

        log_info("全部完成");
        return 0;
    } catch (const std::exception &ex) {
        log_error(std::string("异常: ") + ex.what());
        return 1;
    }
}
