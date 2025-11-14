#pragma once
#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>

struct GlobalConfig {

    // ---------- Disk / IO 配置 ----------
    uint32_t LOGIC_BLOCK_SIZE = 0;
    uint64_t DISK_SIZE = 0;

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

    // ---------- 单例入口 ----------
    static GlobalConfig& instance() {
        static GlobalConfig inst;
        return inst;
    }

private:
    GlobalConfig() = default;
};

struct DatasetInfo {
    std::string dataset_name;
    std::string hdf5_file_name;
    std::string base_data_path;
    std::string query_data_path;
    int M;
    int ef_construction;
    int search_ef;

    // 自动生成的路径
    std::string data_path;
    std::string gt_directory;
    std::string index_path;
};

struct ParamConfig{
    std::string dev_path;
    std::string hd5f_file_root_dir;
    std::string fvecs_gt_root_dir;
    std::string hnsw_index_root_dir;

    int topk;
    int gt_batch_size;
    int gt_gt_k;
    int query_cnt;
    int io_depths;
    bool is_debug;
    int iovec_ext_number;
    bool mocktest;
    int num_threads;
    int parallel_mode;

    std::string query_or_base;

    DatasetInfo dataset;

    static ParamConfig& instance() {
        static ParamConfig cfg;
        return cfg;
    }
};