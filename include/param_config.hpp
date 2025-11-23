#pragma once
#include <string>
#include <limits>
#include <json.hpp>

struct DatasetInfo {
    std::string dataset_name;
    std::string hdf5_file_name;
    std::string base_data_path;
    std::string query_data_path;
    int M;
    int ef_construction;
    int search_ef;
    int hop_diff_limit = std::numeric_limits<int>::max();
    int stable_hops = std::numeric_limits<int>::max();
    float break_percent = 0.0f;

    // 状态机参数（可选，如果未配置则使用默认值）
    float easy_stable_hops_multiplier = 0.5f;
    float easy_break_percent_multiplier = 1.5f;
    float hard_stable_hops_multiplier = 3.0f;
    float hard_break_percent_multiplier = 0.5f;
    int easy_to_hard_threshold = 3;
    int recent_hops_window = 10;
    int large_jump_threshold = 2;

    // 自动生成的路径
    std::string data_path;
    std::string gt_directory;
    std::string index_path;
};

struct ParamConfig{
public:
    std::string dev_path;
    std::string hd5f_file_root_dir;
    std::string fvecs_gt_root_dir;
    std::string hnsw_index_root_dir;
    std::string raw_or_basefs;
    std::string fs_data_dir_path;
    int max_elements;
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
public:
    static ParamConfig& instance() {
        static ParamConfig inst;
        return inst;
    }

    ParamConfig(const ParamConfig&) = delete;
    ParamConfig& operator=(const ParamConfig&) = delete;
    void load_from_json_global(const nlohmann::json& js);
    void load_from_json(const nlohmann::json& js, const std::string& dataset_name_arg);

private:
    ParamConfig() = default;
};