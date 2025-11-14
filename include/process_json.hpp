#pragma once
#include <fstream>
#include <string>
#include <stdexcept>
#include <json.hpp>
#include "global_config.hpp"
using json = nlohmann::json;
// ---------------------------
// 从文件读取 JSON
// ---------------------------
json read_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + path);
    }
    json j;
    file >> j;
    return j;
}
inline void load_config(const std::string& json_path)
{
    auto& cfg = ParamConfig::instance();
    json js = read_json(json_path);

    // 基础配置
    cfg.dev_path = js.value("dev_path", "");
    cfg.hd5f_file_root_dir = js.value("hd5f_file_root_dir", "");
    cfg.fvecs_gt_root_dir  = js.value("fvecs_gt_root_dir", "");
    cfg.hnsw_index_root_dir = js.value("hnsw_index_root_dir", "");

    cfg.topk = js.value("topk", 100);
    cfg.gt_batch_size = js.value("gt_batch_size", 20000);
    cfg.gt_gt_k = js.value("gt_gt_k", 100);
    cfg.query_cnt = js.value("query_cnt", 10000);
    cfg.io_depths = js.value("io_depths", 256);
    cfg.is_debug = js.value("is_debug", false);
    cfg.iovec_ext_number = js.value("iovec_ext_number", 1);
    cfg.mocktest = js.value("mocktest", true);
    cfg.num_threads = js.value("num_threads", 1);
    cfg.parallel_mode = js.value("parallel_mode", 2);
    cfg.query_or_base = js.value("query_or_base", "query");

    // 解析 dataset_list
    std::string dataset_name;
    if (js.contains("dataset_name") && !js["dataset_name"].is_null()) {
        dataset_name = js.value("dataset_name", "");
    } else {
        // 如果 JSON 中没有 dataset_name，则回退到已经设置在 ParamConfig 中的值
        dataset_name = cfg.dataset.dataset_name;
    }

    if (dataset_name.empty()) {
        throw std::runtime_error("dataset_name not provided in JSON or as program argument");
    }

    if (!js.contains("dataset_list") || js["dataset_list"].is_null()) {
        throw std::runtime_error("dataset_list missing in configuration JSON");
    }
    auto dataset_list = js["dataset_list"];

    auto it = std::find_if(dataset_list.begin(), dataset_list.end(),
        [&](const json& item) {
            return item.value("dataset_name", "") == dataset_name;
        });

    if (it == dataset_list.end()) {
        throw std::runtime_error("dataset_name not found in dataset_list");
    }

    const json& dcfg = *it;

    // 填充 DatasetInfo
    cfg.dataset.dataset_name = dataset_name;
    cfg.dataset.hdf5_file_name = dcfg.value("hdf5_file_name", "");
    cfg.dataset.base_data_path = dcfg.value("base_data_path", "");
    cfg.dataset.query_data_path = dcfg.value("query_data_path", "");
    cfg.dataset.M = dcfg.value("M", 16);
    cfg.dataset.ef_construction = dcfg.value("ef_construction", 200);
    cfg.dataset.search_ef = dcfg.value("search_ef", 500);

    // 自动拼接路径
    cfg.dataset.data_path =
        (cfg.query_or_base == "query")
        ? cfg.fvecs_gt_root_dir + cfg.dataset.query_data_path
        : cfg.fvecs_gt_root_dir + cfg.dataset.base_data_path;

    cfg.dataset.gt_directory =
        cfg.fvecs_gt_root_dir + dataset_name + "_" +
        cfg.query_or_base + "_to_unique_base_gt_ip";

    cfg.dataset.index_path =
        cfg.hnsw_index_root_dir +
        "M" + std::to_string(cfg.dataset.M) +
        "_efc" + std::to_string(cfg.dataset.ef_construction) +
        "/" + dataset_name + ".bin";
}
