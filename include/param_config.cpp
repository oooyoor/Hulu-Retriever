#include "param_config.hpp"
#include <stdexcept>

using json = nlohmann::json;
void ParamConfig::load_from_json(const json& js, const std::string& dataset_name_arg)
{
    // 根配置
    dev_path = js.value("dev_path", "");
    hd5f_file_root_dir = js.value("hd5f_file_root_dir", "");
    fvecs_gt_root_dir  = js.value("fvecs_gt_root_dir", "");
    hnsw_index_root_dir = js.value("hnsw_index_root_dir", "");
    raw_or_basefs = js.value("raw_or_basefs", "raw");
    fs_data_dir_path = js.value("fs_data_dir_path","");
    max_elements = js.value("max_elements", 1000000);
    topk = js.value("topk", 100);
    gt_batch_size = js.value("gt_batch_size", 20000);
    gt_gt_k = js.value("gt_gt_k", 100);
    query_cnt = js.value("query_cnt", 10000);
    io_depths = js.value("io_depths", 256);
    is_debug = js.value("is_debug", false);
    iovec_ext_number = js.value("iovec_ext_number", 1);
    mocktest = js.value("mocktest", true);
    num_threads = js.value("num_threads", 1);
    parallel_mode = js.value("parallel_mode", 2);
    query_or_base = js.value("query_or_base", "query");

    // 选用 dataset_name
    std::string dataset_name = dataset_name_arg;
    if (dataset_name.empty()) {
        dataset_name = js.value("dataset_name", "");
    }
    if (dataset_name.empty()) {
        throw std::runtime_error("dataset_name missing");
    }

    // 在 dataset_list 中寻找对应项
    const auto& dataset_list = js.at("dataset_list");
    auto it = std::find_if(dataset_list.begin(), dataset_list.end(),
        [&](const json& item) {
            return item.value("dataset_name", "") == dataset_name;
        });

    if (it == dataset_list.end()) {
        throw std::runtime_error("dataset_name not found in dataset_list");
    }

    const json& dcfg = *it;

    // 填 DatasetInfo
    dataset.dataset_name = dataset_name;
    dataset.hdf5_file_name = dcfg.value("hdf5_file_name", "");
    dataset.base_data_path = dcfg.value("base_data_path", "");
    dataset.query_data_path = dcfg.value("query_data_path", "");
    dataset.M = dcfg.value("M", 16);
    dataset.ef_construction = dcfg.value("ef_construction", 200);
    dataset.search_ef = dcfg.value("search_ef", 500);
    dataset.stable_hops = dcfg.value("stable_hops", 5);
    dataset.hop_diff_limit = dcfg.value("hop_diff_limit", 3);
    dataset.break_percent = dcfg.value("break_percent", 0.1);
    // 自动拼接路径
    dataset.data_path =
        (query_or_base == "query")
        ? fvecs_gt_root_dir + dataset.query_data_path
        : fvecs_gt_root_dir + dataset.base_data_path;

    dataset.gt_directory =
        fvecs_gt_root_dir + dataset_name + "_" +
        query_or_base + "_to_unique_base_gt_ip";

    dataset.index_path =
        hnsw_index_root_dir +
        "M" + std::to_string(dataset.M) +
        "_efc" + std::to_string(dataset.ef_construction) +
        "/" + dataset_name + ".bin";
}

void ParamConfig::load_from_json_global(const json& js) {
    dev_path = js.value("dev_path", "");
    iovec_ext_number = js.value("iovec_ext_number", 1);
    io_depths = js.value("io_depths", 256);
}
