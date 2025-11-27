#include <spdlog/spdlog.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <hnswlib/hnswlib.h>
#include <string>
#include <filesystem>
#include <cstddef>
#include <algorithm>
#include "param_config.hpp"
#include "process_json.hpp"
#include "logger_init.hpp"
#include "disk_info.hpp"
#include "parallel.hpp"
#include "data_reader.hpp"
#include "iouring_manager.hpp"
namespace fs = std::filesystem;

// Index to FS path mapper
class IndexToPathMapper {
private:
    std::string base_path;
    bool is_tree_structure;
    int fanout;
    size_t num_files;
    size_t total_leaves;
    size_t per_leaf;
    size_t remainder;
    std::unordered_map<size_t, std::string> path_cache; // Optional cache for performance

    // Calculate tree path from global index
    std::string index_to_tree_path(size_t index) const {
        size_t leaf_index = 0;
        size_t local_index = 0;
        
        // Calculate which leaf directory and local file index
        // First 'remainder' leaves have (per_leaf + 1) files each
        // Remaining leaves have per_leaf files each
        size_t files_in_first_leaves = remainder * (per_leaf + 1);
        
        if (index < files_in_first_leaves) {
            // Index is in one of the first 'remainder' leaves
            leaf_index = index / (per_leaf + 1);
            local_index = index % (per_leaf + 1);
        } else {
            // Index is in the remaining leaves
            size_t remaining_index = index - files_in_first_leaves;
            leaf_index = remainder + remaining_index / per_leaf;
            local_index = remaining_index % per_leaf;
        }
        
        // Convert leaf_index to i, j, k (three-level tree structure)
        size_t plane = static_cast<size_t>(fanout) * static_cast<size_t>(fanout);
        int i = static_cast<int>(leaf_index / plane);
        size_t tmp = leaf_index % plane;
        int j = static_cast<int>(tmp / fanout);
        int k = static_cast<int>(tmp % fanout);
        
        // Build path: base/tree/l{i}/l{j}/l{k}/f_{local_index}.bin
        return base_path + "/l" + std::to_string(i) + 
               "/l" + std::to_string(j) + 
               "/l" + std::to_string(k) + 
               "/f_" + std::to_string(local_index) + ".bin";
    }

    // Calculate flat path from index
    std::string index_to_flat_path(size_t index) const {
        return base_path + "/" + std::to_string(index) + ".bin";
    }

public:
    IndexToPathMapper(const std::string& fs_data_dir_path, size_t max_elements, int fanout_val = 100)
        : base_path(fs_data_dir_path), num_files(max_elements), fanout(fanout_val) {
        // Check if it's tree structure (path contains "/tree")
        // Otherwise, assume flat structure (lab1 or similar)
        is_tree_structure = (base_path.find("/tree") != std::string::npos);
        
        if (is_tree_structure) {
            total_leaves = static_cast<size_t>(fanout) * static_cast<size_t>(fanout) * static_cast<size_t>(fanout);
            per_leaf = num_files / total_leaves;
            remainder = num_files % total_leaves;
        } else {
            // Flat structure: /mnt/nvme0n1/lab1/{index}.bin
            total_leaves = 0;
            per_leaf = 0;
            remainder = 0;
        }
    }

    // Convert index to file path
    std::string get_path(size_t index) const {
        if (index >= num_files) {
            return ""; // Invalid index
        }
        
        if (is_tree_structure) {
            return index_to_tree_path(index);
        } else {
            return index_to_flat_path(index);
        }
    }

    // Build dictionary for a range of indices (optional, for batch operations)
    std::unordered_map<size_t, std::string> build_dict(size_t start = 0, size_t end = SIZE_MAX) const {
        std::unordered_map<size_t, std::string> dict;
        size_t end_idx = std::min(end, num_files);
        for (size_t i = start; i < end_idx; ++i) {
            dict[i] = get_path(i);
        }
        return dict;
    }
};

int warmup_mode() {
    auto& pc = ParamConfig::instance();
    auto& cfg = GlobalConfig::instance();

    cfg.logger->info("Warmup Mode: SSD + io_uring + DMA + buffer warmup");
    // 1) 获取 block size 以及 disk size 信息
     auto block_res = get_block_size(pc.dev_path, pc.iovec_ext_number);

    IOuringManager ioer(
        pc.io_depths,
        {pc.dev_path},
        std::get<1>(block_res)
    );

    // 固定偏移读 256 次（1MB 或 4MB 等）
    std::vector<off_t> offsets;
    for (int i = 0; i < 256; i++) {
        offsets.push_back(i * cfg.LOGIC_BLOCK_SIZE);
    }

    cfg.logger->info("Issuing warmup IO...");

    for (int i = 0; i < 32; i++) {   // 执行 32 轮
        ioer.batch_read_offset(offsets);
    }

    cfg.logger->info("Warmup completed.");
    return 0;
}

int main(int argc, char *argv[])
{
    // ============================
    // Warmup 模式
    // ============================
    if (argc == 3 && std::string(argv[1]) == "warmup") {
        std::string config_json_path = argv[2];

        // 初始化 logger
        init_logger(true);

        // 载入 JSON（但无需 dataset_name）
        auto js = read_json(config_json_path);
        ParamConfig::instance().load_from_json_global(js); 
        return warmup_mode();
    }

    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << " <config_json_path> <dataset_name> [repeat_id] [num_threads]\n";
        return 1;
    }
    std::string config_json_path = argv[1];
    std::string dataset_name = argv[2];
    int repeat_id = argc > 3 ? std::stoi(argv[3]) : 1;
    auto js = read_json(config_json_path);
    ParamConfig::instance().load_from_json(js, dataset_name);
    init_logger(ParamConfig::instance().is_debug);

    auto& cfg = GlobalConfig::instance();
    auto& pc = ParamConfig::instance();
    
    // 如果提供了线程数参数，覆盖配置中的值
    if (argc > 4) {
        pc.num_threads = std::stoi(argv[4]);
        cfg.logger->info("Overriding num_threads from command line: {}", pc.num_threads);
    }

    cfg.logger->info("Dataset loaded: {}", pc.dataset.dataset_name);
    cfg.logger->info("Data path: {}", pc.dataset.data_path);
    cfg.logger->info("GT directory: {}", pc.dataset.gt_directory);
    cfg.logger->info("Index path: {}", pc.dataset.index_path);
    
    int num_threads = pc.num_threads;
    int query_cnt = pc.query_cnt;
    int parallel_mode = pc.parallel_mode;

    auto vecs = read_fvecs<float>(pc.dataset.data_path, 1000000);
    if (vecs.empty()) { cfg.logger->error("No vectors read."); return 1; }
    normalize_rows<float>(vecs);
    cfg.logger->info(fmt::format("vecs.size(): {}", vecs.size()));
    int dim = static_cast<int>(vecs[0].size());
    size_t max_elements = vecs.size();
    cfg.logger->info(fmt::format("max_elements: {}", max_elements));
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, pc.dataset.index_path);
    alg_hnsw->setEf(pc.dataset.search_ef);

    GroundTruthReader gt_reader(pc.dataset.gt_directory, pc.gt_batch_size, max_elements, pc.gt_gt_k);

    if(max_elements < pc.query_cnt){
        cfg.logger->warn("max_elements < query_cnt, set query_cnt to max_elements: {}", max_elements);
        pc.query_cnt = max_elements;
    }
    // 更新局部变量 query_cnt，确保与调整后的 pc.query_cnt 一致
    query_cnt = pc.query_cnt;
    // std::vector<std::vector<double>> costs(3, std::vector<double>(pc.query_cnt - 1, 0));
    std::vector<std::vector<double>> costs(3, std::vector<double>(pc.query_cnt, 0));
    std::vector<std::vector<double>> recalls(1, std::vector<double>(pc.query_cnt, 0));
    std::vector<std::vector<double>> iter_dist_counts(2, std::vector<double>(pc.query_cnt, 0));
    auto block_res = get_block_size(pc.dev_path, pc.iovec_ext_number);
    std::vector<IOuringManager*> ioers;
    for (int i = 0; i < num_threads; i++) ioers.push_back(new IOuringManager(pc.io_depths, {pc.dev_path}, std::get<1>(block_res)));
    
    // Create index to path mapper
    IndexToPathMapper path_mapper(pc.fs_data_dir_path, 1000001, pc.fanout);
    cfg.logger->info("FS path mapper initialized: base_path={}, is_tree={}, fanout={}", 
                     pc.fs_data_dir_path, 
                     (pc.fs_data_dir_path.find("/tree") != std::string::npos),
                     pc.fanout);
    auto path_dict=path_mapper.build_dict(0,max_elements);
    
    // Validate that path_dict was built correctly
    size_t empty_path_count = 0;
    for(size_t i = 0; i < std::min(static_cast<size_t>(1000), max_elements); ++i) {
        if (path_dict.find(i) == path_dict.end() || path_dict[i].empty()) {
            if (empty_path_count == 0) {
                cfg.logger->error("Path dict validation failed. Index {} has empty or missing path.", i);
            }
            empty_path_count++;
        }
    }
    if (empty_path_count > 0) {
        cfg.logger->error("Path dict validation: {} out of {} sample indices have empty paths. "
                         "This indicates a problem with path mapping logic.", 
                         empty_path_count, std::min(static_cast<size_t>(1000), max_elements));
    }
    
    // Validate that some sample paths exist on filesystem
    size_t sample_size = std::min(static_cast<size_t>(100), max_elements);
    size_t missing_count = 0;
    for(size_t i = 0; i < sample_size; ++i) {
        if (path_dict.find(i) != path_dict.end() && !path_dict[i].empty()) {
            if (!fs::exists(path_dict[i])) {
                if (missing_count == 0) {
                    cfg.logger->error("Sample path validation failed. First missing file: {}", path_dict[i]);
                }
                missing_count++;
            }
        }
    }
    if (missing_count > 0) {
        cfg.logger->error("Path validation: {} out of {} sample files are missing. Please ensure files are created using prepare_files.", 
                         missing_count, sample_size);
        cfg.logger->error("Base path: {}", pc.fs_data_dir_path);
        // Don't exit, but log the issue - the actual open() will fail with better error message
    } else {
        cfg.logger->info("Path validation: All {} sample files exist", sample_size);
    }
    
    // Debug: print first few paths
    for(int i=0;i<std::min(3, static_cast<int>(max_elements));i++){
        if (path_dict.find(i) != path_dict.end()) {
            cfg.logger->debug("Path[{}] = {}", i, path_dict[i]);
        } else {
            cfg.logger->warn("Path[{}] not found in dict", i);
        }
    }
    // 根据parallel_mode选择不同的并行执行器
    auto parallel_executor = [&](auto fn) {
        switch (parallel_mode) {
            case 0:
                OptimizedParallelFor(0, query_cnt, num_threads, fn);
                break;
            case 1:
                WorkStealingParallelFor(0, query_cnt, num_threads, fn);
                break;
            case 2:
            default:
                StaticPartitionParallelFor(0, query_cnt, num_threads, fn);
                break;
        }
    };

    auto allstart = std::chrono::high_resolution_clock::now();

    

    parallel_executor([&](size_t row, size_t threadId) {
        auto gt_top_k_indices = gt_reader.getTopKResults(row);
        int cur_k = std::min(static_cast<int>(gt_top_k_indices.size()), pc.topk);
        std::set<hnswlib::labeltype> gtset;
        for(int i=0;i<cur_k;i++){
            gtset.insert(gt_top_k_indices[i]);
        }
    //     CPUProfiler cpu_profiler;
        auto hnswst = std::chrono::high_resolution_clock::now();
    //     cpu_profiler.start();
        // int iter_count=0;
        // int dist_count=0;
        auto hnsw_result = alg_hnsw->searchKnn((void*)(vecs[row].data()), cur_k);
    //     cpu_profiler.stop();
        auto hnswed = std::chrono::high_resolution_clock::now();
        auto hnswcst = std::chrono::duration_cast<std::chrono::microseconds>(hnswed - hnswst).count();
        // std::vector<std::string> offset_list;
        std::vector<std::string> path_list;  // New: store fs paths
        std::vector<ull> indexs;
        while (!hnsw_result.empty()) {
            auto tmp = hnsw_result.top().second; hnsw_result.pop();
            indexs.emplace_back(static_cast<ull>(tmp));
            // offset_list.emplace_back(tmp<<cfg.OFF_BITS_LEN);
            // Convert index to fs-path using mapper
            std::string fs_path;
            if (path_dict.find(tmp) != path_dict.end()) {
                fs_path = path_dict[tmp];
            } else {
                // Index not in dict, generate path dynamically
                fs_path = path_mapper.get_path(static_cast<size_t>(tmp));
                if (fs_path.empty()) {
                    cfg.logger->error("Invalid index {} (max_elements={}) for query {}. "
                                     "This may indicate HNSW index contains out-of-range indices.", 
                                     tmp, max_elements, row);
                    continue; // Skip this invalid index
                }
            }
            if (fs_path.empty()) {
                cfg.logger->error("Empty path for index {} in query {} (max_elements={})", 
                                 tmp, row, max_elements);
                continue; // Skip empty paths
            }
            path_list.emplace_back(fs_path);
        }
        recalls[0][row] = get_recall<ull, int64_t>(indexs, gt_top_k_indices, cur_k);
        // iter_dist_counts[0][row]=iter_count;
        // iter_dist_counts[1][row]=dist_count;
        TimePoint io_start, io_end;
        auto iost = std::chrono::high_resolution_clock::now();
        // Use path_list (fs paths) instead of offset_list
        ioers[threadId]->batch_read_file(path_list);
        auto ioed = std::chrono::high_resolution_clock::now();
        auto iocst = std::chrono::duration_cast<std::chrono::microseconds>(ioed - iost).count();

    //     // 可复现记录 HNSW + IO + CPU profiler
    //     TrackerQueryRecord record;
    //     record.queryId = row;
    //     record.threadId = threadId;
    //     record.cpuCore = sched_getcpu();
    //     record.wallTimeUs = hnswcst;
    //     record.userTimeUs = cpu_profiler.getUserTimeUs();
    //     record.systemTimeUs = cpu_profiler.getSystemTimeUs();
    //     record.cpuEfficiency = cpu_profiler.getCpuEfficiency();

        // if (row > 0) {
        // costs[0][row - 1] = hnswcst;
        // costs[1][row - 1] = iocst;
        // costs[2][row - 1] = hnswcst + iocst;
        // }
        costs[0][row] = hnswcst;
        costs[1][row] = iocst;
        costs[2][row] = hnswcst + iocst;
        // iocnts[0][row] = 1.0;     // mocktest
    });

    auto allend = std::chrono::high_resolution_clock::now();
    auto allcst = std::chrono::duration_cast<std::chrono::microseconds>(allend - allstart).count();

    // 输出 JSON / CSV
    std::string out_dir = "/home/zqf/Hulu-Retriever/ATimeResults/multi_fs_results/" + dataset_name + "/" + std::to_string(num_threads)+"_"
        +std::to_string(pc.dataset.search_ef)+"_"+std::to_string(pc.io_depths)+"/"+std::to_string(query_cnt)+"/"+std::to_string(repeat_id);
    if (!std::filesystem::exists(out_dir)) std::filesystem::create_directories(out_dir);
    generate_json_multi_T<double>(costs, {"hnsw","io","hnswio"}, query_cnt, out_dir + "/HNSW_fsIO.json");
    // generate_json_multi_T<double>(costs, {"hnsw","io","hnswio"}, query_cnt-1, out_dir + "/HNSWIO.json");
    generate_json_multi_T<double>(recalls, {"recall"}, query_cnt, out_dir + "/HNSWIO_Recall.json");
    // hulu::generate_json_multi_T<double>(iocnts, {"avg_iocnt"}, query_cnt, out_dir + "/HNSWIO_IOCnt" + mode_suffix + ".json");

    vecs.clear();
    // for (auto ioer : ioers) delete ioer;
    delete alg_hnsw;
    return 0;
}

