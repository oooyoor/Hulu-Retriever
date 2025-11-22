#include <spdlog/spdlog.h>
#include <vector>
#include <map>
#include <hnswlib/hnswlib.h>
#include <string>
#include <filesystem>
#include "param_config.hpp"
#include "process_json.hpp"
#include "logger_init.hpp"
#include "disk_info.hpp"
#include "parallel.hpp"
#include "data_reader.hpp"
#include "iouring_manager.hpp"
namespace fs = std::filesystem;

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
    std::vector<std::vector<int>> iter_dist_counts(2, std::vector<int>(pc.query_cnt, 0));
    auto block_res = get_block_size(pc.dev_path, pc.iovec_ext_number);
    std::vector<IOuringManager*> ioers;
    for (int i = 0; i < num_threads; i++) ioers.push_back(new IOuringManager(pc.io_depths, {pc.dev_path}, std::get<1>(block_res)));
    
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
        std::set<hnswlib::labeltype> gt_set;
        for (int i=0;i<cur_k;i++) {
            gt_set.insert(static_cast<hnswlib::labeltype>(gt_top_k_indices[i]));
        }
        int iter_count = 0;
        int dist_count = 0;
    //     CPUProfiler cpu_profiler;
        auto hnswst = std::chrono::high_resolution_clock::now();
    //     cpu_profiler.start();
        auto hnsw_result = alg_hnsw->searchKnnRecallCost((void*)(vecs[row].data()), cur_k, gt_set, iter_count, dist_count);
    //     cpu_profiler.stop();
        auto hnswed = std::chrono::high_resolution_clock::now();
        auto hnswcst = std::chrono::duration_cast<std::chrono::microseconds>(hnswed - hnswst).count();
        std::vector<off_t> offset_list;
        std::vector<ull> indexs;
        while (!hnsw_result.empty()) {
            auto tmp = hnsw_result.top().second; hnsw_result.pop();
            indexs.emplace_back(static_cast<ull>(tmp));
            offset_list.emplace_back(tmp<<cfg.OFF_BITS_LEN);
        }
        recalls[0][row] = get_recall<ull, int64_t>(indexs, gt_top_k_indices, cur_k);
        iter_dist_counts[0][row] = iter_count;
        iter_dist_counts[1][row] = dist_count;
        TimePoint io_start, io_end;
        auto iost = std::chrono::high_resolution_clock::now();
        ioers[threadId]->batch_read_offset(offset_list);
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
    std::string out_dir = "./search_difficulty_results/"+pc.raw_or_basefs+"_results/Offset_results/" + dataset_name + "/" + std::to_string(num_threads)+"_"
        +std::to_string(pc.dataset.search_ef)+"_"+std::to_string(pc.io_depths)+"/"+std::to_string(query_cnt)+"/"+std::to_string(repeat_id);
    if (!std::filesystem::exists(out_dir)) std::filesystem::create_directories(out_dir);
    generate_json_multi_T<double>(costs, {"hnsw","io","hnswio"}, query_cnt, out_dir + "/HNSWIO.json");
    // generate_json_multi_T<double>(costs, {"hnsw","io","hnswio"}, query_cnt-1, out_dir + "/HNSWIO.json");
    generate_json_multi_T<double>(recalls, {"recall"}, query_cnt, out_dir + "/HNSWIO_Recall.json");
    generate_json_multi_T<int>(iter_dist_counts, {"iter_count","dist_count"}, query_cnt, out_dir + "/HNSWIO_IterDistCount.json");

    vecs.clear();
    // for (auto ioer : ioers) delete ioer;
    delete alg_hnsw;
    return 0;
}

