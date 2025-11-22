#pragma once
#include <fstream>
#include <string>
#include <json.hpp>
#include "global_config.hpp"

using json = nlohmann::json;

inline json read_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("cannot open: " + path);

    json j;
    file >> j;
    return j;
}

/**
* @brief 生成单个数据类型的 JSON 文件
* @param costs - 每个查询的代价数据
* @param name_labels - 代价名称标签
* @param searchcnt - 查询数量
* @param output_path - 输出文件路径
*/
// ---------- 生成多项指标的 JSON 文件 ----------
template <typename T>
void generate_json_multi_T(const std::vector<std::vector<T>> &costs,
                            const std::vector<std::string> &name_labels,
                            const int searchcnt, const std::string &output_path)
{
    int comp_size = costs.size();
    assert(comp_size == name_labels.size());
    std::vector<double> avgcosts(comp_size, 0);
    nlohmann::json entries = nlohmann::json::array(); // array
    for (int i = 0; i < searchcnt; i++)
    {
        nlohmann::json entry; // json item
        for (int j = 0; j < comp_size; j++)
        {
            entry[name_labels[j]] = costs[j][i];
            avgcosts[j] += costs[j][i];
        }
        entries.push_back(entry);
    }
    nlohmann::json avgcostjson;
    for (int j = 0; j < comp_size; j++)
    {
        avgcostjson["avg_" + name_labels[j]] = avgcosts[j] / searchcnt;
    }
    nlohmann::json data = {
        {"avgcost", avgcostjson},
        {"entries", entries}};
    GlobalConfig::instance().logger->info("avgcost: {}", avgcostjson.dump());
    std::ofstream fout(output_path);
    fout << data.dump(4);
    fout.close();
}


