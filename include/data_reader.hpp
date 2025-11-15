#pragma once
#include <set>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>    // std::runtime_error
#include <cstring>      // std::strncmp
#include <cctype>       // std::isdigit
#include <iostream>     // std::cerr
#include <stdint.h>     // uint8_t, uint16_t
#include <iomanip>      // std::setw, std::setfill


template <typename T, typename F>
float get_recall(std::vector<T> r1, std::vector<F> r2, int K)
{
    std::set<T> a(r1.begin(), r1.begin() + K);
    std::set<T> b(r2.begin(), r2.begin() + K);
    std::set<T> result;
    set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));
    return (float)result.size() / a.size();
}

/**
 * @brief 读取 fvecs 文件
 * @param path - 文件路径
 * @param max_vectors - 最大读取向量数量，0 表示读取所有
 */
template <typename T>
std::vector<std::vector<T>> read_fvecs(const std::string& path, size_t max_vectors=0){
    std::vector<std::vector<T>> vecs;
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return vecs;
    }

    size_t i = 0;
    while (true) {
        int dim = 0;
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!fin.good()) break;
        if (dim <= 0) {
            std::cerr << "Invalid dim read: " << dim << " at index " << i << std::endl;
            break;
        }
        std::vector<float> arr(dim);
        fin.read(reinterpret_cast<char*>(arr.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!fin.good()) break;

        vecs.push_back(std::move(arr));
        ++i;
        if (max_vectors > 0 && vecs.size() >= max_vectors) break;
    }

    fin.close();
    return vecs;
}

/**
 * @brief 按行归一化
 * @param X - 待归一化的二维向量
 * @param eps - 防止除零的小常数
 */
template <typename T>
void normalize_rows(std::vector<std::vector<T>>& X, float eps=1e-10f){
    for (auto& row : X) {
        double norm2 = 0.0;
        for (float v : row) norm2 += double(v) * double(v);
        float norm = static_cast<float>(std::sqrt(norm2));
        if (norm < eps) norm = eps;
        for (float& v : row) v /= norm;
    }
}

// ---------- 2. NPY 文件读取器 ----------
class NPYReader {
    private:
        struct NPYHeader {
            std::string dtype;
            std::vector<size_t> shape;
            bool fortran_order;
        };

        // 解析 NPY 头部
        NPYHeader parseHeader(std::ifstream& file) {
            NPYHeader header;
            
            // 读取魔数
            char magic[6];
            file.read(magic, 6);
            if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
                throw std::runtime_error("Invalid NPY file format");
            }
            
            // 读取版本
            uint8_t major_version, minor_version;
            file.read(reinterpret_cast<char*>(&major_version), 1);
            file.read(reinterpret_cast<char*>(&minor_version), 1);
            
            // 读取头部长度
            uint16_t header_len;
            file.read(reinterpret_cast<char*>(&header_len), 2);
            
            // 读取头部字典
            std::string header_str(header_len, '\0');
            file.read(&header_str[0], header_len);
            
            // 解析 dtype
            size_t descr_pos = header_str.find("'descr':");
            if (descr_pos != std::string::npos) {
                size_t start = header_str.find("'", descr_pos + 8) + 1;
                size_t end = header_str.find("'", start);
                header.dtype = header_str.substr(start, end - start);
            }
            
            // 解析 shape
            size_t shape_pos = header_str.find("'shape':");
        if (shape_pos != std::string::npos) {
            size_t start = header_str.find("(", shape_pos) + 1;
            size_t end = header_str.find(")", start);
            std::string shape_str = header_str.substr(start, end - start);
            
            std::stringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // 移除前后的空白字符
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                
                // 检查是否为空或只包含非数字字符
                if (!token.empty() && token.find_first_of("0123456789") != std::string::npos) {
                    try {
                        // 只保留数字字符
                        std::string num_only;
                        for (char c : token) {
                            if (std::isdigit(c)) {
                                num_only += c;
                            }
                        }
                        if (!num_only.empty()) {
                            header.shape.push_back(std::stoull(num_only));
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to parse shape token '" << token << "': " << e.what() << std::endl;
                        // 继续处理其他tokens，不抛出异常
                    }
                }
            }
        }
        
        header.fortran_order = header_str.find("'fortran_order': True") != std::string::npos;
        return header;
    }

public:
    // 读取 int32 类型的 npy 文件
    std::vector<std::vector<int32_t>> readInt32Array(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<i4" && header.dtype != ">i4" && header.dtype != "i4") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<int32_t>> result(rows, std::vector<int32_t>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(int32_t));
        }
        
        return result;
    }

    // 读取 int64 类型的 npy 文件
    std::vector<std::vector<int64_t>> readInt64Array(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<i8" && header.dtype != ">i8" && header.dtype != "i8") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<int64_t>> result(rows, std::vector<int64_t>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(int64_t));
        }
        
        return result;
    }
    
    // 读取 float32 类型的 npy 文件
    std::vector<std::vector<float>> readFloat32Array(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        NPYHeader header = parseHeader(file);
        
        if (header.dtype != "<f4" && header.dtype != ">f4" && header.dtype != "f4") {
            throw std::runtime_error("Unsupported dtype: " + header.dtype);
        }
        
        if (header.shape.size() != 2) {
            throw std::runtime_error("Expected 2D array");
        }
        
        size_t rows = header.shape[0];
        size_t cols = header.shape[1];
        
        std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
        
        for (size_t i = 0; i < rows; ++i) {
            file.read(reinterpret_cast<char*>(result[i].data()), cols * sizeof(float));
        }
        
        return result;
    }
};

    
// ---------- 3. Ground Truth 读取器 ----------
class GroundTruthReader {
private:
    std::string gt_dir;
    size_t batch_size;
    size_t total_queries;
    size_t k;
    
public:
    GroundTruthReader(const std::string& directory, size_t batch_sz, size_t total_q, size_t topk) 
        : gt_dir(directory), batch_size(batch_sz), total_queries(total_q), k(topk) {}
    
    // 获取指定 query id 的 top-k 结果 ID
    std::vector<int64_t> getTopKResults(size_t query_id) {
        if (query_id >= total_queries) {
            throw std::runtime_error("Query ID out of range");
        }
        
        size_t batch_idx = query_id / batch_size;
        size_t local_idx = query_id % batch_size;
        
        std::ostringstream oss;
        oss << gt_dir << "/indices_" << std::setfill('0') << std::setw(4) << batch_idx << ".npy";
        std::string filename = oss.str();
        
        try {
            NPYReader reader;
            auto indices = reader.readInt64Array(filename);
            
            if (local_idx >= indices.size()) {
                throw std::runtime_error("Local index out of range in batch file");
            }
            
            return indices[local_idx];
        } catch (const std::exception& e) {
            std::cerr << "Error reading " << filename << ": " << e.what() << std::endl;
            return std::vector<int64_t>();
        }
    }
    
    // 获取指定 query id 的距离
    std::vector<float> getTopKDistances(size_t query_id) {
        if (query_id >= total_queries) {
            throw std::runtime_error("Query ID out of range");
        }
        
        size_t batch_idx = query_id / batch_size;
        size_t local_idx = query_id % batch_size;
        
        std::ostringstream oss;
        oss << gt_dir << "/distances_" << std::setfill('0') << std::setw(4) << batch_idx << ".npy";
        std::string filename = oss.str();
        
        try {
            NPYReader reader;
            auto distances = reader.readFloat32Array(filename);
            
            if (local_idx >= distances.size()) {
                throw std::runtime_error("Local index out of range in batch file");
            }
            
            return distances[local_idx];
        } catch (const std::exception& e) {
            std::cerr << "Error reading " << filename << ": " << e.what() << std::endl;
            return std::vector<float>();
        }
    }
};

    