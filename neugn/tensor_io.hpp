#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::string relative_path;
};

std::vector<float> read_binary_float32(const std::string& path);
std::vector<int64_t> read_binary_int64(const std::string& path);
void write_binary_float32(const std::string& path, const std::vector<float>& data);

std::unordered_map<std::string, std::string> parse_config_txt(const std::string& path);
std::unordered_map<std::string, TensorInfo> parse_manifest_tsv(const std::string& path);

std::vector<int64_t> parse_shape_csv(const std::string& csv);
size_t numel_of_shape(const std::vector<int64_t>& shape);
