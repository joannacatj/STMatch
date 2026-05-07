#include "tensor_io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<float> read_binary_float32(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open float32 binary: " + path);
    in.seekg(0, std::ios::end);
    auto nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(float) != 0) throw std::runtime_error("Invalid float32 binary size: " + path);
    std::vector<float> out(nbytes / sizeof(float));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nbytes));
    return out;
}

std::vector<int64_t> read_binary_int64(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open int64 binary: " + path);
    in.seekg(0, std::ios::end);
    auto nbytes = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (nbytes % sizeof(int64_t) != 0) throw std::runtime_error("Invalid int64 binary size: " + path);
    std::vector<int64_t> out(nbytes / sizeof(int64_t));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nbytes));
    return out;
}

void write_binary_float32(const std::string& path, const std::vector<float>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output binary: " + path);
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
}

std::vector<int64_t> parse_shape_csv(const std::string& csv) {
    std::vector<int64_t> shape;
    if (csv.empty()) return shape;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoll(item));
    }
    return shape;
}

size_t numel_of_shape(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

std::unordered_map<std::string, std::string> parse_config_txt(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open config.txt: " + path);
    std::unordered_map<std::string, std::string> cfg;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        cfg[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return cfg;
}

std::unordered_map<std::string, TensorInfo> parse_manifest_tsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open manifest.tsv: " + path);
    std::unordered_map<std::string, TensorInfo> m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string name, dtype, shape_csv, rel;
        if (!std::getline(ss, name, '\t')) continue;
        if (!std::getline(ss, dtype, '\t')) continue;
        if (!std::getline(ss, shape_csv, '\t')) continue;
        if (!std::getline(ss, rel, '\t')) continue;
        TensorInfo info{name, dtype, parse_shape_csv(shape_csv), rel};
        m[name] = info;
    }
    return m;
}
