#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

namespace STMatch {

  inline std::unordered_map<uint64_t, int>& global_label_map() {
    static std::unordered_map<uint64_t, int> label_map;
    return label_map;
  }

  inline int get_compact_label(uint64_t raw_label) {
    auto& label_map = global_label_map();
    auto it = label_map.find(raw_label);
    if (it != label_map.end()) return it->second;

    int compact_label = static_cast<int>(label_map.size());
    if (compact_label >= 32) {
      std::cerr << "Too many vertex labels: bitarray32 supports at most 32 labels\n";
      assert(false);
      exit(1);
    }
    label_map.emplace(raw_label, compact_label);
    return compact_label;
  }

}
