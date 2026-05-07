
#pragma once

#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <cassert>
#include <unordered_map>
#include "config.h"
#include "label_map.h"


namespace STMatch {

  typedef struct {

    pattern_node_t nnodes = 0;
    int rowptr[PAT_SIZE];
    int degree[PAT_SIZE];
    bitarray32 slot_labels[MAX_SLOT_NUM];
    bitarray32 partial[MAX_SLOT_NUM];
    set_op_t set_ops[MAX_SLOT_NUM];
  } Pattern;


  struct PatternPreprocessor {

    Pattern pat;

    int PatternMultiplicity;
    int adj_matrix_[PAT_SIZE][PAT_SIZE];
    int vertex_order_[PAT_SIZE];
    int order_map_[PAT_SIZE];
    std::vector<std::vector<int>> L_adj_matrix_;
    std::vector<std::vector<int>> board;

    bitarray32 slot_labels[PAT_SIZE][PAT_SIZE];
    bitarray32 partial[PAT_SIZE][PAT_SIZE];
    set_op_t set_ops[PAT_SIZE][PAT_SIZE];

    std::vector<int> vertex_labels;

    int length[PAT_SIZE];
    int edge[PAT_SIZE][PAT_SIZE];

    PatternPreprocessor(std::string filename) {
      readfile(filename);
      get_matching_order();
      get_partial_order();
      get_set_ops();
      propagate_partial_order();
      get_labels();
      convert2oned();

      //std::cout << "Pattern read complete. Pattern size: " << (int)pat.nnodes << std::endl;
    }

    Pattern* to_gpu() {
      Pattern* patcopy;
      cudaMalloc(&patcopy, sizeof(Pattern));
      cudaMemcpy(patcopy, &pat, sizeof(Pattern), cudaMemcpyHostToDevice);
      return patcopy;
    }

    void readfile(std::string& filename) {
      std::ifstream fin(filename);
      if (!fin.good()) {
        std::cerr << "Failed to open pattern file: " << filename << "\n";
        exit(1);
      }

      std::vector<std::string> lines;
      std::string line;
      while (std::getline(fin, line)) {
        size_t first_char = line.find_first_not_of(" \t\r\n");
        if (first_char == std::string::npos) continue;
        if (line[first_char] == '#') continue;
        lines.push_back(line.substr(first_char));
      }
      if (lines.empty()) {
        std::cerr << "Pattern file is empty: " << filename << "\n";
        exit(1);
      }

      memset(adj_matrix_, 0, sizeof(adj_matrix_));
      vertex_labels.clear();
      std::unordered_map<uint64_t, int> id_map;
      std::vector<int> input_degrees;
      bool new_format = false;
      uint64_t expected_vertices = 0, expected_edges = 0;
      size_t start_line = 0;

      std::istringstream first(lines[0]);
      char first_tag;
      first >> first_tag;
      if (first_tag == 't') {
        if (!(first >> expected_vertices >> expected_edges)) {
          std::cerr << "New pattern format must start with: t N M\n";
          exit(1);
        }
        if (expected_vertices > PAT_SIZE) {
          std::cerr << "Pattern has too many vertices for PAT_SIZE: " << expected_vertices << "\n";
          exit(1);
        }
        new_format = true;
        pat.nnodes = static_cast<pattern_node_t>(expected_vertices);
        input_degrees.assign(expected_vertices, -1);
        start_line = 1;
      }
      else if (first_tag != 'v') {
        std::cerr << "Pattern file must start with a vertex line or t N M\n";
        exit(1);
      }

      uint64_t vertex_count = 0, edge_count = 0;
      for (size_t i = start_line; i < lines.size(); i++) {
        std::istringstream sin(lines[i]);
        char tag;
        sin >> tag;
        if (tag == 'v') {
          uint64_t external_id, raw_label, degree = 0;
          if (new_format) {
            if (!(sin >> external_id >> raw_label >> degree)) {
              std::cerr << "Invalid new-format pattern vertex line: " << lines[i] << "\n";
              exit(1);
            }
            if (vertex_count >= expected_vertices) {
              std::cerr << "More pattern vertices than declared\n";
              exit(1);
            }
          }
          else if (!(sin >> external_id >> raw_label)) {
            std::cerr << "Invalid old-format pattern vertex line: " << lines[i] << "\n";
            exit(1);
          }

          int internal_id = static_cast<int>(vertex_count++);
          if (internal_id >= static_cast<int>(PAT_SIZE)) {
            std::cerr << "Pattern has too many vertices for PAT_SIZE\n";
            exit(1);
          }
          if (!id_map.emplace(external_id, internal_id).second) {
            std::cerr << "Duplicate pattern vertex id: " << external_id << "\n";
            exit(1);
          }
          vertex_labels.push_back(LABELED ? get_compact_label(raw_label) : 1);
          if (new_format) input_degrees[internal_id] = static_cast<int>(degree);
        }
        else if (tag == 'e') {
          uint64_t external_u, external_v;
          if (!(sin >> external_u >> external_v)) {
            std::cerr << "Invalid pattern edge line: " << lines[i] << "\n";
            exit(1);
          }
          auto it_u = id_map.find(external_u);
          auto it_v = id_map.find(external_v);
          if (it_u == id_map.end() || it_v == id_map.end()) {
            std::cerr << "Pattern edge references an unknown vertex: " << lines[i] << "\n";
            exit(1);
          }
          adj_matrix_[it_u->second][it_v->second] = 1;
          adj_matrix_[it_v->second][it_u->second] = 1;
          edge_count++;
        }
        else {
          std::cerr << "Unknown line type in pattern file: " << lines[i] << "\n";
          exit(1);
        }
      }

      if (new_format) {
        if (vertex_count != expected_vertices || edge_count != expected_edges) {
          std::cerr << "Pattern count mismatch: declared (" << expected_vertices << ", " << expected_edges
                    << ") but read (" << vertex_count << ", " << edge_count << ")\n";
          exit(1);
        }
      }
      else {
        pat.nnodes = static_cast<pattern_node_t>(vertex_count);
      }
      assert(vertex_count <= PAT_SIZE);

      if (new_format) {
        for (int i = 0; i < pat.nnodes; i++) {
          int actual_degree = 0;
          for (int j = 0; j < pat.nnodes; j++) actual_degree += (adj_matrix_[i][j] > 0);
          if (input_degrees[i] != actual_degree) {
            std::cerr << "Pattern degree mismatch for internal vertex " << i << ": declared "
                      << input_degrees[i] << ", actual " << actual_degree << "\n";
            exit(1);
          }
        }
      }
    }


    // input from dryadic is alreay reordered 
    void get_matching_order() {
      /* int root = 0;
      int max_degree = 0;
      for (int i = 0; i < pat.nnodes; i++) {
        int d = 0;
        for (int j = 0; j < pat.nnodes; j++) {
          if (adj_matrix_[i][j] > 0) d++;
        }
        if (d > max_degree) {
          root = i;
          max_degree = d;
        }
      }

      std::vector<int> q;
      q.push_back(root);
      int i = 0;
      std::vector<int> visited(pat.nnodes, 0);
      while (!q.empty()) {
        int a = q.back();
        q.pop_back();
        if (!visited[a]) {
          vertex_order_[i++] = a;
        }
        visited[a] = 1;
        for (int b = 0; b < pat.nnodes; b++) {
          if (adj_matrix_[a][b] > 0 && !visited[b])
            q.push_back(b);
        }
      }

      for (int i = 0; i < pat.nnodes; i++)
        order_map_[vertex_order_[i]] = i;*/


      

      for (int i = 0; i < pat.nnodes; i++) {
        vertex_order_[i] = i;
        order_map_[vertex_order_[i]] = i;
      }

      for (int i = 0; i < pat.nnodes; i++) {
        int d = 0;
        for (int j = 0; j < pat.nnodes; j++) {
          if (adj_matrix_[i][j] > 0) d++;
        }
        pat.degree[order_map_[i]] = d;
      }
    }

    void _permutation(
      std::vector<std::vector<int>>& all,
      std::vector<int>& a, int l, int r) {
      // Base case
      if (l == r)
        all.push_back(a);
      else {
        // Permutations made
        for (int i = l; i <= r; i++) {
          // Swapping done
          std::swap(a[l], a[i]);
          // Recursion called
          _permutation(all, a, l + 1, r);
          // backtrack
          std::swap(a[l], a[i]);
        }
      }
    }

    void get_set_ops() {

      board.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      board[0][0] = 1;

      for (int i = 1; i < pat.nnodes - 1; i++) {
        int ops = 0;
        for (int j = 0; j <= i; j++) {
          if (adj_matrix_[vertex_order_[i + 1]][vertex_order_[j]]) ops |= (1 << (i - j));
        }
        board[i][0] = ops;
      }

      memset(length, 0, sizeof(length));
      for (int i = 0; i < pat.nnodes; i++) length[i] = 1;

      memset(set_ops, 0, sizeof(set_ops));
      for (int j = 0; j < pat.nnodes - 1; j++) {
        for (int i = pat.nnodes - 2 - j; i >= 0; i--) {
          // 0 means empty slot in board
          if (board[i][j] == 0) continue;

          int op1 = board[i][j] & 1;
          int op2 = (board[i][j] >> 1);

          if (op2 > 0) {
            bool exist = false;
            // k starts from 1 to make sure candidate sets are not used for computing slots 
            //int startk = ((!LABELED && partial[i - 1][0] == 0) ? 0 : 1);
            int startk = 1;
            for (int k = startk; k < length[i - 1]; k++) {
              if (op2 == board[i - 1][k]) {
                exist = true;
                set_ops[i][j] += k;
                set_ops[i][j] += (op1 << 5);
                break;
              }
            }
            if (!exist) {
              set_ops[i][j] += length[i - 1];
              set_ops[i][j] += (op1 << 5);
              board[i - 1][length[i - 1]++] = op2;
            }
          }
          else {
            set_ops[i][j] |= 0x10;
          }
        }
      }
      // mark the end of slot
      for (int i = 0; i < pat.nnodes - 1; i++) {
        set_ops[i][length[i]] |= 0x80;
      }
    }

    void get_partial_order() {

      std::vector<int> p1;
      for (int i = 0; i < pat.nnodes; i++) {
        p1.push_back(i);
      }
      std::vector<std::vector<int>> permute, valid_permute;
      _permutation(permute, p1, 0, pat.nnodes - 1);

      for (auto& pp : permute) {
        std::vector<std::set<int>> adj_tmp(pat.nnodes);
        for (int i = 0; i < pat.nnodes; i++) {
          std::set<int> tp;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 0) continue;
            tp.insert(pp[j]);
          }
          adj_tmp[pp[i]] = tp;
        }
        bool valid = true;
        for (int i = 0; i < pat.nnodes; i++) {
          bool equal = true;
          int c = 0;
          for (int j = 0; j < pat.nnodes; j++) {
            if (adj_matrix_[i][j] == 1) {
              c++;
              if (adj_tmp[i].find(j) == adj_tmp[i].end()) equal = false;
            }
          }
          if (!equal || c != adj_tmp[i].size()) {
            valid = false;
            break;
          }
        }
        if (valid)
          valid_permute.push_back(pp);
      }

      PatternMultiplicity = valid_permute.size();

      L_adj_matrix_.resize(pat.nnodes, std::vector<int>(pat.nnodes, 0));
      std::set<std::pair<int, int>> L;
      for (int i = 0; i < pat.nnodes; i++) {
        int v = vertex_order_[i];
        std::vector<std::vector<int>> stabilized_aut;
        for (auto& x : valid_permute) {
          if (x[v] == v) {
            stabilized_aut.push_back(x);
          }
          else {
            L_adj_matrix_[order_map_[v]][order_map_[x[v]]] = 1;
          }
        }
        valid_permute = stabilized_aut;
      }

      memset(partial, 0, sizeof(partial));
      for (int level = 1; level < pat.nnodes; level++) {
        for (int j = level - 1; j >= 0; j--) {
          if (L_adj_matrix_[j][level] == 1) {
            partial[level - 1][0] |= (1 << j);
          }
        }
      }
    }

    int bitidx(bitarray32 a) {
      for (int i = 0; i < 32; i++) {
        if (a & (1 << i)) return i;
      }
      return -1;
    }

    void propagate_partial_order() {
      // propagate partial order of candiate sets to all slots
      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {
          int m = 0;
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            if (set_ops[i + 1][k] & 0x20) {
              // if the slot depends on the current slot and the operation is intersection
              if ((set_ops[i + 1][k] & 0xF) == j) {
                if (partial[i + 1][k] != 0) {
                  // we add the upper bound of that slot to the current slot
                  // the upper bound has to be vertex above level i 
                  m |= (partial[i + 1][k] & (((1 << (i + 1)) - 1)));
                }
                else {
                  m = 0;
                  break;
                }
              }
            }
            else {
              m = 0;
              break;
            }
          }
          partial[i][j] = m;
        }
      }
    }

    void get_labels() {

      memset(slot_labels, 0, sizeof(slot_labels));

      for (int i = 0; i < pat.nnodes; i++) {
        slot_labels[i][0] = (1 << vertex_labels[i + 1]);
      }

      for (int i = pat.nnodes - 3; i >= 0; i--) {
        for (int j = 1; j < length[i]; j++) {

          bitarray32 m = 0;
          //if(j==0) m = pat.partial[i][j];
          // for all slots in the next level, 
          for (int k = 0; k < length[i + 1]; k++) {
            // if the slot depends on the current slot and the operation is intersection
            if ((set_ops[i + 1][k] & 0xF) == j) {
              // we add the upper bound of that slot to the current slot
              // the upper bound has to be vertex above level i 
              m |= slot_labels[i + 1][k];
            }
          }
          slot_labels[i][j] = m;
        }
      }
    }

    void convert2oned() {

      int onedidx[PAT_SIZE][PAT_SIZE];
      memset(onedidx, 0, sizeof(onedidx));

      int count = 1;
      pat.rowptr[0] = 0;
      pat.rowptr[1] = 1;
      // this is used for filtering the edges in job queue
      pat.partial[0] = partial[0][0];
      for (int i = 1; i < pat.nnodes - 1; i++) {
        for (int j = 0; j < PAT_SIZE; j++) {
          if (set_ops[i][j] < 0) break;
          onedidx[i][j] = count;
          pat.slot_labels[count] = slot_labels[i][j];
          pat.partial[count] = partial[i][j];
          int idx = 0;
          if (i > 1) idx = onedidx[i - 1][(set_ops[i][j] & 0x0F)];
          assert(idx < 31);
          pat.set_ops[count] = ((set_ops[i][j] & 0x30) << 1) + idx;
          count++;
        }
        pat.rowptr[i + 1] = count;
      }
      //std::cout << "total number of slots: " << count << std::endl;
      assert(count <= MAX_SLOT_NUM);
    }
  };
}