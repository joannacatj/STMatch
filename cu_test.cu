#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "src/gpu_match.cuh"
#include "neugn/neug_model.hpp"
#include "neugn/tensor_io.hpp"
#include "src/neugn_bridge.h"

using namespace std;
using namespace STMatch;

namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

std::vector<int> build_path_fallback(std::vector<std::vector<int>> query_adj) {
  const int n = static_cast<int>(query_adj.size());
  for (auto& adj : query_adj) std::sort(adj.begin(), adj.end());
  std::vector<int> path;
  std::vector<int> visited(n, 0);

  auto dfs = [&](auto&& self, int u) -> void {
    visited[u] = 1;
    for (int v : query_adj[u]) {
      if (!visited[v]) {
        path.push_back(v);
        self(self, v);
        path.push_back(u);
      }
    }
  };

  if (n > 0) {
    path.push_back(0);
    dfs(dfs, 0);
    for (int u = 0; u < n; u++) {
      if (!visited[u]) {
        path.push_back(u);
        dfs(dfs, u);
      }
    }
  }
  if (path.empty() && n > 0) path.push_back(0);
  return path;
}

std::vector<int> build_node2sub(const std::vector<int>& path_nodes, int qn, int sub_node_id_size) {
  std::vector<int> node2sub(qn, 0);
  std::vector<int> seen(qn, 0);
  const int mod = std::max(1, sub_node_id_size);
  for (int i = 0; i < static_cast<int>(path_nodes.size()); i++) {
    int q = path_nodes[i];
    if (q >= 0 && q < qn && !seen[q]) {
      node2sub[q] = i % mod;
      seen[q] = 1;
    }
  }
  return node2sub;
}

int read_sub_node_id_size(const std::string& export_dir) {
  try {
    auto cfg = parse_config_txt(export_dir + "/config.txt");
    auto it = cfg.find("sub_node_id_size");
    if (it != cfg.end()) return std::stoi(it->second);
  } catch (...) {
  }
  return 32;
}

}  // namespace

int main(int argc, char* argv[]) {
  if ((!USE_NEUGN && argc < 3) || (USE_NEUGN && argc < 4)) {
    std::cerr << "Usage: " << argv[0] << " <data_graph> <query_graph>";
    if (USE_NEUGN) std::cerr << " <neugn_export_dir>";
    std::cerr << std::endl;
    return 1;
  }

  check_cuda(cudaSetDevice(0), "cudaSetDevice failed");

  STMatch::GraphPreprocessor g(argv[1]);
  STMatch::PatternPreprocessor p(argv[2]);

  NeuGNCudaModel neugn;
  int model_num_nodes = 0;
  int model_token_len = 0;
  int vocab_size = 0;
  int edge_stride = 0;
  int valid_token_len = 0;
  std::vector<int> query_path;
  std::vector<int> node2sub;

  if (USE_NEUGN) {
    const std::string neugn_export_dir = argv[3];
    neugn.load_model(neugn_export_dir);
    model_num_nodes = neugn.num_nodes();
    model_token_len = neugn.token_len();
    vocab_size = neugn.vocab_size();
    edge_stride = neugn.max_edges_with_self_loops();

    if (p.query_n_for_neugn > model_num_nodes) {
      std::cerr << "Query nodes exceed NeuGN model capacity: " << p.query_n_for_neugn
                << " > " << model_num_nodes << std::endl;
      return 1;
    }
    if (static_cast<int>(p.query_edge_src_for_neugn.size()) + model_num_nodes > edge_stride) {
      std::cerr << "Query edge inputs exceed NeuGN edge stride" << std::endl;
      return 1;
    }

    query_path = build_path_fallback(p.query_adj_for_neugn);
    int sub_node_id_size = read_sub_node_id_size(neugn_export_dir);
    node2sub = build_node2sub(query_path, p.query_n_for_neugn, sub_node_id_size);
    valid_token_len = std::min(static_cast<int>(query_path.size()) + 1, model_token_len);
  }

  // copy graph and pattern to GPU global memory
  Graph* gpu_graph = g.to_gpu();
  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = JobQueuePreprocessor(g.g, p).to_gpu();
  CallStack* gpu_callstack;

  graph_node_t* slot_storage;
  check_cuda(cudaMalloc(&slot_storage, sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE),
             "cudaMalloc slot_storage failed");

  std::vector<CallStack> stk(NWARPS_TOTAL);
  for (int i = 0; i < NWARPS_TOTAL; i++) {
    auto& s = stk[i];
    s.active = 1;
    s.paused_for_neugn = 0;
    memset(s.iter, 0, sizeof(s.iter));
    memset(s.uiter, 0, sizeof(s.uiter));
    memset(s.slot_size, 0, sizeof(s.slot_size));
    s.level = 0;
    s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])((char*)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
  }
  check_cuda(cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack)), "cudaMalloc gpu_callstack failed");
  check_cuda(cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice),
             "cudaMemcpy gpu_callstack failed");

  size_t* gpu_res;
  check_cuda(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL), "cudaMalloc gpu_res failed");
  check_cuda(cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL), "cudaMemset gpu_res failed");
  size_t* res = new size_t[NWARPS_TOTAL];

  int* gpu_found;
  check_cuda(cudaMalloc(&gpu_found, sizeof(int)), "cudaMalloc gpu_found failed");
  check_cuda(cudaMemset(gpu_found, 0, sizeof(int)), "cudaMemset gpu_found failed");

  unsigned long long* gpu_fms;
  check_cuda(cudaMalloc(&gpu_fms, sizeof(unsigned long long)), "cudaMalloc gpu_fms failed");
  check_cuda(cudaMemset(gpu_fms, 0, sizeof(unsigned long long)), "cudaMemset gpu_fms failed");

  int* idle_warps;
  check_cuda(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM), "cudaMalloc idle_warps failed");
  check_cuda(cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM), "cudaMemset idle_warps failed");

  int* idle_warps_count;
  check_cuda(cudaMalloc(&idle_warps_count, sizeof(int)), "cudaMalloc idle_warps_count failed");
  check_cuda(cudaMemset(idle_warps_count, 0, sizeof(int)), "cudaMemset idle_warps_count failed");

  int* global_mutex;
  check_cuda(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM), "cudaMalloc global_mutex failed");
  check_cuda(cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM), "cudaMemset global_mutex failed");

  NeuGNRequest* d_neugn_requests = nullptr;
  int* d_neugn_request_count = nullptr;
  int* d_active_warps = nullptr;
  int64_t *d_edge_src_packed = nullptr, *d_edge_dst_packed = nullptr, *d_feat_id_packed = nullptr;
  int64_t *d_tokens_packed = nullptr, *d_subnode_packed = nullptr;
  int *d_q_edge_src = nullptr, *d_q_edge_dst = nullptr, *d_q_labels = nullptr, *d_query_path = nullptr, *d_node2sub = nullptr;

  if (USE_NEUGN) {
    check_cuda(cudaMalloc(&d_neugn_requests, sizeof(NeuGNRequest) * NEUGN_BATCH_CAPACITY), "cudaMalloc d_neugn_requests failed");
    check_cuda(cudaMalloc(&d_neugn_request_count, sizeof(int)), "cudaMalloc d_neugn_request_count failed");
    check_cuda(cudaMalloc(&d_active_warps, sizeof(int)), "cudaMalloc d_active_warps failed");
    check_cuda(cudaMalloc(&d_edge_src_packed, sizeof(int64_t) * NEUGN_BATCH_CAPACITY * edge_stride), "cudaMalloc d_edge_src_packed failed");
    check_cuda(cudaMalloc(&d_edge_dst_packed, sizeof(int64_t) * NEUGN_BATCH_CAPACITY * edge_stride), "cudaMalloc d_edge_dst_packed failed");
    check_cuda(cudaMalloc(&d_feat_id_packed, sizeof(int64_t) * NEUGN_BATCH_CAPACITY * model_num_nodes), "cudaMalloc d_feat_id_packed failed");
    check_cuda(cudaMalloc(&d_tokens_packed, sizeof(int64_t) * NEUGN_BATCH_CAPACITY * model_token_len), "cudaMalloc d_tokens_packed failed");
    check_cuda(cudaMalloc(&d_subnode_packed, sizeof(int64_t) * NEUGN_BATCH_CAPACITY * model_token_len), "cudaMalloc d_subnode_packed failed");

    check_cuda(cudaMalloc(&d_q_edge_src, sizeof(int) * std::max<size_t>(1, p.query_edge_src_for_neugn.size())), "cudaMalloc d_q_edge_src failed");
    check_cuda(cudaMalloc(&d_q_edge_dst, sizeof(int) * std::max<size_t>(1, p.query_edge_dst_for_neugn.size())), "cudaMalloc d_q_edge_dst failed");
    check_cuda(cudaMalloc(&d_q_labels, sizeof(int) * std::max(1, p.query_n_for_neugn)), "cudaMalloc d_q_labels failed");
    check_cuda(cudaMalloc(&d_query_path, sizeof(int) * std::max<size_t>(1, query_path.size())), "cudaMalloc d_query_path failed");
    check_cuda(cudaMalloc(&d_node2sub, sizeof(int) * std::max(1, p.query_n_for_neugn)), "cudaMalloc d_node2sub failed");
    if (!p.query_edge_src_for_neugn.empty()) {
      check_cuda(cudaMemcpy(d_q_edge_src, p.query_edge_src_for_neugn.data(), sizeof(int) * p.query_edge_src_for_neugn.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_q_edge_src failed");
      check_cuda(cudaMemcpy(d_q_edge_dst, p.query_edge_dst_for_neugn.data(), sizeof(int) * p.query_edge_dst_for_neugn.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_q_edge_dst failed");
    }
    if (p.query_n_for_neugn > 0) {
      check_cuda(cudaMemcpy(d_q_labels, p.query_labels_for_neugn.data(), sizeof(int) * p.query_n_for_neugn, cudaMemcpyHostToDevice), "cudaMemcpy d_q_labels failed");
      check_cuda(cudaMemcpy(d_node2sub, node2sub.data(), sizeof(int) * p.query_n_for_neugn, cudaMemcpyHostToDevice), "cudaMemcpy d_node2sub failed");
    }
    if (!query_path.empty()) {
      check_cuda(cudaMemcpy(d_query_path, query_path.data(), sizeof(int) * query_path.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_query_path failed");
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int launch_count = 0;
  if (USE_NEUGN) {
    while (true) {
      launch_count++;
      check_cuda(cudaMemset(d_neugn_request_count, 0, sizeof(int)), "cudaMemset d_neugn_request_count failed");
      check_cuda(cudaMemset(d_active_warps, 0, sizeof(int)), "cudaMemset d_active_warps failed");
      check_cuda(cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM), "cudaMemset idle_warps failed");
      check_cuda(cudaMemset(idle_warps_count, 0, sizeof(int)), "cudaMemset idle_warps_count failed");
      check_cuda(cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM), "cudaMemset global_mutex failed");

      launch_parallel_match(gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res,
                            idle_warps, idle_warps_count, global_mutex, gpu_found, gpu_fms,
                            d_neugn_requests, d_neugn_request_count, NEUGN_BATCH_CAPACITY, d_active_warps);
      check_cuda(cudaGetLastError(), "Kernel launch failed");
      check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");

      int h_req_count_raw = 0;
      check_cuda(cudaMemcpy(&h_req_count_raw, d_neugn_request_count, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy request count failed");
      int h_req_count = std::min(h_req_count_raw, NEUGN_BATCH_CAPACITY);

      if (h_req_count > 0) {
        build_neugn_batch_inputs_kernel<<<h_req_count, 256>>>(
            d_neugn_requests, h_req_count, d_q_edge_src, d_q_edge_dst,
            static_cast<int>(p.query_edge_src_for_neugn.size()), d_q_labels, p.query_n_for_neugn,
            d_query_path, static_cast<int>(query_path.size()), d_node2sub, g.g.nnodes,
            model_num_nodes, model_token_len, edge_stride, d_edge_src_packed, d_edge_dst_packed,
            d_feat_id_packed, d_tokens_packed, d_subnode_packed);
        check_cuda(cudaGetLastError(), "build_neugn_batch_inputs_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "build_neugn_batch_inputs_kernel failed");

        std::vector<int> edge_counts(h_req_count, static_cast<int>(p.query_edge_src_for_neugn.size()) + model_num_nodes);
        std::vector<int> token_mask_lens(h_req_count, valid_token_len);
        neugn.forward_batch_from_device(d_edge_src_packed, d_edge_dst_packed, edge_stride,
                                        d_feat_id_packed, model_num_nodes,
                                        d_tokens_packed, model_token_len,
                                        d_subnode_packed, model_token_len,
                                        edge_counts, token_mask_lens, h_req_count, false);

        apply_neugn_ranking_kernel<<<h_req_count, 1>>>(d_neugn_requests, h_req_count, gpu_callstack,
                                                       neugn.device_output(), vocab_size);
        check_cuda(cudaGetLastError(), "apply_neugn_ranking_kernel launch failed");
        clear_neugn_pause_kernel<<<(h_req_count + 255) / 256, 256>>>(d_neugn_requests, h_req_count, gpu_callstack);
        check_cuda(cudaGetLastError(), "clear_neugn_pause_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "NeuGN ranking/clear kernels failed");
      }

      int h_active = 0;
      check_cuda(cudaMemcpy(&h_active, d_active_warps, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy active_warps failed");
      if (FIND_FIRST) {
        int found = 0;
        check_cuda(cudaMemcpy(&found, gpu_found, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy found failed");
        if (found) break;
      }
      if (h_active == 0 && h_req_count == 0) break;
    }
  }
  else {
    launch_count = 1;
    launch_parallel_match(gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res,
                          idle_warps, idle_warps_count, global_mutex, gpu_found, gpu_fms);
    check_cuda(cudaGetLastError(), "Kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");
  }

  cudaEventRecord(stop);
  check_cuda(cudaEventSynchronize(stop), "Timing synchronize failed");

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  check_cuda(cudaMemcpy(res, gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost), "cudaMemcpy results failed");

  unsigned long long tot_count = 0;
  for (int i = 0; i < NWARPS_TOTAL; i++) tot_count += res[i];

  unsigned long long fms = 0;
  if (FIND_FIRST) {
    int found = 0;
    check_cuda(cudaMemcpy(&found, gpu_found, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy found failed");
    check_cuda(cudaMemcpy(&fms, gpu_fms, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy fms failed");
    tot_count = found ? 1 : 0;
  }
  else if (!LABELED) {
    tot_count = tot_count * p.PatternMultiplicity;
  }

  if (USE_NEUGN && FIND_FIRST) {
    printf("%s\t%f\t%llu\t%llu\t%d\n", argv[2], milliseconds, tot_count, fms, launch_count);
  }
  else if (FIND_FIRST) {
    printf("%s\t%f\t%llu\t%llu\n", argv[2], milliseconds, tot_count, fms);
  }
  else {
    printf("%s\t%f\t%llu\n", argv[2], milliseconds, tot_count);
  }

  return 0;
}
