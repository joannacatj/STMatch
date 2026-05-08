#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
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

struct MatchResult {
  float milliseconds = 0.0f;
  unsigned long long count = 0;
  unsigned long long fms = 0;
  int launch_count = 0;
};

struct GeneratedQuery {
  std::vector<graph_node_t> data_vertices;
  std::vector<std::vector<int>> adj_matrix;
  std::vector<int> compact_labels;
};

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(1);
  }
}

void log_progress(const std::string& message) {
  std::cerr << "[STMatch][progress] " << message << std::endl;
}

int bitidx(bitarray32 a) {
  for (int i = 0; i < 32; i++) {
    if (a & (1u << i)) return i;
  }
  return 0;
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

void free_gpu_graph(Graph* gpu_graph) {
  if (gpu_graph == nullptr) return;
  Graph h{};
  check_cuda(cudaMemcpy(&h, gpu_graph, sizeof(Graph), cudaMemcpyDeviceToHost), "cudaMemcpy gpu graph for free failed");
  cudaFree(h.vertex_label);
  cudaFree(h.rowptr);
  cudaFree(h.colidx);
  cudaFree(gpu_graph);
}

void free_gpu_job_queue(JobQueue* gpu_queue) {
  if (gpu_queue == nullptr) return;
  JobQueue h{};
  check_cuda(cudaMemcpy(&h, gpu_queue, sizeof(JobQueue), cudaMemcpyDeviceToHost), "cudaMemcpy gpu queue for free failed");
  cudaFree(h.q);
  cudaFree(gpu_queue);
}

GeneratedQuery generate_random_walk_query(const Graph& g, int query_nodes, std::mt19937& rng) {
  if (query_nodes < 3 || query_nodes > static_cast<int>(PAT_SIZE)) {
    std::cerr << "Random-walk query node count must be in [3, " << PAT_SIZE << "]\n";
    std::exit(1);
  }
  if (query_nodes > g.nnodes) {
    std::cerr << "Random-walk query node count exceeds data graph nodes\n";
    std::exit(1);
  }

  std::vector<graph_node_t> non_isolated;
  for (graph_node_t v = 0; v < g.nnodes; v++) {
    if (g.rowptr[v] < g.rowptr[v + 1]) non_isolated.push_back(v);
  }
  if (non_isolated.empty()) {
    std::cerr << "Random-walk query generation requires at least one edge in the data graph\n";
    std::exit(1);
  }

  std::uniform_int_distribution<size_t> start_dist(0, non_isolated.size() - 1);
  std::vector<graph_node_t> selected;
  std::vector<int> in_selected(g.nnodes, 0);

  for (int attempt = 0; attempt < 1000 && static_cast<int>(selected.size()) < query_nodes; attempt++) {
    selected.clear();
    std::fill(in_selected.begin(), in_selected.end(), 0);
    graph_node_t cur = non_isolated[start_dist(rng)];
    selected.push_back(cur);
    in_selected[cur] = 1;

    for (int step = 0; step < g.nnodes * 20 && static_cast<int>(selected.size()) < query_nodes; step++) {
      graph_edge_t begin = g.rowptr[cur];
      graph_edge_t end = g.rowptr[cur + 1];
      if (begin == end) break;
      std::uniform_int_distribution<graph_edge_t> edge_dist(begin, end - 1);
      graph_node_t nxt = g.colidx[edge_dist(rng)];
      cur = nxt;
      if (!in_selected[cur]) {
        selected.push_back(cur);
        in_selected[cur] = 1;
      }
    }
  }

  if (static_cast<int>(selected.size()) < query_nodes) {
    std::cerr << "Failed to sample a connected random-walk query with " << query_nodes << " nodes\n";
    std::exit(1);
  }

  std::vector<int> pos(g.nnodes, -1);
  for (int i = 0; i < query_nodes; i++) pos[selected[i]] = i;

  GeneratedQuery q;
  q.data_vertices = selected;
  q.adj_matrix.assign(query_nodes, std::vector<int>(query_nodes, 0));
  q.compact_labels.assign(query_nodes, 0);
  for (int i = 0; i < query_nodes; i++) {
    graph_node_t u = selected[i];
    q.compact_labels[i] = LABELED ? bitidx(g.vertex_label[u]) : 1;
    for (graph_edge_t e = g.rowptr[u]; e < g.rowptr[u + 1]; e++) {
      graph_node_t v = g.colidx[e];
      if (v >= 0 && v < g.nnodes && pos[v] >= 0) {
        q.adj_matrix[i][pos[v]] = 1;
      }
    }
  }
  for (int i = 0; i < query_nodes; i++) q.adj_matrix[i][i] = 0;
  return q;
}

std::string describe_generated_query(const GeneratedQuery& q) {
  std::ostringstream out;
  out << "vertices=";
  for (size_t i = 0; i < q.data_vertices.size(); i++) {
    if (i) out << ",";
    out << q.data_vertices[i];
  }
  return out.str();
}

MatchResult run_match(GraphPreprocessor& g, Graph* gpu_graph, PatternPreprocessor& p,
                      const std::string& query_name, bool enable_neugn_runtime,
                      NeuGNCudaModel* neugn, const std::string& neugn_export_dir) {
  MatchResult result;

  if (enable_neugn_runtime) {
    if (neugn == nullptr) {
      std::cerr << "NeuGN runtime requested but model is not loaded\n";
      std::exit(1);
    }
    if (p.query_n_for_neugn > neugn->num_nodes()) {
      std::cerr << "Query nodes exceed NeuGN model capacity: " << p.query_n_for_neugn
                << " > " << neugn->num_nodes() << std::endl;
      std::exit(1);
    }
    if (static_cast<int>(p.query_edge_src_for_neugn.size()) + neugn->num_nodes() > neugn->max_edges_with_self_loops()) {
      std::cerr << "Query edge inputs exceed NeuGN edge stride" << std::endl;
      std::exit(1);
    }
  }

  log_progress(std::string("Preparing GPU state for query: ") + query_name +
               (enable_neugn_runtime ? " [with NeuGN]" : " [without NeuGN]"));

  Pattern* gpu_pattern = p.to_gpu();
  JobQueue* gpu_queue = JobQueuePreprocessor(g.g, p).to_gpu();
  CallStack* gpu_callstack = nullptr;

  graph_node_t* slot_storage = nullptr;
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

  size_t* gpu_res = nullptr;
  check_cuda(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL), "cudaMalloc gpu_res failed");
  check_cuda(cudaMemset(gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL), "cudaMemset gpu_res failed");
  std::vector<size_t> res(NWARPS_TOTAL, 0);

  int* gpu_found = nullptr;
  check_cuda(cudaMalloc(&gpu_found, sizeof(int)), "cudaMalloc gpu_found failed");
  check_cuda(cudaMemset(gpu_found, 0, sizeof(int)), "cudaMemset gpu_found failed");

  unsigned long long* gpu_fms = nullptr;
  check_cuda(cudaMalloc(&gpu_fms, sizeof(unsigned long long)), "cudaMalloc gpu_fms failed");
  check_cuda(cudaMemset(gpu_fms, 0, sizeof(unsigned long long)), "cudaMemset gpu_fms failed");

  int* idle_warps = nullptr;
  check_cuda(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM), "cudaMalloc idle_warps failed");
  check_cuda(cudaMemset(idle_warps, 0, sizeof(int) * GRID_DIM), "cudaMemset idle_warps failed");

  int* idle_warps_count = nullptr;
  check_cuda(cudaMalloc(&idle_warps_count, sizeof(int)), "cudaMalloc idle_warps_count failed");
  check_cuda(cudaMemset(idle_warps_count, 0, sizeof(int)), "cudaMemset idle_warps_count failed");

  int* global_mutex = nullptr;
  check_cuda(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM), "cudaMalloc global_mutex failed");
  check_cuda(cudaMemset(global_mutex, 0, sizeof(int) * GRID_DIM), "cudaMemset global_mutex failed");

  NeuGNRequest* d_neugn_requests = nullptr;
  int* d_neugn_request_count = nullptr;
  int* d_active_warps = nullptr;
  int64_t *d_edge_src_packed = nullptr, *d_edge_dst_packed = nullptr, *d_feat_id_packed = nullptr;
  int64_t *d_tokens_packed = nullptr, *d_subnode_packed = nullptr;
  int *d_q_edge_src = nullptr, *d_q_edge_dst = nullptr, *d_q_labels = nullptr, *d_query_path = nullptr, *d_node2sub = nullptr;

  int model_num_nodes = 0;
  int model_token_len = 0;
  int vocab_size = 0;
  int edge_stride = 0;
  int valid_token_len = 0;
  std::vector<int> query_path;
  std::vector<int> node2sub;

  if (enable_neugn_runtime) {
    model_num_nodes = neugn->num_nodes();
    model_token_len = neugn->token_len();
    vocab_size = neugn->vocab_size();
    edge_stride = neugn->max_edges_with_self_loops();
    query_path = build_path_fallback(p.query_adj_for_neugn);
    int sub_node_id_size = read_sub_node_id_size(neugn_export_dir);
    node2sub = build_node2sub(query_path, p.query_n_for_neugn, sub_node_id_size);
    valid_token_len = std::min(static_cast<int>(query_path.size()) + 1, model_token_len);

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
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");
  check_cuda(cudaEventRecord(start), "cudaEventRecord start failed");

  if (enable_neugn_runtime) {
    while (true) {
      result.launch_count++;
      log_progress("Launching STMatch DFS kernel, iteration=" + std::to_string(result.launch_count));
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
      log_progress("STMatch kernel finished, iteration=" + std::to_string(result.launch_count) +
                   ", neugn_requests=" + std::to_string(h_req_count));

      if (h_req_count > 0) {
        log_progress("Building NeuGN batch inputs for " + std::to_string(h_req_count) + " requests");
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
        log_progress("Running NeuGN forward_batch_from_device for " + std::to_string(h_req_count) + " requests");
        neugn->forward_batch_from_device(d_edge_src_packed, d_edge_dst_packed, edge_stride,
                                         d_feat_id_packed, model_num_nodes,
                                         d_tokens_packed, model_token_len,
                                         d_subnode_packed, model_token_len,
                                         edge_counts, token_mask_lens, h_req_count, false);

        log_progress("Applying NeuGN ranking and resuming paused warps");
        apply_neugn_ranking_kernel<<<h_req_count, 1>>>(d_neugn_requests, h_req_count, gpu_callstack,
                                                       neugn->device_output(), vocab_size);
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
        if (found) {
          log_progress("Find-first match found; stopping search loop");
          break;
        }
      }
      if (h_active == 0 && h_req_count == 0) {
        log_progress("No active warps and no NeuGN requests remain; stopping search loop");
        break;
      }
    }
  }
  else {
    result.launch_count = 1;
    log_progress("Launching STMatch DFS kernel without NeuGN");
    launch_parallel_match(gpu_graph, gpu_pattern, gpu_callstack, gpu_queue, gpu_res,
                          idle_warps, idle_warps_count, global_mutex, gpu_found, gpu_fms);
    check_cuda(cudaGetLastError(), "Kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "Kernel execution failed");
    log_progress("STMatch DFS kernel without NeuGN finished");
  }

  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  check_cuda(cudaEventSynchronize(stop), "Timing synchronize failed");
  check_cuda(cudaEventElapsedTime(&result.milliseconds, start, stop), "cudaEventElapsedTime failed");

  check_cuda(cudaMemcpy(res.data(), gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost), "cudaMemcpy results failed");
  for (int i = 0; i < NWARPS_TOTAL; i++) result.count += res[i];

  if (FIND_FIRST) {
    int found = 0;
    check_cuda(cudaMemcpy(&found, gpu_found, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy found failed");
    check_cuda(cudaMemcpy(&result.fms, gpu_fms, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy fms failed");
    result.count = found ? 1 : 0;
  }
  else if (!LABELED) {
    result.count = result.count * p.PatternMultiplicity;
  }

  log_progress("Query complete: " + query_name +
               (enable_neugn_runtime ? " [with NeuGN]" : " [without NeuGN]") +
               ", elapsed_ms=" + std::to_string(result.milliseconds) +
               ", count=" + std::to_string(result.count) +
               (FIND_FIRST ? ", fms=" + std::to_string(result.fms) : "") +
               ", launches=" + std::to_string(result.launch_count));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_neugn_requests);
  cudaFree(d_neugn_request_count);
  cudaFree(d_active_warps);
  cudaFree(d_edge_src_packed);
  cudaFree(d_edge_dst_packed);
  cudaFree(d_feat_id_packed);
  cudaFree(d_tokens_packed);
  cudaFree(d_subnode_packed);
  cudaFree(d_q_edge_src);
  cudaFree(d_q_edge_dst);
  cudaFree(d_q_labels);
  cudaFree(d_query_path);
  cudaFree(d_node2sub);
  cudaFree(global_mutex);
  cudaFree(idle_warps_count);
  cudaFree(idle_warps);
  cudaFree(gpu_fms);
  cudaFree(gpu_found);
  cudaFree(gpu_res);
  cudaFree(gpu_callstack);
  cudaFree(slot_storage);
  free_gpu_job_queue(gpu_queue);
  cudaFree(gpu_pattern);

  return result;
}

void run_random_walk_compare(int argc, char* argv[]) {
  if (!USE_NEUGN) {
    std::cerr << "Random-walk compare mode requires a USE_NEUGN build target.\n";
    std::exit(1);
  }
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0]
              << " <data_graph> --random-walk <query_nodes> <num_queries> <neugn_export_dir> [seed]\n";
    std::exit(1);
  }

  const int query_nodes = std::stoi(argv[3]);
  const int num_queries = std::stoi(argv[4]);
  const std::string neugn_export_dir = argv[5];
  const unsigned int seed = (argc >= 7) ? static_cast<unsigned int>(std::stoul(argv[6])) : 1u;
  if (num_queries <= 0) {
    std::cerr << "num_queries must be positive\n";
    std::exit(1);
  }

  log_progress("Selecting CUDA device 0");
  check_cuda(cudaSetDevice(0), "cudaSetDevice failed");
  log_progress(std::string("Loading data graph: ") + argv[1]);
  GraphPreprocessor g(argv[1]);
  Graph* gpu_graph = g.to_gpu();

  log_progress(std::string("Loading NeuGN model from: ") + neugn_export_dir);
  NeuGNCudaModel neugn;
  neugn.load_model(neugn_export_dir);
  log_progress("NeuGN model loaded: num_nodes=" + std::to_string(neugn.num_nodes()) +
               ", token_len=" + std::to_string(neugn.token_len()) +
               ", vocab_size=" + std::to_string(neugn.vocab_size()) +
               ", edge_stride=" + std::to_string(neugn.max_edges_with_self_loops()));

  std::mt19937 rng(seed);
  printf("query_id\tquery_vertices\tno_neugn_ms\tno_neugn_count\tno_neugn_fms\tneugn_ms\tneugn_count\tneugn_fms\tneugn_launches\n");
  for (int qi = 0; qi < num_queries; qi++) {
    GeneratedQuery generated = generate_random_walk_query(g.g, query_nodes, rng);
    PatternPreprocessor p(generated.adj_matrix, generated.compact_labels);
    const std::string qdesc = "random_walk_" + std::to_string(qi) + "(" + describe_generated_query(generated) + ")";
    log_progress("Generated query " + std::to_string(qi) + ": " + describe_generated_query(generated));

    MatchResult baseline = run_match(g, gpu_graph, p, qdesc, false, nullptr, "");
    MatchResult ranked = run_match(g, gpu_graph, p, qdesc, true, &neugn, neugn_export_dir);
    printf("%d\t%s\t%f\t%llu\t%llu\t%f\t%llu\t%llu\t%d\n",
           qi, describe_generated_query(generated).c_str(),
           baseline.milliseconds, baseline.count, baseline.fms,
           ranked.milliseconds, ranked.count, ranked.fms, ranked.launch_count);
    fflush(stdout);
  }
  free_gpu_graph(gpu_graph);
}

}  // namespace

int main(int argc, char* argv[]) {
  if (USE_NEUGN && argc >= 3 && std::string(argv[2]) == "--random-walk") {
    run_random_walk_compare(argc, argv);
    return 0;
  }

  if ((!USE_NEUGN && argc < 3) || (USE_NEUGN && argc < 4)) {
    std::cerr << "Usage: " << argv[0] << " <data_graph> <query_graph>";
    if (USE_NEUGN) {
      std::cerr << " <neugn_export_dir>\n"
                << "       " << argv[0]
                << " <data_graph> --random-walk <query_nodes> <num_queries> <neugn_export_dir> [seed]";
    }
    std::cerr << std::endl;
    return 1;
  }

  log_progress("Selecting CUDA device 0");
  check_cuda(cudaSetDevice(0), "cudaSetDevice failed");

  log_progress(std::string("Loading data graph: ") + argv[1]);
  GraphPreprocessor g(argv[1]);
  log_progress(std::string("Loading query graph: ") + argv[2]);
  PatternPreprocessor p(argv[2]);
  Graph* gpu_graph = g.to_gpu();

  NeuGNCudaModel neugn;
  NeuGNCudaModel* neugn_ptr = nullptr;
  std::string neugn_export_dir;
  bool enable_neugn_runtime = false;
  if (USE_NEUGN) {
    neugn_export_dir = argv[3];
    log_progress(std::string("Loading NeuGN model from: ") + neugn_export_dir);
    neugn.load_model(neugn_export_dir);
    log_progress("NeuGN model loaded: num_nodes=" + std::to_string(neugn.num_nodes()) +
                 ", token_len=" + std::to_string(neugn.token_len()) +
                 ", vocab_size=" + std::to_string(neugn.vocab_size()) +
                 ", edge_stride=" + std::to_string(neugn.max_edges_with_self_loops()));
    neugn_ptr = &neugn;
    enable_neugn_runtime = true;
  }

  MatchResult result = run_match(g, gpu_graph, p, argv[2], enable_neugn_runtime, neugn_ptr, neugn_export_dir);
  free_gpu_graph(gpu_graph);

  if (USE_NEUGN && FIND_FIRST) {
    printf("%s\t%f\t%llu\t%llu\t%d\n", argv[2], result.milliseconds, result.count, result.fms, result.launch_count);
  }
  else if (FIND_FIRST) {
    printf("%s\t%f\t%llu\t%llu\n", argv[2], result.milliseconds, result.count, result.fms);
  }
  else {
    printf("%s\t%f\t%llu\n", argv[2], result.milliseconds, result.count);
  }

  return 0;
}
