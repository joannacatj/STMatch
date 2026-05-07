#include "neugn_bridge.h"

namespace STMatch {

__global__ void build_neugn_batch_inputs_kernel(
    const NeuGNRequest* requests,
    int request_count,
    const int* query_edge_src,
    const int* query_edge_dst,
    int query_edge_count,
    const int* query_labels,
    int query_n,
    const int* query_path,
    int path_len,
    const int* node2sub,
    int data_node_count,
    int model_num_nodes,
    int model_token_len,
    int edge_stride,
    int64_t* edge_src_packed,
    int64_t* edge_dst_packed,
    int64_t* feat_id_packed,
    int64_t* tokens_packed,
    int64_t* subnode_packed) {
  int rid = blockIdx.x;
  if (rid >= request_count) return;
  const NeuGNRequest req = requests[rid];
  int tid = threadIdx.x;

  int edge_count_with_self = query_edge_count + model_num_nodes;
  for (int e = tid; e < edge_stride; e += blockDim.x) {
    int64_t src = 0;
    int64_t dst = 0;
    if (e < query_edge_count) {
      src = query_edge_src[e];
      dst = query_edge_dst[e];
    } else if (e < edge_count_with_self) {
      int node = e - query_edge_count;
      src = node;
      dst = node;
    }
    edge_src_packed[rid * edge_stride + e] = src;
    edge_dst_packed[rid * edge_stride + e] = dst;
  }

  for (int i = tid; i < model_num_nodes; i += blockDim.x) {
    feat_id_packed[rid * model_num_nodes + i] = (i < query_n) ? query_labels[i] : 0;
  }

  const int64_t padding_id = static_cast<int64_t>(data_node_count);
  const int64_t sos_id = static_cast<int64_t>(data_node_count) + 1;
  for (int i = tid; i < model_token_len; i += blockDim.x) {
    int64_t tok = padding_id;
    int64_t sub = 0;
    if (i == 0) {
      tok = sos_id;
      sub = (req.qnode >= 0 && req.qnode < query_n) ? node2sub[req.qnode] : 0;
    } else {
      int path_pos = i - 1;
      if (path_pos < path_len) {
        int pn = query_path[path_pos];
        if (pn >= 0 && pn < query_n) {
          sub = node2sub[pn];
          if (pn == req.qnode) {
            tok = sos_id;
          } else if (req.mapping[pn] >= 0) {
            tok = req.mapping[pn];
          }
        }
      }
    }
    tokens_packed[rid * model_token_len + i] = tok;
    subnode_packed[rid * model_token_len + i] = sub;
  }
}

__global__ void apply_neugn_ranking_kernel(
    const NeuGNRequest* requests,
    int request_count,
    CallStack* callstacks,
    const float* logits,
    int vocab_size) {
  int rid = blockIdx.x;
  if (rid >= request_count || threadIdx.x != 0) return;
  const NeuGNRequest req = requests[rid];
  CallStack& stk = callstacks[req.warp_id];
  graph_node_t* cand = &stk.slot_storage[req.slot][req.unroll_id][0];
  int n = req.cand_len;
  if (n > GRAPH_DEGREE) n = GRAPH_DEGREE;

  for (int i = 1; i < n; i++) {
    graph_node_t key = cand[i];
    float key_score = (key >= 0 && key < vocab_size) ? logits[rid * vocab_size + key] : -1.0e30f;
    int j = i - 1;
    while (j >= 0) {
      graph_node_t prev = cand[j];
      float prev_score = (prev >= 0 && prev < vocab_size) ? logits[rid * vocab_size + prev] : -1.0e30f;
      if (prev_score >= key_score) break;
      cand[j + 1] = cand[j];
      j--;
    }
    cand[j + 1] = key;
  }
}

__global__ void clear_neugn_pause_kernel(
    const NeuGNRequest* requests,
    int request_count,
    CallStack* callstacks) {
  int rid = blockIdx.x * blockDim.x + threadIdx.x;
  if (rid >= request_count) return;
  const int warp_id = requests[rid].warp_id;
  callstacks[warp_id].paused_for_neugn = 0;
  callstacks[warp_id].active = 1;
}

}  // namespace STMatch
