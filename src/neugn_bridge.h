#pragma once

#include <cstdint>
#include "callstack.h"

namespace STMatch {

struct NeuGNRequest {
  int warp_id;
  int level;
  int slot;
  int unroll_id;
  int cand_len;
  int qnode;
  graph_node_t mapping[PAT_SIZE];
};

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
    int64_t* subnode_packed);

__global__ void apply_neugn_ranking_kernel(
    const NeuGNRequest* requests,
    int request_count,
    CallStack* callstacks,
    const float* logits,
    int vocab_size);

__global__ void clear_neugn_pause_kernel(
    const NeuGNRequest* requests,
    int request_count,
    CallStack* callstacks);

}  // namespace STMatch
