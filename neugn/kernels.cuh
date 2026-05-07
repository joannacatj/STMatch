#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void launch_zero_float(float* x, int n, cudaStream_t stream = 0);
void launch_zero_int(int* x, int n, cudaStream_t stream = 0);
void launch_embedding_lookup_kernel(const int64_t* ids, const float* table, float* out, int n_ids, int dim, cudaStream_t stream = 0);
void launch_add_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream = 0);
void launch_add_inplace_kernel(float* a, const float* b, int n, cudaStream_t stream = 0);
void launch_add_row_vector_inplace(float* x, const float* vec, int rows, int cols, cudaStream_t stream = 0);
void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream = 0);
void launch_copy_row_kernel(const float* in, float* out, int row_idx, int cols, cudaStream_t stream = 0);
void launch_linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim, cudaStream_t stream = 0);
void launch_relu_kernel(float* x, int n, cudaStream_t stream = 0);
void launch_gelu_kernel(float* x, int n, cudaStream_t stream = 0);
void launch_silu_mul_kernel(float* a, const float* b, int n, cudaStream_t stream = 0);
void launch_rmsnorm_kernel(const float* x, const float* w, float* y, int rows, int dim, float eps, cudaStream_t stream = 0);
void launch_degree_kernel(const int64_t* dst, int* deg, int e, cudaStream_t stream = 0);
void launch_gcn_aggregate_kernel(const int64_t* src, const int64_t* dst, const int* deg, const float* x, float* out, int e, int dim, cudaStream_t stream = 0);
void launch_max_pool_kernel(const float* x, float* out, int rows, int dim, cudaStream_t stream = 0);
void launch_attention_scores_kernel(const float* q, const float* k, float* scores, int seq, int q_heads, int kv_heads, int head_dim, cudaStream_t stream = 0);
void launch_attention_mask_row_kernel(float* scores, int seq, int heads, int valid_rows, cudaStream_t stream = 0);
void launch_softmax_rows_kernel(float* scores, int rows, int cols, cudaStream_t stream = 0);
void launch_attention_weighted_sum_kernel(const float* scores, const float* v, float* context, int seq, int q_heads, int kv_heads, int head_dim, cudaStream_t stream = 0);
