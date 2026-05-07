#include "kernels.cuh"

#include <float.h>
#include <math.h>

__global__ void zero_float_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}
__global__ void zero_int_kernel(int* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0;
}

__global__ void embedding_lookup_kernel(const int64_t* ids, const float* table, float* out, int n_ids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_ids * dim;
    if (idx < total) {
        int r = idx / dim;
        int c = idx % dim;
        int64_t id = ids[r];
        out[idx] = table[id * dim + c];
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
__global__ void add_inplace_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void add_row_vector_kernel(float* x, const float* vec, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int c = idx % cols;
        x[idx] += vec[c];
    }
}

__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

__global__ void copy_row_kernel(const float* in, float* out, int row_idx, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cols) out[i] = in[row_idx * cols + i];
}

__global__ void linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim) {
    int r = blockIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && o < out_dim) {
        float acc = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            acc += x[r * in_dim + i] * w[o * in_dim + i];
        }
        y[r * out_dim + o] = acc;
    }
}

__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void gelu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        x[idx] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

__global__ void silu_mul_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a[i];
        float silu = v / (1.0f + expf(-v));
        a[i] = silu * b[i];
    }
}

__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int rows, int dim, float eps) {
    int r = blockIdx.x;
    if (r < rows) {
        float mean_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float v = x[r * dim + i];
            mean_sq += v * v;
        }
        mean_sq /= (float)dim;
        float inv = rsqrtf(mean_sq + eps);
        for (int i = 0; i < dim; ++i) {
            y[r * dim + i] = x[r * dim + i] * inv * w[i];
        }
    }
}

__global__ void degree_kernel(const int64_t* dst, int* deg, int e) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < e) {
        atomicAdd(&deg[dst[i]], 1);
    }
}

__global__ void gcn_aggregate_kernel(const int64_t* src, const int64_t* dst, const int* deg, const float* x, float* out, int e, int dim) {
    int edge_i = blockIdx.x;
    int d = threadIdx.x;
    if (edge_i < e && d < dim) {
        int64_t s = src[edge_i];
        int64_t t = dst[edge_i];
        float dsrc = rsqrtf(fmaxf((float)deg[s], 1.0f));
        float ddst = rsqrtf(fmaxf((float)deg[t], 1.0f));
        float msg = x[s * dim + d] * dsrc * ddst;
        atomicAdd(&out[t * dim + d], msg);
    }
}

__global__ void max_pool_kernel(const float* x, float* out, int rows, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) {
        float m = -FLT_MAX;
        for (int r = 0; r < rows; ++r) {
            float v = x[r * dim + d];
            if (v > m) m = v;
        }
        out[d] = m;
    }
}

__global__ void attention_scores_kernel(const float* q, const float* k, float* scores, int seq, int q_heads, int kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = q_heads * seq * seq;
    if (idx < total) {
        int j = idx % seq;
        int i = (idx / seq) % seq;
        int h = idx / (seq * seq);
        int n_rep = q_heads / kv_heads;
        int hk = h / n_rep;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int qidx = i * (q_heads * head_dim) + h * head_dim + d;
            int kidx = j * (kv_heads * head_dim) + hk * head_dim + d;
            acc += q[qidx] * k[kidx];
        }
        acc /= sqrtf((float)head_dim);
        scores[idx] = acc;
    }
}

__global__ void attention_mask_row_kernel(float* scores, int seq, int heads, int valid_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = heads * seq * seq;
    if (idx < total) {
        int i = (idx / seq) % seq;
        if (i >= valid_rows) scores[idx] = -1e30f;
    }
}

__global__ void softmax_rows_kernel(float* scores, int rows, int cols) {
    int r = blockIdx.x;
    if (r < rows) {
        float m = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            float v = scores[r * cols + c];
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(scores[r * cols + c] - m);
            scores[r * cols + c] = e;
            sum += e;
        }
        float inv = 1.0f / fmaxf(sum, 1e-12f);
        for (int c = 0; c < cols; ++c) {
            scores[r * cols + c] *= inv;
        }
    }
}

__global__ void attention_weighted_sum_kernel(const float* scores, const float* v, float* context, int seq, int q_heads, int kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * q_heads * head_dim;
    if (idx < total) {
        int d = idx % head_dim;
        int h = (idx / head_dim) % q_heads;
        int i = idx / (head_dim * q_heads);
        int n_rep = q_heads / kv_heads;
        int hk = h / n_rep;

        float acc = 0.0f;
        int row = h * seq + i;
        for (int j = 0; j < seq; ++j) {
            float p = scores[row * seq + j];
            int vidx = j * (kv_heads * head_dim) + hk * head_dim + d;
            acc += p * v[vidx];
        }
        context[i * (q_heads * head_dim) + h * head_dim + d] = acc;
    }
}

// Launchers
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void launch_zero_float(float* x, int n, cudaStream_t stream) { zero_float_kernel<<<ceil_div(n,256),256,0,stream>>>(x,n); }
void launch_zero_int(int* x, int n, cudaStream_t stream) { zero_int_kernel<<<ceil_div(n,256),256,0,stream>>>(x,n); }
void launch_embedding_lookup_kernel(const int64_t* ids, const float* table, float* out, int n_ids, int dim, cudaStream_t stream) { embedding_lookup_kernel<<<ceil_div(n_ids*dim,256),256,0,stream>>>(ids,table,out,n_ids,dim);} 
void launch_add_kernel(const float* a, const float* b, float* out, int n, cudaStream_t stream) { add_kernel<<<ceil_div(n,256),256,0,stream>>>(a,b,out,n);} 
void launch_add_inplace_kernel(float* a, const float* b, int n, cudaStream_t stream) { add_inplace_kernel<<<ceil_div(n,256),256,0,stream>>>(a,b,n);} 
void launch_add_row_vector_inplace(float* x, const float* vec, int rows, int cols, cudaStream_t stream) { add_row_vector_kernel<<<ceil_div(rows*cols,256),256,0,stream>>>(x,vec,rows,cols);} 
void launch_copy_kernel(const float* in, float* out, int n, cudaStream_t stream) { copy_kernel<<<ceil_div(n,256),256,0,stream>>>(in,out,n);} 
void launch_copy_row_kernel(const float* in, float* out, int row_idx, int cols, cudaStream_t stream) { copy_row_kernel<<<ceil_div(cols,256),256,0,stream>>>(in,out,row_idx,cols);} 
void launch_linear_kernel(const float* x, const float* w, const float* b, float* y, int rows, int in_dim, int out_dim, cudaStream_t stream) { dim3 grid(ceil_div(out_dim,256), rows, 1); linear_kernel<<<grid,256,0,stream>>>(x,w,b,y,rows,in_dim,out_dim);} 
void launch_relu_kernel(float* x, int n, cudaStream_t stream) { relu_kernel<<<ceil_div(n,256),256,0,stream>>>(x,n);} 
void launch_gelu_kernel(float* x, int n, cudaStream_t stream) { gelu_kernel<<<ceil_div(n,256),256,0,stream>>>(x,n);} 
void launch_silu_mul_kernel(float* a, const float* b, int n, cudaStream_t stream) { silu_mul_kernel<<<ceil_div(n,256),256,0,stream>>>(a,b,n);} 
void launch_rmsnorm_kernel(const float* x, const float* w, float* y, int rows, int dim, float eps, cudaStream_t stream) { rmsnorm_kernel<<<rows,1,0,stream>>>(x,w,y,rows,dim,eps);} 
void launch_degree_kernel(const int64_t* dst, int* deg, int e, cudaStream_t stream) { degree_kernel<<<ceil_div(e,256),256,0,stream>>>(dst,deg,e);} 
void launch_gcn_aggregate_kernel(const int64_t* src, const int64_t* dst, const int* deg, const float* x, float* out, int e, int dim, cudaStream_t stream) { gcn_aggregate_kernel<<<e,dim,0,stream>>>(src,dst,deg,x,out,e,dim);} 
void launch_max_pool_kernel(const float* x, float* out, int rows, int dim, cudaStream_t stream) { max_pool_kernel<<<ceil_div(dim,256),256,0,stream>>>(x,out,rows,dim);} 
void launch_attention_scores_kernel(const float* q, const float* k, float* scores, int seq, int q_heads, int kv_heads, int head_dim, cudaStream_t stream) { attention_scores_kernel<<<ceil_div(q_heads*seq*seq,256),256,0,stream>>>(q,k,scores,seq,q_heads,kv_heads,head_dim);} 
void launch_attention_mask_row_kernel(float* scores, int seq, int heads, int valid_rows, cudaStream_t stream) { attention_mask_row_kernel<<<ceil_div(heads*seq*seq,256),256,0,stream>>>(scores,seq,heads,valid_rows);} 
void launch_softmax_rows_kernel(float* scores, int rows, int cols, cudaStream_t stream) { softmax_rows_kernel<<<rows,1,0,stream>>>(scores,rows,cols);} 
void launch_attention_weighted_sum_kernel(const float* scores, const float* v, float* context, int seq, int q_heads, int kv_heads, int head_dim, cudaStream_t stream) { attention_weighted_sum_kernel<<<ceil_div(seq*q_heads*head_dim,256),256,0,stream>>>(scores,v,context,seq,q_heads,kv_heads,head_dim);} 
