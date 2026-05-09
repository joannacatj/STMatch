#pragma once

#include "tensor_io.hpp"

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

class NeuGNCudaModel {
public:
    NeuGNCudaModel() = default;
    ~NeuGNCudaModel();

    NeuGNCudaModel(const NeuGNCudaModel&) = delete;
    NeuGNCudaModel& operator=(const NeuGNCudaModel&) = delete;

    // Backward-compatible path: load config/manifest/weights and also read input/*.bin once.
    void load(const std::string& export_dir);

    // New path: load config/manifest/weights once, allocate work buffers, keep all weights resident on GPU.
    // This does NOT read export_dir/input/*.bin.
    void load_model(const std::string& export_dir);

    // Set one sample from host memory without writing input/*.bin.
    // edge_src_no_self_loop / edge_dst_no_self_loop are query edges WITHOUT self loops;
    // this method appends one self-loop per model node to match the old load() behavior.
    void set_input_from_host(
        const std::vector<int64_t>& edge_src_no_self_loop,
        const std::vector<int64_t>& edge_dst_no_self_loop,
        const std::vector<int64_t>& feat_id,
        const std::vector<int64_t>& tokens,
        const std::vector<int64_t>& subnode_ids,
        const std::vector<int64_t>& token_mask_len
    );

    // Convenience overload for edge_index flattened as [src..., dst...].
    void set_input_from_host_edge_index(
        const std::vector<int64_t>& edge_index_2row_flat,
        const std::vector<int64_t>& feat_id,
        const std::vector<int64_t>& tokens,
        const std::vector<int64_t>& subnode_ids,
        const std::vector<int64_t>& token_mask_len
    );

    // Forward for the current single-sample input resident in d_src_/d_dst_/d_feat_id_/d_tokens_/d_subnode_.
    void forward_full_model();

    // Batched forward from already-device-resident inputs. This does not read/write input/*.bin.
    // edge_src_packed / edge_dst_packed must already include self loops if the model expects them.
    // Layout:
    //   edge_src_packed: [batch_size, edge_stride]
    //   edge_dst_packed: [batch_size, edge_stride]
    //   feat_id_packed:  [batch_size, feat_stride]  usually feat_stride == num_nodes()
    //   tokens_packed:   [batch_size, token_stride] usually token_stride == token_len()
    //   subnode_packed:  [batch_size, subnode_stride] usually subnode_stride == token_len()
    // edge_counts_with_self_loops and token_mask_lens are host-side metadata used for kernel launches.
    void forward_batch_from_device(
        const int64_t* d_edge_src_packed,
        const int64_t* d_edge_dst_packed,
        int edge_stride,
        const int64_t* d_feat_id_packed,
        int feat_stride,
        const int64_t* d_tokens_packed,
        int token_stride,
        const int64_t* d_subnode_packed,
        int subnode_stride,
        const std::vector<int>& edge_counts_with_self_loops,
        const std::vector<int>& token_mask_lens,
        int batch_size,
        bool copy_to_host = true
    );

    void save_output(const std::string& path) const;
    std::vector<float> first_values(int k) const;

    const std::vector<int64_t>& output_shape() const { return output_shape_; }
    const std::vector<float>& output_values() const { return output_host_; }
    const float* device_output() const { return d_last_output_; }

    int num_nodes() const { return num_nodes_; }
    int token_len() const { return token_len_; }
    int vocab_size() const { return vocab_; }
    int max_edges_without_self_loops() const { return max_edges_without_self_loop_; }
    int max_edges_with_self_loops() const { return input_edge_capacity_; }
    int encoder_label_vocab_size() const { return encoder_label_vocab_; }
    int token_vocab_size() const { return token_vocab_; }
    int subnode_vocab_size() const { return subnode_vocab_; }

private:
    std::string export_dir_;
    std::unordered_map<std::string, std::string> config_;
    std::unordered_map<std::string, TensorInfo> manifest_;

    // Host copies for backward compatibility / validation.
    std::vector<int64_t> edge_src_h_, edge_dst_h_, feat_id_h_, tokens_h_, subnode_h_, token_mask_len_h_;
    int num_nodes_ = 0;
    int num_edges_ = 0;                  // runtime edge count including self loops for current single input
    int token_len_ = 0;
    int max_edges_without_self_loop_ = 0;
    int input_edge_capacity_ = 0;        // max_edges_without_self_loop_ + num_nodes_

    // Config dims.
    int dim_ = 0, enc_in_dim_ = 0, n_layers_ = 0, n_heads_ = 0, kv_heads_ = 0, kv_dim_ = 0, head_dim_ = 0;
    int encoder_layers_ = 0;
    int ffn_dim_ = 0;
    int out_hidden_dim_ = 0;
    int vocab_ = 0;
    int encoder_label_vocab_ = 0;
    int token_vocab_ = 0;
    int subnode_vocab_ = 0;
    float norm_eps_ = 1e-5f;

    // Output.
    std::vector<int64_t> output_shape_;
    std::vector<float> output_host_;
    float* d_last_output_ = nullptr;

    // Device input buffers for single-sample compatibility path.
    int64_t *d_src_ = nullptr, *d_dst_ = nullptr, *d_feat_id_ = nullptr, *d_tokens_ = nullptr, *d_subnode_ = nullptr;

    // Work buffers reused across all forwards. Current implementation executes batched samples sequentially
    // over these buffers, while keeping inputs/weights on device.
    int* d_deg_ = nullptr;
    float *d_h_ = nullptr, *d_tmp_ = nullptr, *d_graph_ = nullptr, *d_masked_h_ = nullptr;
    float *d_q_ = nullptr, *d_k_ = nullptr, *d_v_ = nullptr, *d_scores_ = nullptr, *d_ctx_ = nullptr;
    float *d_ffn1_ = nullptr, *d_ffn3_ = nullptr, *d_ffn_hidden_ = nullptr;
    float *d_logits_ = nullptr;
    float *d_batch_logits_ = nullptr;
    int batch_capacity_ = 0;

    // Resident model weights.
    float* d_val_emb_ = nullptr;
    std::vector<float*> d_enc_w_, d_enc_b_;

    float *d_tok_ = nullptr, *d_node_ = nullptr, *d_type_ = nullptr, *d_pos_ = nullptr;
    float* d_norm_w_ = nullptr;

    std::vector<float*> d_attn_norm_w_, d_ffn_norm_w_;
    std::vector<float*> d_wq_, d_wk_, d_wv_, d_wo_;
    std::vector<float*> d_w1_, d_w2_, d_w3_;

    float *d_ow0_ = nullptr, *d_ob0_ = nullptr, *d_ow2_ = nullptr, *d_ob2_ = nullptr;

    void require_weight(const std::string& name) const;
    void clear_cuda();
    void allocate_work_buffers();
    void load_resident_weights();
    float* load_weight_to_device(const std::string& name);
    void ensure_batch_output_capacity(int batch_size);

    void forward_one_device(
        const int64_t* d_src,
        const int64_t* d_dst,
        int edge_count_with_self_loops,
        const int64_t* d_feat_id,
        const int64_t* d_tokens,
        const int64_t* d_subnode,
        int valid_rows,
        float* d_logits_out
    );
};
