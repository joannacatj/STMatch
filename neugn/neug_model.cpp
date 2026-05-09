#include "neug_model.hpp"

#include "kernels.cuh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<int64_t> read_shape_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open shape file: " + path);
    std::string csv;
    std::getline(in, csv);
    return parse_shape_csv(csv);
}

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return static_cast<bool>(in);
}

int cfg_int(const std::unordered_map<std::string, std::string>& cfg, const std::string& key) {
    auto it = cfg.find(key);
    if (it == cfg.end()) throw std::runtime_error("Missing config key: " + key);
    return std::stoi(it->second);
}

float cfg_float(const std::unordered_map<std::string, std::string>& cfg, const std::string& key) {
    auto it = cfg.find(key);
    if (it == cfg.end()) throw std::runtime_error("Missing config key: " + key);
    return std::stof(it->second);
}

std::vector<float> load_weight_by_name(
    const std::unordered_map<std::string, TensorInfo>& manifest,
    const std::string& base,
    const std::string& name
) {
    auto it = manifest.find(name);
    if (it == manifest.end()) throw std::runtime_error("Missing required weight in manifest: " + name);
    return read_binary_float32(base + "/" + it->second.relative_path);
}

void checked_cuda(cudaError_t err, const std::string& where) {
    if (err != cudaSuccess) throw std::runtime_error(where + ": " + std::string(cudaGetErrorString(err)));
}

void upload_to_device(const std::vector<float>& h, float** d) {
    const size_t nbytes = h.size() * sizeof(float);
    if (nbytes == 0) {
        *d = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(d), nbytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error(
            "cudaMalloc failed for float upload (" + std::to_string(nbytes) + " bytes): " +
            std::string(cudaGetErrorString(err)) +
            ", free=" + std::to_string(free_b) + ", total=" + std::to_string(total_b)
        );
    }
    checked_cuda(cudaMemcpy(*d, h.data(), nbytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D failed for float upload");
}

void upload_or_realloc_i64(const std::vector<int64_t>& h, int64_t** d, size_t capacity, const std::string& name) {
    if (h.size() > capacity) {
        throw std::runtime_error(name + " size " + std::to_string(h.size()) + " exceeds capacity " + std::to_string(capacity));
    }
    if (!*d) {
        checked_cuda(cudaMalloc(reinterpret_cast<void**>(d), capacity * sizeof(int64_t)), "cudaMalloc " + name);
    }
    if (!h.empty()) {
        checked_cuda(cudaMemcpy(*d, h.data(), h.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "cudaMemcpy H2D " + name);
    }
}

std::vector<int64_t> make_self_looped(const std::vector<int64_t>& edge, int num_nodes) {
    std::vector<int64_t> out = edge;
    out.reserve(edge.size() + static_cast<size_t>(num_nodes));
    for (int i = 0; i < num_nodes; ++i) out.push_back(i);
    return out;
}

size_t checked_count_bytes(size_t count, size_t elem_size, const std::string& name) {
    if (count == 0) return 0;
    if (count > std::numeric_limits<size_t>::max() / elem_size) {
        throw std::runtime_error("Allocation size overflow for " + name);
    }
    return count * elem_size;
}

void checked_cuda_malloc(void** ptr, size_t nbytes, const std::string& name) {
    if (nbytes == 0) {
        *ptr = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(ptr, nbytes);
    if (err != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        throw std::runtime_error(
            "cudaMalloc failed for " + name + " (" + std::to_string(nbytes) + " bytes): " +
            std::string(cudaGetErrorString(err)) +
            ", free=" + std::to_string(free_b) + ", total=" + std::to_string(total_b)
        );
    }
}

void check_index_bounds(const std::vector<int64_t>& ids, int64_t limit, const std::string& name) {
    if (limit <= 0) {
        throw std::runtime_error("Invalid bound for " + name + ": " + std::to_string(limit));
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= limit) {
            throw std::runtime_error(
                name + " out of range at index " + std::to_string(i) +
                ": value=" + std::to_string(ids[i]) +
                ", expected in [0, " + std::to_string(limit - 1) + "]"
            );
        }
    }
}

void check_last_cuda_error(const std::string& where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(where + ": " + std::string(cudaGetErrorString(err)));
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 != shape.size()) s += ", ";
    }
    s += "]";
    return s;
}

void require_1d_shape(
    const std::unordered_map<std::string, TensorInfo>& manifest,
    const std::string& name,
    int64_t d0
) {
    auto it = manifest.find(name);
    if (it == manifest.end()) throw std::runtime_error("Missing required weight in manifest: " + name);
    const auto& s = it->second.shape;
    if (s.size() != 1 || s[0] != d0) {
        throw std::runtime_error(
            "Unsupported weight shape for " + name +
            ", expected [" + std::to_string(d0) + "] got " + shape_to_string(s)
        );
    }
}

void require_2d_shape(
    const std::unordered_map<std::string, TensorInfo>& manifest,
    const std::string& name,
    int64_t d0,
    int64_t d1
) {
    auto it = manifest.find(name);
    if (it == manifest.end()) throw std::runtime_error("Missing required weight in manifest: " + name);
    const auto& s = it->second.shape;
    if (s.size() != 2 || s[0] != d0 || s[1] != d1) {
        throw std::runtime_error(
            "Unsupported weight shape for " + name +
            ", expected [" + std::to_string(d0) + ", " + std::to_string(d1) + "] got " + shape_to_string(s)
        );
    }
}

void free_ptr(float*& p) {
    if (p) cudaFree(p);
    p = nullptr;
}

void free_ptr_i64(int64_t*& p) {
    if (p) cudaFree(p);
    p = nullptr;
}

void free_ptr_int(int*& p) {
    if (p) cudaFree(p);
    p = nullptr;
}

void free_vec(std::vector<float*>& v) {
    for (float*& p : v) free_ptr(p);
    v.clear();
}
}  // namespace

NeuGNCudaModel::~NeuGNCudaModel() {
    clear_cuda();
}

void NeuGNCudaModel::require_weight(const std::string& name) const {
    if (manifest_.find(name) == manifest_.end()) {
        throw std::runtime_error("Missing required weight in manifest: " + name);
    }
}

void NeuGNCudaModel::clear_cuda() {
    free_ptr_i64(d_src_);
    free_ptr_i64(d_dst_);
    free_ptr_i64(d_feat_id_);
    free_ptr_i64(d_tokens_);
    free_ptr_i64(d_subnode_);

    free_ptr_int(d_deg_);
    free_ptr(d_h_);
    free_ptr(d_tmp_);
    free_ptr(d_graph_);
    free_ptr(d_masked_h_);
    free_ptr(d_q_);
    free_ptr(d_k_);
    free_ptr(d_v_);
    free_ptr(d_scores_);
    free_ptr(d_ctx_);
    free_ptr(d_ffn1_);
    free_ptr(d_ffn3_);
    free_ptr(d_ffn_hidden_);
    free_ptr(d_logits_);
    free_ptr(d_batch_logits_);

    free_ptr(d_val_emb_);
    free_vec(d_enc_w_);
    free_vec(d_enc_b_);

    free_ptr(d_tok_);
    free_ptr(d_node_);
    free_ptr(d_type_);
    free_ptr(d_pos_);
    free_ptr(d_norm_w_);

    free_vec(d_attn_norm_w_);
    free_vec(d_ffn_norm_w_);
    free_vec(d_wq_);
    free_vec(d_wk_);
    free_vec(d_wv_);
    free_vec(d_wo_);
    free_vec(d_w1_);
    free_vec(d_w2_);
    free_vec(d_w3_);

    free_ptr(d_ow0_);
    free_ptr(d_ob0_);
    free_ptr(d_ow2_);
    free_ptr(d_ob2_);

    d_last_output_ = nullptr;
    batch_capacity_ = 0;
    output_host_.clear();
    output_shape_.clear();
    encoder_label_vocab_ = 0;
    token_vocab_ = 0;
    subnode_vocab_ = 0;
}

float* NeuGNCudaModel::load_weight_to_device(const std::string& name) {
    auto h = load_weight_by_name(manifest_, export_dir_, name);
    float* d = nullptr;
    upload_to_device(h, &d);
    return d;
}

void NeuGNCudaModel::load_model(const std::string& export_dir) {
    clear_cuda();
    cudaError_t init_err = cudaFree(0);
    if (init_err != cudaSuccess) {
        throw std::runtime_error("CUDA runtime initialization failed: " + std::string(cudaGetErrorString(init_err)));
    }

    export_dir_ = export_dir;
    config_ = parse_config_txt(export_dir + "/config.txt");
    manifest_ = parse_manifest_tsv(export_dir + "/manifest.tsv");

    if (config_.at("encoder_name") != "gcn") throw std::runtime_error("Only encoder_name=gcn supported");
    if (config_.at("decoder_type") != "llama") throw std::runtime_error("Only decoder_type=llama supported");

    num_nodes_ = cfg_int(config_, "num_nodes");
    token_len_ = cfg_int(config_, "token_len");
    dim_ = cfg_int(config_, "decoder_dim");
    n_layers_ = cfg_int(config_, "n_layers");
    n_heads_ = cfg_int(config_, "n_heads");
    encoder_layers_ = cfg_int(config_, "encoder_layers");
    norm_eps_ = cfg_float(config_, "norm_eps");
    vocab_ = cfg_int(config_, "output_dim");
    max_edges_without_self_loop_ = cfg_int(config_, "num_edges");
    input_edge_capacity_ = max_edges_without_self_loop_ + num_nodes_;

    if (num_nodes_ <= 0 || token_len_ <= 0 || dim_ <= 0 || n_layers_ <= 0 || encoder_layers_ <= 0) {
        throw std::runtime_error("Invalid model sizes in config.txt");
    }
    if (n_heads_ <= 0 || dim_ % n_heads_ != 0) {
        throw std::runtime_error(
            "Invalid attention dims: decoder_dim=" + std::to_string(dim_) +
            ", n_heads=" + std::to_string(n_heads_)
        );
    }
    head_dim_ = dim_ / n_heads_;
    kv_heads_ = n_heads_;
    kv_dim_ = dim_;

    // Required weights.
    require_weight("encoder.value_embedding.weight");
    require_weight("decoder.tok_embeddings.weight");
    require_weight("decoder.node_embeddings.ne");
    require_weight("decoder.type_embeddings.weight");
    require_weight("decoder.pos_embeddings.pe");
    require_weight("decoder.norm.weight");
    require_weight("decoder.output.0.weight");
    require_weight("decoder.output.0.bias");
    require_weight("decoder.output.2.weight");
    require_weight("decoder.output.2.bias");
    for (int i = 0; i < n_layers_; ++i) {
        const std::string p = "decoder.layers." + std::to_string(i) + ".";
        require_weight(p + "attention.wq.weight");
        require_weight(p + "attention.wk.weight");
        require_weight(p + "attention.wv.weight");
        require_weight(p + "attention.wo.weight");
        require_weight(p + "attention_norm.weight");
        require_weight(p + "ffn_norm.weight");
        require_weight(p + "feed_forward.w1.weight");
        require_weight(p + "feed_forward.w2.weight");
        require_weight(p + "feed_forward.w3.weight");
    }

    // Validate embeddings.
    encoder_label_vocab_ = static_cast<int>(manifest_.at("encoder.value_embedding.weight").shape.at(0));
    enc_in_dim_ = static_cast<int>(manifest_.at("encoder.value_embedding.weight").shape.at(1));
    token_vocab_ = static_cast<int>(manifest_.at("decoder.tok_embeddings.weight").shape.at(0));
    subnode_vocab_ = static_cast<int>(manifest_.at("decoder.node_embeddings.ne").shape.at(0));
    if (encoder_label_vocab_ <= 0 || token_vocab_ <= 0 || subnode_vocab_ <= 0) {
        throw std::runtime_error("Invalid embedding vocabulary sizes in manifest.tsv");
    }
    if (enc_in_dim_ <= 0) throw std::runtime_error("Invalid encoder embedding dim: " + std::to_string(enc_in_dim_));
    require_2d_shape(manifest_, "encoder.value_embedding.weight", encoder_label_vocab_, enc_in_dim_);
    require_2d_shape(manifest_, "decoder.tok_embeddings.weight", token_vocab_, dim_);
    require_2d_shape(manifest_, "decoder.node_embeddings.ne", subnode_vocab_, dim_);
    require_2d_shape(manifest_, "decoder.type_embeddings.weight", manifest_.at("decoder.type_embeddings.weight").shape.at(0), dim_);

    auto pos_it = manifest_.find("decoder.pos_embeddings.pe");
    const auto& pos_shape = pos_it->second.shape;
    int64_t pos_rows = -1;
    int64_t pos_dim = -1;
    if (pos_shape.size() == 2) {
        pos_rows = pos_shape[0];
        pos_dim = pos_shape[1];
    } else if (pos_shape.size() == 3 && pos_shape[0] == 1) {
        pos_rows = pos_shape[1];
        pos_dim = pos_shape[2];
    } else {
        throw std::runtime_error(
            "Unsupported shape for decoder.pos_embeddings.pe, expected [max_len, dim] or [1, max_len, dim], got " +
            shape_to_string(pos_shape)
        );
    }
    if (pos_dim != dim_) throw std::runtime_error("Unsupported positional embedding dim for decoder.pos_embeddings.pe");
    if (pos_rows < 1 + token_len_) {
        throw std::runtime_error(
            "Positional embedding too short: need at least " + std::to_string(1 + token_len_) +
            " rows, got " + std::to_string(pos_rows)
        );
    }

    // Infer kv projection width from layer 0.
    {
        const auto& wk0_shape = manifest_.at("decoder.layers.0.attention.wk.weight").shape;
        if (wk0_shape.size() != 2 || wk0_shape[1] != dim_) {
            throw std::runtime_error(
                "Unsupported wk shape at layer 0, expected [kv_dim, " + std::to_string(dim_) +
                "] got " + shape_to_string(wk0_shape)
            );
        }
        kv_dim_ = static_cast<int>(wk0_shape[0]);
        if (kv_dim_ <= 0 || kv_dim_ % head_dim_ != 0) {
            throw std::runtime_error(
                "Unsupported kv_dim=" + std::to_string(kv_dim_) +
                ", must be positive and divisible by head_dim=" + std::to_string(head_dim_)
            );
        }
        kv_heads_ = kv_dim_ / head_dim_;
        if (n_heads_ % kv_heads_ != 0) {
            throw std::runtime_error(
                "Unsupported grouped attention: n_heads=" + std::to_string(n_heads_) +
                " is not divisible by kv_heads=" + std::to_string(kv_heads_)
            );
        }
    }

    for (int i = 0; i < n_layers_; ++i) {
        const std::string p = "decoder.layers." + std::to_string(i) + ".";
        require_1d_shape(manifest_, p + "attention_norm.weight", dim_);
        require_1d_shape(manifest_, p + "ffn_norm.weight", dim_);
        require_2d_shape(manifest_, p + "attention.wq.weight", dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wk.weight", kv_dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wv.weight", kv_dim_, dim_);
        require_2d_shape(manifest_, p + "attention.wo.weight", dim_, dim_);
    }

    // Encoder layer widths.
    for (int l = 0; l < encoder_layers_; ++l) {
        std::string p = "encoder.convs." + std::to_string(l) + ".linear.";
        int in_dim = (l == 0) ? enc_in_dim_ : dim_;
        require_2d_shape(manifest_, p + "weight", dim_, in_dim);
        require_1d_shape(manifest_, p + "bias", dim_);
    }

    // FFN/output shapes.
    ffn_dim_ = static_cast<int>(manifest_.at("decoder.layers.0.feed_forward.w1.weight").shape[0]);
    if (ffn_dim_ <= 0) throw std::runtime_error("Invalid FFN dim");
    for (int l = 0; l < n_layers_; ++l) {
        const std::string p = "decoder.layers." + std::to_string(l) + ".";
        require_2d_shape(manifest_, p + "feed_forward.w1.weight", ffn_dim_, dim_);
        require_2d_shape(manifest_, p + "feed_forward.w3.weight", ffn_dim_, dim_);
        require_2d_shape(manifest_, p + "feed_forward.w2.weight", dim_, ffn_dim_);
    }
    require_1d_shape(manifest_, "decoder.norm.weight", dim_);
    out_hidden_dim_ = static_cast<int>(manifest_.at("decoder.output.0.bias").shape[0]);
    require_2d_shape(manifest_, "decoder.output.0.weight", out_hidden_dim_, dim_);
    require_1d_shape(manifest_, "decoder.output.0.bias", out_hidden_dim_);
    require_2d_shape(manifest_, "decoder.output.2.weight", vocab_, out_hidden_dim_);
    require_1d_shape(manifest_, "decoder.output.2.bias", vocab_);

    allocate_work_buffers();
    load_resident_weights();

    if (file_exists(export_dir + "/python_output.shape")) {
        output_shape_ = read_shape_file(export_dir + "/python_output.shape");
    }
}

void NeuGNCudaModel::load(const std::string& export_dir) {
    load_model(export_dir);

    auto edge_all = read_binary_int64(export_dir + "/input/graph_edge_index.bin");
    if (edge_all.size() % 2 != 0) throw std::runtime_error("graph_edge_index.bin invalid");
    int raw_edges = static_cast<int>(edge_all.size() / 2);
    std::vector<int64_t> src(edge_all.begin(), edge_all.begin() + raw_edges);
    std::vector<int64_t> dst(edge_all.begin() + raw_edges, edge_all.end());

    auto feat_id = read_binary_int64(export_dir + "/input/graph_feat_id.bin");
    auto tokens = read_binary_int64(export_dir + "/input/tokens.bin");
    auto subnode = read_binary_int64(export_dir + "/input/subnode_ids.bin");
    auto token_mask_len = read_binary_int64(export_dir + "/input/token_mask_len.bin");
    set_input_from_host(src, dst, feat_id, tokens, subnode, token_mask_len);
}

void NeuGNCudaModel::allocate_work_buffers() {
    // Single-sample input buffers.
    checked_cuda_malloc(reinterpret_cast<void**>(&d_src_), checked_count_bytes(static_cast<size_t>(input_edge_capacity_), sizeof(int64_t), "d_src_"), "d_src_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_dst_), checked_count_bytes(static_cast<size_t>(input_edge_capacity_), sizeof(int64_t), "d_dst_"), "d_dst_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_feat_id_), checked_count_bytes(static_cast<size_t>(num_nodes_), sizeof(int64_t), "d_feat_id_"), "d_feat_id_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_tokens_), checked_count_bytes(static_cast<size_t>(token_len_), sizeof(int64_t), "d_tokens_"), "d_tokens_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_subnode_), checked_count_bytes(static_cast<size_t>(token_len_), sizeof(int64_t), "d_subnode_"), "d_subnode_");

    checked_cuda_malloc(reinterpret_cast<void**>(&d_deg_), checked_count_bytes(static_cast<size_t>(num_nodes_), sizeof(int), "d_deg_"), "d_deg_");
    const int encoder_work_dim = std::max(dim_, enc_in_dim_);
    const size_t encoder_elems = static_cast<size_t>(num_nodes_) * static_cast<size_t>(encoder_work_dim);
    const size_t decoder_elems = static_cast<size_t>(1 + token_len_) * static_cast<size_t>(dim_);
    const size_t tmp_elems = std::max(encoder_elems, decoder_elems);
    checked_cuda_malloc(reinterpret_cast<void**>(&d_h_), checked_count_bytes(encoder_elems, sizeof(float), "d_h_"), "d_h_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_tmp_), checked_count_bytes(tmp_elems, sizeof(float), "d_tmp_"), "d_tmp_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_graph_), checked_count_bytes(static_cast<size_t>(dim_), sizeof(float), "d_graph_"), "d_graph_");

    int seq = 1 + token_len_;
    checked_cuda_malloc(reinterpret_cast<void**>(&d_masked_h_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_masked_h_"), "d_masked_h_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_q_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_q_"), "d_q_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_k_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(kv_dim_), sizeof(float), "d_k_"), "d_k_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_v_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(kv_dim_), sizeof(float), "d_v_"), "d_v_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ctx_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(dim_), sizeof(float), "d_ctx_"), "d_ctx_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_scores_), checked_count_bytes(static_cast<size_t>(n_heads_) * static_cast<size_t>(seq) * static_cast<size_t>(seq), sizeof(float), "d_scores_"), "d_scores_");

    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn1_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim_), sizeof(float), "d_ffn1_"), "d_ffn1_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn3_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim_), sizeof(float), "d_ffn3_"), "d_ffn3_");
    checked_cuda_malloc(reinterpret_cast<void**>(&d_ffn_hidden_), checked_count_bytes(static_cast<size_t>(seq) * static_cast<size_t>(ffn_dim_), sizeof(float), "d_ffn_hidden_"), "d_ffn_hidden_");

    checked_cuda_malloc(reinterpret_cast<void**>(&d_logits_), checked_count_bytes(static_cast<size_t>(vocab_), sizeof(float), "d_logits_"), "d_logits_");
}

void NeuGNCudaModel::load_resident_weights() {
    d_val_emb_ = load_weight_to_device("encoder.value_embedding.weight");

    d_enc_w_.resize(encoder_layers_, nullptr);
    d_enc_b_.resize(encoder_layers_, nullptr);
    for (int l = 0; l < encoder_layers_; ++l) {
        const std::string p = "encoder.convs." + std::to_string(l) + ".linear.";
        d_enc_w_[l] = load_weight_to_device(p + "weight");
        d_enc_b_[l] = load_weight_to_device(p + "bias");
    }

    d_tok_ = load_weight_to_device("decoder.tok_embeddings.weight");
    d_node_ = load_weight_to_device("decoder.node_embeddings.ne");
    d_type_ = load_weight_to_device("decoder.type_embeddings.weight");
    d_pos_ = load_weight_to_device("decoder.pos_embeddings.pe");
    d_norm_w_ = load_weight_to_device("decoder.norm.weight");

    d_attn_norm_w_.resize(n_layers_, nullptr);
    d_ffn_norm_w_.resize(n_layers_, nullptr);
    d_wq_.resize(n_layers_, nullptr);
    d_wk_.resize(n_layers_, nullptr);
    d_wv_.resize(n_layers_, nullptr);
    d_wo_.resize(n_layers_, nullptr);
    d_w1_.resize(n_layers_, nullptr);
    d_w2_.resize(n_layers_, nullptr);
    d_w3_.resize(n_layers_, nullptr);
    for (int l = 0; l < n_layers_; ++l) {
        const std::string p = "decoder.layers." + std::to_string(l) + ".";
        d_attn_norm_w_[l] = load_weight_to_device(p + "attention_norm.weight");
        d_ffn_norm_w_[l] = load_weight_to_device(p + "ffn_norm.weight");
        d_wq_[l] = load_weight_to_device(p + "attention.wq.weight");
        d_wk_[l] = load_weight_to_device(p + "attention.wk.weight");
        d_wv_[l] = load_weight_to_device(p + "attention.wv.weight");
        d_wo_[l] = load_weight_to_device(p + "attention.wo.weight");
        d_w1_[l] = load_weight_to_device(p + "feed_forward.w1.weight");
        d_w2_[l] = load_weight_to_device(p + "feed_forward.w2.weight");
        d_w3_[l] = load_weight_to_device(p + "feed_forward.w3.weight");
    }

    d_ow0_ = load_weight_to_device("decoder.output.0.weight");
    d_ob0_ = load_weight_to_device("decoder.output.0.bias");
    d_ow2_ = load_weight_to_device("decoder.output.2.weight");
    d_ob2_ = load_weight_to_device("decoder.output.2.bias");
}

void NeuGNCudaModel::set_input_from_host_edge_index(
    const std::vector<int64_t>& edge_index_2row_flat,
    const std::vector<int64_t>& feat_id,
    const std::vector<int64_t>& tokens,
    const std::vector<int64_t>& subnode_ids,
    const std::vector<int64_t>& token_mask_len
) {
    if (edge_index_2row_flat.size() % 2 != 0) throw std::runtime_error("edge_index_2row_flat size must be even");
    int e = static_cast<int>(edge_index_2row_flat.size() / 2);
    std::vector<int64_t> src(edge_index_2row_flat.begin(), edge_index_2row_flat.begin() + e);
    std::vector<int64_t> dst(edge_index_2row_flat.begin() + e, edge_index_2row_flat.end());
    set_input_from_host(src, dst, feat_id, tokens, subnode_ids, token_mask_len);
}

void NeuGNCudaModel::set_input_from_host(
    const std::vector<int64_t>& edge_src_no_self_loop,
    const std::vector<int64_t>& edge_dst_no_self_loop,
    const std::vector<int64_t>& feat_id,
    const std::vector<int64_t>& tokens,
    const std::vector<int64_t>& subnode_ids,
    const std::vector<int64_t>& token_mask_len
) {
    if (edge_src_no_self_loop.size() != edge_dst_no_self_loop.size()) {
        throw std::runtime_error("edge_src and edge_dst size mismatch");
    }
    if (static_cast<int>(edge_src_no_self_loop.size()) > max_edges_without_self_loop_) {
        throw std::runtime_error(
            "Too many query edges: got " + std::to_string(edge_src_no_self_loop.size()) +
            ", config capacity=" + std::to_string(max_edges_without_self_loop_)
        );
    }
    if (feat_id.size() != static_cast<size_t>(num_nodes_)) {
        throw std::runtime_error("graph_feat_id length mismatch: expected " + std::to_string(num_nodes_) + ", got " + std::to_string(feat_id.size()));
    }
    if (tokens.size() != static_cast<size_t>(token_len_)) throw std::runtime_error("tokens length mismatch");
    if (subnode_ids.size() != static_cast<size_t>(token_len_)) throw std::runtime_error("subnode_ids length mismatch");
    if (token_mask_len.empty()) throw std::runtime_error("token_mask_len is empty");

    const int64_t value_vocab = manifest_.at("encoder.value_embedding.weight").shape.at(0);
    const int64_t token_vocab = manifest_.at("decoder.tok_embeddings.weight").shape.at(0);
    const int64_t subnode_vocab = manifest_.at("decoder.node_embeddings.ne").shape.at(0);
    check_index_bounds(feat_id, value_vocab, "graph_feat_id");
    check_index_bounds(tokens, token_vocab, "tokens");
    check_index_bounds(subnode_ids, subnode_vocab, "subnode_ids");
    check_index_bounds(edge_src_no_self_loop, num_nodes_, "edge_src");
    check_index_bounds(edge_dst_no_self_loop, num_nodes_, "edge_dst");

    edge_src_h_ = make_self_looped(edge_src_no_self_loop, num_nodes_);
    edge_dst_h_ = make_self_looped(edge_dst_no_self_loop, num_nodes_);
    feat_id_h_ = feat_id;
    tokens_h_ = tokens;
    subnode_h_ = subnode_ids;
    token_mask_len_h_ = token_mask_len;
    num_edges_ = static_cast<int>(edge_src_h_.size());

    upload_or_realloc_i64(edge_src_h_, &d_src_, static_cast<size_t>(input_edge_capacity_), "d_src_");
    upload_or_realloc_i64(edge_dst_h_, &d_dst_, static_cast<size_t>(input_edge_capacity_), "d_dst_");
    upload_or_realloc_i64(feat_id_h_, &d_feat_id_, static_cast<size_t>(num_nodes_), "d_feat_id_");
    upload_or_realloc_i64(tokens_h_, &d_tokens_, static_cast<size_t>(token_len_), "d_tokens_");
    upload_or_realloc_i64(subnode_h_, &d_subnode_, static_cast<size_t>(token_len_), "d_subnode_");
}

void NeuGNCudaModel::forward_one_device(
    const int64_t* d_src,
    const int64_t* d_dst,
    int edge_count_with_self_loops,
    const int64_t* d_feat_id,
    const int64_t* d_tokens,
    const int64_t* d_subnode,
    int valid_rows,
    float* d_logits_out
) {
    if (!d_val_emb_) throw std::runtime_error("Model weights are not loaded. Call load_model/load first.");
    if (!d_src || !d_dst || !d_feat_id || !d_tokens || !d_subnode) throw std::runtime_error("Null device input pointer");
    if (edge_count_with_self_loops <= 0) throw std::runtime_error("edge_count_with_self_loops must be positive");

    const int seq = 1 + token_len_;
    valid_rows = std::max(0, std::min(valid_rows, seq));

    // ----- Encoder: GCN -----
    launch_embedding_lookup_kernel(d_feat_id, d_val_emb_, d_h_, num_nodes_, enc_in_dim_);
    check_last_cuda_error("encoder.value_embedding lookup");

    for (int l = 0; l < encoder_layers_; ++l) {
        const int layer_in_dim = (l == 0) ? enc_in_dim_ : dim_;
        launch_zero_int(d_deg_, num_nodes_);
        launch_degree_kernel(d_dst, d_deg_, edge_count_with_self_loops);
        launch_zero_float(d_tmp_, num_nodes_ * layer_in_dim);
        launch_gcn_aggregate_kernel(d_src, d_dst, d_deg_, d_h_, d_tmp_, edge_count_with_self_loops, layer_in_dim);
        launch_linear_kernel(d_tmp_, d_enc_w_[l], d_enc_b_[l], d_h_, num_nodes_, layer_in_dim, dim_);
        launch_relu_kernel(d_h_, num_nodes_ * dim_);
        check_last_cuda_error("encoder.gcn layer " + std::to_string(l));
    }

    launch_max_pool_kernel(d_h_, d_graph_, num_nodes_, dim_);
    check_last_cuda_error("encoder.max_pool");

    // ----- Build decoder input h [seq, dim] -----
    launch_embedding_lookup_kernel(d_tokens, d_tok_, d_masked_h_ + dim_, token_len_, dim_);
    launch_embedding_lookup_kernel(d_subnode, d_node_, d_tmp_, token_len_, dim_);
    launch_add_inplace_kernel(d_masked_h_ + dim_, d_tmp_, token_len_ * dim_);
    launch_add_row_vector_inplace(d_masked_h_ + dim_, d_type_ + 0 * dim_, token_len_, dim_);

    launch_copy_kernel(d_graph_, d_masked_h_, dim_);
    launch_add_inplace_kernel(d_masked_h_, d_type_ + 1 * dim_, dim_);
    launch_add_inplace_kernel(d_masked_h_, d_pos_, seq * dim_);
    check_last_cuda_error("decoder.input embedding build");

    // ----- Transformer layers -----
    for (int l = 0; l < n_layers_; ++l) {
        launch_rmsnorm_kernel(d_masked_h_, d_attn_norm_w_[l], d_tmp_, seq, dim_, norm_eps_);

        launch_linear_kernel(d_tmp_, d_wq_[l], nullptr, d_q_, seq, dim_, dim_);
        launch_linear_kernel(d_tmp_, d_wk_[l], nullptr, d_k_, seq, dim_, kv_dim_);
        launch_linear_kernel(d_tmp_, d_wv_[l], nullptr, d_v_, seq, dim_, kv_dim_);

        launch_attention_scores_kernel(d_q_, d_k_, d_scores_, seq, n_heads_, kv_heads_, head_dim_);
        launch_attention_mask_row_kernel(d_scores_, seq, n_heads_, valid_rows);
        launch_softmax_rows_kernel(d_scores_, n_heads_ * seq, seq);
        launch_attention_weighted_sum_kernel(d_scores_, d_v_, d_ctx_, seq, n_heads_, kv_heads_, head_dim_);

        launch_linear_kernel(d_ctx_, d_wo_[l], nullptr, d_tmp_, seq, dim_, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);
        check_last_cuda_error("decoder.attention layer " + std::to_string(l));

        launch_rmsnorm_kernel(d_masked_h_, d_ffn_norm_w_[l], d_tmp_, seq, dim_, norm_eps_);
        launch_linear_kernel(d_tmp_, d_w1_[l], nullptr, d_ffn1_, seq, dim_, ffn_dim_);
        launch_linear_kernel(d_tmp_, d_w3_[l], nullptr, d_ffn3_, seq, dim_, ffn_dim_);
        launch_copy_kernel(d_ffn1_, d_ffn_hidden_, seq * ffn_dim_);
        launch_silu_mul_kernel(d_ffn_hidden_, d_ffn3_, seq * ffn_dim_);
        launch_linear_kernel(d_ffn_hidden_, d_w2_[l], nullptr, d_tmp_, seq, ffn_dim_, dim_);
        launch_add_inplace_kernel(d_masked_h_, d_tmp_, seq * dim_);
        check_last_cuda_error("decoder.ffn layer " + std::to_string(l));
    }

    launch_rmsnorm_kernel(d_masked_h_, d_norm_w_, d_tmp_, seq, dim_, norm_eps_);
    check_last_cuda_error("decoder.final_norm");

    launch_copy_row_kernel(d_tmp_, d_graph_, 1, dim_);
    check_last_cuda_error("decoder.select_row");

    launch_linear_kernel(d_graph_, d_ow0_, d_ob0_, d_ffn1_, 1, dim_, out_hidden_dim_);
    launch_gelu_kernel(d_ffn1_, out_hidden_dim_);
    launch_linear_kernel(d_ffn1_, d_ow2_, d_ob2_, d_logits_out, 1, out_hidden_dim_, vocab_);
    check_last_cuda_error("decoder.output_mlp");
}

void NeuGNCudaModel::forward_full_model() {
    if (!d_src_ || !d_dst_ || !d_feat_id_ || !d_tokens_ || !d_subnode_ || token_mask_len_h_.empty()) {
        throw std::runtime_error("No input available. Call load() or set_input_from_host() first.");
    }
    const int valid_rows = static_cast<int>(token_mask_len_h_[0]) + 1;
    forward_one_device(d_src_, d_dst_, num_edges_, d_feat_id_, d_tokens_, d_subnode_, valid_rows, d_logits_);

    checked_cuda(cudaDeviceSynchronize(), "CUDA forward failed");
    output_host_.resize(static_cast<size_t>(vocab_));
    checked_cuda(cudaMemcpy(output_host_.data(), d_logits_, static_cast<size_t>(vocab_) * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output failed");
    output_shape_ = {1, 1, vocab_};
    d_last_output_ = d_logits_;
}

void NeuGNCudaModel::ensure_batch_output_capacity(int batch_size) {
    if (batch_size <= batch_capacity_ && d_batch_logits_) return;
    free_ptr(d_batch_logits_);
    checked_cuda_malloc(reinterpret_cast<void**>(&d_batch_logits_), checked_count_bytes(static_cast<size_t>(batch_size) * static_cast<size_t>(vocab_), sizeof(float), "d_batch_logits_"), "d_batch_logits_");
    batch_capacity_ = batch_size;
}

void NeuGNCudaModel::forward_batch_from_device(
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
    bool copy_to_host
) {
    if (batch_size <= 0) throw std::runtime_error("batch_size must be positive");
    if (!d_edge_src_packed || !d_edge_dst_packed || !d_feat_id_packed || !d_tokens_packed || !d_subnode_packed) {
        throw std::runtime_error("Null device input pointer for batched forward");
    }
    if (static_cast<int>(edge_counts_with_self_loops.size()) != batch_size) {
        throw std::runtime_error("edge_counts_with_self_loops size mismatch");
    }
    if (static_cast<int>(token_mask_lens.size()) != batch_size) {
        throw std::runtime_error("token_mask_lens size mismatch");
    }
    if (edge_stride <= 0 || feat_stride < num_nodes_ || token_stride < token_len_ || subnode_stride < token_len_) {
        throw std::runtime_error("Invalid packed input stride");
    }

    ensure_batch_output_capacity(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        const int edge_count = edge_counts_with_self_loops[b];
        if (edge_count <= 0 || edge_count > edge_stride) {
            throw std::runtime_error("Invalid edge_count for batch item " + std::to_string(b));
        }
        const int valid_rows = token_mask_lens[b] + 1;
        const int64_t* src = d_edge_src_packed + static_cast<size_t>(b) * static_cast<size_t>(edge_stride);
        const int64_t* dst = d_edge_dst_packed + static_cast<size_t>(b) * static_cast<size_t>(edge_stride);
        const int64_t* feat = d_feat_id_packed + static_cast<size_t>(b) * static_cast<size_t>(feat_stride);
        const int64_t* toks = d_tokens_packed + static_cast<size_t>(b) * static_cast<size_t>(token_stride);
        const int64_t* sub = d_subnode_packed + static_cast<size_t>(b) * static_cast<size_t>(subnode_stride);
        float* out = d_batch_logits_ + static_cast<size_t>(b) * static_cast<size_t>(vocab_);
        forward_one_device(src, dst, edge_count, feat, toks, sub, valid_rows, out);
    }

    checked_cuda(cudaDeviceSynchronize(), "CUDA batched forward failed");
    d_last_output_ = d_batch_logits_;
    output_shape_ = {batch_size, 1, vocab_};

    if (copy_to_host) {
        output_host_.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(vocab_));
        checked_cuda(cudaMemcpy(output_host_.data(), d_batch_logits_, output_host_.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy batched output failed");
    }
}

void NeuGNCudaModel::save_output(const std::string& path) const {
    if (output_host_.empty()) throw std::runtime_error("No host output available. Run forward with copy_to_host=true first.");
    write_binary_float32(path, output_host_);
}

std::vector<float> NeuGNCudaModel::first_values(int k) const {
    int n = std::min<int>(k, static_cast<int>(output_host_.size()));
    return std::vector<float>(output_host_.begin(), output_host_.begin() + n);
}
