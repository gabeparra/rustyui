// cuDNN's scaled-dot-product attention with an additive bias, built as PyTorch 2.11 builds it
// (aten/src/ATen/native/cudnn/MHA.cpp: build_graph and run_cudnn_SDP_fprop, inference, no
// dropout, not causal) on the cudnn-frontend version it pins (v1.18.0, third_party/). ComfyUI's
// masked attention (LTXVAddGuide's guide mask) runs through torch's cuDNN backend, whose kernels
// cuDNN generates and compiles at run time, so the same graph on the same cuDNN is how rustyui
// computes what ComfyUI does.

#include <cudnn_frontend.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace {

// MHA.cpp's UIDS
enum Uid : int64_t { Q, K, V, O, BIAS, SCALE };

struct Built {
    std::unique_ptr<fe::graph::Graph> graph;
    int64_t workspace;
};

// one handle and one graph cache per thread, as torch keeps them
cudnnHandle_t handle(char *err, int err_len, bool &ok) {
    thread_local cudnnHandle_t h = nullptr;
    ok = true;
    if (!h) {
        cudnnStatus_t s = cudnnCreate(&h);
        if (s != CUDNN_STATUS_SUCCESS) {
            snprintf(err, err_len, "cudnnCreate: %s", cudnnGetErrorString(s));
            h = nullptr;
            ok = false;
        }
    }
    return h;
}

std::unordered_map<std::string, Built> &cache() {
    thread_local auto *c = new std::unordered_map<std::string, Built>();
    return *c;
}

std::vector<int64_t> vec4(const int64_t *p) { return {p[0], p[1], p[2], p[3]}; }

// fixSizeOneDimStrideSDPA: a size-1 dimension with stride 0 gets the product of the strides after it
std::vector<int64_t> fix_size_one(const int64_t *dims, const int64_t *strides) {
    std::vector<int64_t> s = vec4(strides);
    for (int d = 0; d < 4; d++) {
        if (dims[d] == 1 && !s[d]) {
            int64_t cur = 1;
            for (int d2 = d + 1; d2 < 4; d2++) cur *= s[d2];
            s[d] = cur;
        }
    }
    return s;
}

#define TRY(what, x)                                                                   \
    do {                                                                               \
        auto e_ = (x);                                                                 \
        if (e_.is_bad()) {                                                             \
            snprintf(err, err_len, "cudnn_frontend %s: %s", what, e_.get_message().c_str()); \
            return nullptr;                                                            \
        }                                                                              \
    } while (0)

Built *get(cudnnHandle_t h, bool bf16, const int64_t *q_dim, const int64_t *q_stride, const int64_t *k_dim,
           const int64_t *k_stride, const int64_t *v_dim, const int64_t *v_stride, const int64_t *o_dim,
           const int64_t *o_stride, const int64_t *bias_dim, const int64_t *bias_stride, char *err, int err_len) {
    // the cache key: what MHAParams holds (the scale goes by value at execution)
    std::string key = bf16 ? "bf16" : "f16";
    for (const int64_t *a : {q_dim, q_stride, k_dim, k_stride, v_dim, v_stride, o_dim, o_stride}) {
        for (int i = 0; i < 4; i++) key += "," + std::to_string(a[i]);
    }
    if (bias_dim) {
        key += "|bias";
        for (const int64_t *a : {bias_dim, bias_stride}) {
            for (int i = 0; i < 4; i++) key += "," + std::to_string(a[i]);
        }
    }
    auto it = cache().find(key);
    if (it != cache().end()) return &it->second;

    auto g = std::make_unique<fe::graph::Graph>();
    g->set_io_data_type(bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto scale = g->tensor(fe::graph::Tensor_attributes()
                               .set_uid(SCALE)
                               .set_name("Attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));
    auto opts = fe::graph::SDPA_attributes()
                    .set_name("CUDNN_SDPA")
                    .set_generate_stats(false)
                    .set_causal_mask(false)
                    .set_attn_scale(scale);
    auto q = g->tensor(fe::graph::Tensor_attributes().set_uid(Q).set_name("Q"));
    auto k = g->tensor(fe::graph::Tensor_attributes().set_uid(K).set_name("K"));
    auto v = g->tensor(fe::graph::Tensor_attributes().set_uid(V).set_name("V"));
    if (bias_dim) {
        // the bias as torch hands it over: its own (expanded) sizes and strides, unfixed
        opts.set_bias(g->tensor(fe::graph::Tensor_attributes()
                                    .set_uid(BIAS)
                                    .set_name("bias")
                                    .set_dim(vec4(bias_dim))
                                    .set_stride(vec4(bias_stride))));
    }
    auto [o, stats] = g->sdpa(q, k, v, opts);
    (void)stats;
    o->set_uid(O).set_output(true);
    q->set_dim(vec4(q_dim)).set_stride(fix_size_one(q_dim, q_stride));
    k->set_dim(vec4(k_dim)).set_stride(fix_size_one(k_dim, k_stride));
    v->set_dim(vec4(v_dim)).set_stride(fix_size_one(v_dim, v_stride));
    o->set_dim(vec4(o_dim)).set_stride(fix_size_one(o_dim, o_stride));

    TRY("validate", g->validate());
    TRY("build_operation_graph", g->build_operation_graph(h));
    TRY("create_execution_plans", g->create_execution_plans({fe::HeurMode_t::A}));
    TRY("check_support", g->check_support(h));
    TRY("build_plans", g->build_plans(h));
    int64_t ws = 0;
    TRY("get_workspace_size", g->get_workspace_size(ws));
    auto [slot, _] = cache().emplace(key, Built{std::move(g), ws});
    return &slot->second;
}

}  // namespace

extern "C" {

// The workspace this attention needs, building (and caching) its graph first. Dims and strides
// are (B, H, S, D) as torch's 4-D tensors carry them; bias_dim/bias_stride null for no bias.
// Returns 0, or -1 with the reason in err.
int rustyui_sdpa_workspace(int bf16, const int64_t *q_dim, const int64_t *q_stride, const int64_t *k_dim,
                           const int64_t *k_stride, const int64_t *v_dim, const int64_t *v_stride,
                           const int64_t *o_dim, const int64_t *o_stride, const int64_t *bias_dim,
                           const int64_t *bias_stride, int64_t *workspace, char *err, int err_len) {
    bool ok;
    cudnnHandle_t h = handle(err, err_len, ok);
    if (!ok) return -1;
    Built *b = get(h, bf16 != 0, q_dim, q_stride, k_dim, k_stride, v_dim, v_stride, o_dim, o_stride, bias_dim,
                   bias_stride, err, err_len);
    if (!b) return -1;
    *workspace = b->workspace;
    return 0;
}

// Runs the attention on `stream` (a CUstream), writing o. The workspace must hold what
// rustyui_sdpa_workspace said for the same shapes.
int rustyui_sdpa_forward(void *stream, int bf16, const int64_t *q_dim, const int64_t *q_stride,
                         const int64_t *k_dim, const int64_t *k_stride, const int64_t *v_dim,
                         const int64_t *v_stride, const int64_t *o_dim, const int64_t *o_stride,
                         const int64_t *bias_dim, const int64_t *bias_stride, float scale, void *q, void *k,
                         void *v, void *o, void *bias, void *workspace, char *err, int err_len) {
    bool ok;
    cudnnHandle_t h = handle(err, err_len, ok);
    if (!ok) return -1;
    Built *b = get(h, bf16 != 0, q_dim, q_stride, k_dim, k_stride, v_dim, v_stride, o_dim, o_stride, bias_dim,
                   bias_stride, err, err_len);
    if (!b) return -1;
    cudnnStatus_t s = cudnnSetStream(h, static_cast<cudaStream_t>(stream));
    if (s != CUDNN_STATUS_SUCCESS) {
        snprintf(err, err_len, "cudnnSetStream: %s", cudnnGetErrorString(s));
        return -1;
    }
    std::unordered_map<int64_t, void *> pack = {{Q, q}, {K, k}, {V, v}, {SCALE, &scale}, {O, o}};
    if (bias_dim) pack[BIAS] = bias;
    auto e = b->graph->execute(h, pack, workspace);
    if (e.is_bad()) {
        snprintf(err, err_len, "cudnn_frontend execute: %s", e.get_message().c_str());
        return -1;
    }
    return 0;
}

}  // extern "C"
