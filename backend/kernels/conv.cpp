// cuDNN convolution forward as PyTorch 2.11 runs it without benchmarking
// (aten/src/ATen/native/cudnn/Conv_v8.cpp: build_opgraph, get_configs_from_heuristics, try_configs,
// run_conv_plan): one ConvolutionForward operation on tensors 'x', 'w', 'y' with their own
// alignments, a FLOAT compute type for half and bfloat16, the engine configs cuDNN's instant
// heuristics give (those that down-convert inputs left out), tried in order until one builds and
// runs, then its fallback list. ComfyUI leaves benchmarking off, so the plan torch settles on is
// the first of those that works, and the same graph on the same cuDNN settles on the same one.

#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// getAlignment: the largest power of two up to 32 that divides the address
uint8_t alignment_of(const void *p) {
    uintptr_t address = reinterpret_cast<uintptr_t>(p);
    uint8_t alignment = 1;
    for (; alignment < 32; alignment *= 2) {
        if (address % (alignment * 2)) return alignment;
    }
    return alignment;
}

// fixSizeOneDimStride (aten/src/ATen/cudnn/Descriptors.h): size-one dims get the stride their
// place in the memory order gives them, contiguous order (W, H, [D,] C, N) or channels-last
// (C, W, H, [D,] N)
std::vector<int64_t> fixed_strides(int dim, const int64_t *size, const int64_t *stride, bool nhwc) {
    std::vector<int64_t> s(stride, stride + dim);
    int64_t z = 1;
    std::vector<int> permutation;
    if (nhwc) permutation.push_back(1);
    for (int d = dim - 1; d > 1; d--) permutation.push_back(d);
    if (!nhwc) permutation.push_back(1);
    permutation.push_back(0);
    for (int d : permutation) {
        if (size[d] == 1) {
            s[d] = z;
        } else {
            z *= size[d];
        }
    }
    return s;
}

cudnn_frontend::Tensor tensor_desc(int nd, const int64_t *dims, const int64_t *strides, int64_t id, uint8_t alignment,
                                   cudnnDataType_t dt, bool nhwc) {
    auto s = fixed_strides(nd, dims, strides, nhwc);
    return cudnn_frontend::TensorBuilder()
        .setDim(nd, dims)
        .setStrides(nd, s.data())
        .setId(id)
        .setAlignment(alignment)
        .setDataType(dt)
        .setVirtual(false)
        .build();
}

// filterEngineConfigs for a non-deterministic run: engines that down-convert their inputs are left
// out (and, for float inputs, tensor-core engines only when TF32 is off, which ComfyUI never sets)
void filter_configs(cudnn_frontend::EngineConfigList &from, cudnn_frontend::EngineConfigList &to) {
    auto drop = [](cudnnBackendDescriptor_t c) {
        return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c);
    };
    cudnn_frontend::filter(from, to, drop);
}

// one cuDNN handle and plan cache per thread (torch keeps them per thread too)
struct State {
    cudnnHandle_t handle = nullptr;
    std::unordered_map<std::string, cudnn_frontend::ExecutionPlan> plans;
};

State &state() {
    thread_local State *s = new State();
    return *s;
}

// run_conv_plan: the data pointers by uid, the plan's workspace taken from the stream's pool for
// this call and given back after it (stream-ordered, as torch's caching allocator lends it), so no
// convolution keeps a large workspace for the rest of the process. `oom` says the card had no room
// for the workspace.
cudnnStatus_t run_plan(State &st, cudaStream_t stream, cudnn_frontend::ExecutionPlan &plan, void *x, void *y, void *w, bool &oom) {
    oom = false;
    int64_t need = plan.getWorkspaceSize();
    void *ws = nullptr;
    if (need > 0 && cudaMallocAsync(&ws, need, stream) != cudaSuccess) {
        cudaGetLastError();
        oom = true;
        return CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
    }
    void *ptrs[3] = {x, y, w};
    int64_t uids[3] = {'x', 'y', 'w'};
    auto pack = cudnn_frontend::VariantPackBuilder()
                    .setWorkspacePointer(ws)
                    .setDataPointers(3, ptrs)
                    .setUids(3, uids)
                    .build();
    cudnnStatus_t s = cudnnBackendExecute(st.handle, plan.get_raw_desc(), pack.get_raw_desc());
    if (ws) cudaFreeAsync(ws, stream);
    return s;
}

}  // namespace

extern "C" {

// op 0: y = conv(x, w); op 1: x = the backward-data convolution of y (a transposed convolution, as
// torch's cudnn_convolution_transpose runs it), for nd-dimensional (4: NCHW, 5: NCDHW) tensors in
// torch's layout description (sizes and strides in elements), padding, stride and dilation per
// spatial dimension, dtype 0 bfloat16, 1 half, 2 float, nhwc 1 when torch runs the convolution
// channels-last (its input or weight is). Groups follow from x's and w's channels.
// Returns 0, or -1 with the reason in err.
int rustyui_conv(void *stream_, int op_kind, int dtype, int nhwc, int nd, const int64_t *x_dim, const int64_t *x_stride, const int64_t *w_dim,
                 const int64_t *w_stride, const int64_t *y_dim, const int64_t *y_stride, const int64_t *pad,
                 const int64_t *stride, const int64_t *dil, void *x, void *w, void *y, char *err, int err_len) {
    const cudnnBackendDescriptorType_t kind = op_kind == 1 ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
                                                           : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    State &st = state();
    if (!st.handle) {
        cudnnStatus_t s = cudnnCreate(&st.handle);
        if (s != CUDNN_STATUS_SUCCESS) {
            snprintf(err, err_len, "cudnnCreate: %s", cudnnGetErrorString(s));
            st.handle = nullptr;
            return -1;
        }
    }
    cudnnStatus_t s = cudnnSetStream(st.handle, stream);
    if (s != CUDNN_STATUS_SUCCESS) {
        snprintf(err, err_len, "cudnnSetStream: %s", cudnnGetErrorString(s));
        return -1;
    }
    cudnnDataType_t dt = dtype == 0 ? CUDNN_DATA_BFLOAT16 : dtype == 1 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    int ns = nd - 2;
    uint8_t ax = alignment_of(x), aw = alignment_of(w), ay = alignment_of(y);

    // the cache key: what torch's CacheKey holds for a forward convolution
    std::string key = std::to_string(op_kind) + ":" + std::to_string(dtype) + ":" + std::to_string(nhwc) + ":" + std::to_string(nd);
    for (const int64_t *a : {x_dim, x_stride, w_dim, w_stride, y_dim, y_stride}) {
        for (int i = 0; i < nd; i++) key += "," + std::to_string(a[i]);
    }
    for (const int64_t *a : {pad, stride, dil}) {
        for (int i = 0; i < ns; i++) key += ";" + std::to_string(a[i]);
    }
    key += "|" + std::to_string(ax) + "," + std::to_string(aw) + "," + std::to_string(ay);

    // no room for a workspace is an out-of-memory error the caller can free memory for and retry;
    // moving on to an engine that needs less (torch's try_configs does) would change the plan, and
    // with it the result, for the rest of the process
    bool oom = false;
    auto out_of_memory = [&](cudnn_frontend::ExecutionPlan &plan) {
        snprintf(err, err_len, "out of memory: no room on the card for cuDNN's %lld-byte workspace", (long long)plan.getWorkspaceSize());
        return -1;
    };
    auto found = st.plans.find(key);
    if (found != st.plans.end()) {
        s = run_plan(st, stream, found->second, x, y, w, oom);
        if (s == CUDNN_STATUS_SUCCESS) return 0;
        if (oom) return out_of_memory(found->second);
        snprintf(err, err_len, "cuDNN convolution: %s", cudnnGetErrorString(s));
        return -1;
    }

    try {
        // build_opgraph
        auto conv = cudnn_frontend::ConvDescBuilder()
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setMathMode(CUDNN_CROSS_CORRELATION)
                        .setNDims(ns)
                        .setStrides(ns, stride)
                        .setPrePadding(ns, pad)
                        .setPostPadding(ns, pad)
                        .setDilation(ns, dil)
                        .build();
        auto op = cudnn_frontend::OperationBuilder(kind)
                      .setxDesc(tensor_desc(nd, x_dim, x_stride, 'x', ax, dt, nhwc != 0))
                      .setyDesc(tensor_desc(nd, y_dim, y_stride, 'y', ay, dt, nhwc != 0))
                      .setwDesc(tensor_desc(nd, w_dim, w_stride, 'w', aw, dt, nhwc != 0))
                      .setcDesc(conv)
                      .build();
        std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
        auto graph = cudnn_frontend::OperationGraphBuilder().setHandle(st.handle).setOperationGraph(ops.size(), ops.data()).build();
        std::string tag = graph.getTag();

        // get_configs_from_heuristics: instant-mode heuristics first, then the fallback list
        auto heuristics_source = [](cudnn_frontend::OperationGraph &g) -> cudnn_frontend::EngineConfigList {
            auto h = cudnn_frontend::EngineHeuristicsBuilder().setOperationGraph(g).setHeurMode(CUDNN_HEUR_MODE_INSTANT).build();
            auto &configs = h.getEngineConfig(h.getEngineConfigCount());
            cudnn_frontend::EngineConfigList kept;
            filter_configs(configs, kept);
            return kept;
        };
        auto fallback_source = [kind](cudnn_frontend::OperationGraph &g) -> cudnn_frontend::EngineConfigList {
            auto f = cudnn_frontend::EngineFallbackListBuilder()
                         .setOperationGraph(g)
                         .setOperation(kind)
                         .build();
            auto &list = f.getFallbackList();
            cudnn_frontend::EngineConfigList kept;
            filter_configs(list, kept);
            return kept;
        };
        for (int pass = 0; pass < 2; pass++) {
            std::vector<cudnn_frontend::GeneratorSource> sources;
            if (pass == 0) {
                sources.push_back(heuristics_source);
            } else {
                sources.push_back(fallback_source);
            }
            cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
            auto configs = generator.generate_engine_config(graph);
            // try_configs: the first that builds and runs is the plan
            for (auto &config : configs) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(st.handle).setEngineConfig(config, tag).build();
                    if (run_plan(st, stream, plan, x, y, w, oom) != CUDNN_STATUS_SUCCESS) {
                        if (oom) return out_of_memory(plan);
                        continue;
                    }
                    st.plans.emplace(key, std::move(plan));
                    return 0;
                } catch (cudnn_frontend::cudnnException &) {
                }
            }
        }
        snprintf(err, err_len, "cuDNN found no engine for this convolution");
        return -1;
    } catch (cudnn_frontend::cudnnException &e) {
        snprintf(err, err_len, "cudnn_frontend: %s", e.what());
        return -1;
    }
}

// The engine a cached plan runs (its tag), for checking which one was chosen; empty if none yet
int rustyui_conv_plans(char *out, int out_len) {
    std::string all;
    for (auto &[key, plan] : state().plans) all += plan.getTag() + "\n";
    snprintf(out, out_len, "%s", all.c_str());
    return static_cast<int>(state().plans.size());
}

}  // extern "C"
