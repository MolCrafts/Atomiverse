/// @file cuda/device/graph_ring.cuh
/// @brief Several executable instances of one CUDA graph, launched in turn.
///
/// A launch of an executable graph is ordered behind the previous launch of
/// the same instance, and the host call itself waits for that instance to
/// retire. Re-launching one instance every step therefore keeps the host at
/// most one step ahead, which is a host synchronization by another name
/// (iron law 10). Rotating over a few instances of the same graph keeps the
/// stream order (all launches go to one stream) while the host runs ahead.
/// A graph with conditional nodes cannot be instantiated twice; `build()`
/// captures one graph per slot for those.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cuda {

class GraphRing {
  public:
    static constexpr int kDepth = 3;

    GraphRing() = default;
    GraphRing(const GraphRing&) = delete;
    GraphRing& operator=(const GraphRing&) = delete;
    GraphRing(GraphRing&& o) noexcept { *this = static_cast<GraphRing&&>(o); }
    GraphRing& operator=(GraphRing&& o) noexcept {
        if (this != &o) {
            reset();
            for (int i = 0; i < kDepth; ++i) {
                exec_[i] = o.exec_[i];
                o.exec_[i] = nullptr;
            }
            next_ = o.next_;
        }
        return *this;
    }
    ~GraphRing() { reset(); }

    /// @brief Instantiate every slot from `graph` (the caller keeps it).
    void instantiate(cudaGraph_t graph) {
        reset();
        for (auto& exec : exec_) {
            const cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
            if (e != cudaSuccess) {
                exec = nullptr;
                reset();
                throw std::runtime_error(std::string("GraphRing: instantiate: ")
                                         + cudaGetErrorString(e));
            }
        }
    }

    /// @brief Instantiate every slot from its own graph, built by `make()`
    ///        once per slot. Graphs with conditional nodes may be
    ///        instantiated only once, so each slot needs a fresh capture.
    template <typename Make>
    void build(Make&& make) {
        reset();
        for (auto& exec : exec_) {
            cudaGraph_t graph = make();
            const cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
            cudaGraphDestroy(graph);
            if (e != cudaSuccess) {
                exec = nullptr;
                reset();
                throw std::runtime_error(std::string("GraphRing: instantiate: ")
                                         + cudaGetErrorString(e));
            }
        }
    }

    bool ready() const noexcept { return exec_[0] != nullptr; }

    /// @brief Launch the next instance on `stream`.
    cudaError_t launch(cudaStream_t stream) {
        const cudaError_t e = cudaGraphLaunch(exec_[next_], stream);
        next_ = (next_ + 1) % kDepth;
        return e;
    }

    void reset() noexcept {
        for (auto& exec : exec_)
            if (exec) {
                cudaGraphExecDestroy(exec);
                exec = nullptr;
            }
        next_ = 0;
    }

  private:
    cudaGraphExec_t exec_[kDepth] = {};
    int next_ = 0;
};

namespace graph_capture {

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("graph capture: ") + what + ": "
                                 + cudaGetErrorString(e));
}

/// @brief True while `stream` records into a graph (a step segment).
inline bool active(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &status), "status");
    return status == cudaStreamCaptureStatusActive;
}

/// @brief Make `node` the only tail of the capture on `stream`.
inline void set_tail(cudaStream_t stream, cudaGraphNode_t node) {
    check(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1,
                                              cudaStreamSetCaptureDependencies),
          "update dependencies");
}

/// @brief Embed a prebuilt graph into the capture on `stream` as a child
///        node, so a component with its own graph joins the step graph
///        instead of costing a separate launch.
inline void splice(cudaStream_t stream, cudaGraph_t child) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    std::size_t count = 0;
    check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps,
                                   nullptr, &count),
          "capture info");
    cudaGraphNode_t node = nullptr;
    check(cudaGraphAddChildGraphNode(&node, graph, deps, count, child),
          "child node");
    set_tail(stream, node);
}

/// @brief Append `gate(handle)` and an IF node to the capture on `stream`;
///        `body(side)` records the conditional body on the idle `side`
///        stream. The gate kernel sets the handle on the device.
template <typename Gate, typename Body>
void splice_if(cudaStream_t stream, cudaStream_t side, Gate&& gate,
               Body&& body) {
    cudaStreamCaptureStatus status;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    std::size_t count = 0;
    check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps,
                                   nullptr, &count),
          "capture info");
    cudaGraphConditionalHandle handle;
    check(cudaGraphConditionalHandleCreate(&handle, graph, 0,
                                           cudaGraphCondAssignDefault),
          "conditional handle");
    gate(handle);
    check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, &deps,
                                   nullptr, &count),
          "capture info");
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
    check(cudaGraphAddNode(&node, graph, deps, nullptr, count, &params),
          "conditional node");
    set_tail(stream, node);
    check(cudaStreamBeginCaptureToGraph(side, params.conditional.phGraph_out[0],
                                        nullptr, nullptr, 0,
                                        cudaStreamCaptureModeThreadLocal),
          "capture body");
    try {
        body(side);
    } catch (...) {
        cudaGraph_t discard = nullptr;
        cudaStreamEndCapture(side, &discard);
        throw;
    }
    cudaGraph_t captured = nullptr;
    check(cudaStreamEndCapture(side, &captured), "end body");
}

} // namespace graph_capture

} // namespace cuda
} // namespace ATV
