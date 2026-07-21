/// @file torch/neighbor/neighbor_list.cu
/// @brief NeighborList for torch backend -- holds ATV::NeighborList.
///
/// The neighbor pair list is owned by `torch::NeighborList` (see the nested
/// `PairList`), not by `torch::Frame`. update() runs the CUDA pair search and
/// wraps the device-resident `cuda::PairList` into self-contained Tensors.

#ifdef ATV_HAS_TORCH

#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/torch/frame/transfer.cuh"
#include "atomiverse/torch/neighbor/neighbor_list.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace ATV {
namespace torch {

// -- NeighborList class -------------------------------------------------

NeighborList::NeighborList(double cutoff, double skin)
    : cutoff_(cutoff), skin_(skin) {}

NeighborList::~NeighborList() {
    if (built_ && nl_) {
        DestroyContext dctx{ctx_, cframe_};
        // Frees the CUDA NeighborList's owned PairList + internal buffers.
        nl_->destroy(dctx);
        // Free the cuda::Frame arrays allocated by cpu::to_cuda.
        cuda::destroy(cframe_);
    }
}

NeighborList::NeighborList(NeighborList&&) noexcept = default;
NeighborList& NeighborList::operator=(NeighborList&&) noexcept = default;

void NeighborList::build(Frame& frame) {
    // Set up CUDA context using the current Torch stream
    auto stream = ::c10::cuda::getCurrentCUDAStream();
    ctx_.stream = static_cast<void*>(stream.stream());
    ctx_.defer_stream = ctx_.stream;

    // Transfer torch::Frame -> cpu::Frame -> cuda::Frame (core allocator).
    // This ensures the cuda::Frame uses cudaMalloc, which the NeighborList
    // pipeline expects.
    cpu::Frame cpu_frame(frame.n_atoms());
    to_cpu(frame, cpu_frame);
    cframe_ = cpu::to_cuda(cpu_frame);

    // Create and build the CUDA NeighborList. build() allocates the
    // component-owned cuda::PairList and publishes it to ctx_.md.pair_list.
    nl_ = std::make_unique<ATV::NeighborList>(cutoff_, skin_);
    BuildContext bctx{ctx_, cframe_};
    nl_->build(bctx);

    built_ = true;
}

void NeighborList::update(Frame& frame) {
    if (!built_) {
        build(frame);
    }

    // Copy current positions from torch tensors to cuda::Frame arrays
    // (the cuda::Frame owns its own cudaMalloc memory)
    cudaStream_t stream = static_cast<cudaStream_t>(ctx_.stream);

    int n = frame.n_atoms();
    cudaMemcpyAsync(cframe_.atoms.x, frame.atoms.x.data_ptr<cuda::real_t>(),
                    n * sizeof(cuda::real_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cframe_.atoms.y, frame.atoms.y.data_ptr<cuda::real_t>(),
                    n * sizeof(cuda::real_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cframe_.atoms.z, frame.atoms.z.data_ptr<cuda::real_t>(),
                    n * sizeof(cuda::real_t), cudaMemcpyDeviceToDevice, stream);
    if (frame.box.defined() && cframe_.box) {
        cudaMemcpyAsync(cframe_.box, frame.box.data_ptr<cuda::prec_t>(),
                        9 * sizeof(cuda::prec_t), cudaMemcpyDeviceToDevice,
                        stream);
    }

    // Launch pair search. launch() re-publishes the owned pair list to
    // ctx_.md.pair_list.
    LaunchContext lctx{ctx_, cframe_, 0, 0};
    nl_->launch(lctx);

    cudaStreamSynchronize(stream);

    // Read the device-resident pair list owned by the CUDA NeighborList.
    const cuda::PairList& dpairs = nl_->pairs();

    // Read n_pairs from device
    int h_n_pairs = 0;
    if (dpairs.n_pairs) {
        cudaMemcpy(&h_n_pairs, dpairs.n_pairs, sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
    pairs_.n_pairs_ = h_n_pairs;

    // Wrap pair list as self-contained Tensors (cloned) into the owned
    // PairList. Cloning decouples the Tensors from the CUDA NeighborList's
    // buffers, which are freed in destroy().
    if (h_n_pairs > 0) {
        auto real_opts =
            ::torch::TensorOptions().dtype(kRealDtype).device(::torch::kCUDA);
        auto int_opts =
            ::torch::TensorOptions().dtype(kIntDtype).device(::torch::kCUDA);
        int64_t np = static_cast<int64_t>(h_n_pairs);

        pairs_.pair_i = ::torch::from_blob(dpairs.i, {np}, int_opts).clone();
        pairs_.pair_j = ::torch::from_blob(dpairs.j, {np}, int_opts).clone();
        pairs_.pair_dx = ::torch::from_blob(dpairs.dx, {np}, real_opts).clone();
        pairs_.pair_dy = ::torch::from_blob(dpairs.dy, {np}, real_opts).clone();
        pairs_.pair_dz = ::torch::from_blob(dpairs.dz, {np}, real_opts).clone();
        pairs_.pair_dist =
            ::torch::from_blob(dpairs.dist, {np}, real_opts).clone();
    } else {
        pairs_.pair_i = Tensor();
        pairs_.pair_j = Tensor();
        pairs_.pair_dx = Tensor();
        pairs_.pair_dy = Tensor();
        pairs_.pair_dz = Tensor();
        pairs_.pair_dist = Tensor();
    }
}

// -- Convenience free function ------------------------------------------

NeighborList neighbor_list(Frame& frame, double cutoff, double skin) {
    NeighborList nl(cutoff, skin);
    nl.build(frame);
    nl.update(frame);
    return nl;
}

} // namespace torch
} // namespace ATV

#endif // ATV_HAS_TORCH
