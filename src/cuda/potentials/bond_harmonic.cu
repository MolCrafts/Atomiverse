/// @file cuda/potentials/bond_harmonic.cu
/// @brief CUDA BondHarmonic — V = K(r − r₀)² (kernel + class).

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {

namespace cuda {
namespace kernels {

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void harmonic_bond_k(
    const pos_t* __restrict__ x, const pos_t* __restrict__ y,
    const pos_t* __restrict__ z, fixed_t* __restrict__ fx,
    fixed_t* __restrict__ fy, fixed_t* __restrict__ fz, ForceFault* fault,
    acc_t* __restrict__ energy, acc_t* __restrict__ stress,
    const int* __restrict__ bi, const int* __restrict__ bj,
    const int* __restrict__ bt, const force_t* __restrict__ K,
    const force_t* __restrict__ r0, cuda::Box box_arg, int n_bonds) {
    const auto box = box_arg.current().as<force_t>();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tail lanes contribute zero and still participate in the CUB collective.
    acc_t v = 0;
    acc_t my_v[6] = {};
    DeferredForceAdds<6> adds;

    if (idx < n_bonds) {
        int i = bi[idx], j = bj[idx], t = bt[idx];

        force_t dx = x[j] - x[i];
        force_t dy = y[j] - y[i];
        force_t dz = z[j] - z[i];

        // Minimum image convention via cached inv_l on the by-value
        // Box — no per-topology division.
        box.min_image(dx, dy, dz);

        force_t r = sqrt(dx * dx + dy * dy + dz * dz);
        force_t dr = r - r0[t];

        // V = K(r - r0)^2
        v = acc_t(K[t] * dr * dr);

        // F = -2K(r - r0) * r_hat  (on atom j; opposite on atom i)
        force_t f_over_r = force_t(-2) * K[t] * dr / r;
        force_t f_x = f_over_r * dx;
        force_t f_y = f_over_r * dy;
        force_t f_z = f_over_r * dz;

        adds.add(0, fx, j, f_x, fault);
        adds.add(1, fy, j, f_y, fault);
        adds.add(2, fz, j, f_z, fault);
        adds.add(3, fx, i, -f_x, fault);
        adds.add(4, fy, i, -f_y, fault);
        adds.add(5, fz, i, -f_z, fault);

        // Bonded virial: w_ab = r_a · F_b (with our sign: dx = r_j - r_i,
        // F_b on atom j = f_over_r * d_ab). Filled here, reduced below.
        my_v[0] = acc_t(dx * f_x);
        my_v[1] = acc_t(dy * f_y);
        my_v[2] = acc_t(dz * f_z);
        my_v[3] = acc_t(dx * f_y);
        my_v[4] = acc_t(dx * f_z);
        my_v[5] = acc_t(dy * f_z);
    }

    // Reduce energy and six virial components in one CUB collective.
    ::cuda::std::array<acc_t, 7> my_vals = {v,       my_v[0], my_v[1], my_v[2],
                                            my_v[3], my_v[4], my_v[5]};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 7>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out =
        Reduce(reduction).Reduce(my_vals, SumComponents<acc_t, 7>{});
    if (threadIdx.x == 0) {
        atomicAdd(energy, out[0]);
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], out[c + 1]);
        }
    }
    adds.check(fault);
}

} // namespace kernels
} // namespace cuda

namespace cuda {

struct BondHarmonic::State {
    force_t* d_K = nullptr;
    force_t* d_r0 = nullptr;
    int n_types = 0;
};

BondHarmonic::BondHarmonic(std::vector<double> K, std::vector<double> r0)
    : K_(std::move(K)), r0_(std::move(r0)), state_(std::make_unique<State>()) {}

BondHarmonic::~BondHarmonic() = default;
BondHarmonic::BondHarmonic(BondHarmonic&&) noexcept = default;
BondHarmonic& BondHarmonic::operator=(BondHarmonic&&) noexcept = default;

void BondHarmonic::build(const BuildContext& /*ctx*/) {
    int nt = static_cast<int>(K_.size());
    state_->n_types = nt;
    const std::vector<force_t> device_K(K_.begin(), K_.end());
    const std::vector<force_t> device_r0(r0_.begin(), r0_.end());
    cudaMalloc(&state_->d_K, nt * sizeof(force_t));
    cudaMalloc(&state_->d_r0, nt * sizeof(force_t));
    cudaMemcpy(state_->d_K, device_K.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_r0, device_r0.data(), nt * sizeof(force_t),
               cudaMemcpyHostToDevice);
}

void BondHarmonic::compute(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& f = ctx.frame;
    int nb = f.n_bonds();
    if (nb == 0)
        return;

    int block = kDefaultBlock;
    int grid = grid_size(static_cast<int>(nb));
    constexpr int shmem = 0;
    kernels::harmonic_bond_k<<<grid, block, shmem, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        f.force_fault, &f.scalars->energy, &f.scalars->stress[0], f.bonds.atomi,
        f.bonds.atomj, f.bonds.type, state_->d_K, state_->d_r0,
        ctx.runtime.host_box, nb);
}

void BondHarmonic::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_K);
    cudaFree(state_->d_r0);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV
