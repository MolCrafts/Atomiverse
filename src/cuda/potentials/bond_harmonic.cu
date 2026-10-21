/// @file cuda/potentials/bond_harmonic.cu
/// @brief CUDA BondHarmonic — V = K(r − r₀)² (kernel + class).

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/harmonic_bond_kernel.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace kernels {

__global__ __launch_bounds__(ATV::cuda::kDefaultBlock, 4) void harmonic_bond_k(
    const real_t* __restrict__ x, const real_t* __restrict__ y,
    const real_t* __restrict__ z, real_t* __restrict__ fx,
    real_t* __restrict__ fy, real_t* __restrict__ fz,
    prec_t* __restrict__ energy, prec_t* __restrict__ stress,
    const int* __restrict__ bi, const int* __restrict__ bj,
    const int* __restrict__ bt, const work_t* __restrict__ K,
    const work_t* __restrict__ r0, cuda::Box box, int n_bonds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-bond locals — initialised to zero so out-of-range threads
    // contribute nothing to the block-level energy and virial reductions
    // below. CRITICAL: an early-return for `idx >= n_bonds` would skip
    // the `__syncthreads()` calls inside the reduction, leaving the
    // `extern __shared__ sdata` slots of out-of-range threads
    // UNINITIALISED — the block reduction would then sum garbage into
    // `energy` (~20× over-count vs CPU/LAMMPS on stage 1 of the MSU PE
    // deck) and the same garbage into `stress[]` (catastrophic NPT
    // pressure spikes → barostat blowup → T → 0 in stage 2).
    work_t v = work_t(0);
    work_t my_v[6] = {work_t(0), work_t(0), work_t(0),
                      work_t(0), work_t(0), work_t(0)};

    if (idx < n_bonds) {
        int i = bi[idx], j = bj[idx], t = bt[idx];

        work_t dx = x[j] - x[i];
        work_t dy = y[j] - y[i];
        work_t dz = z[j] - z[i];

        // Minimum image convention via cached inv_l on the by-value
        // Box — no per-topology FP64 division
        // (cuda-md-review-cleanup-04-box-by-value-pbc).
        box.min_image(dx, dy, dz);

        work_t r = sqrt(dx * dx + dy * dy + dz * dz);
        work_t dr = r - r0[t];

        // V = K(r - r0)^2
        v = K[t] * dr * dr;

        // F = -2K(r - r0) * r_hat  (on atom j; opposite on atom i)
        work_t f_over_r = work_t(-2) * K[t] * dr / r;
        work_t f_x = f_over_r * dx;
        work_t f_y = f_over_r * dy;
        work_t f_z = f_over_r * dz;

        atomicAdd(&fx[j], static_cast<real_t>(f_x));
        atomicAdd(&fy[j], static_cast<real_t>(f_y));
        atomicAdd(&fz[j], static_cast<real_t>(f_z));
        atomicAdd(&fx[i], static_cast<real_t>(-f_x));
        atomicAdd(&fy[i], static_cast<real_t>(-f_y));
        atomicAdd(&fz[i], static_cast<real_t>(-f_z));

        // Bonded virial: w_ab = r_a · F_b (with our sign: dx = r_j - r_i,
        // F_b on atom j = f_over_r * d_ab). Filled here, reduced below.
        my_v[0] = dx * f_x;
        my_v[1] = dy * f_y;
        my_v[2] = dz * f_z;
        my_v[3] = dx * f_y;
        my_v[4] = dx * f_z;
        my_v[5] = dy * f_z;
    }

    // Packed warp-shuffle reduction: 7 scalars (energy + 6 stress
    // components) reduced via cg_reduce
    // (cuda-md-review-cleanup-05-reductions-and-atomics).
    extern __shared__ work_t warp_scratch[]; // per-warp staging cells
    work_t my_vals[7] = {v,       my_v[0], my_v[1], my_v[2],
                         my_v[3], my_v[4], my_v[5]};
    work_t out[7];
    block_reduce_sum_n<7>(my_vals, warp_scratch, out);
    if (threadIdx.x == 0) {
        atomicAdd(energy, static_cast<prec_t>(out[0]));
        if (stress != nullptr) {
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(&stress[c], static_cast<prec_t>(out[c + 1]));
        }
    }
}

} // namespace kernels
} // namespace cuda

namespace cuda {

struct BondHarmonic::State {
    work_t* d_K = nullptr;
    work_t* d_r0 = nullptr;
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
    const std::vector<work_t> device_K(K_.begin(), K_.end());
    const std::vector<work_t> device_r0(r0_.begin(), r0_.end());
    cudaMalloc(&state_->d_K, nt * sizeof(work_t));
    cudaMalloc(&state_->d_r0, nt * sizeof(work_t));
    cudaMemcpy(state_->d_K, device_K.data(), nt * sizeof(work_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(state_->d_r0, device_r0.data(), nt * sizeof(work_t),
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
    int shmem = warps_per_block(block) * sizeof(work_t);
    kernels::harmonic_bond_k<<<grid, block, shmem, stream>>>(
        f.atoms.x, f.atoms.y, f.atoms.z, f.atoms.fx, f.atoms.fy, f.atoms.fz,
        &f.scalars->energy, &f.scalars->stress[0], f.bonds.atomi, f.bonds.atomj,
        f.bonds.type, state_->d_K, state_->d_r0, ctx.runtime.host_box, nb);
}

void BondHarmonic::destroy(const DestroyContext& /*ctx*/) {
    cudaFree(state_->d_K);
    cudaFree(state_->d_r0);
    *state_ = {};
}

} // namespace cuda
} // namespace ATV
