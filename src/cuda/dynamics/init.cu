/// @file cuda/dynamics/init.cu
/// @brief Initializer implementations: ZeroForces, ZeroEnergy, ZeroKinetic, InitVelocity.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/dynamics/kinetic_energy.cuh"
#include <cublas_v2.h>
#include <memory>
#include <type_traits>

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ATV {

namespace cuda {
namespace md {

// ── Zero initializers ────────────────────────────────────────────────

void ZeroForces::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    if (n > 0) {
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(fixed_t);
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fx, 0, bytes, stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fy, 0, bytes, stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.atoms.fz, 0, bytes, stream));
    }
}

void ZeroEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Zero `energy` + 4 per-term fields. The pair/bonded virial lives on
    // `ZeroStress` (same split as `cpu::ZeroEnergy` / `cpu::ZeroStress`).
    cudaMemsetAsync(&frame.scalars->energy, 0, 5 * sizeof(double), stream);
}

void ZeroStress::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Without this clear, force kernels (LJ + bond/angle/dihedral)
    // accumulate virial across steps and `NoseHooverNPT` reads runaway
    // pressure.
    cudaMemsetAsync(&frame.scalars->stress[0], 0, 6 * sizeof(double), stream);
}

void ZeroKinetic::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    cudaMemsetAsync(&frame.scalars->kinetic, 0, sizeof(acc_t), stream);
}

// ── InitVelocity ─────────────────────────────────────────────────────

namespace {

/// @brief Generate Gaussian-distributed velocities at target temperature.
///
/// v_i ~ N(0, sigma) where sigma = sqrt(kB*T / (ATV::units::mass_v2_to_energy * m_i)).
/// Uses cuRAND device API (per-thread state, Philox generator).
__global__ void
generate_velocities(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                    mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
                    const std::int64_t* __restrict__ tag,
                    mixed_t kbt_over_mvv2e, unsigned long long seed,
                    const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(tag[i]), 0, &state);
    const double2 xy = curand_normal2_double(&state);
    const double2 zw = curand_normal2_double(&state);
    const mixed_t sigma = sqrt(kbt_over_mvv2e / mass[i]);
    vx[i] = mixed_t(xy.x) * sigma;
    vy[i] = mixed_t(xy.y) * sigma;
    vz[i] = mixed_t(zw.x) * sigma;
}

/// @brief CUB block sum of momentum/mass, one atomic per component/block.
__global__ void accumulate_momentum(
    const mixed_t* __restrict__ vx, const mixed_t* __restrict__ vy,
    const mixed_t* __restrict__ vz, const mixed_t* __restrict__ mass,
    acc_t* __restrict__ px, acc_t* __restrict__ py, acc_t* __restrict__ pz,
    acc_t* __restrict__ total_mass, const int* __restrict__ n_ptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    using Values = ::cuda::std::array<acc_t, 4>;
    using Reduce = cub::BlockReduce<Values, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    Values values{};
    if (i < *n_ptr) {
        const acc_t m = static_cast<acc_t>(mass[i]);
        values = {m * acc_t(vx[i]), m * acc_t(vy[i]), m * acc_t(vz[i]), m};
    }
    const auto sum =
        Reduce(reduction).Reduce(values, SumComponents<acc_t, 4>{});
    if (threadIdx.x == 0) {
        atomicAdd(px, sum[0]);
        atomicAdd(py, sum[1]);
        atomicAdd(pz, sum[2]);
        atomicAdd(total_mass, sum[3]);
    }
}

/// @brief Subtract center-of-mass velocity from all atoms (host-provided v_cm).
__global__ void remove_com(mixed_t* __restrict__ vx, mixed_t* __restrict__ vy,
                           mixed_t* __restrict__ vz, acc_t vcm_x, acc_t vcm_y,
                           acc_t vcm_z, const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    vx[i] -= static_cast<mixed_t>(vcm_x);
    vy[i] -= static_cast<mixed_t>(vcm_y);
    vz[i] -= static_cast<mixed_t>(vcm_z);
}

/// @brief Subtract COM velocity computed from device-resident
///        `scratch = {Σ m·v_x, Σ m·v_y, Σ m·v_z, Σ m}` (4 acc_t words).
///
/// Pure GPU — replaces the D2H + sync + host-divide + H2D-as-arg pattern
/// used by the build()-time path with a single per-atom division. Cost:
/// 3 FP64 divisions per atom (~60 cycles total on Hopper), dwarfed by the
/// surrounding atomic-accumulate cost. Used by `ZeroMomentum::launch()`
/// to keep the hot path GPU-pure (`CLAUDE.md ## Architecture § CUDA
/// execution model` "launch() must be GPU-pure" rule).
///
/// Each thread reads the 4-word scratch as broadcast — coalesced across
/// the whole grid, served from L1 after the first warp.
__global__ void remove_com_from_scratch(mixed_t* __restrict__ vx,
                                        mixed_t* __restrict__ vy,
                                        mixed_t* __restrict__ vz,
                                        const acc_t* __restrict__ scratch_pM,
                                        const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    acc_t M = scratch_pM[3];
    if (!(M > acc_t(0))) // zero or NaN: no-op
        return;

    acc_t inv_M = acc_t(1) / M;
    acc_t vcm_x = scratch_pM[0] * inv_M;
    acc_t vcm_y = scratch_pM[1] * inv_M;
    acc_t vcm_z = scratch_pM[2] * inv_M;

    vx[i] -= static_cast<mixed_t>(vcm_x);
    vy[i] -= static_cast<mixed_t>(vcm_y);
    vz[i] -= static_cast<mixed_t>(vcm_z);
}

} // anonymous namespace

void InitVelocity::build(const BuildContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    if (n == 0)
        return;

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    // sigma^2 = kB * T / (ATV::units::mass_v2_to_energy * m)  =>  pass kB*T/ATV::units::mass_v2_to_energy as common factor
    const mixed_t kbt_over_mvv2e = static_cast<mixed_t>(
        ATV::units::kB_ev_per_k * temperature_ / ATV::units::mass_v2_to_energy);

    // 1. Generate Gaussian velocities
    generate_velocities<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass,
        frame.atoms.tag, kbt_over_mvv2e, seed_, frame.atoms.d_size);

    // 2. Compute total momentum + total mass
    acc_t* d_px;
    acc_t* d_py;
    acc_t* d_pz;
    acc_t* d_M;
    cudaMalloc(&d_px, sizeof(acc_t));
    cudaMalloc(&d_py, sizeof(acc_t));
    cudaMalloc(&d_pz, sizeof(acc_t));
    cudaMalloc(&d_M, sizeof(acc_t));
    cudaMemsetAsync(d_px, 0, sizeof(acc_t), stream);
    cudaMemsetAsync(d_py, 0, sizeof(acc_t), stream);
    cudaMemsetAsync(d_pz, 0, sizeof(acc_t), stream);
    cudaMemsetAsync(d_M, 0, sizeof(acc_t), stream);

    accumulate_momentum<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass, d_px,
        d_py, d_pz, d_M, frame.atoms.d_size);

    // 3. D2H momentum + mass (build-time D2H allowed)
    acc_t px = 0, py = 0, pz = 0, M = 0;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&px, d_px, sizeof(acc_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&py, d_py, sizeof(acc_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pz, d_pz, sizeof(acc_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&M, d_M, sizeof(acc_t), cudaMemcpyDeviceToHost);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    cudaFree(d_M);

    // 4. Remove center-of-mass velocity
    if (M > acc_t(0)) {
        acc_t vcm_x = px / M;
        acc_t vcm_y = py / M;
        acc_t vcm_z = pz / M;
        remove_com<<<grid, block, 0, stream>>>(frame.atoms.vx, frame.atoms.vy,
                                               frame.atoms.vz, vcm_x, vcm_y,
                                               vcm_z, frame.atoms.d_size);
        cudaStreamSynchronize(stream);
    }
    // Match the CPU recipe: the random draw is followed by exact thermal
    // rescaling with the backend's published global DOF. Reuse the existing
    // fused CUB kinetic reduction and cuBLAS vector scaling.
    if (ctx.runtime.md.n_dof_global > 0) {
        ATV_CUDA_CHECK(cudaMemsetAsync(&frame.scalars->kinetic, 0,
                                       sizeof(double), stream));
        ATV_CUDA_CHECK(cudaMemsetAsync(frame.scalars->stress_kinetic, 0,
                                       6 * sizeof(double), stream));
        KineticEnergy kinetic;
        kinetic.launch(
            {ctx.runtime, frame, slot_of(ATV::md::Phase::end_of_step), 0});
        double ke = 0;
        ATV_CUDA_CHECK(cudaMemcpyAsync(&ke, &frame.scalars->kinetic,
                                       sizeof(double), cudaMemcpyDeviceToHost,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (ke > 0) {
            const mixed_t factor =
                mixed_t(sqrt(temperature_ * double(ctx.runtime.md.n_dof_global)
                             * units::kB_ev_per_k / (2 * ke)));
            cublasHandle_t handle = nullptr;
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("InitVelocity: cublasCreate failed");
            std::unique_ptr<std::remove_pointer_t<cublasHandle_t>,
                            decltype(&cublasDestroy)>
                guard(handle, cublasDestroy);
            if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(
                    "InitVelocity: cublasSetStream failed");
            auto scale = [&]<typename T>(T* values, const T* alpha) {
                if constexpr (std::is_same_v<T, double>)
                    return cublasDscal(handle, n, alpha, values, 1);
                else
                    return cublasSscal(handle, n, alpha, values, 1);
            };
            for (auto* values :
                 {frame.atoms.vx, frame.atoms.vy, frame.atoms.vz})
                if (scale(values, &factor) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error(
                        "InitVelocity: cuBLAS scaling failed");
            ATV_CUDA_CHECK(cudaMemsetAsync(&frame.scalars->kinetic, 0,
                                           sizeof(double), stream));
            ATV_CUDA_CHECK(cudaMemsetAsync(frame.scalars->stress_kinetic, 0,
                                           6 * sizeof(double), stream));
            kinetic.launch(
                {ctx.runtime, frame, slot_of(ATV::md::Phase::end_of_step), 0});
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
}

void InitVelocity::launch(const LaunchContext& /*ctx*/) {
    // No-op: velocities initialized once in build().
}

// ── ZeroMomentum ────────────────────────────────────────────────────
//
// CUDA sibling of `cpu::md::ZeroLinearMomentum`. Drops the net COM
// linear momentum every `freq` `end_of_step` invocations. Same kernels
// as `InitVelocity::build`'s helpers (`accumulate_momentum`,
// `remove_com` in the anonymous namespace above) — they are file-local
// reductions reused here for the same `Σ m v_α + Σ m → broadcast v_cm
// → subtract` flow.

void ZeroMomentum::build(const BuildContext& /*ctx*/) {
    // 4 contiguous acc_t = {Σ mv_x, Σ mv_y, Σ mv_z, Σ m}. One alloc per
    // component instance; preserved across launches.
    acc_t* scratch = nullptr;
    cudaMalloc(&scratch, 4 * sizeof(acc_t));
    d_scratch_ = scratch;
}

void ZeroMomentum::launch(const LaunchContext& ctx) {
    // Frequency gate matches `cpu::md::ZeroLinearMomentum`: skip every
    // step that is not a multiple of `freq_`. step counter is local so
    // a shared lvalue ZeroMomentum can be re-registered into successive
    // per-stage drivers without the gate getting confused by out-of-band
    // step indices.
    if (step_++ % freq_ != 0)
        return;

    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = frame.n_atoms();
    if (n <= 0)
        return;

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    auto* scratch = static_cast<acc_t*>(d_scratch_);
    acc_t* d_px = scratch + 0;
    acc_t* d_py = scratch + 1;
    acc_t* d_pz = scratch + 2;
    acc_t* d_M = scratch + 3;

    // 1. Zero the 4 accumulator words on the compute stream.
    cudaMemsetAsync(scratch, 0, 4 * sizeof(acc_t), stream);

    // 2. CUB sums of m_i v_i (3 axes) + m_i. Reusing the
    //    same kernel as InitVelocity::build (anonymous-namespace helper
    //    declared earlier in this file).
    accumulate_momentum<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass, d_px,
        d_py, d_pz, d_M, frame.atoms.d_size);

    // 3. On-device v_cm broadcast — reads the 4-word scratch, computes
    //    v_cm = Σmv/Σm, subtracts. No D2H, no sync — keeps launch()
    //    GPU-pure per the CLAUDE.md "## Architecture § CUDA execution
    //    model" contract (cuda-md-review-cleanup-03-launch-purity).
    //    The previous host-divide + H2D-as-arg pattern was the last
    //    `cudaStreamSynchronize` on the MD hot path; this rewrite
    //    eliminates it. For multi-rank operation a single-GPU range
    //    (umbrella scope decision §1) means no MPI_Allreduce is needed;
    //    device-local sums are already the global sums.
    remove_com_from_scratch<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, scratch,
        frame.atoms.d_size);
}

void ZeroMomentum::destroy(const DestroyContext& /*ctx*/) {
    if (d_scratch_) {
        cudaFree(d_scratch_);
        d_scratch_ = nullptr;
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
