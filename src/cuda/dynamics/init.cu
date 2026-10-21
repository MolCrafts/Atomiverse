/// @file cuda/dynamics/init.cu
/// @brief Initializer implementations: ZeroForces, ZeroEnergy, ZeroKinetic, InitVelocity.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ATV {

using cuda::motion_t;
using cuda::prec_t;
using cuda::real_t;

namespace cuda {
namespace md {

// ── Zero initializers ────────────────────────────────────────────────

namespace {
__global__ void zero_forces(real_t* fx, real_t* fy, real_t* fz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        fx[i] = fy[i] = fz[i] = real_t(0);
}
__global__ void zero_kinetic(SystemScalars* scalars) {
    if (threadIdx.x == 0)
        scalars->kinetic = 0;
    if (threadIdx.x < 6)
        scalars->stress_kinetic[threadIdx.x] = 0;
}
} // namespace

void ZeroForces::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = frame.n_atoms();
    if (n <= 0)
        return;
    if (n > 0) {
        zero_forces<<<grid_size(n), kDefaultBlock, 0, stream>>>(
            frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, n);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}

void ZeroEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Zero `energy` + all five per-term fields. The pair/bonded virial lives on
    // `ZeroStress` (same split as `cpu::ZeroEnergy` / `cpu::ZeroStress`).
    ATV_CUDA_CHECK(
        cudaMemsetAsync(&frame.scalars->energy, 0, 6 * sizeof(double), stream));
}

void ZeroStress::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Without this clear, force kernels (LJ + bond/angle/dihedral)
    // accumulate virial across steps and `NoseHooverNPT` reads runaway
    // pressure.
    ATV_CUDA_CHECK(cudaMemsetAsync(&frame.scalars->stress[0], 0,
                                   6 * sizeof(double), stream));
}

void ZeroKinetic::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    zero_kinetic<<<1, 32, 0, stream>>>(frame.scalars);
    ATV_CUDA_CHECK(cudaGetLastError());
}

// ── InitVelocity ─────────────────────────────────────────────────────

namespace {

/// @brief Generate Gaussian-distributed velocities at target temperature.
///
/// v_i ~ N(0, sigma) where sigma = sqrt(kB*T / (ATV::units::mass_v2_to_energy * m_i)).
/// Uses cuRAND device API (per-thread state, Philox generator).
__global__ void
generate_velocities(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
                    motion_t* __restrict__ vz,
                    const motion_t* __restrict__ mass, motion_t kbt_over_mvv2e,
                    unsigned long long seed, const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    curandState state;
    curand_init(seed, i, 0, &state);

    motion_t sigma = sqrt(kbt_over_mvv2e / static_cast<motion_t>(mass[i]));
    vx[i] = static_cast<motion_t>(
        static_cast<motion_t>(curand_normal_double(&state)) * sigma);
    vy[i] = static_cast<motion_t>(
        static_cast<motion_t>(curand_normal_double(&state)) * sigma);
    vz[i] = static_cast<motion_t>(
        static_cast<motion_t>(curand_normal_double(&state)) * sigma);
}

/// @brief Accumulate total momentum and total mass via atomicAdd.
///
/// Build-time only (atomicAdd precision acceptable for initialization).
__global__ void accumulate_momentum(
    const motion_t* __restrict__ vx, const motion_t* __restrict__ vy,
    const motion_t* __restrict__ vz, const motion_t* __restrict__ mass,
    motion_t* __restrict__ px, motion_t* __restrict__ py,
    motion_t* __restrict__ pz, motion_t* __restrict__ total_mass,
    const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    atomicAdd(px,
              static_cast<motion_t>(mass[i]) * static_cast<motion_t>(vx[i]));
    atomicAdd(py,
              static_cast<motion_t>(mass[i]) * static_cast<motion_t>(vy[i]));
    atomicAdd(pz,
              static_cast<motion_t>(mass[i]) * static_cast<motion_t>(vz[i]));
    atomicAdd(total_mass, static_cast<motion_t>(mass[i]));
}

/// @brief Subtract center-of-mass velocity from all atoms (host-provided v_cm).
__global__ void remove_com(motion_t* __restrict__ vx, motion_t* __restrict__ vy,
                           motion_t* __restrict__ vz, motion_t vcm_x,
                           motion_t vcm_y, motion_t vcm_z,
                           const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    vx[i] -= static_cast<motion_t>(vcm_x);
    vy[i] -= static_cast<motion_t>(vcm_y);
    vz[i] -= static_cast<motion_t>(vcm_z);
}

/// @brief Subtract COM velocity computed from device-resident
///        `scratch = {Σ m·v_x, Σ m·v_y, Σ m·v_z, Σ m}` (4 motion_t words).
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
__global__ void remove_com_from_scratch(motion_t* __restrict__ vx,
                                        motion_t* __restrict__ vy,
                                        motion_t* __restrict__ vz,
                                        const motion_t* __restrict__ scratch_pM,
                                        const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    motion_t M = scratch_pM[3];
    if (!(M > motion_t(0))) // zero or NaN: no-op
        return;

    motion_t inv_M = motion_t(1) / M;
    motion_t vcm_x = scratch_pM[0] * inv_M;
    motion_t vcm_y = scratch_pM[1] * inv_M;
    motion_t vcm_z = scratch_pM[2] * inv_M;

    vx[i] -= static_cast<motion_t>(vcm_x);
    vy[i] -= static_cast<motion_t>(vcm_y);
    vz[i] -= static_cast<motion_t>(vcm_z);
}

} // anonymous namespace

void InitVelocity::build(const BuildContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();
    if (n <= 0)
        return;

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    // sigma^2 = kB * T / (ATV::units::mass_v2_to_energy * m)  =>  pass kB*T/ATV::units::mass_v2_to_energy as common factor
    motion_t kbt_over_mvv2e =
        ATV::units::kB_ev_per_k * temperature_ / ATV::units::mass_v2_to_energy;

    // 1. Generate Gaussian velocities
    generate_velocities<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass,
        kbt_over_mvv2e, seed_, frame.atoms.d_size);
    ATV_CUDA_CHECK(cudaGetLastError());

    // 2. Compute total momentum + total mass
    motion_t* d_px;
    motion_t* d_py;
    motion_t* d_pz;
    motion_t* d_M;
    ATV_CUDA_CHECK(cudaMalloc(&d_px, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_py, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_pz, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMalloc(&d_M, sizeof(motion_t)));
    ATV_CUDA_CHECK(cudaMemsetAsync(d_px, 0, sizeof(motion_t), stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(d_py, 0, sizeof(motion_t), stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(d_pz, 0, sizeof(motion_t), stream));
    ATV_CUDA_CHECK(cudaMemsetAsync(d_M, 0, sizeof(motion_t), stream));

    accumulate_momentum<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass, d_px,
        d_py, d_pz, d_M, frame.atoms.d_size);

    ATV_CUDA_CHECK(cudaGetLastError());

    // 3. D2H momentum + mass (build-time D2H allowed)
    motion_t px = 0, py = 0, pz = 0, M = 0;
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    ATV_CUDA_CHECK(
        cudaMemcpy(&px, d_px, sizeof(motion_t), cudaMemcpyDeviceToHost));
    ATV_CUDA_CHECK(
        cudaMemcpy(&py, d_py, sizeof(motion_t), cudaMemcpyDeviceToHost));
    ATV_CUDA_CHECK(
        cudaMemcpy(&pz, d_pz, sizeof(motion_t), cudaMemcpyDeviceToHost));
    ATV_CUDA_CHECK(
        cudaMemcpy(&M, d_M, sizeof(motion_t), cudaMemcpyDeviceToHost));

    ATV_CUDA_CHECK(cudaFree(d_px));
    ATV_CUDA_CHECK(cudaFree(d_py));
    ATV_CUDA_CHECK(cudaFree(d_pz));
    ATV_CUDA_CHECK(cudaFree(d_M));

    // 4. Remove center-of-mass velocity
    if (M > motion_t(0)) {
        motion_t vcm_x = px / M;
        motion_t vcm_y = py / M;
        motion_t vcm_z = pz / M;
        remove_com<<<grid, block, 0, stream>>>(frame.atoms.vx, frame.atoms.vy,
                                               frame.atoms.vz, vcm_x, vcm_y,
                                               vcm_z, frame.atoms.d_size);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
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
    // 4 contiguous motion_t = {Σ mv_x, Σ mv_y, Σ mv_z, Σ m}. One alloc per
    // component instance; preserved across launches.
    motion_t* scratch = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&scratch, 4 * sizeof(motion_t)));
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
    auto* scratch = static_cast<motion_t*>(d_scratch_);
    motion_t* d_px = scratch + 0;
    motion_t* d_py = scratch + 1;
    motion_t* d_pz = scratch + 2;
    motion_t* d_M = scratch + 3;

    // 1. Zero the 4 accumulator words on the compute stream.
    ATV_CUDA_CHECK(cudaMemsetAsync(scratch, 0, 4 * sizeof(motion_t), stream));

    // 2. Σ_i m_i v_i (3 axes) + Σ_i m_i via atomicAdd. Reusing the
    //    same kernel as InitVelocity::build (anonymous-namespace helper
    //    declared earlier in this file).
    accumulate_momentum<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass, d_px,
        d_py, d_pz, d_M, frame.atoms.d_size);
    ATV_CUDA_CHECK(cudaGetLastError());

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
    ATV_CUDA_CHECK(cudaGetLastError());
}

void ZeroMomentum::destroy(const DestroyContext& /*ctx*/) {
    if (d_scratch_) {
        ATV_CUDA_CHECK(cudaFree(d_scratch_));
        d_scratch_ = nullptr;
    }
}

} // namespace md
} // namespace cuda
} // namespace ATV
