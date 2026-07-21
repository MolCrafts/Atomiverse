/// @file md/init.cu
/// @brief Initializer implementations: ZeroForces, ZeroEnergy, ZeroKinetic, InitVelocity.

#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/md/init.cuh"

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

// ── Zero initializers ────────────────────────────────────────────────

void ZeroForces::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    auto n = static_cast<size_t>(ctx.frame.n_atoms());
    cudaMemsetAsync(frame.atoms.fx, 0, n * sizeof(real_t), stream);
    cudaMemsetAsync(frame.atoms.fy, 0, n * sizeof(real_t), stream);
    cudaMemsetAsync(frame.atoms.fz, 0, n * sizeof(real_t), stream);
}

void ZeroEnergy::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    // Zero `energy` + 4 per-term fields and the 6-component pair/bonded
    // virial tensor `stress[6]`. Without the stress clear, force kernels
    // (LJ + bond/angle/dihedral) accumulate across steps and
    // `NoseHooverNPT` reads runaway pressure → barostat blowup → stage-2
    // collapse.
    cudaMemsetAsync(&frame.scalars->energy, 0, 5 * sizeof(double), stream);
    cudaMemsetAsync(&frame.scalars->stress[0], 0, 6 * sizeof(double), stream);
}

void ZeroKinetic::launch(const LaunchContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    cudaMemsetAsync(&frame.scalars->kinetic, 0, sizeof(prec_t), stream);
}

// ── InitVelocity ─────────────────────────────────────────────────────

namespace {

/// @brief Generate Gaussian-distributed velocities at target temperature.
///
/// v_i ~ N(0, sigma) where sigma = sqrt(kB*T / (kMvv2e * m_i)).
/// Uses cuRAND device API (per-thread state, Philox generator).
__global__ void
generate_velocities(real_t* __restrict__ vx, real_t* __restrict__ vy,
                    real_t* __restrict__ vz, const real_t* __restrict__ mass,
                    work_t kbt_over_mvv2e, unsigned long long seed,
                    const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    curandState state;
    curand_init(seed, i, 0, &state);

    work_t sigma = sqrt(kbt_over_mvv2e / static_cast<work_t>(mass[i]));
    vx[i] = static_cast<real_t>(
        static_cast<work_t>(curand_normal_double(&state)) * sigma);
    vy[i] = static_cast<real_t>(
        static_cast<work_t>(curand_normal_double(&state)) * sigma);
    vz[i] = static_cast<real_t>(
        static_cast<work_t>(curand_normal_double(&state)) * sigma);
}

/// @brief Accumulate total momentum and total mass via atomicAdd.
///
/// Build-time only (atomicAdd precision acceptable for initialization).
__global__ void accumulate_momentum(
    const real_t* __restrict__ vx, const real_t* __restrict__ vy,
    const real_t* __restrict__ vz, const real_t* __restrict__ mass,
    work_t* __restrict__ px, work_t* __restrict__ py, work_t* __restrict__ pz,
    work_t* __restrict__ total_mass, const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    atomicAdd(px, static_cast<work_t>(mass[i]) * static_cast<work_t>(vx[i]));
    atomicAdd(py, static_cast<work_t>(mass[i]) * static_cast<work_t>(vy[i]));
    atomicAdd(pz, static_cast<work_t>(mass[i]) * static_cast<work_t>(vz[i]));
    atomicAdd(total_mass, static_cast<work_t>(mass[i]));
}

/// @brief Subtract center-of-mass velocity from all atoms (host-provided v_cm).
__global__ void remove_com(real_t* __restrict__ vx, real_t* __restrict__ vy,
                           real_t* __restrict__ vz, work_t vcm_x, work_t vcm_y,
                           work_t vcm_z, const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    vx[i] -= static_cast<real_t>(vcm_x);
    vy[i] -= static_cast<real_t>(vcm_y);
    vz[i] -= static_cast<real_t>(vcm_z);
}

/// @brief Subtract COM velocity computed from device-resident
///        `scratch = {Σ m·v_x, Σ m·v_y, Σ m·v_z, Σ m}` (4 work_t words).
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
__global__ void remove_com_from_scratch(real_t* __restrict__ vx,
                                        real_t* __restrict__ vy,
                                        real_t* __restrict__ vz,
                                        const work_t* __restrict__ scratch_pM,
                                        const int* __restrict__ n_ptr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_ptr)
        return;

    work_t M = scratch_pM[3];
    if (!(M > work_t(0))) // zero or NaN: no-op
        return;

    work_t inv_M = work_t(1) / M;
    work_t vcm_x = scratch_pM[0] * inv_M;
    work_t vcm_y = scratch_pM[1] * inv_M;
    work_t vcm_z = scratch_pM[2] * inv_M;

    vx[i] -= static_cast<real_t>(vcm_x);
    vy[i] -= static_cast<real_t>(vcm_y);
    vz[i] -= static_cast<real_t>(vcm_z);
}

} // anonymous namespace

void InitVelocity::build(const BuildContext& ctx) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = ctx.frame.n_atoms();

    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(static_cast<int>(n));

    // sigma^2 = kB * T / (kMvv2e * m)  =>  pass kB*T/kMvv2e as common factor
    work_t kbt_over_mvv2e = kBoltzmann * temperature_ / kMvv2e;

    // 1. Generate Gaussian velocities
    generate_velocities<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass,
        kbt_over_mvv2e, seed_, frame.atoms.d_size);

    // 2. Compute total momentum + total mass
    work_t* d_px;
    work_t* d_py;
    work_t* d_pz;
    work_t* d_M;
    cudaMalloc(&d_px, sizeof(work_t));
    cudaMalloc(&d_py, sizeof(work_t));
    cudaMalloc(&d_pz, sizeof(work_t));
    cudaMalloc(&d_M, sizeof(work_t));
    cudaMemsetAsync(d_px, 0, sizeof(work_t), stream);
    cudaMemsetAsync(d_py, 0, sizeof(work_t), stream);
    cudaMemsetAsync(d_pz, 0, sizeof(work_t), stream);
    cudaMemsetAsync(d_M, 0, sizeof(work_t), stream);

    accumulate_momentum<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.mass, d_px,
        d_py, d_pz, d_M, frame.atoms.d_size);

    // 3. D2H momentum + mass (build-time D2H allowed)
    work_t px = 0, py = 0, pz = 0, M = 0;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&px, d_px, sizeof(work_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&py, d_py, sizeof(work_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pz, d_pz, sizeof(work_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&M, d_M, sizeof(work_t), cudaMemcpyDeviceToHost);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    cudaFree(d_M);

    // 4. Remove center-of-mass velocity
    if (M > work_t(0)) {
        work_t vcm_x = px / M;
        work_t vcm_y = py / M;
        work_t vcm_z = pz / M;
        remove_com<<<grid, block, 0, stream>>>(frame.atoms.vx, frame.atoms.vy,
                                               frame.atoms.vz, vcm_x, vcm_y,
                                               vcm_z, frame.atoms.d_size);
        cudaStreamSynchronize(stream);
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
    // 4 contiguous work_t = {Σ mv_x, Σ mv_y, Σ mv_z, Σ m}. One alloc per
    // component instance; preserved across launches.
    work_t* scratch = nullptr;
    cudaMalloc(&scratch, 4 * sizeof(work_t));
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
    auto* scratch = static_cast<work_t*>(d_scratch_);
    work_t* d_px = scratch + 0;
    work_t* d_py = scratch + 1;
    work_t* d_pz = scratch + 2;
    work_t* d_M = scratch + 3;

    // 1. Zero the 4 accumulator words on the compute stream.
    cudaMemsetAsync(scratch, 0, 4 * sizeof(work_t), stream);

    // 2. Σ_i m_i v_i (3 axes) + Σ_i m_i via atomicAdd. Reusing the
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
