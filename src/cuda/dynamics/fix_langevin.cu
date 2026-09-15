/// @file cuda/dynamics/fix_langevin.cu
/// @brief CUDA Langevin thermostat kernel.

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace ATV {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace md {

namespace {

/// @brief splitmix64 — single-step bijective hash. Matches the CPU
/// `cpu::md::FixLangevin` per-atom stream so both backends draw from
/// the same family of `(seed, step, tag)`-keyed Gaussians.
__device__ inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/// @brief Convert a 64-bit hash into `(0, 1]`-valued double.
__device__ inline work_t u01_from(std::uint64_t h) {
    // Map upper 53 bits into [0, 1); avoid exact 0 for log() safety.
    work_t u = static_cast<work_t>((h >> 11) | std::uint64_t(1))
               * (work_t(1) / work_t(9007199254740992.0)); // 2^53
    return u;
}

/// @brief Box-Muller transform: 2 uniforms → 2 standard normals.
__device__ inline void box_muller(work_t u1, work_t u2, work_t& z1,
                                  work_t& z2) {
    work_t r = sqrt(work_t(-2) * log(u1));
    work_t theta = work_t(6.283185307179586) * u2; // 2π
    z1 = r * cos(theta);
    z2 = r * sin(theta);
}

__global__ void
apply_langevin(const real_t* __restrict__ vx, const real_t* __restrict__ vy,
               const real_t* __restrict__ vz, real_t* __restrict__ fx,
               real_t* __restrict__ fy, real_t* __restrict__ fz,
               const real_t* __restrict__ mass,
               const std::int64_t* __restrict__ tag, std::uint64_t base_seed,
               work_t T_live, work_t damp, work_t dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    work_t m = static_cast<work_t>(mass[i]);
    // gfactor1 (drag): `-m / (damp · ftm2v)`. Force = gfactor1 · v.
    work_t gfactor1 = -m / (damp * ATV::units::force_to_accel);
    // gfactor2 (random): `sqrt(2 m kB T_live / (damp · dt · mvv2e)) / ftm2v`.
    work_t gfactor2_inside = work_t(2) * m * ATV::units::kB_ev_per_k * T_live
                             / (damp * dt * ATV::units::mass_v2_to_energy);
    work_t gfactor2 = sqrt(gfactor2_inside) / ATV::units::force_to_accel;

    // Per-atom (seed, step, tag)-keyed RNG. Multiply the global tag by
    // a large odd prime BEFORE adding it to the (seed+step) base — same
    // pattern as cpu::md::FixLangevin's `GaussStream`. The naive
    // `base ^ (tag + magic)` keying fails because (atom_i, step_j) and
    // (atom_j, step_i) collide whenever `i ^ j` produces the same bit
    // pattern via small-carry symmetry, dramatically reducing the
    // unique RNG-stream count and leaving the ensemble net-correlated
    // (system heats up instead of thermalising). The multiplication by
    // 0xD6E8FEB86659FD93 spreads adjacent tags by ~2^63 distance in
    // key-space; combined with step in `base_seed` (low 32 bits) the
    // (tag, step) → key map is collision-free.
    std::uint64_t t = static_cast<std::uint64_t>(tag[i]);
    std::uint64_t k = base_seed + t * 0xD6E8FEB86659FD93ull;
    std::uint64_t h1 = splitmix64(k);
    std::uint64_t h2 = splitmix64(h1);
    std::uint64_t h3 = splitmix64(h2);
    std::uint64_t h4 = splitmix64(h3);
    work_t u1 = u01_from(h1), u2 = u01_from(h2);
    work_t u3 = u01_from(h3), u4 = u01_from(h4);
    work_t rx, ry, rz, r_unused;
    box_muller(u1, u2, rx, ry);
    box_muller(u3, u4, rz, r_unused);

    fx[i] = static_cast<real_t>(static_cast<work_t>(fx[i])
                                + gfactor1 * static_cast<work_t>(vx[i])
                                + gfactor2 * rx);
    fy[i] = static_cast<real_t>(static_cast<work_t>(fy[i])
                                + gfactor1 * static_cast<work_t>(vy[i])
                                + gfactor2 * ry);
    fz[i] = static_cast<real_t>(static_cast<work_t>(fz[i])
                                + gfactor1 * static_cast<work_t>(vz[i])
                                + gfactor2 * rz);
}

} // anonymous namespace

void FixLangevin::launch(const LaunchContext& ctx) {
    // ── Host-side T_live update (matches CPU compute_target) ────────
    double delta = 0.0;
    if (nlast_ > nfirst_) {
        delta = static_cast<double>(ctx.step - nfirst_)
                / static_cast<double>(nlast_ - nfirst_);
        if (delta < 0.0)
            delta = 0.0;
        if (delta > 1.0)
            delta = 1.0;
    }
    T_live_ = T_start_ + delta * (T_stop_ - T_start_);

    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    int n = frame.n_atoms();
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);

    // Per-step seed: mix the 32-bit user seed with the step counter via
    // PRIME multiplication (matches cpu::md::FixLangevin). The previous
    // `(seed << 32) ^ step` keying caused consecutive steps' base_seed
    // to differ by only ~1 bit, which left a small but DEFINITE positive
    // autocorrelation in the per-step Gaussian stream after splitmix.
    // Over many steps the cross-terms compounded into the no-drag
    // accumulating-variance test, producing T_eq ≈ 4× target. CPU keying
    // gives ~2^60 separation between consecutive steps' base, so splitmix
    // avalanches into a fully independent stream.
    std::uint64_t base_seed =
        static_cast<std::uint64_t>(seed_) * 0x9E3779B97F4A7C15ULL
        + static_cast<std::uint64_t>(ctx.step) * 0xBF58476D1CE4E5B9ULL;

    apply_langevin<<<grid, block, 0, stream>>>(
        frame.atoms.vx, frame.atoms.vy, frame.atoms.vz, frame.atoms.fx,
        frame.atoms.fy, frame.atoms.fz, frame.atoms.mass, frame.atoms.tag,
        base_seed, static_cast<work_t>(T_live_), static_cast<work_t>(damp_),
        static_cast<work_t>(dt_), n);
}

} // namespace md
} // namespace cuda
} // namespace ATV
