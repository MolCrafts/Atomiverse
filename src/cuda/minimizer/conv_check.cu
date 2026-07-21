/// @file minimizer/conv_check.cu
/// @brief Fused convergence-metric reduction + host five-criteria verdict.

#include <cmath>

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/minimizer/conv_check.cuh"
#include <cuda_runtime.h>

namespace ATV {

using cuda::real_t;
using cuda::work_t;

namespace cuda {
namespace minimizer {

namespace {

/// @brief Fused per-atom force + displacement metric reduction.
///
/// One pass produces all four convergence metrics:
///   metrics[0] = max_i |F_i|            (atomic max, non-negative)
///   metrics[1] = Σ_i |F_i|²             (atomic add)
///   metrics[2] = max_i,c |x_c − x_prev| (atomic max, non-negative)
///   metrics[3] = Σ_i |x − x_prev|²      (atomic add)
///
/// Displacements are minimum-imaged through the by-value Box: the step
/// kernels wrap positions back into the cell (MD-integrator form), so a
/// raw `x − x_prev` across the boundary would report a spurious ~L jump
/// that the CPU components (which never wrap) cannot see.
///
/// `x_prev` may be null (step 0) — displacement metrics stay 0.
__global__ void
conv_metrics(const real_t* __restrict__ fx, const real_t* __restrict__ fy,
             const real_t* __restrict__ fz, const real_t* __restrict__ x,
             const real_t* __restrict__ y, const real_t* __restrict__ z,
             const work_t* __restrict__ x_prev, cuda::Box box,
             work_t* __restrict__ metrics, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    work_t f2 = work_t(0), fmag = work_t(0);
    work_t d2 = work_t(0), dmax = work_t(0);
    if (i < n) {
        work_t fxi = static_cast<work_t>(fx[i]);
        work_t fyi = static_cast<work_t>(fy[i]);
        work_t fzi = static_cast<work_t>(fz[i]);
        f2 = fxi * fxi + fyi * fyi + fzi * fzi;
        fmag = sqrt(f2);
        if (x_prev != nullptr) {
            work_t dx = static_cast<work_t>(x[i]) - x_prev[i];
            work_t dy = static_cast<work_t>(y[i]) - x_prev[n + i];
            work_t dz = static_cast<work_t>(z[i]) - x_prev[2 * n + i];
            box.min_image(dx, dy, dz);
            d2 = dx * dx + dy * dy + dz * dz;
            dmax = fmax(fabs(dx), fmax(fabs(dy), fabs(dz)));
        }
    }

    extern __shared__ work_t warp_scratch[];
    work_t sums[2] = {f2, d2};
    work_t out[2];
    ATV::cuda::block_reduce_sum_n<2>(sums, warp_scratch, out);
    work_t fmag_max = ATV::cuda::block_reduce_max(fmag, warp_scratch);
    work_t dmax_max = ATV::cuda::block_reduce_max(dmax, warp_scratch);
    if (threadIdx.x == 0) {
        ATV::cuda::atomic_max_nonneg(&metrics[0], fmag_max);
        atomicAdd(&metrics[1], out[0]);
        ATV::cuda::atomic_max_nonneg(&metrics[2], dmax_max);
        atomicAdd(&metrics[3], out[1]);
    }
}

} // anonymous namespace

void ConvCheck::build(int n_atoms) {
    n_ = n_atoms;
    cudaMalloc(&d_metrics_, 4 * sizeof(work_t));
    cudaMallocHost(&h_metrics_, 4 * sizeof(work_t));
    cudaMallocHost(&h_energy_, sizeof(double));
    energy_prev_ = 0;
    last_energy_ = 0;
    max_force_ = 0;
}

void ConvCheck::destroy() {
    cudaFree(d_metrics_);
    cudaFreeHost(h_metrics_);
    cudaFreeHost(h_energy_);
    d_metrics_ = nullptr;
    h_metrics_ = nullptr;
    h_energy_ = nullptr;
}

void ConvCheck::measure(const LaunchContext& ctx, const work_t* x_prev,
                        int step, std::vector<OptRecord>& trace) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = n_;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    int shmem = cuda::warps_per_block(block) * sizeof(work_t);

    // A displacement is measurable only when the optimizer kept the pre-step geometry
    // (SteepestDescent does not) AND has actually taken a step. The component knows;
    // the handler is told, through `has_displacement`. It must never guess from the
    // iteration counter.
    const bool has_disp = (x_prev != nullptr) && (step > 0);

    cudaMemsetAsync(d_metrics_, 0, 4 * sizeof(work_t), stream);
    conv_metrics<<<grid, block, shmem, stream>>>(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, frame.atoms.x,
        frame.atoms.y, frame.atoms.z, has_disp ? x_prev : nullptr,
        ctx.runtime.host_box, d_metrics_, n);
    cudaMemcpyAsync(h_metrics_, d_metrics_, 4 * sizeof(work_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_energy_, &frame.scalars->energy, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    // The one host sync of the minimizer iteration: the metrics must land on the host,
    // where the driver polls its handlers after step_once().
    cudaStreamSynchronize(stream);

    max_force_ = static_cast<double>(h_metrics_[0]);
    const double rms_force =
        std::sqrt(static_cast<double>(h_metrics_[1]) / (3.0 * n));
    const double max_disp = has_disp ? static_cast<double>(h_metrics_[2]) : 0.0;
    const double rms_disp =
        has_disp ? std::sqrt(static_cast<double>(h_metrics_[3]) / (3.0 * n))
                 : 0.0;
    const double energy = *h_energy_;
    // No previous energy at step 0: publish a sentinel so that no handler can call a
    // run converged before it has moved (|ΔE| < etol would otherwise be trivially
    // true). Same value, same reason, as the CPU optimizers.
    const double energy_change =
        (step > 0) ? std::abs(energy - energy_prev_) : 1e30;

    OptRecord rec{};
    rec.step = step;
    rec.energy = energy;
    rec.max_force = max_force_;
    rec.rms_force = rms_force;
    rec.max_displacement = max_disp;
    rec.rms_displacement = rms_disp;
    rec.energy_change = energy_change;
    trace.push_back(rec);

    // PUBLISH, do not decide. `cuda::ForceConverged` (cuda/driver/loop_handler.cuh)
    // owns the tolerances and returns the Verdict; this class owns the reduction.
    // Until this line, ConvCheck wrote `outer_status.converged = true` itself, so the
    // criteria could not be changed without touching the optimizer — and a nested
    // loop's convergence could stop the loop above it.
    if (LoopMetrics* m = ctx.runtime.loop.metrics) {
        m->energy = energy;
        m->max_force = max_force_;
        m->rms_force = rms_force;
        m->max_displacement = max_disp;
        m->rms_displacement = rms_disp;
        m->energy_change = energy_change;
        m->has_displacement = has_disp;
    }

    energy_prev_ = energy;
    last_energy_ = energy;
}

} // namespace minimizer
} // namespace cuda
} // namespace ATV
