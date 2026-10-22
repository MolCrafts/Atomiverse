/// @file cuda/minimize/conv_check.cu
/// @brief Fused convergence-metric reduction + host five-criteria verdict.

#include <cmath>

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/minimize/conv_check.cuh"
#include <cuda_runtime.h>

namespace ATV {
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
/// Forces are decoded from fixed point and every metric reduces in acc_t.
/// Displacements use the reconstructed coordinate (primary + correction) in
/// mixed_t and are minimum-imaged through the by-value FP64 Box: the step
/// kernels wrap positions back into the cell (MD-integrator form), so a
/// raw `x − x_prev` across the boundary would report a spurious ~L jump
/// that the CPU components (which never wrap) cannot see.
///
/// `x_prev` may be null (step 0) — displacement metrics stay 0.
__global__ void
conv_metrics(const fixed_t* __restrict__ fx, const fixed_t* __restrict__ fy,
             const fixed_t* __restrict__ fz, const pos_t* __restrict__ x,
             const pos_t* __restrict__ y, const pos_t* __restrict__ z,
             const pos_t* __restrict__ xc, const pos_t* __restrict__ yc,
             const pos_t* __restrict__ zc, const mixed_t* __restrict__ x_prev,
             cuda::Box box, acc_t* __restrict__ metrics, int n) {
    box = box.current();
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t f2 = acc_t(0), fmag = acc_t(0);
    acc_t d2 = acc_t(0), dmax = acc_t(0);
    if (i < n) {
        const acc_t fxi = decode_force<acc_t>(fx[i]);
        const acc_t fyi = decode_force<acc_t>(fy[i]);
        const acc_t fzi = decode_force<acc_t>(fz[i]);
        f2 = fxi * fxi + fyi * fyi + fzi * fzi;
        fmag = sqrt(f2);
        if (x_prev != nullptr) {
            mixed_t dx = load_position(x, xc, i) - x_prev[i];
            mixed_t dy = load_position(y, yc, i) - x_prev[n + i];
            mixed_t dz = load_position(z, zc, i) - x_prev[2 * n + i];
            box.min_image(dx, dy, dz);
            const acc_t ax = acc_t(dx), ay = acc_t(dy), az = acc_t(dz);
            d2 = ax * ax + ay * ay + az * az;
            dmax = fmax(fabs(ax), fmax(fabs(ay), fabs(az)));
        }
    }

    ::cuda::std::array<acc_t, 2> sums = {f2, d2};

    using Reduce =
        cub::BlockReduce<::cuda::std::array<acc_t, 2>, kDefaultBlock>;
    __shared__ typename Reduce::TempStorage reduction;
    const auto out = Reduce(reduction).Reduce(sums, SumComponents<acc_t, 2>{});
    using Max = cub::BlockReduce<acc_t, kDefaultBlock>;
    __shared__ typename Max::TempStorage max_reduction;
    const acc_t fmag_max = Max(max_reduction).Reduce(fmag, ::cuda::maximum<>{});
    __syncthreads(); // CUB temporary storage is reused by the second maximum.
    const acc_t dmax_max = Max(max_reduction).Reduce(dmax, ::cuda::maximum<>{});
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
    cudaMalloc(&d_metrics_, 4 * sizeof(acc_t));
    cudaMallocHost(&h_metrics_, 4 * sizeof(acc_t));
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

void ConvCheck::measure(const LaunchContext& ctx, const mixed_t* x_prev,
                        BigInt step,
                        std::vector<ATV::minimizer::OptRecord>& trace) {
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int n = n_;
    int block = cuda::kDefaultBlock;
    int grid = cuda::grid_size(n);
    constexpr int shmem = 0;

    // A displacement is measurable only when the optimizer kept the pre-step geometry
    // (SteepestDescent does not) AND has actually taken a step. The component knows;
    // the handler is told, through `has_displacement`. It must never guess from the
    // iteration counter.
    const bool has_disp = (x_prev != nullptr) && (step > 0);

    cudaMemsetAsync(d_metrics_, 0, 4 * sizeof(acc_t), stream);
    const auto& a = frame.atoms;
    conv_metrics<<<grid, block, shmem, stream>>>(
        a.fx, a.fy, a.fz, a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr,
        has_disp ? x_prev : nullptr, ctx.runtime.host_box, d_metrics_, n);
    cudaMemcpyAsync(h_metrics_, d_metrics_, 4 * sizeof(acc_t),
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
    // A step counter does not establish a previous energy sample. The first
    // measurement publishes a sentinel, even after an initial-force-driven move.
    const double energy_change =
        !trace.empty() ? std::abs(energy - energy_prev_) : 1e30;

    ATV::minimizer::OptRecord rec{};
    rec.step = step;
    rec.energy = energy;
    rec.max_force = max_force_;
    rec.rms_force = rms_force;
    rec.max_displacement = max_disp;
    rec.rms_displacement = rms_disp;
    rec.energy_change = energy_change;
    trace.push_back(rec);

    // PUBLISH, do not decide. `cuda::ForceConverged` (cuda/driver/criterion.cuh)
    // owns the tolerances and returns the Verdict; this class owns the reduction.
    // Until this line, ConvCheck wrote `outer_status.converged = true` itself, so the
    // criteria could not be changed without touching the optimizer — and a nested
    // loop's convergence could stop the loop above it.
    if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
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
