/// @file cpu/minimize/opt_log.cpp
/// @brief OptLog -- TSV writer for geometry-optimization progress.
///        MPI-aware: collective metric reduction, rank-0 I/O.

#include "atomiverse/cpu/minimize/opt_log.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/minimize/global_reduce.h"

#include <cmath>
#include <cstdio>

namespace ATV {
namespace cpu {
namespace minimizer {

OptLog::OptLog(std::string filename, bool also_stdout)
    : filename_(std::move(filename)), also_stdout_(also_stdout) {}

void OptLog::build(const cpu::BuildContext& ctx) {
    // Position reference sized to the frame capacity (≥ n_local under
    // multi-rank) so launch() never allocates on the steady-state path.
    const int cap = ctx.frame.nlocal();
    x_prev_.assign(cap, 0.0);
    y_prev_.assign(cap, 0.0);
    z_prev_.assign(cap, 0.0);
    has_prev_ = false;
    n_rows_ = 0;
    last_generation_ = ~std::uint64_t{0};

    if (ctx.runtime.mpi_rank != 0)
        return;
    os_.open(filename_, std::ios::out | std::ios::trunc);
    os_ << "step\tenergy_eV\tmax_force\trms_force\tstep_norm\n";
    if (also_stdout_)
        std::printf("%6s  %15s  %12s  %12s  %12s\n", "step", "E[eV]", "max_F",
                    "rms_F", "|dx|");
}

void OptLog::launch(const cpu::LaunchContext& ctx) {
    // NOTE: no rank gate here — the metric reductions below are
    // collectives that every rank must enter; only the I/O at the end is
    // rank-0-only.
    const auto& f = ctx.frame;
    const int n = owned_count(ctx);

    // Hoisted once for the whole launch -- the launch-boundary rule.
    const StorageReal* __restrict sp_x = f.span(kAtom_x);
    const StorageReal* __restrict sp_y = f.span(kAtom_y);
    const StorageReal* __restrict sp_z = f.span(kAtom_z);
    const AccReal* __restrict sp_fx = f.span(kAtom_fx);
    const AccReal* __restrict sp_fy = f.span(kAtom_fy);
    const AccReal* __restrict sp_fz = f.span(kAtom_fz);

    // Migration reshuffles local rows: the position reference is stale.
    {
        const auto* dd = ctx.runtime.md.dd;
        const std::uint64_t gen =
            (dd && ctx.runtime.mpi_size > 1) ? dd->neighbor_generation : 0;
        if (gen != last_generation_) {
            if (last_generation_ != ~std::uint64_t{0})
                has_prev_ = false;
            last_generation_ = gen;
        }
    }

    if (n > static_cast<int>(x_prev_.size())) {
        x_prev_.resize(n, 0.0);
        y_prev_.resize(n, 0.0);
        z_prev_.resize(n, 0.0);
    }

    double sumsq = 0, maxsq = 0;
#pragma omp parallel for schedule(static) reduction(+ : sumsq)                 \
    reduction(max : maxsq)
    for (int i = 0; i < n; ++i) {
        const double fx = sp_fx[i];
        const double fy = sp_fy[i];
        const double fz = sp_fz[i];
        const double m2 = fx * fx + fy * fy + fz * fz;
        sumsq += m2;
        maxsq = std::max(maxsq, m2);
    }

    double step_acc = 0;
    if (has_prev_) {
#pragma omp parallel for schedule(static) reduction(+ : step_acc)
        for (int i = 0; i < n; ++i) {
            const double dx = sp_x[i] - x_prev_[i];
            const double dy = sp_y[i] - y_prev_[i];
            const double dz = sp_z[i] - z_prev_[i];
            step_acc += dx * dx + dy * dy + dz * dz;
        }
    }

    // Batched cross-rank reduction (single-rank no-op). All ranks enter.
    double sums[3] = {sumsq, step_acc, static_cast<double>(n)};
    global_sum(ctx, sums, 3);
    maxsq = global_max(ctx, maxsq);

    const double n_global = std::max(sums[2], 1.0);
    const double max_force = std::sqrt(maxsq);
    const double rms_force = std::sqrt(sums[0] / n_global);
    const double step_norm = has_prev_ ? std::sqrt(sums[1]) : 0.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x_prev_[i] = sp_x[i];
        y_prev_[i] = sp_y[i];
        z_prev_[i] = sp_z[i];
    }
    has_prev_ = true;

    if (ctx.runtime.mpi_rank != 0)
        return;
    if (os_.is_open()) {
        os_ << ctx.step << '\t' << f.meta.get(kEnergy) << '\t' << max_force
            << '\t' << rms_force << '\t' << step_norm << '\n';
        os_.flush();
    }
    if (also_stdout_) {
        std::printf("%6d  %15.8e  %12.6e  %12.6e  %12.6e\n", ctx.step,
                    f.meta.get(kEnergy), max_force, rms_force, step_norm);
    }
    ++n_rows_;
}

void OptLog::destroy(const cpu::DestroyContext& /*ctx*/) {
    if (os_.is_open())
        os_.close();
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
