/// @file src/cpu/communication/rebalance.cpp
/// @brief `Rebalance` -- histogram-based non-uniform brick rebalance.
///
/// Algorithm per launch:
///   1. Each rank builds three 1D histograms (`n_bins_` bins each) of its
///      owned atoms' wrapped positions along x, y, z.
///   2. `MPI_Allreduce(SUM)` produces an identical global histogram on every
///      rank, so the cut-plane computation is deterministic across ranks.
///   3. For each axis the cumulative histogram is bisected into `dims[d]`
///      equal-count slabs by linear interpolation between bin edges. The
///      resulting `dims[d] + 1` fractional cut positions are stored in
///      `dd->cuts[d]`.
///   4. `compute_sublo_subhi(frame.box)` re-derives this rank's `sublo/subhi`
///      from the new cuts; `neighbor_generation` and `partition_dirty` tell
///      bonded caches and Verlet nflag that owners may need to move.
///
/// Bit-identical cuts on every rank are essential: migration would otherwise
/// see disagreeing destinations and lose atoms. The `MPI_Allreduce` plus
/// purely-deterministic post-processing guarantees this by construction.
///
/// Algorithm transplanted from LAMMPS `src/balance.cpp` (`balance dynamic xyz`,
/// checkout `78a7ddf`).
///
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.

#include "atomiverse/cpu/communication/rebalance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

Rebalance::Rebalance(int every, int n_bins)
    : every_(std::max(1, every)), n_bins_(std::max(8, n_bins)) {}

#ifdef ATV_HAS_MPI

namespace {

// Build a `n_bins`-bin histogram of wrapped positions along a given axis.
void build_local_histogram(const Frame& frame, int n_local, int axis,
                           const cpu::Box& box, int n_bins,
                           std::vector<std::int64_t>& hist) {
    std::fill(hist.begin(), hist.end(), 0);
    const double L = box.length(axis);
    if (L <= 0.0 || n_local <= 0)
        return;
    const double inv_bin = static_cast<double>(n_bins) / L;
    // Hoisted ONCE per call (axis is fixed for the whole loop below) --
    // never resolve a span inside the per-atom loop.
    const StorageReal* __restrict q_ptr = nullptr;
    switch (axis) {
    case 0:
        q_ptr = frame.span(kAtom_x);
        break;
    case 1:
        q_ptr = frame.span(kAtom_y);
        break;
    case 2:
        q_ptr = frame.span(kAtom_z);
        break;
    default:
        break;
    }
    for (int i = 0; i < n_local; ++i) {
        const double q = (q_ptr != nullptr) ? q_ptr[i] : 0.0;
        const double wrapped = box.wrap_axis(q, axis);
        int bin = static_cast<int>(wrapped * inv_bin);
        if (bin < 0)
            bin = 0;
        if (bin >= n_bins)
            bin = n_bins - 1;
        hist[static_cast<std::size_t>(bin)] += 1;
    }
}

// Compute cut planes (as box-length fractions in [0, 1]) for one axis from
// a global histogram of owner counts. The resulting `cuts` vector has size
// `dims_d + 1`, with `cuts.front() == 0` and `cuts.back() == 1`. Targets
// `total / dims_d` atoms per slab; ties broken by linear interpolation
// inside the bin where the cumulative crosses the target.
std::vector<double> cuts_from_histogram(const std::vector<std::int64_t>& hist,
                                        int n_bins, int dims_d) {
    std::vector<double> cuts(static_cast<std::size_t>(dims_d + 1), 0.0);
    cuts.back() = 1.0;
    if (dims_d <= 1)
        return cuts;

    std::int64_t total = 0;
    for (auto v : hist)
        total += v;
    if (total <= 0) {
        // No atoms (degenerate); fall back to uniform cuts.
        for (int k = 0; k <= dims_d; ++k) {
            cuts[static_cast<std::size_t>(k)] =
                static_cast<double>(k) / static_cast<double>(dims_d);
        }
        return cuts;
    }

    const double per_slab =
        static_cast<double>(total) / static_cast<double>(dims_d);
    std::int64_t cum = 0;
    int next_cut = 1; // cuts[0] is always 0 by construction
    const double inv_bins = 1.0 / static_cast<double>(n_bins);

    for (int b = 0; b < n_bins && next_cut < dims_d; ++b) {
        const std::int64_t cum_after = cum + hist[static_cast<std::size_t>(b)];
        // Place every cut whose target falls inside bin b.
        while (next_cut < dims_d) {
            const double target = per_slab * static_cast<double>(next_cut);
            if (static_cast<double>(cum_after) >= target) {
                // Linearly interpolate inside bin b.
                const double bin_start = static_cast<double>(b);
                const double in_bin =
                    (hist[static_cast<std::size_t>(b)] > 0)
                        ? (target - static_cast<double>(cum))
                              / static_cast<double>(
                                  hist[static_cast<std::size_t>(b)])
                        : 0.0;
                const double clamped_in_bin = std::clamp(in_bin, 0.0, 1.0);
                const double frac = (bin_start + clamped_in_bin) * inv_bins;
                cuts[static_cast<std::size_t>(next_cut)] =
                    std::clamp(frac, 0.0, 1.0);
                ++next_cut;
            } else {
                break;
            }
        }
        cum = cum_after;
    }
    // Any leftover cuts (if cumulative under-shot) clamp to 1.0; this
    // shouldn't happen mathematically but keeps cuts strictly monotonic.
    for (int k = next_cut; k < dims_d; ++k) {
        cuts[static_cast<std::size_t>(k)] = 1.0;
    }

    // Enforce monotonicity (FP safety).
    for (int k = 1; k <= dims_d; ++k) {
        if (cuts[static_cast<std::size_t>(k)]
            < cuts[static_cast<std::size_t>(k - 1)]) {
            cuts[static_cast<std::size_t>(k)] =
                cuts[static_cast<std::size_t>(k - 1)];
        }
    }
    return cuts;
}

} // namespace

void Rebalance::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1)
        return;
    if (ctx.step <= 0)
        return; // skip step 0 -- positions are user-supplied, not equilibrated
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    if (dd->dims[0] <= 1 && dd->dims[1] <= 1 && dd->dims[2] <= 1)
        return;

    const Frame& frame = ctx.frame;
    const double Lx = frame.box.length(0);
    const double Ly = frame.box.length(1);
    const double Lz = frame.box.length(2);
    if (Lx <= 0.0 || Ly <= 0.0 || Lz <= 0.0)
        return;

    const int n_local = dd->n_local;

    // 3 axis-aligned histograms. Packed into one 3*n_bins buffer for a single
    // allreduce.
    std::vector<std::int64_t> hist_local(static_cast<std::size_t>(3 * n_bins_),
                                         0);
    std::vector<std::int64_t> hist_global(static_cast<std::size_t>(3 * n_bins_),
                                          0);

    {
        std::vector<std::int64_t> tmp(static_cast<std::size_t>(n_bins_), 0);
        build_local_histogram(frame, n_local, 0, frame.box, n_bins_, tmp);
        std::copy(tmp.begin(), tmp.end(), hist_local.begin());
        build_local_histogram(frame, n_local, 1, frame.box, n_bins_, tmp);
        std::copy(tmp.begin(), tmp.end(), hist_local.begin() + n_bins_);
        build_local_histogram(frame, n_local, 2, frame.box, n_bins_, tmp);
        std::copy(tmp.begin(), tmp.end(), hist_local.begin() + 2 * n_bins_);
    }

    MPI_Allreduce(hist_local.data(), hist_global.data(), 3 * n_bins_,
                  MPI_INT64_T, MPI_SUM, dd->cart_comm);

    // Bisect each axis independently. dims[d] == 1 leaves cuts[d] empty so
    // the uniform-axis path stays in compute_sublo_subhi (no per-axis cost).
    for (int d = 0; d < 3; ++d) {
        if (dd->dims[d] <= 1) {
            dd->cuts[d].clear();
            continue;
        }
        std::vector<std::int64_t> axis_hist(hist_global.begin() + d * n_bins_,
                                            hist_global.begin()
                                                + (d + 1) * n_bins_);
        dd->cuts[d] = cuts_from_histogram(axis_hist, n_bins_, dd->dims[d]);
    }

    // Refresh this rank's slab from the new cuts.
    dd->compute_sublo_subhi(frame.box);

    ++dd->neighbor_generation;
    dd->partition_dirty = true;
}

#else // !ATV_HAS_MPI

void Rebalance::launch(const cpu::LaunchContext& /*ctx*/) {}

#endif // ATV_HAS_MPI

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
