/// @file src/cpu/gpw/internal/xc_grid.cpp
/// @brief Implementation of XC energy/potential on the GPW uniform grid.

#include "atomiverse/cpu/gpw/internal/xc_grid.h"

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/xc/pade.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {
/// Chunks for the grid-energy reductions here. The accumulator is one double
/// per chunk, so the count is a plain constant — wide enough that a dynamic
/// schedule still balances at any thread count we run.
constexpr long kXcChunks = 256;
} // namespace

XcOnGrid xc_grid_pade(const UniformGrid& grid,
                      const std::vector<double>& rho_r) {
    const long ng = grid.npts();
    const double w = grid.weight();
    XcOnGrid out;
    out.vxc_r.assign(static_cast<std::size_t>(ng), 0.0);
    // Reproducible reduction: fixed chunks, folded in ascending chunk order.
    // `reduction(+ : e)` left both the split and the merge order to libgomp, so
    // E_xc differed run to run at one thread count (see chunked_range.h).
    const device::ChunkedRange range(ng, kXcChunks);
    std::vector<double> part(static_cast<std::size_t>(range.chunks()), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < range.chunks(); ++c) {
        double s = 0.0;
        for (long g = range.begin(c); g < range.end(c); ++g) {
            const double rho = rho_r[g];
            // Spin-unpolarized: split evenly. pade_lsd applies its own
            // low-density cutoff, returning zeros below threshold.
            const ATV::cpu::xc::PadeXC xc =
                ATV::cpu::xc::pade_lsd(0.5 * rho, 0.5 * rho);
            s += w * rho * xc.exc;
            out.vxc_r[g] = xc.vrho_a; // = vrho_b in the unpolarized case
        }
        part[static_cast<std::size_t>(c)] = s;
    }
    double e = 0.0;
    for (double p : part)
        e += p;
    out.e_xc = e;
    return out;
}

XcOnGridSpin xc_grid_pade_lsd(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b) {
    const long ng = grid.npts();
    const double w = grid.weight();
    XcOnGridSpin out;
    out.vxc_a.assign(static_cast<std::size_t>(ng), 0.0);
    out.vxc_b.assign(static_cast<std::size_t>(ng), 0.0);
    const device::ChunkedRange range(ng, kXcChunks);
    std::vector<double> part(static_cast<std::size_t>(range.chunks()), 0.0);
    // Vacuum short-circuit: pade_lsd returns zeros for rho_total < 1e-12; skip
    // the call (and leave pre-zeroed V_xc) on empty regions of large MT boxes.
    constexpr double kRhoCut = 1.0e-12;
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < range.chunks(); ++c) {
        double s = 0.0;
        for (long g = range.begin(c); g < range.end(c); ++g) {
            const double ra = rho_a[static_cast<std::size_t>(g)];
            const double rb = rho_b[static_cast<std::size_t>(g)];
            if (ra + rb < kRhoCut)
                continue;
            const ATV::cpu::xc::PadeXC xc = ATV::cpu::xc::pade_lsd(ra, rb);
            s += w * (ra + rb) * xc.exc;
            out.vxc_a[static_cast<std::size_t>(g)] = xc.vrho_a;
            out.vxc_b[static_cast<std::size_t>(g)] = xc.vrho_b;
        }
        part[static_cast<std::size_t>(c)] = s;
    }
    double e = 0.0;
    for (double p : part)
        e += p;
    out.e_xc = e;
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
