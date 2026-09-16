/// @file src/cpu/gpw/internal/local_pp.cpp
/// @brief GTH local short-range PP on the grid + its Hellmann-Feynman force.
///        See the header for why these two live together.

#include "atomiverse/cpu/gpw/internal/local_pp.h"

#include "atomiverse/cpu/device/chunked_range.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

/// Minimum-image displacement r - R_A, matching the SCF collocation convention.
inline void mic_disp(const double r[3], const double pos[3], const double L[3],
                     bool periodic, double d[3]) {
    for (int k = 0; k < 3; ++k) {
        d[k] = r[k] - pos[k];
        if (periodic && L[k] > 0.0)
            d[k] -= L[k] * std::nearbyint(d[k] / L[k]);
    }
}

/// Chunks for the grid force reduction (one 3-vector per chunk).
constexpr long kForceChunks = 256;

} // namespace

std::vector<double> local_pp_grid(const UniformGrid& grid,
                                  const std::vector<GpwAtom>& atoms,
                                  bool periodic) {
    const long ng = grid.npts();
    std::vector<double> v(static_cast<std::size_t>(ng), 0.0);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double s = 0.0;
        for (const auto& a : atoms) {
            if (a.cloc.empty())
                continue;
            double d[3];
            mic_disp(r, a.pos, grid.L, periodic, d);
            const double t =
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (a.rloc * a.rloc);
            double poly = 0.0, tp = 1.0;
            for (double c : a.cloc) {
                poly += c * tp;
                tp *= t;
            }
            s += std::exp(-0.5 * t) * poly;
        }
        v[static_cast<std::size_t>(g)] = s;
    }
    return v;
}

void add_local_pp_hf_forces(const UniformGrid& grid,
                            const std::vector<GpwAtom>& atoms, bool periodic,
                            const std::vector<double>& rho,
                            std::vector<std::array<double, 3>>& forces) {
    const long ng = grid.npts();
    const double wgt = grid.weight();
    const int na = static_cast<int>(atoms.size());

    for (int A = 0; A < na; ++A) {
        const GpwAtom& a = atoms[static_cast<std::size_t>(A)];
        if (a.cloc.empty())
            continue;
        const double inv_s2 = 1.0 / (a.rloc * a.rloc);
        // Reproducible reduction over the grid: fixed chunks, folded in
        // ascending chunk order (see cpu/device/chunked_range.h). `reduction(+
        // : fx, fy, fz)` made this force differ run to run at one thread count.
        const device::ChunkedRange range(ng, kForceChunks);
        std::vector<std::array<double, 3>> part(
            static_cast<std::size_t>(range.chunks()), {0.0, 0.0, 0.0});
#pragma omp parallel for schedule(dynamic, 1)
        for (long c = 0; c < range.chunks(); ++c) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (long g = range.begin(c); g < range.end(c); ++g) {
                const double rg = rho[static_cast<std::size_t>(g)];
                if (rg == 0.0)
                    continue;
                double r[3], d[3];
                grid.point(g, r);
                mic_disp(r, a.pos, grid.L, periodic, d);
                const double t =
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) * inv_s2;

                // poly(t) = Σ_k C_k t^k and poly'(t) = Σ_k k C_k t^(k-1). Carry
                // t^(k-1) explicitly — dividing t^k by t blows up at the
                // nucleus (t = 0).
                double poly = 0.0, dpoly = 0.0;
                double tp = 1.0, tpm1 = 0.0; // t^k, t^(k-1)
                for (std::size_t k = 0; k < a.cloc.size(); ++k) {
                    poly += a.cloc[k] * tp;
                    if (k > 0)
                        dpoly += static_cast<double>(k) * a.cloc[k] * tpm1;
                    tpm1 = tp;
                    tp *= t;
                }
                // dV/dt = exp(-t/2) [poly'(t) - ½ poly(t)]; ∂t/∂R_A,k =
                // -2 d_k/rloc². F_A,k = -∂/∂R_A,k ∫ρV = ∫ρ · dV/dt · 2 d_k/rloc².
                const double dvdt = std::exp(-0.5 * t) * (dpoly - 0.5 * poly);
                const double cc = rg * dvdt * 2.0 * inv_s2;
                sx += cc * d[0];
                sy += cc * d[1];
                sz += cc * d[2];
            }
            part[static_cast<std::size_t>(c)] = {sx, sy, sz};
        }
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (const auto& p : part) {
            fx += p[0];
            fy += p[1];
            fz += p[2];
        }
        forces[static_cast<std::size_t>(A)][0] += wgt * fx;
        forces[static_cast<std::size_t>(A)][1] += wgt * fy;
        forces[static_cast<std::size_t>(A)][2] += wgt * fz;
    }
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
