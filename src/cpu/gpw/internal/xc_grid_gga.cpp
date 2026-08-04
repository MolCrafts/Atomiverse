/// @file src/cpu/gpw/internal/xc_grid_gga.cpp
/// @brief Implementation of pure GGA XC paths on the grid (see header).
///
/// GGA potential derivation (closed shell). The XC energy is
///   E_xc = integral e(rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb) dr,  e = rho*exc,
/// with sigma_aa=|grad rho_a|^2, sigma_ab=grad rho_a . grad rho_b, etc. The alpha
/// functional derivative is
///   v_a = de/drho_a - div[ de/d(grad rho_a) ]
///       = vrho_a - div[ 2 vsigma_aa grad rho_a + vsigma_ab grad rho_b ].
/// For rho_a = rho_b = rho/2 (so grad rho_a = grad rho_b = grad rho / 2),
///   v_a = vrho_a - div[ (vsigma_aa + 1/2 vsigma_ab) grad rho ].
/// The closed-shell KS XC block uses v = v_a (= v_b).

#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"

#include <array>
#include <stdexcept>

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {
/// Chunks for the grid-energy reductions here. One double per chunk, so the
/// count is a plain constant — wide enough that a dynamic schedule still
/// balances at any thread count we run. See cpu/device/chunked_range.h for why
/// `reduction(+ : e)` cannot be used on a path an SCF line search reads.
constexpr long kXcChunks = 256;
} // namespace

XcOnGrid xc_grid_gga(const UniformGrid& grid, const std::vector<double>& rho_r,
                     cpu::scf::XCType xc) {
    if (!cpu::scf::is_gga(xc)
        || cpu::scf::hybrid_exchange_fraction(xc) != 0.0) {
        throw std::invalid_argument(
            "xc_grid_gga: only pure GGA XCType (PBE, PBEsol, BLYP, …) "
            "supported");
    }
    const long ng = grid.npts();
    const double w = grid.weight();

    const std::array<std::vector<double>, 3> grad = gradient(grid, rho_r);

    XcOnGrid out;
    out.vxc_r.assign(static_cast<std::size_t>(ng), 0.0);
    std::array<std::vector<double>, 3>
        flux; // (vsigma_aa + 1/2 vsigma_ab) grad rho
    for (int d = 0; d < 3; ++d)
        flux[d].assign(static_cast<std::size_t>(ng), 0.0);

    const device::ChunkedRange range(ng, kXcChunks);
    std::vector<double> part(static_cast<std::size_t>(range.chunks()), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < range.chunks(); ++c) {
        double acc = 0.0;
        for (long g = range.begin(c); g < range.end(c); ++g) {
            const double rho = rho_r[g];
            const double gx = grad[0][g], gy = grad[1][g], gz = grad[2][g];
            const double sigma = gx * gx + gy * gy + gz * gz;
            // Spin-unpolarized: rho_a=rho_b=rho/2, grad rho_a=grad rho/2 =>
            // sigma_aa = sigma_ab = sigma_bb = sigma/4.
            const double ra = 0.5 * rho, rb = 0.5 * rho;
            const double s = 0.25 * sigma;
            const ATV::cpu::scf::XCOutput xco =
                ATV::cpu::scf::evaluate_xc(xc, ra, rb, s, s, s);
            acc += w * rho * xco.exc;
            out.vxc_r[g] =
                xco.vrho_alpha; // local part; divergence term added below
            const double vsig_eff = xco.vsigma_aa + 0.5 * xco.vsigma_ab;
            flux[0][g] = vsig_eff * gx;
            flux[1][g] = vsig_eff * gy;
            flux[2][g] = vsig_eff * gz;
        }
        part[static_cast<std::size_t>(c)] = acc;
    }
    double e = 0.0;
    for (double p : part)
        e += p;
    out.e_xc = e;

    const std::vector<double> divflux = divergence(grid, flux);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ng; ++g)
        out.vxc_r[g] -= divflux[g];

    return out;
}

XcOnGrid xc_grid_pbe(const UniformGrid& grid,
                     const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::PBE);
}

XcOnGrid xc_grid_pbesol(const UniformGrid& grid,
                        const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::PBEsol);
}

XcOnGrid xc_grid_revpbe(const UniformGrid& grid,
                        const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::RevPBE);
}

XcOnGrid xc_grid_rpbe(const UniformGrid& grid,
                      const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::RPBE);
}

XcOnGrid xc_grid_blyp(const UniformGrid& grid,
                      const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::BLYP);
}

XcOnGrid xc_grid_optb88_vdw(const UniformGrid& grid,
                            const std::vector<double>& rho_r) {
    return xc_grid_gga(grid, rho_r, cpu::scf::XCType::OptB88VdW);
}

// Spin-resolved GGA: v_σ = vrho_σ − div[ 2 vσ_σσ ∇ρ_σ + vσ_αβ ∇ρ_σ̄ ].
XcOnGridSpin xc_grid_gga_spin(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b,
                              cpu::scf::XCType xc) {
    if (!cpu::scf::is_gga(xc)
        || cpu::scf::hybrid_exchange_fraction(xc) != 0.0) {
        throw std::invalid_argument(
            "xc_grid_gga_spin: only pure GGA XCType (PBE, PBEsol, BLYP, …) "
            "supported");
    }
    const long ng = grid.npts();
    if (static_cast<long>(rho_a.size()) != ng
        || static_cast<long>(rho_b.size()) != ng) {
        throw std::invalid_argument(
            "xc_grid_gga_spin: rho_a/rho_b length must equal grid.npts()");
    }
    const double w = grid.weight();

    const std::array<std::vector<double>, 3> ga = gradient(grid, rho_a);
    const std::array<std::vector<double>, 3> gb = gradient(grid, rho_b);

    XcOnGridSpin out;
    out.vxc_a.assign(static_cast<std::size_t>(ng), 0.0);
    out.vxc_b.assign(static_cast<std::size_t>(ng), 0.0);
    std::array<std::vector<double>, 3> flux_a;
    std::array<std::vector<double>, 3> flux_b;
    for (int d = 0; d < 3; ++d) {
        flux_a[d].assign(static_cast<std::size_t>(ng), 0.0);
        flux_b[d].assign(static_cast<std::size_t>(ng), 0.0);
    }

    const device::ChunkedRange range(ng, kXcChunks);
    std::vector<double> part(static_cast<std::size_t>(range.chunks()), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < range.chunks(); ++c) {
        double acc = 0.0;
        for (long g = range.begin(c); g < range.end(c); ++g) {
            const double ra = rho_a[static_cast<std::size_t>(g)];
            const double rb = rho_b[static_cast<std::size_t>(g)];
            const double gax = ga[0][g], gay = ga[1][g], gaz = ga[2][g];
            const double gbx = gb[0][g], gby = gb[1][g], gbz = gb[2][g];
            const double saa = gax * gax + gay * gay + gaz * gaz;
            const double sbb = gbx * gbx + gby * gby + gbz * gbz;
            const double sab = gax * gbx + gay * gby + gaz * gbz;
            const ATV::cpu::scf::XCOutput xco =
                ATV::cpu::scf::evaluate_xc(xc, ra, rb, saa, sab, sbb);
            acc += w * (ra + rb) * xco.exc;
            out.vxc_a[static_cast<std::size_t>(g)] = xco.vrho_alpha;
            out.vxc_b[static_cast<std::size_t>(g)] = xco.vrho_beta;
            // flux_α = 2 vσ_αα ∇ρ_α + vσ_αβ ∇ρ_β
            // flux_β = 2 vσ_ββ ∇ρ_β + vσ_αβ ∇ρ_α
            const double caa = 2.0 * xco.vsigma_aa;
            const double cbb = 2.0 * xco.vsigma_bb;
            const double cab = xco.vsigma_ab;
            flux_a[0][g] = caa * gax + cab * gbx;
            flux_a[1][g] = caa * gay + cab * gby;
            flux_a[2][g] = caa * gaz + cab * gbz;
            flux_b[0][g] = cbb * gbx + cab * gax;
            flux_b[1][g] = cbb * gby + cab * gay;
            flux_b[2][g] = cbb * gbz + cab * gaz;
        }
        part[static_cast<std::size_t>(c)] = acc;
    }
    double e = 0.0;
    for (double p : part)
        e += p;
    out.e_xc = e;

    const std::vector<double> div_a = divergence(grid, flux_a);
    const std::vector<double> div_b = divergence(grid, flux_b);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ng; ++g) {
        out.vxc_a[static_cast<std::size_t>(g)] -= div_a[g];
        out.vxc_b[static_cast<std::size_t>(g)] -= div_b[g];
    }
    return out;
}

XcOnGridSpin xc_grid_pbe_spin(const UniformGrid& grid,
                              const std::vector<double>& rho_a,
                              const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::PBE);
}

XcOnGridSpin xc_grid_pbesol_spin(const UniformGrid& grid,
                                 const std::vector<double>& rho_a,
                                 const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::PBEsol);
}

XcOnGridSpin xc_grid_revpbe_spin(const UniformGrid& grid,
                                 const std::vector<double>& rho_a,
                                 const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::RevPBE);
}

XcOnGridSpin xc_grid_rpbe_spin(const UniformGrid& grid,
                               const std::vector<double>& rho_a,
                               const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::RPBE);
}

XcOnGridSpin xc_grid_blyp_spin(const UniformGrid& grid,
                               const std::vector<double>& rho_a,
                               const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::BLYP);
}

XcOnGridSpin xc_grid_optb88_vdw_spin(const UniformGrid& grid,
                                     const std::vector<double>& rho_a,
                                     const std::vector<double>& rho_b) {
    return xc_grid_gga_spin(grid, rho_a, rho_b, cpu::scf::XCType::OptB88VdW);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
