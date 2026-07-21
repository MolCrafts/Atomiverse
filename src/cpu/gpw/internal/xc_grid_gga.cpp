/// @file src/cpu/gpw/internal/xc_grid_gga.cpp
/// @brief Implementation of the PBE GGA XC path on the grid (see header).
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

#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/scf/xc_functional.h" // shared libxc-parity PBE

namespace ATV {
namespace cpu {
namespace gpw {

XcOnGrid xc_grid_pbe(const UniformGrid& grid,
                     const std::vector<double>& rho_r) {
    const long ng = grid.npts();
    const double w = grid.weight();

    const std::array<std::vector<double>, 3> grad = gradient(grid, rho_r);

    XcOnGrid out;
    out.vxc_r.assign(static_cast<std::size_t>(ng), 0.0);
    std::array<std::vector<double>, 3>
        flux; // (vsigma_aa + 1/2 vsigma_ab) grad rho
    for (int d = 0; d < 3; ++d)
        flux[d].assign(static_cast<std::size_t>(ng), 0.0);

    double e = 0.0;
#pragma omp parallel for reduction(+ : e) schedule(static)
    for (long g = 0; g < ng; ++g) {
        const double rho = rho_r[g];
        const double gx = grad[0][g], gy = grad[1][g], gz = grad[2][g];
        const double sigma = gx * gx + gy * gy + gz * gz;
        // Spin-unpolarized: rho_a=rho_b=rho/2, grad rho_a=grad rho/2 =>
        // sigma_aa = sigma_ab = sigma_bb = sigma/4.
        const double ra = 0.5 * rho, rb = 0.5 * rho;
        const double s = 0.25 * sigma;
        const ATV::cpu::scf::XCOutput xc = ATV::cpu::scf::evaluate_xc(
            ATV::cpu::scf::XCType::PBE, ra, rb, s, s, s);
        e += w * rho * xc.exc;
        out.vxc_r[g] = xc.vrho_alpha; // local part; divergence term added below
        const double vsig_eff = xc.vsigma_aa + 0.5 * xc.vsigma_ab;
        flux[0][g] = vsig_eff * gx;
        flux[1][g] = vsig_eff * gy;
        flux[2][g] = vsig_eff * gz;
    }
    out.e_xc = e;

    const std::vector<double> divflux = divergence(grid, flux);
    for (long g = 0; g < ng; ++g)
        out.vxc_r[g] -= divflux[g];

    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
