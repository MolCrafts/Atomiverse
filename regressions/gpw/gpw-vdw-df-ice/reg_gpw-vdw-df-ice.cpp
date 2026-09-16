/// @file regressions/gpw/gpw-vdw-df-ice/reg_gpw-vdw-df-ice.cpp
/// @brief Periodic water-dimer GPW PBE + vdW-DF (DRSLL) E_disp offline pin.
///
/// Out-of-library consumer. E_disp is post-SCF nonlocal correlation on the
/// final density (REFERENCE.md). Kernel unit-tested vs libvdwxc; full-SCF
/// pin is ATV dual-frozen after unit-test gate (no live oracle in ctest).

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::XcKind;

    constexpr double kA2B = 1.0 / 0.52917720859;
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");

    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kA2B, ang[i][1] * kA2B,
                               ang[i][2] * kA2B};
        if (Z[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }

    // Offline pin (REFERENCE.md): post-SCF E_c^{nl} DRSLL on Γ-PBE density.
    constexpr double kEDispRef = 1.5194390480303990e-01; // Ha
    constexpr double kTol = 5.0e-5; // Ha (post-SCF density + float32 kernel)

    GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 40;
    gpw.tol = 1e-7;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);

    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp vdW-DF DRSLL water dimer "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (!std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: e_total non-finite\n");
        return 1;
    }

    std::printf("PASS: E_disp PBE+vdW-DF(DRSLL) water dimer = %.14e Ha "
                "(want %.14e +/- %.1e); e_total=%.8f Ha (SCF iters=%d)\n",
                res.e_disp, kEDispRef, kTol, res.e_total, res.iters);
    return 0;
}
