/// @file regressions/gpw/gpw-vdw-df-ice/reg_gpw-vdw-df-ice.cpp
/// @brief Periodic water-dimer GPW PBE + vdW-DF (DRSLL) E_disp offline pin.
///
/// Out-of-library consumer. E_disp is post-SCF nonlocal correlation on the
/// final density (REFERENCE.md). Kernel unit-tested vs libvdwxc; full-SCF
/// pin is ATV dual-frozen after unit-test gate (no live oracle in ctest).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::XcKind;

    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = Z[i];
        frame.span(kAtom_x)[i] = ang[i][0];
        frame.span(kAtom_y)[i] = ang[i][1];
        frame.span(kAtom_z)[i] = ang[i][2];
    }
    frame.box = cpu::Box::cubic(12.0);

    // Offline pin (REFERENCE.md): post-SCF E_c^{nl} DRSLL on Γ-PBE density.
    constexpr double kEDispRef = 1.5194390480303990e-01; // Ha
    constexpr double kTol = 5.0e-5; // Ha (post-SCF density + float32 kernel)

    GpwDft gpw(
        {cpu::gpw::Kind{molrs::Element::O,
                        GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                        GthPotential(molrs::Element::O, "GTH-PBE-q6")},
         cpu::gpw::Kind{molrs::Element::H,
                        GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                        GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
        200.0, XcKind::PBE);
    gpw.n_occ = 8;
    gpw.max_iter = 40;
    gpw.tol = 1e-7;
    gpw.mix = 0.4;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;

    gpw.build(frame);
    const auto& res = gpw.run(frame);

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
