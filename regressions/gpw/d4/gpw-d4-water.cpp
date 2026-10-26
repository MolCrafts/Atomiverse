/// @file regressions/gpw/gpw-d4-water/reg_gpw-d4-water.cpp
/// @brief Periodic water-dimer GPW PBE-D4 E_disp vs offline dftd4 pin.
///
/// Out-of-library consumer: public GPW headers only. Geometry and E_disp pin
/// are in REFERENCE.md (dftd4 3.7.0 offline; no live oracle in ctest).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_d4_water {

int run() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::Kind;
    using cpu::gpw::XcKind;

    // Geometry [Å] (REFERENCE.md water dimer).
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
    frame.box = cpu::Box::cubic(12.0); // Å

    // Offline dftd4 3.7.0 pin (REFERENCE.md): Dispersion energy, two-body.
    constexpr double kEDispRef = -0.00099458842745846774; // Ha
    constexpr double kTol = 1.0e-8;                       // Ha

    // Low cutoff: e_disp is geometry-only; short SCF only.
    GpwDft gpw{{Kind{molrs::Element::O,
                     GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                     GthPotential(molrs::Element::O, "GTH-PBE-q6")},
                Kind{molrs::Element::H,
                     GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                     GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
               200.0,
               XcKind::PBE};
    gpw.n_occ = 8;
    gpw.max_iter = 8;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.nonperiodic = false;
    gpw.d4 = true;

    gpw.build(frame);
    const auto& res = gpw.run(frame);

    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp D4/PBE periodic water dimer "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (!std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: e_total non-finite\n");
        return 1;
    }
    if (res.e_disp >= 0.0) {
        std::fprintf(stderr, "FAIL: expected attractive E_disp, got %.6e\n",
                     res.e_disp);
        return 1;
    }

    std::printf("PASS: E_disp PBE-D4 periodic water dimer = %.14e Ha "
                "(want %.14e +/- %.1e) [dftd4 3.7.0 offline pin]; "
                "e_total=%.8f Ha (SCF iters=%d)\n",
                res.e_disp, kEDispRef, kTol, res.e_total, res.iters);
    return 0;
}

} // namespace atv_reg_gpw_gpw_d4_water

ATV_REGRESSION("gpw/d4", "gpw-d4-water", atv_reg_gpw_gpw_d4_water::run);
