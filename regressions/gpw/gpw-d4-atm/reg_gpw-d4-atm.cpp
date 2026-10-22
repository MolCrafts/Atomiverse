/// @file regressions/gpw/gpw-d4-atm/reg_gpw-d4-atm.cpp
/// @brief Periodic water-dimer GPW PBE-D4 + ATM (s9=1) E_disp vs offline pin.
///
/// Out-of-library consumer: public GPW headers only. Geometry and E_disp pin
/// are in REFERENCE.md (dftd4 3.7.0 offline; no live oracle in ctest).

#include <cmath>
#include <cstdio>
#include <vector>

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

    // Geometry [Å] (REFERENCE.md water dimer; same as gpw-d4-water).
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    const cpu::gpw::Kind O{
        molrs::Element::O,
        GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = Z[i];
        frame.span(kAtom_x)[i] = ang[i][0];
        frame.span(kAtom_y)[i] = ang[i][1];
        frame.span(kAtom_z)[i] = ang[i][2];
    }

    // Offline dftd4 3.7.0 pbe-D4-ATM pin (REFERENCE.md).
    constexpr double kEDispRef = -9.9554475834240288e-04; // Ha
    constexpr double kTol = 1.0e-8;                       // Ha
    // Two-body-only context (must differ when ATM is on).
    constexpr double kEDispTwoBody = -9.9458842745846774e-04;

    frame.box = cpu::Box::cubic(12.0);

    // Low cutoff: e_disp is geometry-only; short SCF only.
    GpwDft gpw{{O, H}, 200.0, XcKind::PBE};
    gpw.n_occ = 8;
    gpw.max_iter = 8;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.nonperiodic = false;
    gpw.d4 = true;
    gpw.d4_atm = true;

    gpw.build(frame);
    const auto& res = gpw.run(frame);

    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp D4+ATM/PBE periodic water dimer "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (std::abs(res.e_disp - kEDispTwoBody) < 1e-10) {
        std::fprintf(stderr,
                     "FAIL: E_disp matches two-body-only pin; ATM not active "
                     "(got=%.14e)\n",
                     res.e_disp);
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

    std::printf("PASS: E_disp PBE-D4+ATM periodic water dimer = %.14e Ha "
                "(want %.14e +/- %.1e) [dftd4 3.7.0 offline pin]; "
                "e_total=%.8f Ha (SCF iters=%d)\n",
                res.e_disp, kEDispRef, kTol, res.e_total, res.iters);
    return 0;
}
