/// @file regressions/gpw/gpw-mbd-hirshfeld/reg_gpw-mbd-hirshfeld.cpp
/// @brief Water-dimer GPW PBE + Hirshfeld-scaled MBD@rsSCS vs offline pin.
///
/// Out-of-library consumer: public GPW headers only. Geometry and E_disp pin
/// are in REFERENCE.md (libMBD + dual Hirshfeld; no live oracle in ctest).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_mbd_hirshfeld {

int run() {
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
    frame.box = cpu::Box::cubic(12.0); // Å

    // Offline pin (REFERENCE.md): Hirshfeld-scaled MBD@rsSCS, isolated MBD
    // geometry with volumes from the GPW PBE density (cutoff 200 Ry; density-
    // dependent E_disp still basis/cutoff sensitive at mHa level — hard gate
    // is this fixed product setting).
    constexpr double kEDispRef = -1.1479852469005181e-03; // Ha
    constexpr double kTol = 1.0e-8;                       // Ha

    GpwDft gpw{
        {cpu::gpw::Kind{molrs::Element::O,
                        GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                        GthPotential(molrs::Element::O, "GTH-PBE-q6")},
         cpu::gpw::Kind{molrs::Element::H,
                        GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                        GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
        200.0,
        XcKind::PBE};
    gpw.n_occ = 8;
    gpw.max_iter = 16;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.nonperiodic = false;
    gpw.mbd = true;
    gpw.hirshfeld = true;

    gpw.build(frame);
    const auto& res = gpw.run(frame);

    std::printf("PIN_CANDIDATE E_disp=%.16e Ha  volumes:", res.e_disp);
    for (double v : res.volume_ratios)
        std::printf(" %.8f", v);
    std::printf("  e_total=%.10f iters=%d conv=%d\n", res.e_total, res.iters,
                res.converged ? 1 : 0);

    if (res.volume_ratios.size() != 6) {
        std::fprintf(stderr, "FAIL: expected 6 volume ratios, got %zu\n",
                     res.volume_ratios.size());
        return 1;
    }
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i) {
        if (!(res.volume_ratios[i] > 0.3) || !(res.volume_ratios[i] < 1.5)) {
            std::fprintf(stderr, "FAIL: volume_ratios[%zu]=%.6f out of range\n",
                         i, res.volume_ratios[i]);
            return 1;
        }
    }
    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp Hirshfeld-MBD water dimer "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (!std::isfinite(res.e_total) || res.e_disp >= 0.0) {
        std::fprintf(stderr,
                     "FAIL: e_total non-finite or E_disp not attractive\n");
        return 1;
    }

    std::printf("PASS: E_disp PBE-MBD@rsSCS Hirshfeld water dimer = %.14e Ha "
                "(want %.14e +/- %.1e)\n",
                res.e_disp, kEDispRef, kTol);
    return 0;
}

} // namespace atv_reg_gpw_gpw_mbd_hirshfeld

ATV_REGRESSION("gpw/mbd", "gpw-mbd-hirshfeld",
               atv_reg_gpw_gpw_mbd_hirshfeld::run);
