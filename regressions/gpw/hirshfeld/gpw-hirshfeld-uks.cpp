/// @file regressions/gpw/gpw-hirshfeld-uks/reg_gpw-hirshfeld-uks.cpp
/// @brief O₂ triplet Γ-UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS vs offline pin.
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

namespace atv_reg_gpw_gpw_hirshfeld_uks {

int run() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::Kind;
    using cpu::gpw::XcKind;

    // O₂, R = 1.2075 Å, centered in 12 Å cubic cell (MT / nonperiodic).
    constexpr double L_ang = 12.0;
    constexpr double R_ang = 1.2075;
    const double z[2] = {0.5 * L_ang - 0.5 * R_ang, 0.5 * L_ang + 0.5 * R_ang};
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_x)[i] = 0.5 * L_ang;
        frame.span(kAtom_y)[i] = 0.5 * L_ang;
        frame.span(kAtom_z)[i] = z[i];
    }
    frame.box = cpu::Box::cubic(L_ang);

    // Offline pin (REFERENCE.md): Hirshfeld-scaled MBD@rsSCS, isolated MBD
    // geometry with volumes from the Γ-UKS PBE density (cutoff 200 Ry).
    constexpr double kEDispRef = -1.0379964698969957e-04; // Ha (PIN_CANDIDATE)
    constexpr double kTol = 1.0e-8;                       // Ha

    GpwDft gpw{{Kind{molrs::Element::O,
                     GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
                     GthPotential(molrs::Element::O, "GTH-PBE-q6")}},
               200.0,
               XcKind::PBE};
    gpw.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    gpw.n_beta = 5;
    gpw.max_iter = 150;
    gpw.tol = 1e-6;
    gpw.mix = 0.35;
    gpw.diis_depth = 10;
    gpw.nonperiodic = true;
    gpw.mbd = true;
    gpw.hirshfeld = true;

    gpw.build(frame);
    const auto& res = gpw.run(frame);

    std::printf("PIN_CANDIDATE E_disp=%.16e Ha  volumes:", res.e_disp);
    for (double v : res.volume_ratios)
        std::printf(" %.8f", v);
    std::printf("  e_total=%.10f s2=%.6f iters=%d conv=%d\n", res.e_total,
                res.s2, res.iters, res.converged ? 1 : 0);

    if (!res.converged) {
        std::fprintf(stderr, "FAIL: SCF did not converge (iters=%d)\n",
                     res.iters);
        return 1;
    }
    if (res.volume_ratios.size() != 2) {
        std::fprintf(stderr, "FAIL: expected 2 volume ratios, got %zu\n",
                     res.volume_ratios.size());
        return 1;
    }
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i) {
        if (!(res.volume_ratios[i] > 0.2) || !(res.volume_ratios[i] < 2.0)) {
            std::fprintf(stderr, "FAIL: volume_ratios[%zu]=%.6f out of range\n",
                         i, res.volume_ratios[i]);
            return 1;
        }
    }
    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp Hirshfeld-MBD O2 UKS "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (!std::isfinite(res.e_total) || res.e_disp >= 0.0) {
        std::fprintf(stderr,
                     "FAIL: e_total non-finite or E_disp not attractive\n");
        return 1;
    }

    std::printf("PASS: E_disp PBE-MBD@rsSCS Hirshfeld O2 UKS = %.14e Ha "
                "(want %.14e +/- %.1e)  volumes=%.6f %.6f\n",
                res.e_disp, kEDispRef, kTol, res.volume_ratios[0],
                res.volume_ratios[1]);
    return 0;
}

} // namespace atv_reg_gpw_gpw_hirshfeld_uks

ATV_REGRESSION("gpw/hirshfeld", "gpw-hirshfeld-uks",
               atv_reg_gpw_gpw_hirshfeld_uks::run);
