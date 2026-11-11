/// @file regressions/gpw/gpw-vdw-scf-potential/reg_gpw-vdw-scf-potential.cpp
/// @brief Γ GPW PBE + rVV10 self-consistent E_disp + analytic force FD gate.
///
/// Out-of-library consumer. SC nonlocal V_nl in SCF + forces (REFERENCE.md).
/// Offline CP2K 2026.1 TYPE RVV10 pin; no live oracle in ctest.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_vdw_scf_potential {

namespace {

constexpr double kA2B = 1.0 / 0.52917720859;

// Same water dimer as gpw-rvv10-ice / gpw-vdw-df-ice [Å].
const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

ATV::cpu::Frame make_frame(const double ang[6][3]) {
    using namespace ATV;
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_atomic_number)[i] = kZ[i];
        frame.span(kAtom_x)[i] = ang[i][0];
        frame.span(kAtom_y)[i] = ang[i][1];
        frame.span(kAtom_z)[i] = ang[i][2];
    }
    frame.box = cpu::Box::cubic(12.0);
    return frame;
}

ATV::cpu::gpw::GpwDft make_gpw() {
    using namespace ATV::cpu::gpw;
    GpwDft gpw({Kind{molrs::Element::O,
                     GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                     GthPotential(molrs::Element::O, "GTH-PBE-q6")},
                Kind{molrs::Element::H,
                     GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                     GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
               200.0, ATV::cpu::xc::XCType::PBE);
    gpw.n_occ = 8;
    gpw.max_iter = 40;
    gpw.tol = 1e-7;
    gpw.mix = 0.4;
    gpw.nonperiodic = false;
    gpw.rvv10 = true;
    return gpw;
}

} // namespace

int run() {
    using namespace ATV::cpu::gpw;

    // Offline CP2K 2026.1 SC Dispersion (TYPE RVV10) — see REFERENCE.md.
    // Post-SCF ATV was 6.11535e-2; SC should land nearer CP2K 6.11647e-2.
    constexpr double kEDispCp2k = 6.116467960721e-02; // Ha
    constexpr double kEDispTol = 1.0e-4;              // Ha

    ATV::cpu::Frame frame = make_frame(kAng);
    GpwDft gpw = make_gpw();
    gpw.build(frame);
    // One SCF + analytic forces at the SC density (Frame forces in eV/Å).
    const GpwResult res = gpw.forces(frame);

    if (!res.converged) {
        std::fprintf(stderr, "FAIL: SCF did not converge (iters=%d)\n",
                     res.iters);
        return 1;
    }
    if (!std::isfinite(res.e_disp)
        || std::abs(res.e_disp - kEDispCp2k) > kEDispTol) {
        std::fprintf(stderr,
                     "FAIL: SC E_disp rVV10 got=%.14e want=%.14e ±%.1e Ha\n",
                     res.e_disp, kEDispCp2k, kEDispTol);
        return 1;
    }

    // Central FD of E_total w.r.t. O0 x (H-bond axis) — non-trivial component.
    // h in Bohr.
    constexpr double h = 5.0e-4;
    double ang_p[6][3], ang_m[6][3];
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 3; ++k) {
            ang_p[i][k] = kAng[i][k];
            ang_m[i][k] = kAng[i][k];
        }
    ang_p[0][0] += h / kA2B; // Å
    ang_m[0][0] -= h / kA2B;

    ATV::cpu::Frame frame_p = make_frame(ang_p);
    ATV::cpu::Frame frame_m = make_frame(ang_m);
    GpwDft gpw_p = make_gpw();
    GpwDft gpw_m = make_gpw();
    gpw_p.build(frame_p);
    gpw_m.build(frame_m);
    const GpwResult rp = gpw_p.run(frame_p);
    const GpwResult rm = gpw_m.run(frame_m);
    if (!rp.converged || !rm.converged) {
        std::fprintf(stderr, "FAIL: FD SCF did not converge\n");
        return 1;
    }
    // Force = −dE/dR; E in Ha, R in Bohr.
    const double f_fd_x = -(rp.e_total - rm.e_total) / (2.0 * h);
    const double f_ana_x = frame.span(ATV::kAtom_fx)[0]
                           / ATV::units::kHartreePerBohrInEvPerAngstrom;
    const double df = std::abs(f_fd_x - f_ana_x);
    constexpr double kForceTol = 5.0e-4; // Ha/Bohr

    std::printf("SC E_disp=%.14e Ha (CP2K %.14e ± %.1e); "
                "F_O0_x ana=%.6e FD=%.6e |Δ|=%.3e Ha/Bohr\n",
                res.e_disp, kEDispCp2k, kEDispTol, f_ana_x, f_fd_x, df);

    if (!(std::abs(f_ana_x) > 1.0e-5)) {
        std::fprintf(stderr,
                     "FAIL: expected non-trivial F_O0_x, got ana=%.6e\n",
                     f_ana_x);
        return 1;
    }
    if (!std::isfinite(f_ana_x) || !std::isfinite(f_fd_x) || df > kForceTol) {
        std::fprintf(stderr,
                     "FAIL: force FD gate |F_ana - F_FD|=%.3e > %.1e Ha/Bohr "
                     "(ana=%.6e FD=%.6e)\n",
                     df, kForceTol, f_ana_x, f_fd_x);
        return 1;
    }

    std::printf("PASS: gpw-vdw-scf-potential rVV10 SC E_disp + force FD\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_vdw_scf_potential

ATV_REGRESSION("gpw/vdw-df", "gpw-vdw-scf-potential",
               atv_reg_gpw_gpw_vdw_scf_potential::run);
