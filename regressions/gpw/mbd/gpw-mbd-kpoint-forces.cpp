/// @file regressions/gpw/gpw-mbd-kpoint-forces/reg_gpw-mbd-kpoint-forces.cpp
/// @brief Multi-k GPW analytic forces fold free-atom MBD@rsSCS.
///
/// Out-of-library consumer: public GPW headers only. Geometry and pins in
/// REFERENCE.md (libMBD offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_mbd_kpoint_forces {

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

// Offline libMBD Fortran pin (same as gpw-mbd-forces / gpw-mbd-ice).
constexpr double kEDispRef = -1.4928804171407961e-03; // Ha
constexpr double kETol = 1.0e-8;                      // Ha
constexpr double kFTol = 1.0e-6;                      // Ha/Bohr
constexpr double kSumFTol = 1.0e-8;                   // Ha/Bohr
constexpr double kWireTol = 1.0e-8;                   // Ha/Bohr

} // namespace

int run() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::Kind;
    using cpu::gpw::mbd_rsscs_pbe;
    using cpu::gpw::MbdResult;
    using cpu::gpw::XcKind;

    constexpr double kA2B = 1.0 / 0.52917720859;

    // ── A: geometry-only MBD dual-gate (water dimer, free-atom) ───────────
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Zarr[6] = {8, 1, 1, 8, 1, 1};
    std::vector<int> Zw(Zarr, Zarr + 6);
    std::vector<std::array<double, 3>> posw(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            posw[static_cast<std::size_t>(i)][d] = ang[i][d] * kA2B;

    const double cell0[3] = {0.0, 0.0, 0.0};
    const MbdResult mbd = mbd_rsscs_pbe(Zw, posw, cell0, /*periodic=*/false);
    check(std::isfinite(mbd.energy)
              && std::abs(mbd.energy - kEDispRef) <= kETol,
          "A2 |E_disp - libMBD pin|");
    check(mbd.forces.size() == 6, "A MBD forces size 6");

    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = posw;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep =
                cpu::gpw::internal::mbd_rsscs_energy(Zw, pos_fd, cell0, false);
            pos_fd[i][d] = save - h;
            const double em =
                cpu::gpw::internal::mbd_rsscs_energy(Zw, pos_fd, cell0, false);
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - mbd.forces[i][d]));
            sum[d] += mbd.forces[i][d];
        }
    }
    check(max_err <= kFTol, "A3 max|F_product - F_fd|");
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    check(sum_norm <= kSumFTol, "A4 |sum F_MBD|");

    // ── B: multi-k product wiring (H2 1×1×2 + free-atom MBD) ──────────────
    const double rA[3] = {5.5, 6.0, 6.0}; // Bohr
    const double rB[3] = {6.5, 6.0, 6.0}; // Bohr
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        const double* r = (i == 0) ? rA : rB;
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = r[0] * units::kBohrInAngstrom;
        frame.span(kAtom_y)[i] = r[1] * units::kBohrInAngstrom;
        frame.span(kAtom_z)[i] = r[2] * units::kBohrInAngstrom;
    }
    frame.box = cpu::Box::cubic(12.0 * units::kBohrInAngstrom);
    // Frame forces [eV/Å] → Ha/Bohr.
    auto frame_forces_ha_bohr = [&]() {
        std::vector<std::array<double, 3>> F(2);
        for (int i = 0; i < 2; ++i)
            F[static_cast<std::size_t>(i)] = {
                frame.span(kAtom_fx)[i] / units::kHartreePerBohrInEvPerAngstrom,
                frame.span(kAtom_fy)[i] / units::kHartreePerBohrInEvPerAngstrom,
                frame.span(kAtom_fz)[i]
                    / units::kHartreePerBohrInEvPerAngstrom};
        return F;
    };

    auto make_cfg = [](bool mbd) {
        GpwDft m{{Kind{molrs::Element::H,
                       GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                       GthPotential(molrs::Element::H, "GTH-PADE-q1")}},
                 100.0,
                 XcKind::PADE};
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.mbd = mbd;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(frame);
    // SCF + analytic forces (free-atom MBD folded).
    const auto& scf = cfg.forces(frame);
    const auto F_mbd = frame_forces_ha_bohr();
    check(scf.converged, "B multi-k SCF converged");
    // 1×1×2 + TR may collapse to n_kpts==1; Γ path reports n_kpts==0.
    check(scf.n_kpts >= 1 && !scf.C_occ_k.empty(), "B multi-k C_occ_k payload");
    check(std::isfinite(scf.e_disp) && scf.e_disp < 0.0, "B e_disp finite <0");

    // Same electronic state: free-atom MBD is a geometry-only constant that
    // never enters the KS matrix, so the plain SCF walks the identical
    // iteration path and lands on the same density as `scf`.
    GpwDft plain = make_cfg(false);
    plain.build(frame);
    plain.forces(frame);
    const auto F_plain = frame_forces_ha_bohr();
    check(F_mbd.size() == 2 && F_plain.size() == 2, "B two force vectors");

    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {{rA[0], rA[1], rA[2]},
                                               {rB[0], rB[1], rB[2]}};
    const MbdResult geom = mbd_rsscs_pbe(Zh, posh, cell0, /*periodic=*/false);

    double max_wire = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d = (F_mbd[static_cast<std::size_t>(a)][k]
                              - F_plain[static_cast<std::size_t>(a)][k])
                             - geom.forces[static_cast<std::size_t>(a)][k];
            max_wire = std::max(max_wire, std::abs(d));
            check(std::isfinite(F_mbd[static_cast<std::size_t>(a)][k]),
                  "B finite force component");
        }
    }
    check(max_wire <= kWireTol, "A5 multi-k wiring |F_mbd-F_plain-F_geom|");
    check(std::abs(geom.forces[0][0]) > 1e-8, "B nontrivial MBD F0x");

    std::printf("PASS: gpw-mbd-kpoint-forces  E_disp(water)=%.14e  "
                "max|F-F_fd|=%.2e  |sumF|=%.2e  multi-k H2 e_disp=%.6e  "
                "max|wire|=%.2e  n_kpts=%d\n",
                mbd.energy, max_err, sum_norm, scf.e_disp, max_wire,
                scf.n_kpts);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}

} // namespace atv_reg_gpw_gpw_mbd_kpoint_forces

ATV_REGRESSION("gpw/mbd", "gpw-mbd-kpoint-forces",
               atv_reg_gpw_gpw_mbd_kpoint_forces::run);
