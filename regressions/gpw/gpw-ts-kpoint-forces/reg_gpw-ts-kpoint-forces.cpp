/// @file regressions/gpw/gpw-ts-kpoint-forces/reg_gpw-ts-kpoint-forces.cpp
/// @brief Multi-k GPW analytic forces fold free-atom Tkatchenko–Scheffler TS.
///
/// Out-of-library consumer: public GPW headers only. Geometry and pins in
/// REFERENCE.md (libMBD offline; no live oracle in ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

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

// Offline libMBD Fortran pin (same as gpw-ts-forces / gpw-ts-ice).
constexpr double kEDispRef = -5.9764872295816794e-04; // Ha isolated
constexpr double kETol = 1.0e-8;                      // Ha
constexpr double kFTol = 1.0e-6;                      // Ha/Bohr
constexpr double kSumFTol = 1.0e-8;                   // Ha/Bohr
constexpr double kWireTol = 1.0e-8;                   // Ha/Bohr

constexpr double kAngToBohr = 1.0 / 0.52917720859;

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::gpw_forces_from_scf;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::ts_pbe;
    using cpu::gpw::TsResult;
    using cpu::gpw::XcKind;

    // ── A: geometry-only TS dual-gate (water dimer, free-atom isolated) ───
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
            posw[static_cast<std::size_t>(i)][d] = ang[i][d] * kAngToBohr;

    const double cell0[3] = {0.0, 0.0, 0.0};
    const TsResult ts = ts_pbe(Zw, posw, cell0, /*periodic=*/false);
    check(std::isfinite(ts.energy) && std::abs(ts.energy - kEDispRef) <= kETol,
          "A2 |E_disp - libMBD pin|");
    check(ts.forces.size() == 6, "A TS forces size 6");

    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = posw;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep = ts_pbe(Zw, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save - h;
            const double em = ts_pbe(Zw, pos_fd, cell0, false).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - ts.forces[i][d]));
            sum[d] += ts.forces[i][d];
        }
    }
    check(max_err <= kFTol, "A3 max|F_product - F_fd|");
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    check(sum_norm <= kSumFTol, "A4 |sum F_TS|");

    // ── B: multi-k product wiring (H2 1×1×2 + free-atom TS) ───────────────
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, basis, pot, 1),
                                  GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = [](bool with_ts) {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
        m.cutoff_ry = 100.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 60;
        m.nonperiodic = false;
        m.local_pp_on_grid = true;
        m.xc = XcKind::PADE;
        m.eps_default = 1e-8;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.ts = with_ts;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    check(scf.converged, "B multi-k SCF converged");
    // 1×1×2 + TR may collapse to n_kpts==1; Γ path reports n_kpts==0.
    check(scf.n_kpts >= 1 && !scf.C_occ_k.empty(), "B multi-k C_occ_k payload");
    check(std::isfinite(scf.e_disp) && scf.e_disp < 0.0, "B e_disp finite <0");

    // GpwDft is non-copyable; rebuild a plain flag set for the second assembly.
    const auto F_ts = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
    check(F_ts.size() == 2 && F_plain.size() == 2, "B two force vectors");

    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {{rA[0], rA[1], rA[2]},
                                               {rB[0], rB[1], rB[2]}};
    const double cell_h2[3] = {12.0, 12.0, 12.0};
    constexpr double kRcut = 50.0 * kAngToBohr;
    // Product k-forces use periodic pure-RS TS (same as Γ assemble when
    // nonperiodic=false); free-atom volumes (no Hirshfeld).
    const TsResult geom = ts_pbe(Zh, posh, cell_h2, /*periodic=*/true, kRcut);

    double max_wire = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d = (F_ts[static_cast<std::size_t>(a)][k]
                              - F_plain[static_cast<std::size_t>(a)][k])
                             - geom.forces[static_cast<std::size_t>(a)][k];
            max_wire = std::max(max_wire, std::abs(d));
            check(std::isfinite(F_ts[static_cast<std::size_t>(a)][k]),
                  "B finite force component");
        }
    }
    check(max_wire <= kWireTol, "A5 multi-k wiring |F_ts-F_plain-F_geom|");
    check(std::abs(geom.forces[0][0]) > 1e-8, "B nontrivial TS F0x");

    std::printf("PASS: gpw-ts-kpoint-forces  E_disp(water)=%.14e  "
                "max|F-F_fd|=%.2e  |sumF|=%.2e  multi-k H2 e_disp=%.6e  "
                "max|wire|=%.2e  n_kpts=%d\n",
                ts.energy, max_err, sum_norm, scf.e_disp, max_wire, scf.n_kpts);

    if (g_fail) {
        std::fprintf(stderr, "FAILED: %d gate(s)\n", g_fail);
        return 1;
    }
    return 0;
}
