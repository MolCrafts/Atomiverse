/// @file tests/cpu/gpw/test_cpu_gpw_ts.cpp
/// @brief Validate Tkatchenko–Scheffler free-atom dispersion vs offline libMBD
///        pins and finite-difference forces.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::ts_pbe;
using ATV::cpu::gpw::TsResult;

namespace {

constexpr double kAngToBohr = 1.0 / 0.52917720859;

std::vector<std::array<double, 3>> water_dimer_bohr() {
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    std::vector<std::array<double, 3>> pos(6);
    for (int i = 0; i < 6; ++i)
        for (int d = 0; d < 3; ++d)
            pos[i][d] = ang[i][d] * kAngToBohr;
    return pos;
}

const std::vector<int> kZ = {8, 1, 1, 8, 1, 1};

} // namespace

// Isolated free-atom TS vs offline libMBD 0.13-dev (cmbd_ts_energy, fermi,
// s_R=0.94, d=20, free-atom α/C6/R for H,O).
TEST(GpwTs, WaterDimerIsolatedMatchesLibMbd) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const TsResult r = ts_pbe(kZ, pos, cell, /*periodic=*/false);

    const double libmbd = -5.9764872295816794e-04;
    std::printf("[TS] isolated E=%.16e  libMBD=%.16e  diff=%.2e Ha\n", r.energy,
                libmbd, r.energy - libmbd);
    EXPECT_NEAR(r.energy, libmbd, 1e-12);
}

// Periodic pure real-space rcut = 50 Å pin from libMBD Fortran get_ts_energy
// on its no-Ewald branch (ewald_on=false, dipole_cutoff=50 Å; libmbd 88069c6,
// pin driver tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90).
TEST(GpwTs, WaterDimerPeriodicEnergy) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    constexpr double kRcut = 50.0 * kAngToBohr;
    const TsResult r = ts_pbe(kZ, pos, cell, /*periodic=*/true, kRcut);

    const double pin = -6.0996307733860822e-04; // libMBD pure-RS
    std::printf("[TS] periodic E=%.16e  pin=%.16e  diff=%.2e Ha\n", r.energy,
                pin, r.energy - pin);
    EXPECT_NEAR(r.energy, pin, 1e-12);
}

// Strained-cell energies vs libMBD pure-RS pins (gpw-ts-stress family): one
// axis scaled by 1±h at fixed fractional coordinates, h = 2e-3 — the exact
// geometry the fixed-P stress Lagrangian rebuilds.
TEST(GpwTs, StrainedCellEnergiesMatchLibMbd) {
    const double kPins[3][2] = {
        {-6.1278297256613687e-04, -6.0709134569744607e-04}, // x {+h, −h}
        {-6.0992605267944588e-04, -6.1000059619617259e-04}, // y
        {-6.0996696060335368e-04, -6.0995897233070745e-04}, // z
    };
    constexpr double kH = 2.0e-3;
    const double L = 12.0 * kAngToBohr;
    constexpr double kRcut = 50.0 * kAngToBohr;

    double max_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int s = 0; s < 2; ++s) {
            const double scale = 1.0 + (s == 0 ? kH : -kH);
            auto pos = water_dimer_bohr();
            double cell[3] = {L, L, L};
            cell[a] *= scale;
            for (auto& p : pos)
                p[static_cast<std::size_t>(a)] *= scale;
            const double e = ts_pbe(kZ, pos, cell, /*periodic=*/true, kRcut)
                                 .energy;
            max_err = std::max(max_err, std::abs(e - kPins[a][s]));
        }
    }
    std::printf("[TS] strained max|E - libMBD| = %.2e Ha\n", max_err);
    EXPECT_LT(max_err, 1e-12);
}

// Analytic forces must match central FD of the same energy (isolated).
TEST(GpwTs, AnalyticForcesMatchFiniteDifference) {
    auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const TsResult r = ts_pbe(kZ, pos, cell, /*periodic=*/false);

    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep = ts_pbe(kZ, pos, cell, false).energy;
            pos[i][d] = save - h;
            const double em = ts_pbe(kZ, pos, cell, false).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[TS] max |F_analytic - F_fd| = %.2e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

// GpwDft.ts folds e_disp into e_total (short SCF; energy is geometry-only).
TEST(GpwTs, GpwDftReportsEDisp) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;

    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000}, {0.758602, 0.000000, 0.504284},
        {0.758602, 0.000000, -0.504284}, {3.000000, 0.000000, 0.000000},
        {3.758602, 0.000000, 0.504284}, {3.758602, 0.000000, -0.504284},
    };
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    std::vector<GpwAtom> atoms;
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kAngToBohr, ang[i][1] * kAngToBohr,
                               ang[i][2] * kAngToBohr};
        if (i == 0 || i == 3)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }

    GpwDft gpw;
    const double L = 12.0 * kAngToBohr;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 6;
    gpw.tol = 1e-5;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.ts = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);
    const double pin = -6.0996307733860822e-04; // libMBD pure-RS
    EXPECT_NEAR(res.e_disp, pin, 1e-12);
    EXPECT_LT(res.e_disp, 0.0);
    EXPECT_TRUE(std::isfinite(res.e_total));
}

// Periodic pure-RS forces: dual-gate vs independent FD + translation.
TEST(GpwTs, PeriodicForcesMatchFiniteDifference) {
    auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    constexpr double kRcut = 50.0 * kAngToBohr;
    const TsResult r = ts_pbe(kZ, pos, cell, /*periodic=*/true, kRcut);

    const double h = 1e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep =
                ts_pbe(kZ, pos, cell, true, kRcut).energy;
            pos[i][d] = save - h;
            const double em =
                ts_pbe(kZ, pos, cell, true, kRcut).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
            sum[d] += r.forces[i][d];
        }
    }
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf("[TS] periodic max|F-F_fd|=%.2e  |sum F|=%.2e Ha/Bohr\n",
                max_err, sum_norm);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(sum_norm, 1e-8);
    EXPECT_GT(std::abs(r.forces[0][0]), 1e-6);
}

// Γ force assembly must fold free-atom TS (geometry-only). H2: one SCF, two
// assemblies vs standalone ts_pbe.
TEST(GpwTs, ProductForcesFoldTs) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::gpw_forces_from_scf;
    using ATV::cpu::gpw::XcKind;

    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    // Positions [Bohr] in a 12 Bohr cell (same style as MBD k-force unit test).
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
        m.ts = with_ts;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    ASSERT_TRUE(scf.converged) << "H2 TS SCF did not converge";
    EXPECT_TRUE(std::isfinite(scf.e_disp));
    EXPECT_LT(scf.e_disp, 0.0);

    const auto F_ts = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
    ASSERT_EQ(F_ts.size(), 2u);
    ASSERT_EQ(F_plain.size(), 2u);

    std::vector<int> Z = {1, 1};
    std::vector<std::array<double, 3>> pos = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell[3] = {12.0, 12.0, 12.0};
    constexpr double kRcut = 50.0 * kAngToBohr;
    const TsResult geom = ts_pbe(Z, pos, cell, /*periodic=*/true, kRcut);

    double max_wiring = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double delta =
                (F_ts[static_cast<std::size_t>(a)][k]
                 - F_plain[static_cast<std::size_t>(a)][k])
                - geom.forces[static_cast<std::size_t>(a)][k];
            max_wiring = std::max(max_wiring, std::abs(delta));
        }
    }
    std::printf("[TS product fold] e_disp=%.6e  max|wiring|=%.2e Ha/Bohr\n",
                scf.e_disp, max_wiring);
    EXPECT_LT(max_wiring, 1e-8)
        << "Γ assemble must add geometry-only TS forces";
    EXPECT_GT(std::abs(geom.forces[0][0]), 1e-8);
}

// Multi-k force assembly must fold geometry-only free-atom TS (product path
// for gpw-ts-kpoint-forces). H2 1×1×2: one SCF, two assemblies vs geometry
// kernel (periodic pure-RS, same as Γ when nonperiodic=false).
TEST(GpwTs, KpointProductForcesFoldTs) {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::GthPotential;
    using ATV::cpu::gpw::XcKind;
    using ATV::cpu::gpw::gpw_forces_from_scf;

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
    ASSERT_TRUE(scf.converged);
    // 1×1×2 + time-reversal often yields n_kpts==1 irreducible point, but the
    // SCF still uses the multi-k path (C_occ_k payload; Γ has n_kpts==0).
    EXPECT_GE(scf.n_kpts, 1);
    EXPECT_FALSE(scf.C_occ_k.empty());
    EXPECT_TRUE(std::isfinite(scf.e_disp));
    EXPECT_LT(scf.e_disp, 0.0);

    // Force assembly only reads public flags + cell from cfg (GpwDft non-copyable).
    const auto F_ts = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);
    ASSERT_EQ(F_ts.size(), 2u);
    ASSERT_EQ(F_plain.size(), 2u);

    std::vector<int> Z = {1, 1};
    std::vector<std::array<double, 3>> pos = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell[3] = {12.0, 12.0, 12.0};
    constexpr double kRcut = 50.0 * kAngToBohr;
    const TsResult geom = ts_pbe(Z, pos, cell, /*periodic=*/true, kRcut);

    double max_d = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d =
                (F_ts[static_cast<std::size_t>(a)][k]
                 - F_plain[static_cast<std::size_t>(a)][k])
                - geom.forces[static_cast<std::size_t>(a)][k];
            max_d = std::max(max_d, std::abs(d));
        }
    }
    std::printf("[TS k-forces] e_disp=%.6e  max|F_ts-F_plain-F_geom|=%.2e\n",
                scf.e_disp, max_d);
    EXPECT_LT(max_d, 1e-8)
        << "multi-k assemble must add geometry-only TS forces";
    // Non-trivial TS contribution on the compressed H–H bond.
    EXPECT_GT(std::abs(geom.forces[0][0]), 1e-8);
}
