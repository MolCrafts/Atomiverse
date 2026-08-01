/// @file tests/cpu/gpw/test_cpu_gpw_vol_forces.cpp
/// @brief Fixed-volume (Hirshfeld-scaled) TS/MBD force dual-gates + product fold.
///
/// Density-chain ∂v/∂R is out of scope: forces hold \(v_A\) fixed. Geometry-
/// only FD duals stay unit-test class; product wiring uses a 2-electron H₂
/// GPW cell (≤1 s). Full water-dimer product pin lives in
/// `reg_gpw-hirshfeld-forces`.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/molrs.hpp"

using ATV::cpu::gpw::mbd_rsscs_pbe;
using ATV::cpu::gpw::ts_pbe;

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
            pos[static_cast<std::size_t>(i)][d] = ang[i][d] * kAngToBohr;
    return pos;
}

const std::vector<int> kZ = {8, 1, 1, 8, 1, 1};

} // namespace

// Fixed-volume-ratio MBD forces: product kernel vs independent central FD.
TEST(GpwVolForces, VolumeScaledMbdForcesMatchFd) {
    auto pos = water_dimer_bohr();
    const double cell0[3] = {0.0, 0.0, 0.0};
    const std::vector<double> v = {0.74, 0.65, 0.65, 0.72, 0.58, 0.58};
    const auto r = mbd_rsscs_pbe(kZ, pos, cell0, /*periodic=*/false, 0.83, 6.0,
                                 v);
    ASSERT_EQ(r.forces.size(), 6u);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_LT(r.energy, 0.0);
    // Distinct from free-atom (v=1) water-dimer pin ~ -1.493e-3 Ha.
    EXPECT_GT(r.energy, -1.4e-3);
    EXPECT_LT(r.energy, -0.8e-3);

    const double h = 1.0e-5;
    double max_err = 0.0;
    double sum[3] = {0.0, 0.0, 0.0};
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep =
                mbd_rsscs_pbe(kZ, pos_fd, cell0, false, 0.83, 6.0, v).energy;
            pos_fd[i][d] = save - h;
            const double em =
                mbd_rsscs_pbe(kZ, pos_fd, cell0, false, 0.83, 6.0, v).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
            sum[d] += r.forces[i][d];
        }
    }
    const double sum_norm =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf("[Vol F MBD] E=%.6e max|F-F_fd|=%.2e |sum F|=%.2e\n", r.energy,
                max_err, sum_norm);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(sum_norm, 1e-8);
}

// Same dual-gate for volume-scaled TS.
TEST(GpwVolForces, VolumeScaledTsForcesMatchFd) {
    auto pos = water_dimer_bohr();
    const double cell0[3] = {0.0, 0.0, 0.0};
    const std::vector<double> v = {0.74, 0.65, 0.65, 0.72, 0.58, 0.58};
    const auto r = ts_pbe(kZ, pos, cell0, /*periodic=*/false, 94.2, 0.94, 20.0,
                          v);
    ASSERT_EQ(r.forces.size(), 6u);
    EXPECT_TRUE(std::isfinite(r.energy));
    EXPECT_LT(r.energy, 0.0);

    const double h = 1.0e-5;
    double max_err = 0.0;
    auto pos_fd = pos;
    for (std::size_t i = 0; i < 6; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos_fd[i][d];
            pos_fd[i][d] = save + h;
            const double ep =
                ts_pbe(kZ, pos_fd, cell0, false, 94.2, 0.94, 20.0, v).energy;
            pos_fd[i][d] = save - h;
            const double em =
                ts_pbe(kZ, pos_fd, cell0, false, 94.2, 0.94, 20.0, v).energy;
            pos_fd[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[Vol F TS] E=%.6e max|F-F_fd|=%.2e\n", r.energy, max_err);
    EXPECT_LT(max_err, 1e-6);
}

// Product path: Γ assemble must add Hirshfeld-scaled TS forces, not free-atom.
TEST(GpwVolForces, ProductForcesFoldHirshfeldTs) {
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

    auto make_cfg = [](bool with_ts, bool with_hirsh) {
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
        m.hirshfeld = with_hirsh;
        return m;
    };

    GpwDft cfg = make_cfg(true, true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    ASSERT_TRUE(scf.converged);
    ASSERT_EQ(scf.volume_ratios.size(), 2u);
    EXPECT_TRUE(std::isfinite(scf.e_disp));
    EXPECT_LT(scf.e_disp, 0.0);

    const auto F_h = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false, false);
    const auto F_plain = gpw_forces_from_scf(atoms, plain, scf);

    std::vector<int> Z = {1, 1};
    std::vector<std::array<double, 3>> pos = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell[3] = {12.0, 12.0, 12.0};
    constexpr double kRcut = 50.0 * kAngToBohr;
    const auto geom_h =
        ts_pbe(Z, pos, cell, true, kRcut, 0.94, 20.0, scf.volume_ratios);
    const auto geom_free = ts_pbe(Z, pos, cell, true, kRcut);

    EXPECT_NEAR(scf.e_disp, geom_h.energy, 1e-10);
    EXPECT_GT(std::abs(scf.e_disp - geom_free.energy), 1e-10)
        << "Hirshfeld must move E_disp vs free-atom";

    double max_h = 0.0, max_free = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double dF = F_h[static_cast<std::size_t>(a)][k]
                              - F_plain[static_cast<std::size_t>(a)][k];
            max_h = std::max(
                max_h,
                std::abs(dF - geom_h.forces[static_cast<std::size_t>(a)][k]));
            max_free = std::max(
                max_free,
                std::abs(dF
                         - geom_free.forces[static_cast<std::size_t>(a)][k]));
        }
    }
    std::printf(
        "[Vol F product TS] e_disp=%.6e  max|wire_h|=%.2e  max|wire_free|=%.2e\n",
        scf.e_disp, max_h, max_free);
    EXPECT_LT(max_h, 1e-8) << "assemble must add Hirshfeld-scaled TS forces";
    EXPECT_GT(max_free, 1e-9);
}
