/// @file tests/cpu/gpw/test_cpu_gpw_dftd3.cpp
/// @brief Validate the standalone Grimme DFT-D3 (zero damping, PBE) module
///        against a CP2K reference number and finite-difference forces.
///
/// CP2K reference (cp2k 2026.1, TYPE DFTD3 / REFERENCE_FUNCTIONAL PBE / default
/// R_CUTOFF 20 Bohr / EPS_CN 1e-6, dftd3.dat) for a water dimer in a 12 Å NONE
/// (isolated) cell prints:
///     "Dispersion energy:   -0.00081399213205"  [Hartree]
/// The geometry below (Ångström) is the exact &COORD of that run; D3 is
/// translation-invariant, so CP2K's internal centering does not change it.

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"

using ATV::cpu::gpw::D3Params;
using ATV::cpu::gpw::D3Result;
using ATV::cpu::gpw::dftd3_pbe;
using ATV::cpu::gpw::dftd3_pbe_bj;

namespace {

constexpr double kAngToBohr = 1.0 / 0.52917720859;

// Water dimer (Ångström), matching the CP2K reference &COORD.
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

// Energy must reproduce CP2K's printed "Dispersion energy" to ~1e-9 Hartree.
TEST(GpwDftD3, WaterDimerEnergyMatchesCP2K) {
    const D3Params params{};
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0}; // unused: isolated

    const D3Result r = dftd3_pbe(kZ, pos, cell, /*periodic=*/false, params);

    const double cp2k = -0.00081399213205;
    std::printf("[D3] E=%.14f  CP2K=%.14f  diff=%.2e Ha\n", r.energy, cp2k,
                r.energy - cp2k);
    EXPECT_NEAR(r.energy, cp2k, 1e-9);
}

// Analytic forces must match central finite differences of the same energy.
TEST(GpwDftD3, AnalyticForcesMatchFiniteDifference) {
    const D3Params params{};
    auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};

    const D3Result r = dftd3_pbe(kZ, pos, cell, /*periodic=*/false, params);

    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep = dftd3_pbe(kZ, pos, cell, false, params).energy;
            pos[i][d] = save - h;
            const double em = dftd3_pbe(kZ, pos, cell, false, params).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h); // F = -dE/dx
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[D3] max |F_analytic - F_fd| = %.2e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

// Periodic lattice sum: a tight cubic box must give a finite, more-negative
// energy than the isolated cluster (extra image pairs all contribute < 0), and
// its forces must still match finite differences (CN images included).
TEST(GpwDftD3, PeriodicEnergyAndForces) {
    const D3Params params{};
    auto pos = water_dimer_bohr();
    const double L = 8.0 * kAngToBohr;
    const double cell[3] = {L, L, L};

    const D3Result iso = dftd3_pbe(kZ, pos, cell, /*periodic=*/false, params);
    const D3Result per = dftd3_pbe(kZ, pos, cell, /*periodic=*/true, params);
    std::printf("[D3] E_iso=%.10f  E_per=%.10f Ha\n", iso.energy, per.energy);
    EXPECT_LT(per.energy, iso.energy); // images add attraction

    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep = dftd3_pbe(kZ, pos, cell, true, params).energy;
            pos[i][d] = save - h;
            const double em = dftd3_pbe(kZ, pos, cell, true, params).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - per.forces[i][d]));
        }
    }
    std::printf("[D3] periodic max |F_analytic - F_fd| = %.2e Ha/Bohr\n",
                max_err);
    EXPECT_LT(max_err, 1e-7);
}

// ---------------------------------------------------------------------------
// PBE-D3(BJ) with CP2K coordination numbers — periodic water dimer pin
// ---------------------------------------------------------------------------
//
// Offline CP2K 2026.1 (git:5e54ba2), TYPE DFTD3(BJ) / REFERENCE_FUNCTIONAL PBE
// / R_CUTOFF 20 (Å → 37.79 Bohr printed; pair sum uses 2× → ~75.59 Bohr)
// / no C9 / no LRC, water dimer in 12 Å PERIODIC XYZ cell
// (tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/water_dimer_d3bj.inp):
//     Dispersion energy:   -0.00137627769481  [Hartree]

namespace {
// 2 × CP2K R_CUTOFF 20 Å [Bohr] — GPW BJ / offline pin convention.
constexpr double kD3BjRcutBohr = 2.0 * 20.0 * kAngToBohr;
} // namespace

TEST(GpwDftD3BJ, PeriodicWaterDimerEnergyMatchesCP2K) {
    const D3Params params{};
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};

    const D3Result r =
        dftd3_pbe_bj(kZ, pos, cell, /*periodic=*/true, params, kD3BjRcutBohr,
                     1.0, 0.7875, 0.4289, 4.4407, /*cp2k_cn=*/true);

    const double cp2k = -0.00137627769481;
    std::printf("[D3BJ] E=%.14f  CP2K=%.14f  diff=%.2e Ha\n", r.energy, cp2k,
                r.energy - cp2k);
    EXPECT_NEAR(r.energy, cp2k, 1e-8);
}

TEST(GpwDftD3BJ, AnalyticForcesMatchFiniteDifference) {
    const D3Params params{};
    auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const auto eval = [&](const std::vector<std::array<double, 3>>& p) {
        return dftd3_pbe_bj(kZ, p, cell, true, params, kD3BjRcutBohr, 1.0,
                            0.7875, 0.4289, 4.4407, true);
    };

    const D3Result r = eval(pos);
    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep = eval(pos).energy;
            pos[i][d] = save - h;
            const double em = eval(pos).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[D3BJ] max |F_analytic - F_fd| = %.2e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

TEST(GpwDftD3BJ, PeriodicMoreNegativeThanIsolated) {
    const D3Params params{};
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D3Result iso =
        dftd3_pbe_bj(kZ, pos, cell, false, params, kD3BjRcutBohr, 1.0, 0.7875,
                     0.4289, 4.4407, true);
    const D3Result per =
        dftd3_pbe_bj(kZ, pos, cell, true, params, kD3BjRcutBohr, 1.0, 0.7875,
                     0.4289, 4.4407, true);
    std::printf("[D3BJ] E_iso=%.10f  E_per=%.10f Ha\n", iso.energy, per.energy);
    EXPECT_LT(per.energy, iso.energy);
}

// Integration: GpwDft.d3bj folds standalone BJ energy into e_total / e_disp.
// Fast H2 (DZVP-MOLOPT-SR / GTH-PBE) — SCF cheap; ΔE must equal e_disp.
TEST(GpwDftD3BJ, IntegratedIntoGpwScf) {
    using namespace ATV::cpu::gpw;

    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    const double rA[3] = {4.0, 4.0, 4.0};
    const double rB[3] = {4.0, 4.0, 4.0 + 1.6 * kAngToBohr};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, bH, pH, 1),
                                  GpwAtom(rB, bH, pH, 1)};
    const double L = 8.0 * kAngToBohr;

    auto make_solver = [&](bool d3bj_flag) {
        GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = L;
        m.cutoff_ry = 200.0;
        m.n_occ = 1;
        m.tol = 1e-8;
        m.max_iter = 40;
        m.xc = XcKind::PBE;
        m.nonperiodic = true;
        m.d3bj = d3bj_flag;
        m.build(atoms);
        return m;
    };

    GpwDft solver_off = make_solver(false);
    GpwDft solver_on = make_solver(true);
    const auto& r_off = solver_off.run(atoms);
    const auto& r_on = solver_on.run(atoms);

    const std::vector<int> Z = {1, 1};
    const std::vector<std::array<double, 3>> pos = {{rA[0], rA[1], rA[2]},
                                                    {rB[0], rB[1], rB[2]}};
    const double cell[3] = {L, L, L};
    const double ed3 =
        dftd3_pbe_bj(Z, pos, cell, /*periodic=*/false, D3Params{},
                     kD3BjRcutBohr, 1.0, 0.7875, 0.4289, 4.4407,
                     /*cp2k_cn=*/true)
            .energy;

    std::printf(
        "[D3BJ integ] e_on-e_off=%.12f  e_disp=%.12f  standalone=%.12f\n",
        r_on.e_total - r_off.e_total, r_on.e_disp, ed3);
    EXPECT_NEAR(r_on.e_disp, ed3, 1e-9);
    EXPECT_NEAR(r_on.e_total - r_off.e_total, ed3, 1e-9);
    EXPECT_LT(ed3, 0.0);
    EXPECT_NEAR(r_off.e_disp, 0.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Three-body ATM / C9 (CP2K CALCULATE_C9_TERM .TRUE.)
// ---------------------------------------------------------------------------
//
// Offline CP2K 2026.1, TYPE DFTD3 / PBE / R_CUTOFF 20 / CALCULATE_C9_TERM T /
// no LRC, water dimer PERIODIC NONE (tools/cp2k_oracle/inputs/gpw-d3-atm):
//     Dispersion energy:   -0.00081414118193  [Hartree]
// Two-body-only on same geometry: -0.00081399213205 Ha.

TEST(GpwDftD3Atm, WaterDimerEnergyMatchesCP2K) {
    const D3Params params{};
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};

    const D3Result two =
        dftd3_pbe(kZ, pos, cell, /*periodic=*/false, params, 40.0, false);
    const D3Result atm =
        dftd3_pbe(kZ, pos, cell, /*periodic=*/false, params, 40.0, true);

    const double cp2k_atm = -0.00081414118193;
    const double cp2k_two = -0.00081399213205;
    std::printf("[D3-ATM] E_two=%.14f  E_atm=%.14f  CP2K_atm=%.14f  "
                "Δ_ATM=%.3e Ha\n",
                two.energy, atm.energy, cp2k_atm, atm.energy - two.energy);
    EXPECT_NEAR(two.energy, cp2k_two, 1e-9);
    EXPECT_NEAR(atm.energy, cp2k_atm, 1e-9);
    // ATM shifts the two-body pin by a small but finite amount.
    EXPECT_GT(std::abs(atm.energy - two.energy), 1e-10);
}

TEST(GpwDftD3Atm, AnalyticForcesMatchFiniteDifference) {
    const D3Params params{};
    auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const auto eval = [&](const std::vector<std::array<double, 3>>& p) {
        return dftd3_pbe(kZ, p, cell, false, params, 40.0, /*calculate_c9=*/true);
    };

    const D3Result r = eval(pos);
    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + h;
            const double ep = eval(pos).energy;
            pos[i][d] = save - h;
            const double em = eval(pos).energy;
            pos[i][d] = save;
            const double fd = -(ep - em) / (2.0 * h);
            max_err = std::max(max_err, std::abs(fd - r.forces[i][d]));
        }
    }
    std::printf("[D3-ATM] max |F_analytic - F_fd| = %.2e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

// Periodic ATM gradient: analytic F(c9 on) - F(c9 off) vs central FD of
// E(c9 on) - E(c9 off). The two-body part cancels exactly, so this gates the
// three-body gradient alone — including the lattice-image triple classes
// (CP2K doabc CASE 2/3, fac0 = 1/3) that the isolated FD test never reaches.
// A 6 A cell with rcut = 20 Bohr keeps every image class populated and the
// enumeration small enough for the unit tier. The dimer is shifted off the
// x = L/2 wrap boundary: the ATM sum follows CP2K in evaluating pbc()-wrapped
// coordinates through HARD r < rcut leg gates over a finite image window, so
// the truncated energy is discontinuous (at the ~1e-7 boundary-triple level)
// exactly on a wrap seam — the FD identity holds in the interior of a wrap
// branch, and the unshifted CP2K geometry puts O1 exactly on the seam.
TEST(GpwDftD3Atm, PeriodicAtmForcesMatchFiniteDifference) {
    const D3Params params{};
    auto pos = water_dimer_bohr();
    const double L = 6.0 * kAngToBohr;
    const double shift[3] = {0.31 * kAngToBohr, 0.29 * kAngToBohr,
                             0.33 * kAngToBohr};
    for (auto& p : pos)
        for (int d = 0; d < 3; ++d)
            p[d] += shift[d];
    const double cell[3] = {L, L, L};
    const double rcut = 20.0;
    const auto atm_only = [&](const std::vector<std::array<double, 3>>& p,
                              D3Result& on, D3Result& off) {
        on = dftd3_pbe(kZ, p, cell, /*periodic=*/true, params, rcut, true);
        off = dftd3_pbe(kZ, p, cell, /*periodic=*/true, params, rcut, false);
    };

    D3Result on, off;
    atm_only(pos, on, off);
    ASSERT_GT(std::abs(on.energy - off.energy), 1e-10); // images engage
    const double h = 1e-5;
    double max_err = 0.0;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            D3Result pon, poff, mon, moff;
            pos[i][d] = save + h;
            atm_only(pos, pon, poff);
            pos[i][d] = save - h;
            atm_only(pos, mon, moff);
            pos[i][d] = save;
            const double fd =
                -((pon.energy - poff.energy) - (mon.energy - moff.energy))
                / (2.0 * h);
            const double analytic = on.forces[i][d] - off.forces[i][d];
            max_err = std::max(max_err, std::abs(fd - analytic));
        }
    }
    std::printf("[D3-ATM per] dE_atm=%.3e Ha  max |dF_analytic - dF_fd| = "
                "%.2e Ha/Bohr\n",
                on.energy - off.energy, max_err);
    EXPECT_LT(max_err, 1e-7);
}

// GpwDft.d3_atm folds standalone ATM energy into e_disp for the water dimer.
TEST(GpwDftD3Atm, IntegratedIntoGpwScf) {
    using namespace ATV::cpu::gpw;

    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    // Translate to the cell centre: E_disp is translation invariant, but the
    // Martyna-Tuckerman cluster path is not, and GpwDft::build rejects
    // off-centre nonperiodic systems (clipped collocation corrupts the forces).
    auto pos = water_dimer_bohr();
    const double shift[3] = {0.5 * L - 1.5 * kAngToBohr, 0.5 * L, 0.5 * L};
    for (auto& r : pos)
        for (int k = 0; k < 3; ++k)
            r[static_cast<std::size_t>(k)] += shift[k];
    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double p[3] = {pos[i][0], pos[i][1], pos[i][2]};
        if (kZ[static_cast<std::size_t>(i)] == 8)
            atoms.emplace_back(p, bO, pO, 8);
        else
            atoms.emplace_back(p, bH, pH, 1);
    }
    const double ed3 = dftd3_pbe(kZ, pos, cell, /*periodic=*/false, D3Params{},
                                 kD3Cp2kPairCutoffBohr, /*calculate_c9=*/true)
                           .energy;

    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = L;
    m.cutoff_ry = 200.0;
    m.n_occ = 8;
    m.tol = 1e-6;
    m.max_iter = 8;
    m.mix = 0.4;
    m.xc = XcKind::PBE;
    m.nonperiodic = true;
    m.d3 = true;
    m.d3_atm = true;
    m.build(atoms);
    const auto& r = m.run(atoms);
    std::printf("[D3-ATM integ] e_disp=%.14f  standalone=%.14f\n", r.e_disp,
                ed3);
    EXPECT_NEAR(r.e_disp, ed3, 1e-9);
    EXPECT_NEAR(r.e_disp, -0.00081414118193, 1e-9);
}

// Multi-k force assembly folds d3_atm identically to Γ: F(on) − F(off) on
// one multi-k density equals the standalone ATM-only gradient. Cheap unit
// twin for regressions/gpw/gpw-d3-atm-kpoint-forces (product wiring gate).
TEST(GpwDftD3Atm, MultiKProductForcesFoldAtm) {
    using namespace ATV::cpu::gpw;

    const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
    const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, bH, pH, 1),
                                  GpwAtom(rB, bH, pH, 1)};

    auto make_cfg = [](bool atm) {
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
        m.d3 = true;
        m.d3_atm = atm;
        // Short cutoff: unit tier must stay cheap; ATM still engages.
        m.d3_rcut_bohr = 20.0;
        return m;
    };

    GpwDft cfg = make_cfg(true);
    cfg.build(atoms);
    const auto& scf = cfg.run(atoms);
    ASSERT_TRUE(scf.converged);
    ASSERT_GE(scf.n_kpts, 1);
    ASSERT_FALSE(scf.C_occ_k.empty());

    const auto F_on = gpw_forces_from_scf(atoms, cfg, scf);
    GpwDft plain = make_cfg(false);
    plain.build(atoms);
    const auto F_off = gpw_forces_from_scf(atoms, plain, scf);

    std::vector<int> Zh = {1, 1};
    std::vector<std::array<double, 3>> posh = {
        {rA[0], rA[1], rA[2]}, {rB[0], rB[1], rB[2]}};
    const double cell0[3] = {12.0, 12.0, 12.0};
    const D3Result on =
        dftd3_pbe(Zh, posh, cell0, /*periodic=*/true, D3Params{}, 20.0, true);
    const D3Result off =
        dftd3_pbe(Zh, posh, cell0, /*periodic=*/true, D3Params{}, 20.0, false);

    double max_wire = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int k = 0; k < 3; ++k) {
            const double d =
                (F_on[static_cast<std::size_t>(a)][k]
                 - F_off[static_cast<std::size_t>(a)][k])
                - (on.forces[static_cast<std::size_t>(a)][k]
                   - off.forces[static_cast<std::size_t>(a)][k]);
            max_wire = std::max(max_wire, std::abs(d));
        }
    }
    std::printf("[D3-ATM multi-k forces] wiring max|dF|=%.3e Ha/Bohr  "
                "e_disp=%.6e\n",
                max_wire, scf.e_disp);
    EXPECT_LT(max_wire, 1e-8);
}

TEST(GpwDftD3BJ, MutualExclusionWithZeroDamping) {
    using namespace ATV::cpu::gpw;
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");
    const double rA[3] = {4.0, 4.0, 4.0};
    const double rB[3] = {4.0, 4.0, 4.0 + 1.6 * kAngToBohr};
    std::vector<GpwAtom> atoms = {GpwAtom(rA, bH, pH, 1),
                                  GpwAtom(rB, bH, pH, 1)};
    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 8.0 * kAngToBohr;
    m.cutoff_ry = 120.0;
    m.n_occ = 1;
    m.max_iter = 5;
    m.d3 = true;
    m.d3bj = true;
    m.build(atoms);
    EXPECT_THROW(m.run(atoms), std::runtime_error);
}
