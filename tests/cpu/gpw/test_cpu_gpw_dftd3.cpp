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
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

using ATV::cpu::gpw::D3Params;
using ATV::cpu::gpw::D3Result;
using ATV::cpu::gpw::dftd3_pbe;

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
