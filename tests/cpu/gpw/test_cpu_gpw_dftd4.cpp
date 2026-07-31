/// @file tests/cpu/gpw/test_cpu_gpw_dftd4.cpp
/// @brief Periodic + isolated DFT-D4 E_disp and forces vs offline dftd4 3.7.0.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/scf/dftd4.h"

using ATV::cpu::scf::D4Result;
using ATV::cpu::scf::dftd4;
using ATV::cpu::scf::dftd4_energy;

namespace {

constexpr double kAngToBohr = 1.0 / 0.52917720859;

// Water dimer (Å) matching gpw-pbe-d3bj-ice / gpw-d4-water REFERENCE.
// Atom order: O H H O H H (matches isolated XYZ pin, not POSCAR O O H H H H).
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
            pos[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] =
                ang[i][d] * kAngToBohr;
    return pos;
}

const std::vector<int> kZ = {8, 1, 1, 8, 1, 1};

// Offline dftd4 3.7.0 -g on XYZ (order O H H O H H): gradient dE/dR [Ha/Bohr].
// ATV force = −gradient.
const double kGradIso[6][3] = {
    {-5.367487773544261e-05, 0.0, 6.468403283702467e-20},
    {-5.984939684718166e-05, 0.0, -1.0621704377196226e-05},
    {-5.9849396847181536e-05, 0.0, 1.0621704377196153e-05},
    {1.371416991978683e-04, 0.0, 4.748814682847428e-21},
    {1.811598611596879e-05, 0.0, -8.567147309218983e-06},
    {1.8115986115968777e-05, 0.0, 8.567147309218974e-06},
};

// Offline dftd4 3.7.0 -g on POSCAR (order O O H H H H) remapped to O H H O H H.
// POSCAR atoms: O0, O1, H0, H1, H2, H3 → ATV [O0, H0, H1, O1, H2, H3].
const double kGradPer[6][3] = {
    {-5.3738961271913915e-05, -1.5591535100726387e-24, -6.1987516039752966e-21},
    {-5.9550148958160894e-05, -3.0412121452621079e-25, -1.0810592129949165e-05},
    {-5.9550148958160921e-05, -5.7104654456581212e-25, 1.0810592129949182e-05},
    {1.3609823922940861e-04, 6.8060330704498828e-25, -8.4063699110321910e-20},
    {1.8370509979413634e-05, 5.7191630572539285e-25, -8.5497124425277334e-06},
    {1.8370509979413539e-05, 1.2193640572572847e-24, 8.5497124425278096e-06},
};

// Offline dftd4 3.7.0 --grad with ATM on (pbe-D4-ATM, s9 = 1) on XYZ
// (order O H H O H H): tools/cp2k_oracle/inputs/gpw-d4-atm/
// pin_isolated_atm_grad.json (dE = −9.8819741533958229e−04 Ha).
const double kGradIsoAtm[6][3] = {
    {-5.3595225765593731e-05, 0.0, 5.8643149671166952e-20},
    {-5.9875031350020689e-05, 0.0, -1.0591165409066941e-05},
    {-5.9875031350020573e-05, 0.0, 1.0591165409066876e-05},
    {1.3718737385385519e-04, 0.0, 4.2247978288649466e-21},
    {1.8078957305889923e-05, 0.0, -8.5356116662728435e-06},
    {1.8078957305889909e-05, 0.0, 8.5356116662728368e-06},
};

// Same with the periodic POSCAR (O O H H H H remapped to O H H O H H):
// pin_periodic_atm_grad.json (E = −9.9554475834240310e−04 Ha).
//
// NOTE (offline finding, 2026-07-31): dftd4 3.7.0's *periodic* ATM gradient
// is non-conservative by ~2.4e-8 Ha/Bohr against its own printed energy
// (central difference of its s9=1 energy at O0 x ± 0.001 Å gives
// −5.3616541919e−05, its printed g_x is −5.3595427767e−05; the isolated
// gradient is conservative to 2.6e−10 at the same step). ATV's energy
// function equals dftd4's at displaced geometries to a constant
// unit-conversion offset (3.8e−12 Ha, cancels in differences), so the
// conservative analytic gradient can match the printed pin only to that
// oracle defect — hence the 5e−8 direct gate plus a sharp 1e−9 gate against
// the FD of the oracle's own printed energies (kEPerAtmDispO0x*).
const double kGradPerAtm[6][3] = {
    {-5.3595427766884764e-05, -3.3533040430274789e-24, -4.7093130846514167e-22},
    {-5.9463596930597288e-05, -8.8497804033039618e-24, -1.0397154303254306e-05},
    {-5.9463596930597301e-05, -4.9419094613683133e-24, 1.0397154303254314e-05},
    {1.3610085695186597e-04, 1.3230988245241392e-23, -8.4467780181584667e-20},
    {1.8210882338106763e-05, -1.1201532595111321e-24, -8.1642912818385456e-06},
    {1.8210882338106668e-05, 1.8647645948199754e-23, 8.1642912818386235e-06},
};

// dftd4 3.7.0 s9=1 periodic energies with O0 displaced x ± 0.001 Å
// (pin_periodic_atm_dispO0x_{plus,minus}.json): the oracle-printed numbers
// behind the conservative-parity FD gate.
constexpr double kEPerAtmDispO0xPlus = -9.9564614957741663e-04;  // Ha
constexpr double kEPerAtmDispO0xMinus = -9.9544350841746518e-04; // Ha
constexpr double kDispO0xH = 0.001; // Å

} // namespace

// Isolated two-body: dftd4 3.7.0 --mbdscale 0.0 on XYZ.
TEST(GpwDftD4, WaterDimerIsolatedMatchesDftd4) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const double e = dftd4_energy(kZ, pos, cell, /*periodic=*/false);
    const double pin = -9.8816586433966e-04;
    std::printf("[D4 iso] E=%.14e  pin=%.14e  diff=%.2e Ha\n", e, pin, e - pin);
    EXPECT_NEAR(e, pin, 1e-8);
}

// Periodic cubic 12 Å two-body: dftd4 3.7.0 --mbdscale 0.0 on POSCAR.
TEST(GpwDftD4, WaterDimerPeriodicMatchesDftd4) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const double e = dftd4_energy(kZ, pos, cell, /*periodic=*/true);
    const double pin = -9.9458842745847e-04;
    std::printf("[D4 per] E=%.14e  pin=%.14e  diff=%.2e Ha\n", e, pin, e - pin);
    EXPECT_NEAR(e, pin, 1e-8);
    // Images add attraction vs isolated.
    const double e_iso = dftd4_energy(kZ, pos, cell, /*periodic=*/false);
    EXPECT_LT(e, e_iso);
}

TEST(GpwDftD4, IsolatedForcesMatchDftd4Grad) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/false);
    EXPECT_NEAR(r.energy, -9.8816586433966e-04, 1e-8);
    double max_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            // Force = −gradient
            const double want = -kGradIso[i][k];
            const double got = r.forces[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(k)];
            max_err = std::max(max_err, std::abs(got - want));
        }
    }
    std::printf("[D4 iso F] max|F+grad|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

TEST(GpwDftD4, PeriodicForcesMatchDftd4Grad) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/true);
    EXPECT_NEAR(r.energy, -9.9458842745847e-04, 1e-8);
    double max_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = -kGradPer[i][k];
            const double got = r.forces[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(k)];
            max_err = std::max(max_err, std::abs(got - want));
        }
    }
    std::printf("[D4 per F] max|F+grad|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

TEST(GpwDftD4, ForcesMatchEnergyCentralDifference) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/true);
    // Independent FD with a different step confirms internal consistency.
    constexpr double h = 2.0e-4;
    double max_err = 0.0;
    auto p = pos;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double x0 = p[static_cast<std::size_t>(i)]
                               [static_cast<std::size_t>(k)];
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                x0 + h;
            const double ep = dftd4_energy(kZ, p, cell, true);
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                x0 - h;
            const double em = dftd4_energy(kZ, p, cell, true);
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = x0;
            const double f_fd = -(ep - em) / (2.0 * h);
            max_err = std::max(
                max_err, std::abs(r.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)]
                                  - f_fd));
        }
    }
    std::printf("[D4 self-FD] max|F-Ffd|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}

// Isolated ATM: dftd4 3.7.0 -f pbe --mbdscale 1.0 on XYZ.
TEST(GpwDftD4, WaterDimerIsolatedAtmMatchesDftd4) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const double e = dftd4_energy(kZ, pos, cell, /*periodic=*/false,
                                  /*s6=*/1.0, /*s8=*/0.95948085,
                                  /*a1=*/0.38574991, /*a2=*/4.80688534,
                                  /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
                                  /*charge=*/0.0, /*cn_cut=*/30.0,
                                  /*disp2_cut=*/60.0, /*s9=*/1.0);
    const double pin = -9.8819741533958208e-04;
    const double e0 = dftd4_energy(kZ, pos, cell, /*periodic=*/false);
    std::printf("[D4 iso ATM] E=%.14e  pin=%.14e  dE_atm=%.2e Ha\n", e, pin,
                e - e0);
    EXPECT_NEAR(e, pin, 1e-8);
    // ATM contribution must be nonzero on this dimer.
    EXPECT_GT(std::abs(e - e0), 1e-10);
}

// Periodic ATM: dftd4 3.7.0 -f pbe (default s9=1) on POSCAR.
TEST(GpwDftD4, WaterDimerPeriodicAtmMatchesDftd4) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const double e = dftd4_energy(kZ, pos, cell, /*periodic=*/true,
                                  /*s6=*/1.0, /*s8=*/0.95948085,
                                  /*a1=*/0.38574991, /*a2=*/4.80688534,
                                  /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
                                  /*charge=*/0.0, /*cn_cut=*/30.0,
                                  /*disp2_cut=*/60.0, /*s9=*/1.0);
    const double pin = -9.9554475834240288e-04;
    const double e0 = dftd4_energy(kZ, pos, cell, /*periodic=*/true);
    std::printf("[D4 per ATM] E=%.14e  pin=%.14e  dE_atm=%.2e Ha\n", e, pin,
                e - e0);
    EXPECT_NEAR(e, pin, 1e-8);
    EXPECT_GT(std::abs(e - e0), 1e-10);
}

// Strained cell (x scaled by 1+2e-3, fixed fractional coordinates — the
// gpw_stress scale_axis convention), s9=1: dftd4 3.7.0 -f pbe on the strained
// POSCAR (tools/cp2k_oracle/inputs/gpw-d4-atm/pin_strain_x_plus_atm.json).
// This is the unit-level pin behind the gpw-d4-atm-stress regression.
TEST(GpwDftD4, WaterDimerPeriodicStrainedAtmMatchesDftd4) {
    auto pos = water_dimer_bohr();
    constexpr double kH = 2e-3;
    for (auto& p : pos)
        p[0] *= 1.0 + kH;
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L * (1.0 + kH), L, L};
    const double e = dftd4_energy(kZ, pos, cell, /*periodic=*/true,
                                  /*s6=*/1.0, /*s8=*/0.95948085,
                                  /*a1=*/0.38574991, /*a2=*/4.80688534,
                                  /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
                                  /*charge=*/0.0, /*cn_cut=*/30.0,
                                  /*disp2_cut=*/60.0, /*s9=*/1.0);
    const double pin = -9.9375767649976510e-04;
    std::printf("[D4 per ATM strained] E=%.14e  pin=%.14e  diff=%.2e Ha\n", e,
                pin, e - pin);
    EXPECT_NEAR(e, pin, 1e-9);
}

// Analytic s9=1 forces vs dftd4 3.7.0 --grad (isolated). The 1e-9 gate is
// deliberate: the ATM force increment is ~1e-7 Ha/Bohr on this dimer, so a
// 1e-7 gate would pass with the ATM gradient dropped entirely.
TEST(GpwDftD4, IsolatedAtmForcesMatchDftd4Grad) {
    const auto pos = water_dimer_bohr();
    const double cell[3] = {0.0, 0.0, 0.0};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/false, /*s6=*/1.0,
                             /*s8=*/0.95948085, /*a1=*/0.38574991,
                             /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                             /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                             /*disp2_cut=*/60.0, /*s9=*/1.0);
    EXPECT_NEAR(r.energy, -9.8819741533958229e-04, 1e-8);
    double max_err = 0.0;
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 3; ++k)
            max_err = std::max(
                max_err, std::abs(r.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)]
                                  - (-kGradIsoAtm[i][k])));
    std::printf("[D4 iso ATM F] max|F+grad|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-9);
}

// Analytic s9=1 forces vs dftd4 3.7.0 --grad (periodic 12 Å cell). The 5e-8
// direct gate absorbs the oracle's own ~2.4e-8 periodic-ATM
// non-conservativeness (header note) while still failing if the ATM gradient
// is dropped (increment is 4.1e-7); the sharp gate is the FD of the oracle's
// printed displaced energies, which a conservative implementation of the
// shared functional must match to FD truncation (~3e-10).
TEST(GpwDftD4, PeriodicAtmForcesMatchDftd4Grad) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/true, /*s6=*/1.0,
                             /*s8=*/0.95948085, /*a1=*/0.38574991,
                             /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                             /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                             /*disp2_cut=*/60.0, /*s9=*/1.0);
    EXPECT_NEAR(r.energy, -9.9554475834240310e-04, 1e-8);
    double max_err = 0.0;
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 3; ++k)
            max_err = std::max(
                max_err, std::abs(r.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)]
                                  - (-kGradPerAtm[i][k])));
    std::printf("[D4 per ATM F] max|F+grad|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 5e-8);

    // Conservative parity: F_x(O0) vs −FD over the oracle's displaced
    // energies (numbers dftd4 printed; truncation ≈ 3e-10 at this step).
    const double f_fd = -(kEPerAtmDispO0xPlus - kEPerAtmDispO0xMinus)
                        / (2.0 * kDispO0xH * kAngToBohr);
    const double diff = std::abs(r.forces[0][0] - f_fd);
    std::printf("[D4 per ATM F] |F_x(O0)+FD(oracle E)|... diff=%.3e Ha/Bohr\n",
                diff);
    EXPECT_LT(diff, 1e-9);
}

// The ATM-only force increment (s9=1 minus s9=0) against the oracle pin
// delta — isolates the three-body gradient from the (much larger) two-body.
TEST(GpwDftD4, PeriodicAtmForceIncrementMatchesDftd4Delta) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D4Result r0 = dftd4(kZ, pos, cell, /*periodic=*/true);
    const D4Result r1 = dftd4(kZ, pos, cell, /*periodic=*/true, /*s6=*/1.0,
                              /*s8=*/0.95948085, /*a1=*/0.38574991,
                              /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                              /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                              /*disp2_cut=*/60.0, /*s9=*/1.0);
    double max_inc = 0.0;
    double max_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double inc = r1.forces[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(k)]
                               - r0.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)];
            const double want = -(kGradPerAtm[i][k] - kGradPer[i][k]);
            max_inc = std::max(max_inc, std::abs(inc));
            max_err = std::max(max_err, std::abs(inc - want));
        }
    }
    std::printf("[D4 ATM dF] max|dF|=%.3e  max|dF-pin|=%.3e Ha/Bohr\n",
                max_inc, max_err);
    // Non-vacuous: the ATM gradient must actually be there…
    EXPECT_GT(max_inc, 1e-8);
    // …and match the oracle's three-body increment up to the oracle's own
    // periodic-ATM non-conservativeness (~2.4e-8; see kGradPerAtm note).
    EXPECT_LT(max_err, 5e-8);
}

// Analytic-vs-FD of the full s9=1 functional (independent step; FD-limited).
TEST(GpwDftD4, AtmForcesMatchEnergyCentralDifference) {
    const auto pos = water_dimer_bohr();
    const double L = 12.0 * kAngToBohr;
    const double cell[3] = {L, L, L};
    const D4Result r = dftd4(kZ, pos, cell, /*periodic=*/true, /*s6=*/1.0,
                             /*s8=*/0.95948085, /*a1=*/0.38574991,
                             /*a2=*/4.80688534, /*ga=*/3.0, /*gc=*/2.0,
                             /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
                             /*disp2_cut=*/60.0, /*s9=*/1.0);
    constexpr double h = 2.0e-4;
    double max_err = 0.0;
    auto p = pos;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double x0 = p[static_cast<std::size_t>(i)]
                               [static_cast<std::size_t>(k)];
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                x0 + h;
            const double ep = dftd4_energy(
                kZ, p, cell, true, 1.0, 0.95948085, 0.38574991, 4.80688534,
                3.0, 2.0, 6.0, 0.0, 30.0, 60.0, /*s9=*/1.0);
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                x0 - h;
            const double em = dftd4_energy(
                kZ, p, cell, true, 1.0, 0.95948085, 0.38574991, 4.80688534,
                3.0, 2.0, 6.0, 0.0, 30.0, 60.0, /*s9=*/1.0);
            p[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] = x0;
            const double f_fd = -(ep - em) / (2.0 * h);
            max_err = std::max(
                max_err, std::abs(r.forces[static_cast<std::size_t>(i)]
                                          [static_cast<std::size_t>(k)]
                                  - f_fd));
        }
    }
    std::printf("[D4 ATM self-FD] max|F-Ffd|=%.3e Ha/Bohr\n", max_err);
    EXPECT_LT(max_err, 1e-7);
}
