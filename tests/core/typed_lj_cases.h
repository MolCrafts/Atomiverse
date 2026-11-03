/// @file tests/core/typed_lj_cases.h
/// @brief Analytic references shared by the CPU and CUDA typed-LJ unit tests.
///
/// Both suites build the same dimers from the same rows and assert against
/// these closed forms, written from the definitions (not from the kernels'
/// LAMMPS lj1..lj4 / expanded r⁻⁴ constants).
#pragma once

#include <cmath>
#include <vector>

#include "atomiverse/core/lj_type_coeff.h"

namespace typed_lj_cases {

/// Global (Coulomb / default LJ) cutoff [Å] of every case.
inline constexpr double kCutoff = 9.0;
/// Type 0 diagonal: ε [eV], σ [Å], own cutoff [Å].
inline constexpr double kEps0 = 0.01, kSig0 = 3.0, kCut0 = 6.0;
/// Type 1 diagonal: ε [eV], σ [Å], own cutoff [Å] (beyond the global one).
inline constexpr double kEps1 = 0.04, kSig1 = 4.0, kCut1 = 10.0;
/// Neighbor cutoff that covers every case [Å].
inline constexpr double kNeighbor = 10.0;

/// Diagonals with per-type cutoffs; (0, 1) is left to LAMMPS mixing.
inline std::vector<ATV::LJTypeCoeff> diagonal_rows() {
    return {{0, 0, kEps0, kSig0, kCut0}, {1, 1, kEps1, kSig1, kCut1}};
}

/// Plain LJ 12-6 energy [eV].
inline double lj(double eps, double sig, double r) {
    const double s6 = std::pow(sig / r, 6.0);
    return 4.0 * eps * (s6 * s6 - s6);
}
/// LJ energy shifted to zero at rc [eV]; 0 beyond rc.
inline double lj_shifted(double eps, double sig, double rc, double r) {
    return r < rc ? lj(eps, sig, r) - lj(eps, sig, rc) : 0.0;
}
/// Radial LJ force −dU/dr [eV/Å] (positive = repulsive); 0 beyond rc.
inline double lj_force(double eps, double sig, double rc, double r) {
    if (!(r < rc))
        return 0.0;
    const double s6 = std::pow(sig / r, 6.0);
    return 24.0 * eps / r * (2.0 * s6 * s6 - s6);
}
/// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] [eV]; 0 beyond.
inline double r4(double S, double sig, double rc, double r) {
    if (!(r < rc))
        return 0.0;
    const double a = std::pow(sig / r, 4.0), b = std::pow(sig / rc, 4.0);
    return -S * (a - b + 4.0 * b * (r / rc - 1.0));
}
/// −dU/dr of `r4` [eV/Å]; 0 beyond r_c.
inline double r4_force(double S, double sig, double rc, double r) {
    if (!(r < rc))
        return 0.0;
    const double s4 = std::pow(sig, 4.0);
    return S * (-4.0 * s4 / std::pow(r, 5.0) + 4.0 * s4 / std::pow(rc, 5.0));
}

/// LJ tail energy of one ordered type pair times V [eV·Å³]:
/// 2π N_i N_j ∫_rc^∞ r² u(r) dr with u = 4ε[(σ/r)¹² − (σ/r)⁶]
/// (homogeneous g = 1; Allen & Tildesley eq. 2.144).
inline double lj_tail_energy(double eps, double sig, double rc, double ni,
                             double nj) {
    const double pi = 3.14159265358979323846;
    const double integral =
        4.0 * eps
        * (std::pow(sig, 12.0) / (9.0 * std::pow(rc, 9.0))
           - std::pow(sig, 6.0) / (3.0 * std::pow(rc, 3.0)));
    return 2.0 * pi * ni * nj * integral;
}
/// LJ tail virial (each diagonal component) times V [eV·Å³]:
/// −(2π/3) N_i N_j ∫_rc^∞ r³ u′(r) dr, so that P_tail = this / V².
inline double lj_tail_virial(double eps, double sig, double rc, double ni,
                             double nj) {
    const double pi = 3.14159265358979323846;
    const double integral =
        4.0 * eps
        * (-12.0 * std::pow(sig, 12.0) / (9.0 * std::pow(rc, 9.0))
           + 6.0 * std::pow(sig, 6.0) / (3.0 * std::pow(rc, 3.0)));
    return -2.0 * pi / 3.0 * ni * nj * integral;
}

/// Lorentz–Berthelot / arithmetic-cutoff mix of the two diagonals.
inline constexpr double kEps01 = 0.02; // sqrt(0.01 * 0.04)
inline constexpr double kSig01 = 3.5;
inline constexpr double kCut01 = 8.0;

/// The r⁻⁴ row of the CG Li–EO style cases: S [eV], σ [Å], r_c [Å].
inline constexpr double kS = 0.05, kSigR4 = 2.5, kCutR4 = 6.0;

} // namespace typed_lj_cases
