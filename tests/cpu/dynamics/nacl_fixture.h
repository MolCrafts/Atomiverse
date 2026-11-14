/// @file tests/cpu/dynamics/nacl_fixture.h
/// @brief A small hot Tosi–Fumi NaCl configuration shared by the NEMD
///        thermostat tests (metal units).
#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_born.h"

namespace nacl_fixture {

inline constexpr int kNa = 0; ///< type of Na⁺
inline constexpr int kCl = 1; ///< type of Cl⁻
inline constexpr int kSide = 4;
inline constexpr int kN = kSide * kSide * kSide;
inline constexpr double kSpacing = 2.82;     ///< [Å]; L = 11.28 Å
inline constexpr double kCut = 5.5;          ///< [Å] < L/2
inline constexpr double kAlpha = 3.0 / kCut; ///< Ewald α [1/Å]
inline constexpr double kMassNa = 22.9898;   ///< [g/mol]
inline constexpr double kMassCl = 35.453;    ///< [g/mol]

/// @brief Tosi–Fumi NaCl rows; `PairBorn` adds +D/r⁸, so D is negated.
///
/// Fumi and Tosi, J. Phys. Chem. Solids 25, 31 and 45 (1964), cgs: b =
/// 0.338e-12 erg with Pauling factors 1.25, 1.00, 0.75; ρ = 0.317 Å;
/// σ_Na = 1.170, σ_Cl = 1.585 Å; C = 1.68, 11.2, 116 × 1e-60 erg cm⁶; D =
/// 0.8, 13.9, 233 × 1e-76 erg cm⁸ (Na–Na, Na–Cl, Cl–Cl). In eV and Å
/// (1 eV = 1.602176634e-12 erg, 1 cm = 1e8 Å) every coefficient is its cgs
/// mantissa / 1.602176634.
inline std::vector<ATV::cpu::BornTypeCoeff> tosi_fumi() {
    constexpr double rho = 0.317; // [Å]
    return {
        {kNa, kNa, 0.2637038, rho, 2.340, 1.0485735, -0.4993207},
        {kNa, kCl, 0.2109630, rho, 2.755, 6.9904902, -8.6756976},
        {kCl, kCl, 0.1582223, rho, 3.170, 72.4015053, -145.4271614},
    };
}

/// @brief Rock salt of `side`³ ions displaced by up to ±`jitter` Å per axis,
///        Maxwell velocities at `temperature` [K] with zero total momentum.
///        Rows are written in reverse lattice order, so a spatial sort
///        permutes them.
inline ATV::cpu::Frame nacl(double temperature, unsigned seed,
                            double jitter = 0.25, int side = kSide) {
    using namespace ATV;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> shift(-jitter, jitter);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    const int n = side * side * side;
    cpu::Frame f(n);
    int idx = 0;
    double px = 0, py = 0, pz = 0, mass_sum = 0;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k, ++idx) {
                const int row = n - 1 - idx;
                const bool na = (i + j + k) % 2 == 0;
                const double m = na ? kMassNa : kMassCl;
                const double sd =
                    std::sqrt(metal.boltz * temperature / (m * metal.mvv2e));
                f.span(kAtom_x)[row] = i * kSpacing + shift(rng);
                f.span(kAtom_y)[row] = j * kSpacing + shift(rng);
                f.span(kAtom_z)[row] = k * kSpacing + shift(rng);
                f.span(kAtom_vx)[row] = sd * gauss(rng);
                f.span(kAtom_vy)[row] = sd * gauss(rng);
                f.span(kAtom_vz)[row] = sd * gauss(rng);
                f.span(kAtom_charge)[row] = na ? 1.0 : -1.0;
                f.span(kAtom_mass)[row] = m;
                f.span(kAtom_type)[row] = na ? kNa : kCl;
                f.span(kAtom_tag)[row] = idx + 1;
                px += m * f.span(kAtom_vx)[row];
                py += m * f.span(kAtom_vy)[row];
                pz += m * f.span(kAtom_vz)[row];
                mass_sum += m;
            }
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_vx)[i] -= px / mass_sum;
        f.span(kAtom_vy)[i] -= py / mass_sum;
        f.span(kAtom_vz)[i] -= pz / mass_sum;
    }
    f.box = cpu::Box::cubic(side * kSpacing);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

} // namespace nacl_fixture
