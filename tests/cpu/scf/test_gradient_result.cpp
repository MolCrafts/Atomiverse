/// @file tests/cpu/scf/test_gradient_result.cpp
/// @brief scf-review-04-meanfield-dedup (Item 7): the Ha/Bohr -> eV/Angstrom
///        gradient glue — three copy-pasted, factory-named helpers
///        (make_result in gradient.cpp, make_force_result in mp2_gradient.cpp
///        and ccsd_gradient.cpp) — is hoisted onto GradientResult as a
///        tag-disambiguated converting constructor
///        GradientResult(FromHartreeBohr, grad_ha_bohr, energy_ha). This
///        RED-first unit pins that constructor in ISOLATION against
///        hand-computed eV/Angstrom + eV literals; there is no SCF here.
///
/// Golden provenance (hard-coded): the converting ctor must reproduce the
/// deleted make_result semantics exactly —
///   forces(A,d) = -units::convert(grad(A,d), Force::HartreePerBohr,
///                                             Force::eVPerAngstrom)
///   energy      =  units::convert(energy_ha, Energy::Hartree, Energy::eV)
/// The literals below were computed offline to full double precision with the
/// SAME ATV::units::convert path and shipping CODATA constants
/// (kHartreeInEv = 27.211386245988, kAngstromInBohr = 1.8897261245650618,
/// so kBohrInAngstrom = 1/kAngstromInBohr). NOTE: they are NOT
/// grad * (27.211386245988 / 0.529177210903) — that rounded-Bohr shortcut
/// differs from the code path by ~1.6e-10 and would fail this 1e-12 gate; the
/// force factor the code actually applies is 1/kEvPerAngstromInHartreePerBohr
/// = 51.42206747467393 eV/Angstrom per Ha/Bohr.

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/gradient_result.h"

using ATV::cpu::scf::FromHartreeBohr;
using ATV::cpu::scf::GradientResult;

namespace {

// (a) Converting ctor: single atom (1x3), sign flip + unit conversion to the
//     public eV/Angstrom (forces) and eV (energy) currency.
TEST(TestGradientResult, FromHartreeBohrConvertsAndFlipsSign) {
    // grad = dE/dR in Ha/Bohr; the physical force is F = -grad.
    xt::xtensor<double, 2> grad_ha_bohr = {{0.1, -0.2, 0.05}};
    const double energy_ha = -1.5;

    GradientResult g{FromHartreeBohr{}, grad_ha_bohr, energy_ha};

    ASSERT_EQ(g.forces.shape()[0], 1u);
    ASSERT_EQ(g.forces.shape()[1], 3u);

    // force = -grad * (1 / kEvPerAngstromInHartreePerBohr) [eV/Angstrom];
    // sign flipped relative to grad.
    EXPECT_NEAR(g.forces(0, 0), -5.142206747467393, 1e-12);  // -(+0.1)
    EXPECT_NEAR(g.forces(0, 1), 10.284413494934785, 1e-12);  // -(-0.2)
    EXPECT_NEAR(g.forces(0, 2), -2.5711033737336964, 1e-12); // -(+0.05)

    // energy = energy_ha * kHartreeInEv [eV]; no sign flip.
    EXPECT_NEAR(g.energy, -40.817079368982, 1e-12);
}

// (b) Default ctor: user-declared ctors remove aggregate init, so the spec
//     restores GradientResult() = default. Zero energy, empty forces.
TEST(TestGradientResult, DefaultCtorIsZeroEnergyEmptyForces) {
    GradientResult g{};
    EXPECT_DOUBLE_EQ(g.energy, 0.0);
    EXPECT_EQ(g.forces.size(), 0u);
}

// (c) Member-wise eV ctor: the documented brace syntax
//     GradientResult{forces, energy} (docs/tutorial ccsd-grad) stays
//     compilable and stores its arguments verbatim (no conversion).
TEST(TestGradientResult, MemberWiseEvCtorStoresVerbatim) {
    xt::xtensor<double, 2> forces_ev = {{1.0, -2.0, 3.0}, {-4.0, 5.0, -6.0}};

    GradientResult g{forces_ev, 2.0}; // eV/Angstrom, eV — verbatim

    ASSERT_EQ(g.forces.shape()[0], 2u);
    ASSERT_EQ(g.forces.shape()[1], 3u);
    EXPECT_DOUBLE_EQ(g.forces(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(g.forces(0, 1), -2.0);
    EXPECT_DOUBLE_EQ(g.forces(1, 2), -6.0);
    EXPECT_DOUBLE_EQ(g.energy, 2.0);
}

} // namespace

#endif
