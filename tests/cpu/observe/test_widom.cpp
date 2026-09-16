/// @file tests/cpu/observe/test_widom.cpp
/// @brief Unit tests for cpu::Widom (single-function / method cases).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/widom.h"

using namespace ATV;

namespace {

cpu::Frame empty_box(double L) {
    cpu::Frame f(0);
    f.box = cpu::Box::cubic(L);
    return f;
}

cpu::Frame one_atom_at_origin(double L) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(L);
    return f;
}

} // namespace

TEST(Widom, IdealGasEmptyHostMuExZero) {
    cpu::Widom w(300.0, 0.5, 3.0, 8.0, /*n_trials=*/2000, /*seed=*/42);
    auto host = empty_box(20.0);
    w.sample(host);
    EXPECT_EQ(w.n_insertions(), 2000);
    EXPECT_NEAR(w.boltzmann_avg(), 1.0, 1e-15);
    EXPECT_NEAR(w.mu_ex(), 0.0, 1e-12);
}

TEST(Widom, HardSphereFreeVolumeFixedHost) {
    // One sphere of diameter σ at origin; L > 2σ so exclusion ball fits.
    const double sigma = 2.0; // Å
    const double L = 10.0;    // Å
    const double T = 300.0;   // K
    const double kT = units::kB_ev_per_k * T;
    const double V_ex = 4.0 / 3.0 * units::pi * sigma * sigma * sigma;
    const double V = L * L * L;
    const double P_insert = 1.0 - V_ex / V;
    const double mu_ref = -kT * std::log(P_insert);

    cpu::Widom w(T, sigma, /*n_trials=*/200000, /*seed=*/7,
                 cpu::WidomPotential::hard_sphere);
    auto host = one_atom_at_origin(L);
    w.sample(host);

    // Monte Carlo free-volume; statistical noise ~ 1/sqrt(N).
    EXPECT_NEAR(w.boltzmann_avg(), P_insert, 0.01);
    EXPECT_NEAR(w.mu_ex(), mu_ref, 0.002 * kT);
}

TEST(Widom, AveragesBoltzmannNotMu) {
    // Jensen check: with varying ΔU, −kT ln⟨e⟩ ≠ ⟨−kT ln e⟩.
    // Ideal empty host: both are 0 so only assert the API uses boltzmann_avg.
    cpu::Widom w(200.0, 1.0, 3.0, 8.0, 100, 1);
    w.sample(empty_box(15.0));
    const double kT = units::kB_ev_per_k * 200.0;
    EXPECT_NEAR(w.mu_ex(), -kT * std::log(w.boltzmann_avg()), 1e-15);
}

TEST(Widom, AllOverlapsReturnsNaNMu) {
    // Host fills space: one HS in a box smaller than the exclusion diameter
    // is wrong (PBC self-overlap). Use many host atoms densely instead:
    // tiny box, one HS diameter larger than box diagonal → almost all overlap.
    // Simpler: empty is fine; force all-overlap by sampling HS with host atom
    // and diameter > box so every random site is within σ of the host under
    // min-image when L < σ.
    const double sigma = 5.0;
    const double L = 4.0; // L < σ → every point is within σ of origin under PBC
    cpu::Widom w(300.0, sigma, 5000, 1, cpu::WidomPotential::hard_sphere);
    w.sample(one_atom_at_origin(L));
    EXPECT_EQ(w.boltzmann_avg(), 0.0);
    EXPECT_TRUE(std::isnan(w.mu_ex()));
}

TEST(Widom, NameAndObserverKind) {
    cpu::Widom w;
    EXPECT_EQ(w.name(), "widom");
    EXPECT_NE(dynamic_cast<cpu::Observer*>(&w), nullptr);
}
