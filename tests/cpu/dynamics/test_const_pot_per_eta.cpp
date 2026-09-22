/// @file tests/cpu/dynamics/test_const_pot_per_eta.cpp
/// @brief Unit tests for ConstPotBO per-electrode Gaussian width η_e
///        (single methods; analytic Gaussian-charge formulas).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr double kSqrt2OverPi = 0.79788456080286535588; // sqrt(2/π)

/// Two single-site plates at |Δz| = 4 Å in a large vacuum box (no wrap).
cpu::Frame vacuum_pair() {
    cpu::Frame f(2);
    const double z[2] = {18.0, 22.0};
    const int typ[2] = {1, 2};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 196.97;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);
    return f;
}

/// One electrode site (type 1) + one point charge q = 0.5 e at r = 3 Å.
cpu::Frame single_site_plus_ion() {
    cpu::Frame f(2);
    const double z[2] = {15.0, 12.0};
    const double q0[2] = {0.0, 0.5};
    const int typ[2] = {1, 2};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
    }
    f.box = cpu::Box::cubic(30.0);
    return f;
}

void run_once(cpu::md::ConstPotBO& conp, cpu::Frame& f) {
    cpu::Context ctx;
    conp.build({ctx, f});
    conp.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(ConstPotPerEta, ThreeArgOverloadValidatesEta) {
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0, 0.955); // valid per-group width
    EXPECT_THROW(conp.add_electrode(2, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(conp.add_electrode(2, 1.0, -1.5), std::invalid_argument);
}

TEST(ConstPotPerEta, SharedEtaTwinExact) {
    // Registering every group at the shared value via the 3-arg overload must
    // reproduce the 2-arg (shared-η) charges exactly.
    auto f_shared = vacuum_pair();
    cpu::md::ConstPotBO shared(2.0);
    shared.add_electrode(1, 0.0);
    shared.add_electrode(2, 1.0);
    run_once(shared, f_shared);

    auto f_twin = vacuum_pair();
    cpu::md::ConstPotBO twin(2.0);
    twin.add_electrode(1, 0.0, 2.0);
    twin.add_electrode(2, 1.0, 2.0);
    run_once(twin, f_twin);

    ASSERT_EQ(shared.n_electrode(), twin.n_electrode());
    for (std::size_t i = 0; i < shared.charges().size(); ++i)
        EXPECT_NEAR(twin.charges()[i], shared.charges()[i], 1e-14);
}

TEST(ConstPotPerEta, MatrixConditionIsEtaRatio) {
    // A_ii = k √(2/π) η_i ⇒ diag ratio = η_max/η_min (no images, no TF).
    auto f = vacuum_pair();
    cpu::md::ConstPotBO conp(2.0); // shared member unused by these groups
    conp.add_electrode(1, 0.0, 1.979);
    conp.add_electrode(2, 1.0, 0.955);
    run_once(conp, f);
    EXPECT_NEAR(conp.matrix_condition(), 1.979 / 0.955, 1e-12);
}

TEST(ConstPotPerEta, VacuumPairAnalyticCharge) {
    // Q = ΔΨ / (A11 + A22 − 2 A12), A_ii = k√(2/π)η_i,
    // A12 = k erf(η12 d)/d, η12 = η1η2/√(η1²+η2²)
    // (Reed–Lanning–Madden JCP 126, 084704 (2007); Scalfi et al. 2020).
    const double eta1 = 1.979, eta2 = 0.955, d = 4.0, dpsi = 1.0;
    auto f = vacuum_pair();
    cpu::md::ConstPotBO conp(2.0);
    conp.add_electrode(1, 0.0, eta1);
    conp.add_electrode(2, dpsi, eta2);
    conp.electroneutrality = true;
    run_once(conp, f);

    const double k = units::coulomb;
    const double a11 = k * kSqrt2OverPi * eta1;
    const double a22 = k * kSqrt2OverPi * eta2;
    const double eta12 = eta1 * eta2 / std::sqrt(eta1 * eta1 + eta2 * eta2);
    const double a12 = k * std::erf(eta12 * d) / d;
    const double q_expect = dpsi / (a11 + a22 - 2.0 * a12);

    EXPECT_NEAR(q_expect, 0.037722100829787, 1e-12); // hard-coded sanity
    EXPECT_NEAR(conp.total_charge(1), q_expect, 1e-10);
    EXPECT_NEAR(conp.total_charge(0), -q_expect, 1e-10);
    EXPECT_LE(conp.neutrality_residual(), 1e-12);
}

TEST(ConstPotPerEta, GroupEtaOverridesSharedInRhs) {
    // Ψ = 0, no neutrality: A Q = −b ⇒ Q = −q erf(η_g r)/r / (√(2/π) η_g).
    // Shared member η deliberately different — the group width must win.
    const double eta_g = 0.955, qj = 0.5, r = 3.0;
    auto f = single_site_plus_ion();
    cpu::md::ConstPotBO conp(3.0); // shared ≠ group
    conp.add_electrode(1, 0.0, eta_g);
    conp.electroneutrality = false;
    run_once(conp, f);

    const double q_expect =
        -(qj * std::erf(eta_g * r) / r) / (kSqrt2OverPi * eta_g);
    EXPECT_NEAR(q_expect, -0.21871735012424745, 1e-12); // hard-coded sanity
    EXPECT_NEAR(conp.total_charge(0), q_expect, 1e-10);
}

TEST(ConstPotPerEta, ElectrodeForceMatchesRhsKernelGradient) {
    // F_z = Q E_z with E = −∇[k q erf(η_g r)/r] — the gradient of the SAME
    // kernel the RHS used (pins the α = η/√2 → η_i fix).
    const double eta_g = 0.955, qj = 0.5, r = 3.0;
    auto f = single_site_plus_ion();
    cpu::md::ConstPotBO conp(3.0);
    conp.add_electrode(1, 0.0, eta_g);
    conp.electroneutrality = false;
    conp.write_electrode_forces = true;
    run_once(conp, f);

    const double k = units::coulomb;
    constexpr double kSqrtPi = 1.7724538509055160273;
    const double ar = eta_g * r;
    const double dPhi_dr = k * qj
                           * (2.0 * eta_g / kSqrtPi * std::exp(-ar * ar) / r
                              - std::erf(ar) / (r * r));
    // Electrode sits ABOVE the ion (dz = +3): r_hat = +z, E_z = −dΦ/dr.
    const double fz_expect = conp.total_charge(0) * (-dPhi_dr);
    EXPECT_NEAR(static_cast<double>(f.span(kAtom_fz)[0]), fz_expect, 1e-10);
    EXPECT_NE(fz_expect, 0.0);
}
