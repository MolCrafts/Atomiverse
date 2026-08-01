/// @file tests/cpu/gpw/test_gpw_bomd.cpp
/// @brief Unit gate: MD binds GPWEvaluator forces for short BOMD steps.
///
/// One method surface only — public MD + VelocityVerlet + GPWEvaluator.
/// Full NVE energy-drift pin lives in regressions/gpw/gpw-bomd-water (e2e);
/// the multi-k RKS twin is regressions/gpw/gpw-bomd-kpoints; multi-k UKS
/// dual P_k^σ is regressions/gpw/gpw-bomd-uks-kpoints.

#include <gtest/gtest.h>

#include <cmath>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

namespace {

/// Periodic H2 in a cubic 12 Bohr cell, bond 1.0 Bohr along x (same family
/// as regressions/gpw/gpw-kpoint-forces / gpw-bomd-kpoints).
cpu::Frame make_h2_periodic() {
    cpu::Frame f(2);
    constexpr double bohr2ang = 0.52917720859;
    const double x[2] = {5.5 * bohr2ang, 6.5 * bohr2ang};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = 6.0 * bohr2ang;
        f.span(kAtom_z)[i] = 6.0 * bohr2ang;
        f.span(kAtom_vx)[i] = 0.0;
        f.span(kAtom_vy)[i] = 0.0;
        f.span(kAtom_vz)[i] = 0.0;
    }
    f.box = cpu::Box::cubic(12.0 * bohr2ang);
    return f;
}

/// Cheap plain multi-k H2 evaluator (mesh 1x1x2 unreduced, no forces).
cpu::gpw::GPWEvaluator h2_kpoints_eval() {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {H}, /*cutoff_ry*/ 60.0, /*max_iter*/ 60, /*tol*/ 1e-6,
        /*mix*/ 0.3, cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ false);
    eval.k_mesh[0] = 1;
    eval.k_mesh[1] = 1;
    eval.k_mesh[2] = 2;
    eval.k_time_reversal = false; // two sampled k
    return eval;
}

} // namespace

TEST(GpwBomd, MdBindsEvaluatorForcesTwoSteps) {
    // Binding gate only: MD drives GPWEvaluator forces for two VV steps and
    // leaves finite state. Physics pins live in regressions/gpw/
    // gpw-bomd-water (Γ) and gpw-bomd-kpoints (multi-k). Cheap periodic H2
    // Γ system — the previous 200 Ry MT water exceeded the 10 s unit cap.
    auto frame = make_h2_periodic();
    frame.span(kAtom_vx)[0] = -0.001; // antisymmetric stretch: KE > 0
    frame.span(kAtom_vx)[1] = 0.001;

    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {H}, /*cutoff_ry*/ 60.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.5});
    sim.on(md::Phase::force).add(std::move(eval));
    sim.enable_timing(false);
    sim.build();
    ASSERT_EQ(sim.run(frame, 2), 2);

    const double pe = frame.meta.get(kEnergy);
    const double ke = frame.meta.get(kKinetic);
    EXPECT_TRUE(std::isfinite(pe)) << "PE=" << pe;
    EXPECT_TRUE(std::isfinite(ke)) << "KE=" << ke;
    EXPECT_GT(ke, 1e-12);

    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_x)[i])));
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_y)[i])));
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.span(kAtom_z)[i])));
        EXPECT_TRUE(
            std::isfinite(static_cast<double>(frame.span(kAtom_fx)[i])));
        EXPECT_TRUE(
            std::isfinite(static_cast<double>(frame.span(kAtom_fy)[i])));
        EXPECT_TRUE(
            std::isfinite(static_cast<double>(frame.span(kAtom_fz)[i])));
    }
}

TEST(GpwBomd, PlainMultiKRelaunchWarmStartsFromPk) {
    // Plain (non-XL/CPMD/SGCP) multi-k BOMD path: the second launch must seed
    // the SCF from the previous step's converged P_k and reach the same fixed
    // point in fewer iterations — the k-path analogue of the Γ guess_C /
    // guess_P warm start (CP2K `EXTRAPOLATION USE_PREV_P`).
    auto frame = make_h2_periodic();
    auto eval = h2_kpoints_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    ASSERT_NO_THROW(eval.build(bc));
    cpu::LaunchContext lc{rt, frame, ATV::slot_of(ATV::md::Phase::force),
                          /*step=*/0};

    eval.launch(lc); // cold multi-k SCF
    const auto& cold = eval.last_result();
    ASSERT_TRUE(cold.converged);
    ASSERT_EQ(cold.n_kpts, 2);
    ASSERT_EQ(static_cast<int>(cold.P_k.size()), cold.n_kpts);
    const int iters_cold = cold.iters;
    const double e_cold = cold.e_total;
    ASSERT_GE(iters_cold, 2) << "cold SCF trivially converged — warm-start "
                                "iteration gate would be vacuous";

    eval.launch(lc); // same geometry: warm from previous P_k
    const auto& warm = eval.last_result();
    EXPECT_TRUE(warm.converged);
    EXPECT_LT(warm.iters, iters_cold)
        << "second launch did not warm-start from the previous P_k";
    EXPECT_NEAR(warm.e_total, e_cold, 1e-6)
        << "warm-started SCF must reach the cold fixed point";
}

TEST(GpwBomd, PlainMultiKUksRelaunchWarmStartsBothSpins) {
    // Multi-k UKS: the warm start must pass both spin densities (the solver
    // rejects a partial spin guess) and reach the cold fixed point.
    auto frame = make_h2_periodic();
    auto eval = h2_kpoints_eval();
    eval.n_alpha = 1;
    eval.n_beta = 1;
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    ASSERT_NO_THROW(eval.build(bc));
    cpu::LaunchContext lc{rt, frame, ATV::slot_of(ATV::md::Phase::force),
                          /*step=*/0};

    eval.launch(lc);
    const auto& cold = eval.last_result();
    ASSERT_TRUE(cold.converged);
    ASSERT_EQ(cold.n_kpts, 2);
    ASSERT_EQ(static_cast<int>(cold.P_k.size()), cold.n_kpts);
    ASSERT_EQ(static_cast<int>(cold.P_k_beta.size()), cold.n_kpts);
    const int iters_cold = cold.iters;
    const double e_cold = cold.e_total;
    ASSERT_GE(iters_cold, 2);

    eval.launch(lc); // must not throw (no partial spin guess) and be warm
    const auto& warm = eval.last_result();
    EXPECT_TRUE(warm.converged);
    EXPECT_LT(warm.iters, iters_cold);
    EXPECT_NEAR(warm.e_total, e_cold, 1e-6);
}
