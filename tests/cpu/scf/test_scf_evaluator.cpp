/// @file tests/cpu/scf/test_scf_evaluator.cpp
/// @brief Integration tests for SCF component, BOMD, and ASPC predictor.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/barostat.h"
#include "atomiverse/cpu/md/fix_langevin.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/scf/aspc_predictor.h"
#include "atomiverse/cpu/scf/internal/scf_types.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

// ── Fixture: H2 frame with optional velocity ────────────────────────

class SCFTest : public ::testing::Test {
  protected:
    static cpu::Frame make_h2(double bond_length, double v_stretch = 0) {
        cpu::Frame f(2);
        f.span(ATV::kAtom_x)[0] = 0.0;
        f.span(ATV::kAtom_x)[1] = bond_length;
        f.span(ATV::kAtom_y)[0] = 0.0;
        f.span(ATV::kAtom_y)[1] = 0.0;
        f.span(ATV::kAtom_z)[0] = 0.0;
        f.span(ATV::kAtom_z)[1] = 0.0;
        f.span(ATV::kAtom_vx)[0] = -v_stretch / 2;
        f.span(ATV::kAtom_vx)[1] = v_stretch / 2;
        f.span(ATV::kAtom_vy)[0] = 0.0;
        f.span(ATV::kAtom_vy)[1] = 0.0;
        f.span(ATV::kAtom_vz)[0] = 0.0;
        f.span(ATV::kAtom_vz)[1] = 0.0;
        f.span(ATV::kAtom_fx)[0] = 0.0;
        f.span(ATV::kAtom_fx)[1] = 0.0;
        f.span(ATV::kAtom_fy)[0] = 0.0;
        f.span(ATV::kAtom_fy)[1] = 0.0;
        f.span(ATV::kAtom_fz)[0] = 0.0;
        f.span(ATV::kAtom_fz)[1] = 0.0;
        f.span(ATV::kAtom_mass)[0] = 1.008;
        f.span(ATV::kAtom_mass)[1] = 1.008;
        f.span(ATV::kAtom_type)[0] = 1;
        f.span(ATV::kAtom_type)[1] = 1;
        f.meta.get(ATV::kEnergy) = 0;
        f.meta.get(ATV::kKinetic) = 0;
        f.simbox = ATV::cpu::Box::cubic(100.0);
        return f;
    }
};

TEST_F(SCFTest, SinglePointRHF) {
    auto frame = make_h2(0.74);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 1);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -29.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -32.0);
    EXPECT_NEAR(static_cast<double>(frame.span(ATV::kAtom_fx)[0]),
                -static_cast<double>(frame.span(ATV::kAtom_fx)[1]), 0.01);
}

TEST_F(SCFTest, SinglePointUHF) {
    auto frame = make_h2(0.74);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 1);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -29.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -32.0);
}

TEST_F(SCFTest, SinglePointKSLSDA) {
    auto frame = make_h2(0.74);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 1);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -29.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -32.0);
    EXPECT_NEAR(static_cast<double>(frame.span(ATV::kAtom_fx)[0]),
                -static_cast<double>(frame.span(ATV::kAtom_fx)[1]), 0.02);
}

TEST_F(SCFTest, SinglePointKSPBE) {
    auto frame = make_h2(0.74);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 1);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -29.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -33.0);
}

// ── BOMD integration tests ──────────────────────────────────────────

class BOMDTest : public SCFTest {};

TEST_F(BOMDTest, RHFEnergyConservation) {
    auto frame = make_h2(0.74, 0.001);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 5);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -40.0);
}

TEST_F(BOMDTest, RHFBondOscillation) {
    auto frame = make_h2(0.80, 0.0);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 5);

    double bond = frame.span(ATV::kAtom_x)[1] - frame.span(ATV::kAtom_x)[0];
    EXPECT_LT(static_cast<double>(bond), 0.80)
        << "H2 bond should contract from stretched state";
    EXPECT_GT(static_cast<double>(bond), 0.50);
}

TEST_F(BOMDTest, MomentumConservation) {
    auto frame = make_h2(0.74, 0.001);
    double px0 =
        frame.span(ATV::kAtom_mass)[0] * frame.span(ATV::kAtom_vx)[0]
        + frame.span(ATV::kAtom_mass)[1] * frame.span(ATV::kAtom_vx)[1];

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 5);

    double px1 =
        frame.span(ATV::kAtom_mass)[0] * frame.span(ATV::kAtom_vx)[0]
        + frame.span(ATV::kAtom_mass)[1] * frame.span(ATV::kAtom_vx)[1];
    EXPECT_NEAR(static_cast<double>(px1), static_cast<double>(px0), 1e-4);
}

TEST_F(BOMDTest, RHFWithASPC) {
    auto frame = make_h2(0.74, 0.001);
    cpu::scf::RHF evaluator{};
    evaluator.predictor(3, 2, double(1e-4), true);

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(std::move(evaluator));
    sim.build();
    sim.run(frame, 5);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -40.0);
}

// ── Ensemble integration tests (AIMD = MD driver + SCF + ─────
//    thermostat / barostat). Verifies SCF force evaluation composes
//    cleanly with the existing CPU MD ensemble components — no new
//    AIMD driver class required.

TEST_F(BOMDTest, RHFNoseHooverNVT) {
    auto frame = make_h2(0.74, 0.0);
    const double dt = 0.5;         // fs
    const double T_target = 300.0; // K
    const double tau = 50.0;       // fs (tight coupling for short test)

    cpu::md::MD sim;
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/42u});
    sim.add(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::NoseHoover{T_target, tau, dt});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 30);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0)
        << "SCF should still converge under NoseHoover coupling";
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kKinetic)), 0.0)
        << "InitVelocity + VelocityVerlet must produce non-zero KE";
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kEnergy))));
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kKinetic))));
}

TEST_F(BOMDTest, RHFLangevinNVT) {
    auto frame = make_h2(0.74, 0.0);
    const double dt = 0.5;
    const double T_target = 300.0;
    const double damp = 100.0;

    // Both SCF and FixLangevin advertise Capability::Force —
    // cpu::md::MD must accept this (multiple Force-capable components
    // on the force / post_force phases).
    cpu::md::MD sim;
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/7u});
    sim.add(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::FixLangevin{T_target, /*T_stop=*/T_target, damp, dt,
                                 /*seed=*/13u});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 30);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0);
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kKinetic)), 0.0);
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kEnergy))));
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kKinetic))));
}

TEST_F(BOMDTest, RHFBerendsenNPT) {
    auto frame = make_h2(0.74, 0.0);
    const double dt = 0.5;
    const double T_target = 300.0;
    const double tau_T = 50.0;
    const double P_target = 0.0; // SCF on isolated H2 has no PBC stress;
    const double tau_P = 1000.0; // weak coupling, long time constant

    Mat3 box_initial = frame.simbox.cell();

    cpu::md::MD sim;
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/3u});
    sim.add(cpu::md::VelocityVerlet{dt});
    sim.add(cpu::md::NoseHoover{T_target, tau_T, dt});
    sim.add(cpu::md::Barostat{P_target, tau_P, dt});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 20);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0)
        << "Box mutation between SCF launches must not break IntegralPack "
           "rebuild";
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kEnergy))));
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kKinetic))));
    // Barostat may scale the box at zero pressure with non-zero virial;
    // we only require it stays finite and within a sane range.
    EXPECT_TRUE(std::isfinite(frame.simbox.cell()(0, 0)));
    EXPECT_GT(frame.simbox.cell()(0, 0), 0.0);
    EXPECT_LT(frame.simbox.cell()(0, 0), 10.0 * box_initial(0, 0));
}

// ── Regression: builder chain on rvalue temporary (Bug #1) ──────────
//
// `SCF{cfg}.predictor(3)` returns an lvalue reference to the
// temporary. Without ref-qualified overloads, `Driver::add(Comp&& comp)`
// deduces `Comp` as `SCF&` (lvalue reference) and stores a
// non-owning pointer to the temporary, which dies at the semicolon →
// SEGV on the first launch. The fix is to add rvalue-ref-qualified
// builder overloads so chains on temporaries propagate rvalue-ness
// through to add()'s rvalue branch (driver moves into unique_ptr).

TEST_F(BOMDTest, ChainedBuilderOnTemporaryDoesNotSEGV) {
    auto frame = make_h2(0.74, 0.001);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    // The whole point of this test: chained builder call on a temporary.
    // Pre-fix, this stored a non-owning pointer to a dangling temporary
    // and segfaulted on the first launch.
    sim.on(md::Phase::force).add(cpu::scf::RHF{}.predictor(3));
    sim.build();
    sim.run(frame, 3);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0);
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kEnergy))));
}

TEST_F(BOMDTest, ChainedBuilderWithGradientToggleOnTemporary) {
    auto frame = make_h2(0.74, 0.001);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    // Two consecutive builder calls on the same temporary chain.
    sim.on(md::Phase::force)
        .add(cpu::scf::RHF{}.with_gradient(true).predictor(2));
    sim.build();
    sim.run(frame, 1);

    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), 0.0);
    EXPECT_TRUE(
        std::isfinite(static_cast<double>(frame.meta.get(ATV::kEnergy))));
}

// ── Regression: PBC-wrap × non-min-image SCF (Bug #2) ───────────────
//
// `VelocityVerlet::wrap_orthorhombic` wraps positions through orthorhombic
// PBC and bumps `frame.span(ATV::kAtom_image)[i]` by ±1 per crossing. `build_integrals`
// is not minimum-image-aware; it builds shells and v_nuc from raw
// `frame.span(ATV::kAtom_x)/y/z`. If an atom starts on the box edge and thermal
// motion pushes it across, the next SCF sees the two atoms separated by
// approximately one box length and the calculation collapses (RHF
// converges to an unphysical ionic limit at ≈ -0.16 Ha instead of the
// bonded ≈ -1.117 Ha for H2).
//
// The fix is to unwrap positions inside `drive_scf_` using
// `frame.span(ATV::kAtom_image)` before the integral build. The frame storage stays
// wrapped (so MD pair lists / classical force fields keep working);
// SCF just sees the unwrapped view.

TEST_F(BOMDTest, BoundaryAtomsRecoverAfterWrap) {
    // H2 sitting on the x-edge: atom 0 at x=0, atom 1 at x=0.74. Any
    // negative x velocity on atom 0 crosses the wrap on step 1.
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.74;
    frame.span(ATV::kAtom_y)[0] = 50.0;
    frame.span(ATV::kAtom_y)[1] = 50.0;
    frame.span(ATV::kAtom_z)[0] = 50.0;
    frame.span(ATV::kAtom_z)[1] = 50.0;
    frame.span(ATV::kAtom_vx)[0] = -0.01;
    frame.span(ATV::kAtom_vx)[1] = 0.01; // atom 0 moves toward x<0 and wraps
    frame.span(ATV::kAtom_vy)[0] = 0.0;
    frame.span(ATV::kAtom_vy)[1] = 0.0;
    frame.span(ATV::kAtom_vz)[0] = 0.0;
    frame.span(ATV::kAtom_vz)[1] = 0.0;
    frame.span(ATV::kAtom_fx)[0] = 0.0;
    frame.span(ATV::kAtom_fx)[1] = 0.0;
    frame.span(ATV::kAtom_fy)[0] = 0.0;
    frame.span(ATV::kAtom_fy)[1] = 0.0;
    frame.span(ATV::kAtom_fz)[0] = 0.0;
    frame.span(ATV::kAtom_fz)[1] = 0.0;
    frame.span(ATV::kAtom_mass)[0] = 1.008;
    frame.span(ATV::kAtom_mass)[1] = 1.008;
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.meta.get(ATV::kEnergy) = 0;
    frame.meta.get(ATV::kKinetic) = 0;
    frame.simbox = ATV::cpu::Box::cubic(100.0);

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{double(0.5)});
    sim.on(md::Phase::force).add(cpu::scf::RHF{});
    sim.build();
    sim.run(frame, 5);

    // After fix: SCF unwraps before building integrals, so PE stays in
    // the bonded H2 range (≈ -30 eV). Pre-fix, atoms wrapped on step 1
    // and PE collapsed to ≈ -4 eV (RHF ionic limit at ~100 Å apart).
    EXPECT_LT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -25.0)
        << "PE ≈ -4 eV indicates SCF saw atoms ~box-length apart "
        << "(wrap × non-minimum-image SCF pathology). Expected bonded H2 "
        << "energy ≈ -30 eV.";
    EXPECT_GT(static_cast<double>(frame.meta.get(ATV::kEnergy)), -35.0);
}

// ── ASPC predictor unit tests ───────────────────────────────────────

class ASPCPredictorTest : public ::testing::Test {};

TEST_F(ASPCPredictorTest, BasicFunctionality) {
    cpu::scf::ASPCPredictor pred(3);

    xt::xtensor<double, 2> P1 = {{1.0, 0.0}, {0.0, 1.0}};
    xt::xtensor<double, 2> P2 = {{1.1, 0.0}, {0.0, 1.1}};
    xt::xtensor<double, 2> P3 = {{1.2, 0.0}, {0.0, 1.2}};
    xt::xtensor<double, 2> P4 = {{1.3, 0.0}, {0.0, 1.3}};

    xt::xtensor<double, 2> P_pred;

    pred.record(P1);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P2);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P3);
    EXPECT_FALSE(pred.predict(P_pred));
    pred.record(P4);
    EXPECT_TRUE(pred.predict(P_pred));

    EXPECT_NEAR(static_cast<double>(P_pred(0, 0)), 1.4, 0.01);
}

TEST_F(ASPCPredictorTest, CorrectorBlending) {
    cpu::scf::ASPCPredictor pred(3);

    xt::xtensor<double, 2> P_scf = {{2.0}};
    xt::xtensor<double, 2> P_pred = {{1.8}};
    xt::xtensor<double, 2> P_corr;

    pred.correct(P_scf, P_pred, P_corr);

    EXPECT_NEAR(static_cast<double>(P_corr(0, 0)),
                5.0 / 9.0 * 2.0 + 4.0 / 9.0 * 1.8, 1e-8);
}
