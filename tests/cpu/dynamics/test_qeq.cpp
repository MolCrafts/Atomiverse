/// @file tests/cpu/dynamics/test_qeq.cpp
/// @brief Unit tests for QEq fluctuating-charge equilibration (single methods).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/qeq.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// Two-atom diatomic along z at separation r, types 1 and 2, in a large box.
cpu::Frame diatomic(double r) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = 20.0;
        f.span(kAtom_y)[i] = 20.0;
        f.span(kAtom_z)[i] = 20.0 + r * i;
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = i + 1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(40.0);
    return f;
}

void run_once(cpu::md::QEq& qeq, cpu::Frame& f) {
    cpu::Context ctx;
    qeq.build({ctx, f});
    qeq.launch({ctx, f, /*stage=*/0, /*step=*/0});
}

} // namespace

TEST(QEq, TwoAtomClosedForm) {
    // Minimize χ₁q + χ₂(−q) + ½(η₁+η₂)q² − (1/r)q² with q₁=q, q₂=−q:
    //   q = (χ₂−χ₁)/(η₁+η₂−2/r)  — bare-1/r LAMMPS qeq/point kernel.
    const double chi1 = 4.0, eta1 = 12.0, chi2 = 7.0, eta2 = 14.0, r = 1.5;
    auto f = diatomic(r);
    cpu::md::QEq qeq(10.0);
    qeq.set_type(1, chi1, eta1).set_type(2, chi2, eta2);
    run_once(qeq, f);
    const double q_expect = (chi2 - chi1) / (eta1 + eta2 - 2.0 / r);
    EXPECT_NEAR(qeq.charges()[0], q_expect, 1e-12);
    EXPECT_NEAR(qeq.charges()[1], -q_expect, 1e-12);
}

TEST(QEq, ChargesWrittenToFrame) {
    auto f = diatomic(1.5);
    cpu::md::QEq qeq(10.0);
    qeq.set_type(1, 4.0, 12.0).set_type(2, 7.0, 14.0);
    run_once(qeq, f);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[0]),
                     qeq.charges()[0]);
    EXPECT_DOUBLE_EQ(static_cast<double>(f.span(kAtom_charge)[1]),
                     qeq.charges()[1]);
}

TEST(QEq, NetChargeConservedNonzeroQtot) {
    auto f = diatomic(1.5);
    cpu::md::QEq qeq(10.0);
    qeq.set_type(1, 4.0, 12.0).set_type(2, 7.0, 14.0);
    qeq.qtot = -1.0;
    run_once(qeq, f);
    EXPECT_NEAR(qeq.net_charge(), -1.0, 1e-12);
}

TEST(QEq, BeyondCutoffDecouples) {
    // r > cutoff ⇒ J = 0 ⇒ q_i = (λ − χ_i)/η_i with Σq = 0:
    //   q₁ = (χ₂−χ₁)/(η₁+η₂) on atom 1.
    const double chi1 = 4.0, eta1 = 12.0, chi2 = 7.0, eta2 = 14.0;
    auto f = diatomic(8.0);
    cpu::md::QEq qeq(5.0);
    qeq.set_type(1, chi1, eta1).set_type(2, chi2, eta2);
    run_once(qeq, f);
    const double q_expect = (chi2 - chi1) / (eta1 + eta2);
    EXPECT_NEAR(qeq.charges()[0], q_expect, 1e-12);
}

TEST(QEq, EnergyClosedForm) {
    const double chi1 = 4.0, eta1 = 12.0, chi2 = 7.0, eta2 = 14.0, r = 1.5;
    auto f = diatomic(r);
    cpu::md::QEq qeq(10.0);
    qeq.set_type(1, chi1, eta1).set_type(2, chi2, eta2);
    run_once(qeq, f);
    const double q = (chi2 - chi1) / (eta1 + eta2 - 2.0 / r);
    const double e_expect = chi1 * q - chi2 * q
                            + 0.5 * (eta1 + eta2) * q * q - q * q / r;
    EXPECT_NEAR(qeq.qeq_energy(), e_expect, 1e-12);
}

TEST(QEq, ThrowsOnMissingTypeParams) {
    auto f = diatomic(1.5);
    cpu::md::QEq qeq(10.0);
    qeq.set_type(1, 4.0, 12.0); // type 2 present in frame, not registered
    cpu::Context ctx;
    EXPECT_THROW(qeq.build({ctx, f}), std::runtime_error);
}

TEST(QEq, ThrowsOnBadConstruction) {
    EXPECT_THROW(cpu::md::QEq(-1.0), std::invalid_argument);
    cpu::md::QEq qeq(10.0);
    EXPECT_THROW(qeq.set_type(1, 4.0, 0.0), std::invalid_argument);
    cpu::Context ctx;
    auto f = diatomic(1.5);
    cpu::md::QEq empty(10.0); // no set_type at all
    EXPECT_THROW(empty.build({ctx, f}), std::runtime_error);
}

TEST(QEq, DefaultBindingPreForce) {
    cpu::md::QEq qeq(10.0);
    EXPECT_EQ(qeq.name(), "qeq");
    auto b = qeq.default_bindings();
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(b[0].phase, ATV::slot_of(ATV::md::Phase::pre_force));
    EXPECT_EQ(b[0].order, -50);
}
