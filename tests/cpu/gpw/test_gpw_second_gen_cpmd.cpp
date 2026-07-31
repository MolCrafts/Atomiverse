/// @file tests/cpu/gpw/test_gpw_second_gen_cpmd.cpp
/// @brief Unit tests: GPWEvaluator second-generation CP (ASPC + partial SCF)
///        build() guards only. No multi-step MD here (e2e is the
///        regressions/dynamics/second-gen-cpmd-* family).

#include <stdexcept>

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/electronic/second_gen_cp.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/electronic/car_parrinello.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

namespace {
using ATV::cpu::electronic::SecondGenCp;
using ATV::cpu::electronic::CarParrinello;
using ATV::cpu::electronic::XlBomdForce;

ATV::cpu::Frame water_frame() {
    ATV::cpu::Frame f(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        f.span(ATV::kAtom_type)[i] = z[i];
        f.span(ATV::kAtom_x)[i] = x[i];
        f.span(ATV::kAtom_y)[i] = y[i];
        f.span(ATV::kAtom_z)[i] = zc[i];
    }
    f.box = ATV::cpu::Box::cubic(10.0);
    return f;
}

ATV::cpu::gpw::GPWEvaluator water_eval() {
    const ATV::cpu::gpw::Kind O{
        molrs::Element::O,
        ATV::cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        ATV::cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const ATV::cpu::gpw::Kind H{
        molrs::Element::H,
        ATV::cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        ATV::cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};
    return ATV::cpu::gpw::GPWEvaluator(
        {O, H}, /*cutoff_ry*/ 200.0, /*max_iter*/ 60, /*tol*/ 1e-5,
        /*mix*/ 0.3, ATV::cpu::gpw::XcKind::PBE, /*nonperiodic*/ true,
        /*compute_forces*/ true);
}

} // namespace

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsWithUks) {
    // Γ-UKS dual ASPC (product path, wave-7 second-gen-cpmd-uks).
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 5;
    base.n_beta = 3;
    SecondGenCp eval{std::move(base), 3, 5};
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsUksWithOt) {
    // Γ-UKS OT + dual ASPC corrector (product path, second-gen-cpmd-uks-ot).
    auto frame = water_frame();
    auto base = water_eval();
    base.n_alpha = 5;
    base.n_beta = 3;
    base.use_ot = true;
    SecondGenCp eval{std::move(base), 3, 5};
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsWithOt) {
    // RKS OT partial-SCF corrector is product (second-gen-cpmd-ot).
    auto frame = water_frame();
    auto base = water_eval();
    base.use_ot = true;
    SecondGenCp eval{std::move(base), 3, 5};
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, DistinctFromCarParrinelloAndXl) {
    SecondGenCp sg{water_eval(), 3, 5};
    CarParrinello cp{water_eval(), 400.0};
    XlBomdForce xl{water_eval(), 5};
    EXPECT_NE(sg.name(), cp.name());
    EXPECT_NE(sg.name(), xl.name());
}

TEST(GpwEvaluatorSecondGenCp, CtorEnablesForces) {
    auto base = water_eval();
    base.compute_forces = false;
    SecondGenCp eval{std::move(base), 3, 5};
    EXPECT_TRUE(eval.method().compute_forces);
}

TEST(GpwEvaluatorSecondGenCp, CtorRejectsBadOrderOrBudget) {
    EXPECT_THROW((SecondGenCp{water_eval(), -1, 5}), std::invalid_argument);
    EXPECT_THROW((SecondGenCp{water_eval(), 3, 0}), std::invalid_argument);
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsDefaultMembers) {
    auto frame = water_frame();
    auto base = water_eval();
    SecondGenCp eval{std::move(base), 3, 5};
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsMultiK) {
    // Multi-k RKS ASPC on complex P_k (product path, second-gen-cpmd-kpoints).
    auto frame = water_frame();
    // Periodic cell required for a k-mesh.
    frame.box = ATV::cpu::Box::cubic(10.0);
    auto base = water_eval();
    base.nonperiodic = false;
    SecondGenCp eval{std::move(base), 3, 5};
    eval.method().k_mesh[0] = 1;
    eval.method().k_mesh[1] = 1;
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsMultiKWithOt) {
    // Multi-k RKS OT partial-SCF corrector is product — the per-k OT
    // manifold warm-starts from natural orbitals of the ASPC-predicted
    // P_k (second-gen-cpmd-ot-kpoints).
    auto frame = water_frame();
    frame.box = ATV::cpu::Box::cubic(10.0);
    auto base = water_eval();
    base.nonperiodic = false;
    base.use_ot = true;
    SecondGenCp eval{std::move(base), 3, 5};
    eval.method().k_mesh[0] = 1;
    eval.method().k_mesh[1] = 1;
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsMultiKUksWithOt) {
    // Multi-k UKS OT partial-SCF corrector is product — dual-spin per-k OT
    // manifolds warm-start from natural orbitals of the ASPC-predicted
    // P_k^sigma (second-gen-cpmd-ot-uks-kpoints).
    auto frame = water_frame();
    frame.box = ATV::cpu::Box::cubic(10.0);
    auto base = water_eval();
    base.nonperiodic = false;
    base.n_alpha = 5;
    base.n_beta = 3;
    base.use_ot = true;
    SecondGenCp eval{std::move(base), 3, 5};
    eval.method().k_mesh[0] = 1;
    eval.method().k_mesh[1] = 1;
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorSecondGenCp, BuildAcceptsMultiKUks) {
    auto frame = water_frame();
    frame.box = ATV::cpu::Box::cubic(10.0);
    auto base = water_eval();
    base.nonperiodic = false;
    base.n_alpha = 5;
    base.n_beta = 3;
    SecondGenCp eval{std::move(base), 3, 5};
    eval.method().k_mesh[0] = 1;
    eval.method().k_mesh[1] = 1;
    eval.method().k_mesh[2] = 2;
    eval.method().k_time_reversal = false;
    ATV::cpu::Context rt;
    ATV::cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}
