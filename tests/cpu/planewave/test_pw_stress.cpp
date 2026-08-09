/// @file tests/cpu/pw/test_pw_stress.cpp
/// @brief Unit gates for plane-wave cell stress (Nielsen–Martin + FD oracle).

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_h_cubic(double L) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(L);
    return f;
}

PW cheap_cfg() {
    PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    cfg.xc = cpu::scf::XCType::LSDA; // LDA XC stress term is complete
    return cfg;
}

} // namespace

TEST(PWStress, CubicMonoatomicOffDiagonalIsotropy) {
    // Cubic symmetry is exact: shear components vanish and the three normal
    // components are equal. These are structural identities of the analytic
    // stress, so the bound is the arithmetic noise floor — the old 0.5
    // eV/Å³ window was wider than the stress itself (~1e-2) and asserted
    // nothing.
    auto f = make_h_cubic(6.0);
    const auto s = pw_stress(f, cheap_cfg());
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(s[static_cast<std::size_t>(k)])) << k;
    EXPECT_LT(std::abs(s[3]), 1e-9);
    EXPECT_LT(std::abs(s[4]), 1e-9);
    EXPECT_LT(std::abs(s[5]), 1e-9);
    EXPECT_NEAR(s[0], s[1], 1e-9);
    EXPECT_NEAR(s[1], s[2], 1e-9);
}

TEST(PWStress, FdRejectsBadStep) {
    auto f = make_h_cubic(6.0);
    EXPECT_THROW(pw_stress_fd(f, cheap_cfg(), 0.0), std::invalid_argument);
    EXPECT_THROW(pw_stress_fd(f, cheap_cfg(), -1e-3), std::invalid_argument);
}

TEST(PWStress, EvaluatorWithStressWritesKStress) {
    auto frame = make_h_cubic(6.0);
    PWEvaluator pwe{cheap_cfg()};
    pwe.with_stress(true);
    EXPECT_TRUE(pwe.compute_stress());
    EXPECT_FALSE(pwe.compute_forces());

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(pwe);
    eval.enable_timing(false);
    eval.build();
    eval.run(frame);

    EXPECT_TRUE(std::isfinite(frame.meta.get(kEnergy)));
    const auto& st = frame.meta.get(kStress);
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(st[static_cast<std::size_t>(k)]));
}
