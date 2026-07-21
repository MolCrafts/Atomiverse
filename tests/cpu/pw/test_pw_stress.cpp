/// @file tests/cpu/pw/test_pw_stress.cpp
/// @brief Unit gates for plane-wave cell stress (strain-FD v1).

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_evaluator.h"
#include "atomiverse/cpu/pw/pw_solver.h"
#include "atomiverse/facade/stage.h"

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
    f.simbox = cpu::Box::cubic(L);
    return f;
}

PWConfig cheap_cfg() {
    PWConfig cfg;
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    return cfg;
}

} // namespace

TEST(PWStress, CubicMonoatomicOffDiagonalSoft) {
    // Single H in cubic cell: off-diagonal stress ~0 by symmetry.
    auto f = make_h_cubic(6.0);
    const auto s = pw_stress(f, cheap_cfg());
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(s[static_cast<std::size_t>(k)])) << k;
    // xy, xz, yz
    EXPECT_LT(std::abs(s[3]), 0.5);
    EXPECT_LT(std::abs(s[4]), 0.5);
    EXPECT_LT(std::abs(s[5]), 0.5);
    // diagonal components should be similar
    EXPECT_NEAR(s[0], s[1], 0.5);
    EXPECT_NEAR(s[1], s[2], 0.5);
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
