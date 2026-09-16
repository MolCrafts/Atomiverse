/// @file tests/cpu/multiscale/test_classical_nonbonded.cpp
/// @brief Unit: ONIOM low-level all-pairs LJ (subset vs full).
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/internal/classical_nonbonded.h"

using namespace ATV;

TEST(ClassicalNonbonded, TwoAtomShiftedLj) {
    const double eps = 0.01;
    const double sigma = 3.4;
    const double cutoff = 10.0;
    const double r = 5.0;
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(40);

    cpu::multiscale::internal::ClassicalNonbonded nb{{{1, 1, eps, sigma}},
                                                     cutoff};
    f.meta.get(kEnergy) = 0;
    const double e = nb.launch(f, {}, /*forces=*/false);
    const double s6 = std::pow(sigma / r, 6.0);
    const double s6c = std::pow(sigma / cutoff, 6.0);
    const double e_expect =
        4.0 * eps * (s6 * s6 - s6) - 4.0 * eps * (s6c * s6c - s6c);
    EXPECT_NEAR(e, e_expect, 1e-12);
}

TEST(ClassicalNonbonded, ModelSubsetExcludesOuter) {
    // 3 atoms in a line: only rows {0,1} should see pair 0–1, not 0–2/1–2.
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 5;
    f.span(kAtom_x)[2] = 10;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(50);
    cpu::multiscale::internal::ClassicalNonbonded nb{{{1, 1, 0.01, 3.4}}, 12.0};
    const double e_full = nb.launch(f, {}, false);
    const double e_model = nb.launch(f, {0, 1}, false);
    // More pairs at longer range deepen the (shifted) attractive well.
    EXPECT_LT(e_full, e_model);
    EXPECT_TRUE(std::isfinite(e_model));
}
