/// @file tests/cpu/semiempirical/test_xtb_gfn0_hessian.cpp
/// @brief Unit: `cpu::vib::Hessian` accepts GFN0 as a Semiempirical force kind.
///
/// Wiring only — `Hessian::run` (6N displacements) lives in
/// `regressions/semiempirical/gfn0-xtb-analytic-hessian/`.
#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

TEST(XtbGfn0Hessian, BuildAcceptsGfn0ForceKind) {
    cpu::semiempirical::GFN0 gfn0;
    gfn0.with_forces = true;
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(gfn0));
    EXPECT_NO_THROW(hess.build());
}
