/// @file tests/cpu/semiempirical/test_xtb_gfn_hessian.cpp
/// @brief Unit: `cpu::vib::Hessian` accepts GFN1 and GFN2 as force kinds.
///
/// Wiring only — `Hessian::run` (6N displacements) lives in
/// `regressions/semiempirical/gfn-xtb-analytic-hessian/`.
#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

TEST(XtbGfnHessian, BuildAcceptsGfn1ForceKind) {
    cpu::semiempirical::GFN1 gfn1;
    gfn1.with_forces = true;
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(gfn1));
    EXPECT_NO_THROW(hess.build());
}

TEST(XtbGfnHessian, BuildAcceptsGfn2ForceKind) {
    cpu::semiempirical::GFN2 gfn2;
    gfn2.with_forces = true;
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(gfn2));
    EXPECT_NO_THROW(hess.build());
}
