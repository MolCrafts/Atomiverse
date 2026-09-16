/// @file tests/cpu/semiempirical/test_xtb_ipea_hessian.cpp
/// @brief Unit: `cpu::vib::Hessian` accepts IPEA as a Semiempirical force kind.
///
/// Wiring only — `Hessian::run` (6N displacements) lives in
/// `regressions/semiempirical/ipea-xtb-analytic-hessian/`.
#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/semiempirical/ipea.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

TEST(XtbIpeaHessian, BuildAcceptsIpeaForceKind) {
    cpu::semiempirical::IPEA ipea;
    ipea.with_forces = true;
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(ipea));
    EXPECT_NO_THROW(hess.build());
}
