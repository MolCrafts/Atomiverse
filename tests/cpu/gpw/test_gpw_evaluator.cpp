/// @file tests/cpu/gpw/test_gpw_evaluator.cpp
/// @brief Unit tests for cpu::gpw::GPWEvaluator's cell precondition.
///
/// GPW carries the cell as three edge lengths and builds its lattice images
/// from them, so a sheared frame would be solved on `diag(h)` — a different
/// Bravais lattice. Every case here fires before any SCF work, so the file
/// stays inside the seconds-scale unit budget.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"

using namespace ATV;

namespace {

/// H2 in a cubic 12 Bohr cell — the cheapest frame GPW accepts.
cpu::Frame make_h2() {
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
    }
    f.box = cpu::Box::cubic(12.0 * bohr2ang);
    return f;
}

cpu::gpw::GPWEvaluator h2_eval() {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    return cpu::gpw::GPWEvaluator(
        /*kinds*/ {H}, /*cutoff_ry*/ 60.0, /*max_iter*/ 1, /*tol*/ 1e-6,
        /*mix*/ 0.3, cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ false);
}

} // namespace

TEST(GpwEvaluatorCellGuard, BuildAcceptsOrthorhombic) {
    auto frame = make_h2();
    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    EXPECT_NO_THROW(eval.build(bc));
}

TEST(GpwEvaluatorCellGuard, BuildRefusesShearedCell) {
    auto frame = make_h2();
    Mat3 h = frame.box.cell();
    h(0, 1) = 1.0; // shear the a-vector into y
    frame.box.set_cell(h);

    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    try {
        eval.build(bc);
        FAIL() << "expected std::invalid_argument for a sheared cell";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("orthorhombic"), std::string::npos) << msg;
        EXPECT_NE(msg.find("h(0,1)"), std::string::npos) << msg;
    }
}

/// The variable-cell path is the reason a build-time-only guard is unsound:
/// TriclinicCellRelax and ParrinelloRahmanNPT write a general cell matrix back
/// to the Frame between launches.
TEST(GpwEvaluatorCellGuard, LaunchRefusesCellShearedAfterBuild) {
    auto frame = make_h2();
    auto eval = h2_eval();
    cpu::Context rt;
    cpu::BuildContext bc{rt, frame};
    ASSERT_NO_THROW(eval.build(bc));

    Mat3 h = frame.box.cell();
    h(1, 2) = 0.5;
    frame.box.set_cell(h);

    EXPECT_THROW(eval.launch({rt, frame, 0, 0}), std::invalid_argument);
}
