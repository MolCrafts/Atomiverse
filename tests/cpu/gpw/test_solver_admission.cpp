#include "atomiverse/cpu/gpw/gpw_solver.h"
#include <gtest/gtest.h>
#include <limits>

namespace {

ATV::cpu::Frame stress_guard_frame() {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(8);
    frame.meta.get(kEnergy) = 17.5;
    frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = 3.5 + i;
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 4;
        frame.span(kAtom_fx)[i] = 3 + i;
        frame.span(kAtom_fy)[i] = 5 + i;
        frame.span(kAtom_fz)[i] = 7 + i;
    }
    return frame;
}

void expect_stress_frame_preserved(const ATV::cpu::Frame& frame) {
    using namespace ATV;
    EXPECT_DOUBLE_EQ(frame.meta.get(kEnergy), 17.5);
    const MetaF64x6 virial = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(frame.meta.get(kStress), virial);
    for (int i = 0; i < 2; ++i) {
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[i], 3 + i);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fy)[i], 5 + i);
        EXPECT_DOUBLE_EQ(frame.span(kAtom_fz)[i], 7 + i);
    }
}

} // namespace

TEST(GpwAdmission, RejectsInvalidStressStepBeforeEvaluatingFrame) {
    auto frame = stress_guard_frame();
    ATV::cpu::gpw::GpwDft method{{}, 80, ATV::cpu::xc::XCType::PADE};
    // Keep the method unbuilt: a probe error must precede run/build checks,
    // so a regression fails quickly instead of entering any SCF.
    for (double h :
         {0.0, -1.0, 0.1, 0.2, std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        method.stress_strain_h = h;
        EXPECT_THROW(method.stress(frame), std::invalid_argument) << "h=" << h;
        expect_stress_frame_preserved(frame);
    }
}

TEST(GpwAdmission, RejectsMaskedStressProbeBeforeEvaluatingFrame) {
    auto frame = stress_guard_frame();
    ATV::cpu::gpw::GpwDft method{{}, 80, ATV::cpu::xc::XCType::PADE};
    method.stress_axes = {false, false, false};
    EXPECT_THROW(method.stress(frame), std::invalid_argument);
    expect_stress_frame_preserved(frame);
}

TEST(GpwAdmission, RejectsInvalidResidualToleranceBeforeEvaluatingFrame) {
    using namespace ATV;
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(8);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = 3.5 + i;
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 4;
    }
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GpwDft method{{H}, 100, cpu::xc::XCType::PADE};
    method.build(frame);
    frame.meta.get(kEnergy) = 42;
    frame.span(kAtom_fx)[0] = 17;
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        method.residual_tol = invalid;
        EXPECT_THROW(method.run(frame), std::invalid_argument);
        EXPECT_EQ(frame.meta.get(kEnergy), 42);
        EXPECT_EQ(frame.span(kAtom_fx)[0], 17);
    }
}
