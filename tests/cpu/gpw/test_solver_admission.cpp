#include "atomiverse/cpu/gpw/gpw_solver.h"
#include <gtest/gtest.h>
#include <limits>

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
    cpu::gpw::GpwDft method{{H}, 100, cpu::gpw::XcKind::PADE};
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
