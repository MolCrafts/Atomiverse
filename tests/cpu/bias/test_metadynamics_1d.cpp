#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/metadynamics_1d.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include <gtest/gtest.h>
#include <memory>
using namespace ATV;
TEST(Metadynamics1D, HillsGrow) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 2;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1;
    f.box = cpu::Box::cubic(20);
    cpu::Context ctx;
    auto cv = std::make_unique<cpu::bias::DistanceCV>(1, 2);
    cpu::bias::Metadynamics1D meta(std::move(cv), 0.1, 0.2, 300.0, 10.0, 1);
    meta.build({ctx, f});
    for (int i = 0; i < 5; ++i) {
        f.meta.get(kEnergy) = 0;
        meta.launch({ctx, f, 0, i});
    }
    EXPECT_GE(meta.n_hills(), 3);
    EXPECT_GT(meta.last_bias_energy(), 0.0);
}
