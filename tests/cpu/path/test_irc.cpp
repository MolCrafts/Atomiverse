/// @file tests/cpu/path/test_irc.cpp
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/irc.h"

using namespace ATV;

TEST(Irc, EnergyNonIncreasingOnMB) {
    cpu::Frame saddle(1);
    // Near published MB saddle (−0.822, 0.624)
    saddle.span(kAtom_x)[0] = -0.822;
    saddle.span(kAtom_y)[0] = 0.624;
    saddle.span(kAtom_z)[0] = 0;
    saddle.span(kAtom_mass)[0] = 1.0;
    saddle.span(kAtom_type)[0] = 1;
    saddle.span(kAtom_tag)[0] = 1;
    saddle.box = cpu::Box::cubic(100.0);

    cpu::path::Irc irc(0.02, 200, 1e-2, 0.1);
    irc.use_muller_brown();
    irc.run(saddle, 0.0, 1.0);

    auto check_mono = [](const std::vector<double>& E) {
        for (size_t i = 1; i < E.size(); ++i) {
            // Allow small numerical rises of 1.0 energy unit on MB surface.
            EXPECT_LE(E[i], E[i - 1] + 1.0) << "i=" << i;
        }
    };
    check_mono(irc.forward_energies());
    check_mono(irc.backward_energies());
    // Both ends should be deep in a well (MB minima ~ −146 and −108)
    EXPECT_LT(irc.forward_final_energy(), -50.0);
    EXPECT_LT(irc.backward_final_energy(), -50.0);
}
