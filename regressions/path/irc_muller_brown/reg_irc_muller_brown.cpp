/// @file regressions/path/irc_muller_brown/reg_irc_muller_brown.cpp
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/irc.h"

using namespace ATV;

int main() {
    cpu::Frame saddle(1);
    saddle.span(kAtom_x)[0] = -0.822;
    saddle.span(kAtom_y)[0] = 0.624;
    saddle.span(kAtom_z)[0] = 0;
    saddle.span(kAtom_mass)[0] = 1.0;
    saddle.span(kAtom_type)[0] = 1;
    saddle.span(kAtom_tag)[0] = 1;
    saddle.box = cpu::Box::cubic(100.0);

    cpu::path::Irc irc(0.02, 300, 0.05, 0.15);
    irc.use_muller_brown();
    irc.run(saddle, 0.0, 1.0);

    const double Ef = irc.forward_final_energy();
    const double Eb = irc.backward_final_energy();
    std::printf("IRC MB: Ef=%.3f Eb=%.3f\n", Ef, Eb);
    // Both wells deeper than −50 (literature minima ~ −146 and −108)
    if (!(Ef < -50.0 && Eb < -50.0)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
