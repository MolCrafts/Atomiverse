#pragma once
#include "atomiverse/cpu/frame/frame.h"
namespace pw_frozen {
inline ATV::cpu::Frame silicon() {
    using namespace ATV;
    cpu::Frame f(3);
    Mat3 h;
    h(0, 0) = 4.2999999999999998;
    h(1, 0) = 0.12;
    h(2, 0) = -0.080000000000000002;
    h(0, 1) = 0.34999999999999998;
    h(1, 1) = 4.0999999999999996;
    h(2, 1) = 0.16;
    h(0, 2) = -0.22;
    h(1, 2) = 0.28000000000000003;
    h(2, 2) = 3.8999999999999999;
    f.box.set_cell(h);
    constexpr double xyz[3][3] = {
        {0.52490000000000003, 0.77579999999999993, 0.91460000000000008},
        {2.6663999999999994, 2.5983999999999998, 2.1118000000000001},
        {0.80779999999999985, 3.3771999999999998, 3.4620000000000002}};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = 28.0855;
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
    }
    return f;
}
} // namespace pw_frozen
