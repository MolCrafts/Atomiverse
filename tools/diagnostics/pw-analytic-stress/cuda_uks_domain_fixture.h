#pragma once
#include "atomiverse/cpu/frame/frame.h"
namespace pw_domain_fixture {
using namespace ATV;
inline cpu::Frame geometry(double box) {
    cpu::Frame f(2);
    Mat3 h;
    for (int a = 0; a < 3; ++a)
        h(a, a) = box;
    f.box.set_cell(h);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = 1.00794;
        f.span(kAtom_x)[i] = box / 2;
        f.span(kAtom_y)[i] = box / 2;
        f.span(kAtom_z)[i] = box / 2 + (i ? .4 : -.4);
        f.span(kAtom_fx)[i] = 10 + i;
        f.span(kAtom_fy)[i] = 20 + i;
        f.span(kAtom_fz)[i] = 30 + i;
    }
    f.meta.get(kEnergy) = 1234;
    f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    return f;
}
} // namespace pw_domain_fixture
