#pragma once
#include "atomiverse/cpu/planewave/pw_solver.h"
namespace pw_frozen {
inline ATV::cpu::Frame silicon() {
    using namespace ATV;
    cpu::Frame frame(2);
    Mat3 h;
    h(0, 0) = .10;
    h(1, 0) = 2.70;
    h(2, 0) = 2.65;
    h(0, 1) = 2.75;
    h(1, 1) = .05;
    h(2, 1) = 2.70;
    h(0, 2) = 2.65;
    h(1, 2) = 2.75;
    h(2, 2) = .15;
    frame.box.set_cell(h);
    constexpr double p[2][3] = {{.2, .3, .4}, {1.57, 1.66, 1.75}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 28.0855;
        frame.span(kAtom_x)[i] = p[i][0];
        frame.span(kAtom_y)[i] = p[i][1];
        frame.span(kAtom_z)[i] = p[i][2];
    }
    return frame;
}
inline ATV::cpu::pw::PW method() {
    using namespace ATV;
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::Si,
                                                            "GTH-PBE-q4");
    cpu::pw::PW m{pp, 12, 1, cpu::xc::XCType::LSDA};
    m.tol = 1e-10;
    m.max_iter = 200;
    m.mix_beta = .2;
    return m;
}
} // namespace pw_frozen
