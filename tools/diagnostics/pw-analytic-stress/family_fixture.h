#pragma once
#include "atomiverse/cpu/planewave/pw_solver.h"
namespace family_fixture {
using namespace ATV;
cpu::Frame hydrogen(bool spin) {
    cpu::Frame frame(spin ? 3 : 2);
    Mat3 cell;
    cell(0, 0) = 4.2;
    cell(0, 1) = .3;
    cell(0, 2) = .2;
    cell(1, 0) = 0;
    cell(1, 1) = 4;
    cell(1, 2) = .1;
    cell(2, 0) = cell(2, 1) = 0;
    cell(2, 2) = 3.8;
    frame.box.set_cell(cell);
    constexpr double positions[3][3] = {
        {1.2, 1.1, 1.0}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = positions[i][0];
        frame.span(kAtom_y)[i] = positions[i][1];
        frame.span(kAtom_z)[i] = positions[i][2];
    }
    return frame;
}
template <class Method>
void configure(Method& method, bool spin, electronic::Smearing smear) {
    method.spin_polarized = spin;
    method.multiplicity = spin ? 2 : 1;
    method.extra_bands = 3;
    method.tol = 1e-9;
    method.max_iter = 400;
    method.mix_beta = .3;
    method.smearing = smear;
    method.degauss = smear == electronic::Smearing::fermi_dirac ? .02 : .04;
}
} // namespace family_fixture
