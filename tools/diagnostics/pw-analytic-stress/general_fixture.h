#pragma once
#include "../pbe-periodic-oracles/pins.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
namespace fixture {
using namespace ATV;
cpu::Frame frame(const pbe_periodic_qe::Pin& pin) {
    cpu::Frame f(pin.nat);
    constexpr double h[3][3] = {
        {1.2, 1.1, 1}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    constexpr double si[2][3] = {{2, 2, 2}, {3.8, 3.2, 3}};
    if (pin.spin) {
        Mat3 cell{};
        cell(0, 0) = 4.2;
        cell(0, 1) = .3;
        cell(0, 2) = .2;
        cell(1, 1) = 4;
        cell(1, 2) = .1;
        cell(2, 2) = 3.8;
        f.box.set_cell(cell);
    } else
        f.box = cpu::Box::cubic(6);
    for (int i = 0; i < pin.nat; ++i) {
        f.span(kAtom_type)[i] = pin.spin ? 1 : 14;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = pin.spin ? 1.008 : 28.0855;
        f.span(kAtom_x)[i] = pin.spin ? h[i][0] : si[i][0];
        f.span(kAtom_y)[i] = pin.spin ? h[i][1] : si[i][1];
        f.span(kAtom_z)[i] = pin.spin ? h[i][2] : si[i][2];
    }
    return f;
}
cpu::xc::XCType xc(const pbe_periodic_qe::Pin& pin) {
    constexpr cpu::xc::XCType models[] = {cpu::xc::XCType::PBEsol,
                                          cpu::xc::XCType::RevPBE,
                                          cpu::xc::XCType::RPBE};
    return models[pin.functional];
}
inline std::unique_ptr<cpu::pw::PW> method(const pbe_periodic_qe::Pin& p) {
    auto potential = std::make_shared<cpu::pw::GthPseudopotential>(
        p.spin ? molrs::Element::H : molrs::Element::Si,
        p.spin ? "GTH-PBE-q1" : "GTH-PBE-q4");
    auto m =
        std::make_unique<cpu::pw::PW>(potential, p.spin ? 6. : 12., 1, xc(p));
    m->spin_polarized = p.spin;
    m->multiplicity = p.spin ? 2 : 1;
    m->smearing = electronic::Smearing::fermi_dirac;
    m->degauss = .02;
    m->extra_bands = p.spin ? 3 : 2;
    m->max_iter = p.spin ? 250 : 200;
    m->tol = 1e-9;
    m->mix_beta = p.spin ? .35 : .2;
    return m;
}
} // namespace fixture
