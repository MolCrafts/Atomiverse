#pragma once
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "diagnostic_recipe.h"
#include "multi_k_fd_pins.h"
namespace pw_multi_k_fd {
inline ATV::cpu::Frame
frame(const pw_multi_k_fd_qe::Pin& p,
      const pw_analytic_diagnostic::Recipe& recipe = {}) {
    using namespace ATV;
    cpu::Frame f(p.nat);
    if (recipe.empty_beta() && (!p.spin || p.nat != 2 || !recipe.fixed()))
        throw std::invalid_argument(
            "empty-beta diagnostic requires fixed spin H2");
    const bool primitive =
        recipe.fixed()
        || recipe.geometry == pw_analytic_diagnostic::Geometry::primitive_si2;
    if (p.spin) {
        Mat3 h;
        h(0, 0) = recipe.empty_beta() || recipe.isolated_h3 ? 6.2 : 4.2;
        h(0, 1) = .3;
        h(0, 2) = .2;
        h(1, 1) = recipe.empty_beta() || recipe.isolated_h3 ? 6 : 4;
        h(1, 2) = .1;
        h(2, 2) = recipe.empty_beta() || recipe.isolated_h3 ? 5.8 : 3.8;
        f.box.set_cell(h);
    } else if (primitive) {
        Mat3 h;
        constexpr double columns[3][3] = {
            {.10, 2.70, 2.65}, {2.75, .05, 2.70}, {2.65, 2.75, .15}};
        for (int c = 0; c < 3; ++c)
            for (int row = 0; row < 3; ++row)
                h(row, c) = columns[c][row];
        f.box.set_cell(h);
    } else
        f.box = cpu::Box::cubic(6);
    constexpr double h[3][3] = {
        {1.2, 1.1, 1}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    const double si[2][3] = {
        {primitive ? .2 : 2, primitive ? .3 : 2, primitive ? .4 : 2},
        {primitive ? 1.57 : 3.8, primitive ? 1.66 : 3.2, primitive ? 1.75 : 3}};
    const double hn = std::sqrt(1.1 * 1.1 + .5 * .5 + .4 * .4);
    const double h2[2][3] = {{3.35 - 1.1 / hn, 3.05 - .5 / hn, 2.9 - .4 / hn},
                             {3.35 + 1.1 / hn, 3.05 + .5 / hn, 2.9 + .4 / hn}};
    for (int i = 0; i < p.nat; ++i) {
        f.span(kAtom_type)[i] = p.spin ? 1 : 14;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = p.spin ? 1.008 : 28.0855;
        f.span(kAtom_x)[i] = recipe.empty_beta() ? h2[i][0]
                             : p.spin            ? h[i][0]
                                                 : si[i][0];
        f.span(kAtom_y)[i] = recipe.empty_beta() ? h2[i][1]
                             : p.spin            ? h[i][1]
                                                 : si[i][1];
        f.span(kAtom_z)[i] = recipe.empty_beta() ? h2[i][2]
                             : p.spin            ? h[i][2]
                                                 : si[i][2];
        f.span(kAtom_fx)[i] = 10 + i;
        f.span(kAtom_fy)[i] = 20 + i;
        f.span(kAtom_fz)[i] = 30 + i;
    }
    f.meta.get(kEnergy) = 123;
    f.meta.get(kEnergyElectronic) = 456;
    f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    if (recipe.rotate_z90) {
        Mat3 q{};
        q(0, 1) = -1;
        q(1, 0) = 1;
        q(2, 2) = 1;
        f.box.set_cell(q * f.box.cell());
        for (int i = 0; i < f.nlocal(); ++i) {
            auto x = q
                     * Vec3{f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                            f.span(kAtom_z)[i]};
            f.span(kAtom_x)[i] = x[0];
            f.span(kAtom_y)[i] = x[1];
            f.span(kAtom_z)[i] = x[2];
        }
    }
    return f;
}
inline std::unique_ptr<ATV::cpu::pw::PW>
method(const pw_multi_k_fd_qe::Pin& p,
       const pw_analytic_diagnostic::Recipe& recipe = {}) {
    using namespace ATV;
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(
        p.spin ? molrs::Element::H : molrs::Element::Si,
        p.spin ? "GTH-PBE-q1" : "GTH-PBE-q4");
    auto m = std::make_unique<cpu::pw::PW>(pp, p.spin ? 6. : 12., recipe.kmesh,
                                           p.model);
    m->spin_polarized = p.spin;
    m->multiplicity = recipe.empty_beta() ? 3 : p.spin ? 2 : 1;
    m->smearing = recipe.smearing;
    m->degauss = recipe.width_ry;
    m->extra_bands = p.spin ? 3 : 2;
    m->max_iter = p.spin ? 250 : 200;
    m->tol =
        !p.spin
                && (recipe.fixed()
                    || recipe.geometry
                           == pw_analytic_diagnostic::Geometry::primitive_si2)
            ? 1e-10
            : 1e-9;
    m->mix_beta = p.spin ? .35 : .2;
    return m;
}
} // namespace pw_multi_k_fd
