// Shared physical inputs for CPU preflight and CUDA/nested validation.
// Public method data only; no private engine objects or scheduling state.
#pragma once
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include <array>
#include <vector>

namespace ATV::diagnostics::gpw_driver {
struct Case {
    const char* name;
    bool uks, multigrid, kpoints;
};
constexpr Case cases[] = {{"rks", false, false, false},
                          {"uks", true, false, false},
                          {"multigrid", false, true, false},
                          {"kpoints", false, false, true}};
constexpr double cutoff[] = {80, 100};
inline std::vector<cpu::gpw::Kind> kinds() {
    return {{molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")}};
}
inline cpu::xc::XCType xc(const Case& c) {
    return c.uks || c.kpoints ? cpu::xc::XCType::PADE : cpu::xc::XCType::PBE;
}
inline std::vector<double> forces(const cpu::Frame& frame) {
    std::vector<double> result;
    for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            result.push_back(frame.span(column)[i]);
    return result;
}
inline void mark(cpu::Frame& frame) {
    int index = 0;
    for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            frame.span(column)[i] = 10.25 + index++;
    frame.meta.get(kEnergy) = 31415.125;
    frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
}
inline cpu::Frame geometry(bool shifted) {
    cpu::Frame frame(2);
    constexpr double a = units::kBohrInAngstrom;
    frame.box = cpu::Box::orthorhombic(12 * a, 11 * a, 10 * a);
    const double position[2][3] = {{5.45, 5.15, 4.4}, {6.55, 5.65, 4.8}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = position[i][0] * a;
        frame.span(kAtom_y)[i] = position[i][1] * a;
        frame.span(kAtom_z)[i] = position[i][2] * a;
    }
    if (shifted)
        frame.span(kAtom_x)[1] += .017;
    mark(frame);
    return frame;
}
template <class Solver>
void configure(Solver& solver, const Case& c) {
    solver.n_occ = c.uks ? 0 : 1;
    solver.n_alpha = c.uks ? 2 : 0;
    solver.n_beta = 0;
    solver.max_iter = 200;
    solver.tol = 1e-10;
    solver.diis_depth = 6;
    solver.eps_default = 1e-10;
    solver.ngrids = c.multigrid ? 2 : 1;
    solver.rel_cutoff = c.multigrid ? 20 : 0;
    solver.k_mesh[0] = c.kpoints ? 2 : 1;
    solver.k_time_reversal = false;
}
} // namespace ATV::diagnostics::gpw_driver
