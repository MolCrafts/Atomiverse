// Shared opt-in fixtures; H3 matches the actual CP2K UKS capture exactly.
#pragma once
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
namespace pbe_periodic_uks_fixture {
inline constexpr double cutoff_ry = 80;
inline ATV::cpu::Frame h3() {
    using namespace ATV;
    cpu::Frame frame(3);
    constexpr double a = units::kBohrInAngstrom;
    frame.box = cpu::Box::orthorhombic(12 * a, 11 * a, 10 * a);
    constexpr double p[3][3] = {
        {1.2, 1.1, 1}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = p[i][0];
        frame.span(kAtom_y)[i] = p[i][1];
        frame.span(kAtom_z)[i] = p[i][2];
    }
    return frame;
}
inline std::vector<ATV::cpu::gpw::Kind> kinds() {
    using namespace ATV;
    return {{molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}
template <class Method>
void configure(Method& method, int n_beta = 1) {
    method.n_alpha = 2;
    method.n_beta = n_beta;
    method.max_iter = 100;
    method.tol = 1e-10;
    method.diis_depth = 6;
    method.eps_default = 1e-10;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.use_ot = false;
    method.ngrids = 1;
    method.stress_strain_h = .002;
    method.stress_axes = {true, true, true};
}
} // namespace pbe_periodic_uks_fixture
