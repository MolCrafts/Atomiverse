/// @file regressions/forcefield/tip3p_two_water/reg_tip3p_two_water.cpp
/// @brief Potential components on Evaluate for two waters.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    cpu::Frame f(6);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = -0.23999;
    f.span(kAtom_y)[2] = 0.92663;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_x)[3] = 3.5;
    f.span(kAtom_y)[3] = 0.1;
    f.span(kAtom_z)[3] = 0.05;
    f.span(kAtom_x)[4] = 4.4;
    f.span(kAtom_y)[4] = 0.2;
    f.span(kAtom_z)[4] = 0.1;
    f.span(kAtom_x)[5] = 3.2;
    f.span(kAtom_y)[5] = 1.0;
    f.span(kAtom_z)[5] = 0.0;
    int types[6] = {0, 1, 1, 0, 1, 1};
    double mass[6] = {15.999, 1.008, 1.008, 15.999, 1.008, 1.008};
    double q[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.simbox = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(1, 2, 0);
    f.add_bond(1, 3, 0);
    f.add_bond(4, 5, 0);
    f.add_bond(4, 6, 0);
    f.add_angle(2, 1, 3, 0);
    f.add_angle(5, 4, 6, 0);
    f.build_special_list();

    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{10.0, 0.3});
    eval.add_potential(
        cpu::PairLJCut{0.0067, 3.1507, 10.0}.set_special_bonds(0, 0, 0));
    eval.on(evaluate::Phase::evaluate)
        .add(cpu::PairCoulCut{10.0}.set_special_bonds(0, 0, 0));
    eval.add_potential(cpu::BondHarmonic{std::vector<double>{25.0},
                                         std::vector<double>{0.9572}});
    eval.add_potential(cpu::AngleHarmonic{
        std::vector<double>{5.0},
        std::vector<double>{104.52 * 3.141592653589793 / 180.0}});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    std::printf("two-water Potentials E=%.6f eV\n", E);
    if (!(std::isfinite(E)) || !(std::fabs(E) < 50.0)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS: Potential components on Evaluate\n");
    return 0;
}
