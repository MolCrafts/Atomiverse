/// @file regressions/md/four-site-tip4p-foothold/reg_four-site-tip4p-foothold.cpp
/// @brief E2E: TIP4P four-site water dimer vs offline LAMMPS pin.
/// Public headers only. See REFERENCE.md. No live third-party at ctest.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"

using namespace ATV;

int main() {
    // Classic TIP4P (Jorgensen 1983/85)
    const double r_oh = 0.9572;
    const double theta = 104.52 * units::kDegreeInRad;
    const double qdist = 0.15;
    const double eps = cpu::kcal_per_mol_to_ev(0.1550);
    const double sigma = 3.15365;
    const double cut = 10.0;
    const double qO = -1.04, qH = 0.52;

    // Offline pin — LAMMPS metal units (see REFERENCE.md)
    const double E_pin = 0.52924228894955327451;
    const double fx_pin[6] = {
        -0.094382238074752045076, -1.0593571566097745862,
        -0.057823257196248978307, 0.094382238074751600987,
        1.0593571566097748082,    0.057823257196249200351};
    const double fz_pin[6] = {
        0.33187666441899288383,   -0.31522207906906996033,
        -0.016654585349923187176, 0.33187666441899305037,
        -0.31522207906906962727,  -0.01665458534992322534};

    const double half = 0.5 * theta;
    const double s = std::sin(half), c = std::cos(half);
    const double ox2 = 2.8;

    cpu::Frame f(6);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r_oh * s;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = r_oh * c;
    f.span(kAtom_x)[2] = -r_oh * s;
    f.span(kAtom_y)[2] = 0;
    f.span(kAtom_z)[2] = r_oh * c;
    f.span(kAtom_x)[3] = ox2;
    f.span(kAtom_y)[3] = 0;
    f.span(kAtom_z)[3] = 0;
    f.span(kAtom_x)[4] = ox2 - r_oh * s;
    f.span(kAtom_y)[4] = 0;
    f.span(kAtom_z)[4] = r_oh * c;
    f.span(kAtom_x)[5] = ox2 + r_oh * s;
    f.span(kAtom_y)[5] = 0;
    f.span(kAtom_z)[5] = r_oh * c;

    const int types[6] = {0, 1, 1, 0, 1, 1};
    const double mass[6] = {15.9994, 1.008, 1.008, 15.9994, 1.008, 1.008};
    const double q[6] = {qO, qH, qH, qO, qH, qH};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(1, 2, 0);
    f.add_bond(1, 3, 0);
    f.add_bond(4, 5, 0);
    f.add_bond(4, 6, 0);
    f.add_angle(2, 1, 3, 0);
    f.add_angle(5, 4, 6, 0);
    f.build_special_list();

    cpu::PairLJCutTip4pCut tip4p(/*type_O=*/0, /*type_H=*/1, qdist, r_oh, theta,
                                 eps, sigma, cut, cut);
    tip4p.set_special_bonds(0.0, 0.0, 0.0);

    // Dual-gate geometry: OM = qdist for water 1
    double xM, yM, zM;
    tip4p.compute_newsite(
        f.span(kAtom_x)[0], f.span(kAtom_y)[0], f.span(kAtom_z)[0],
        f.span(kAtom_x)[1], f.span(kAtom_y)[1], f.span(kAtom_z)[1],
        f.span(kAtom_x)[2], f.span(kAtom_y)[2], f.span(kAtom_z)[2], xM, yM, zM);
    const double om = std::sqrt(xM * xM + yM * yM + zM * zM);
    std::printf("OM = %.16e Å (qdist=%.2f)\n", om, qdist);
    if (!(std::fabs(om - qdist) <= 1e-10)) {
        std::printf("FAIL: OM geometry\n");
        return 1;
    }

    cpu::Evaluate eval;
    eval.set_neighbor(
        cpu::md::NeighborList<>{tip4p.neighbor_cutoff(), 0.3, 1, 0, true});
    eval.set_evaluator(std::move(tip4p));
    eval.enable_timing(false);
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    std::printf("TIP4P dimer E=%.16e eV (pin=%.16e)\n", E, E_pin);
    if (!(std::fabs(E - E_pin) <= 1e-8)) {
        std::printf("FAIL: energy\n");
        return 1;
    }

    bool ok = true;
    for (int i = 0; i < 6; ++i) {
        const double fx = f.span(kAtom_fx)[i];
        const double fy = f.span(kAtom_fy)[i];
        const double fz = f.span(kAtom_fz)[i];
        std::printf("F%d  % .16e % .16e % .16e\n", i + 1, fx, fy, fz);
        if (!(std::fabs(fx - fx_pin[i]) <= 1e-6) || !(std::fabs(fy) <= 1e-10)
            || !(std::fabs(fz - fz_pin[i]) <= 1e-6)) {
            std::printf("FAIL: force atom %d\n", i + 1);
            ok = false;
        }
    }
    if (!ok)
        return 1;

    std::printf("PASS: four-site-tip4p-foothold vs LAMMPS offline pin\n");
    return 0;
}
