/// @file regressions/md/virtual-site-multi-topology/reg_virtual-site-multi-topology.cpp
/// @brief E2E: general VirtualSites TIP4P dimer vs offline LAMMPS tip4p pin +
///        TIP5P geometry dual-gate. Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/virtual_sites.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // Classic TIP4P (Jorgensen 1983/85)
    const double r_oh = 0.9572;
    const double theta = 104.52 * units::kDegreeInRad;
    const double qdist = 0.15;
    const double eps = cpu::kcal_per_mol_to_ev(0.1550);
    const double sigma = 3.15365;
    const double cut = 10.0;
    const double qM = -1.04, qH = 0.52;

    // Offline pin — LAMMPS metal (four-site-tip4p-foothold / this slug twin)
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
    const double alpha = cpu::md::VirtualSites::tip4p_alpha(qdist, r_oh, theta);

    // 8 atoms: water1 O H1 H2 M, water2 O H1 H2 M
    // tags 1..8; types 0=O, 1=H, 2=M
    cpu::Frame f(8);
    // water1
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r_oh * s;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = r_oh * c;
    f.span(kAtom_x)[2] = -r_oh * s;
    f.span(kAtom_y)[2] = 0;
    f.span(kAtom_z)[2] = r_oh * c;
    f.span(kAtom_x)[3] = 0; // M placed by VirtualSites
    f.span(kAtom_y)[3] = 0;
    f.span(kAtom_z)[3] = 0;
    // water2
    f.span(kAtom_x)[4] = ox2;
    f.span(kAtom_y)[4] = 0;
    f.span(kAtom_z)[4] = 0;
    f.span(kAtom_x)[5] = ox2 - r_oh * s;
    f.span(kAtom_y)[5] = 0;
    f.span(kAtom_z)[5] = r_oh * c;
    f.span(kAtom_x)[6] = ox2 + r_oh * s;
    f.span(kAtom_y)[6] = 0;
    f.span(kAtom_z)[6] = r_oh * c;
    f.span(kAtom_x)[7] = ox2;
    f.span(kAtom_y)[7] = 0;
    f.span(kAtom_z)[7] = 0;

    const int types[8] = {0, 1, 1, 2, 0, 1, 1, 2};
    const double mass[8] = {15.9994, 1.008, 1.008, 0.0,
                            15.9994, 1.008, 1.008, 0.0};
    const double q[8] = {0.0, qH, qH, qM, 0.0, qH, qH, qM};
    for (int i = 0; i < 8; ++i) {
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    // Topology so special 0,0,0 excludes all intra-water pairs including M
    // water1: O-H1, O-H2, O-M; angle H1-O-H2
    f.add_bond(1, 2, 0);
    f.add_bond(1, 3, 0);
    f.add_bond(1, 4, 0);
    f.add_angle(2, 1, 3, 0);
    // water2
    f.add_bond(5, 6, 0);
    f.add_bond(5, 7, 0);
    f.add_bond(5, 8, 0);
    f.add_angle(6, 5, 7, 0);
    f.build_special_list();

    cpu::md::VirtualSites vs;
    vs.add_weighted3(/*M=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, alpha);
    vs.add_weighted3(/*M=*/8, /*O=*/5, /*H1=*/6, /*H2=*/7, alpha);

    // Dual-gate geometry before energy
    vs.place(f);
    const double om =
        std::sqrt(static_cast<double>(f.span(kAtom_x)[3]) * f.span(kAtom_x)[3]
                  + f.span(kAtom_y)[3] * f.span(kAtom_y)[3]
                  + f.span(kAtom_z)[3] * f.span(kAtom_z)[3]);
    std::printf("OM = %.16e Å (qdist=%.2f)\n", om, qdist);
    if (!(std::fabs(om - qdist) <= 1e-10)) {
        std::printf("FAIL: OM geometry\n");
        return 1;
    }

    cpu::md::VirtualSites::Place place{vs};
    cpu::md::VirtualSites::Redistribute redist{vs};

    cpu::PairLJCut lj{{{0, 0, eps, sigma}}, cut};
    lj.set_special_bonds(0.0, 0.0, 0.0);
    cpu::PairCoulCut coul{cut};
    coul.set_special_bonds(0.0, 0.0, 0.0);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::prepare).add_ref(place, +100);
    eval.set_neighbor(cpu::md::NeighborList<>{cut, 0.3, 1, 0, true});
    eval.set_evaluator(std::move(lj));
    eval.set_evaluator(std::move(coul));
    eval.on(evaluate::Phase::finalize).add_ref(redist, -100);
    eval.enable_timing(false);
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    std::printf("TIP4P dimer (VirtualSites) E=%.16e eV (pin=%.16e)\n", E,
                E_pin);
    if (!(std::fabs(E - E_pin) <= 1e-8)) {
        std::printf("FAIL: energy\n");
        return 1;
    }

    // Parent forces: map O/H rows (skip M) to pin order O1 H1 H2 O2 H3 H4
    const int parent_row[6] = {0, 1, 2, 4, 5, 6};
    bool ok = true;
    for (int k = 0; k < 6; ++k) {
        const int i = parent_row[k];
        const double fx = f.span(kAtom_fx)[i];
        const double fy = f.span(kAtom_fy)[i];
        const double fz = f.span(kAtom_fz)[i];
        std::printf("F_parent%d  % .16e % .16e % .16e\n", k + 1, fx, fy, fz);
        if (!(std::fabs(fx - fx_pin[k]) <= 1e-6) || !(std::fabs(fy) <= 1e-10)
            || !(std::fabs(fz - fz_pin[k]) <= 1e-6)) {
            std::printf("FAIL: force parent %d\n", k + 1);
            ok = false;
        }
    }
    // M forces must be zero after redistribute
    for (int i : {3, 7}) {
        if (!(std::fabs(f.span(kAtom_fx)[i]) <= 1e-12)
            || !(std::fabs(f.span(kAtom_fy)[i]) <= 1e-12)
            || !(std::fabs(f.span(kAtom_fz)[i]) <= 1e-12)) {
            std::printf("FAIL: residual force on site row %d\n", i);
            ok = false;
        }
    }
    if (!ok)
        return 1;

    // TIP5P geometry dual-gate (one water)
    const double r_ol = 0.70;
    const double theta_lol = 109.47 * units::kDegreeInRad;
    double a1, b1, c1, a2, b2, c2;
    cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(r_ol, theta_lol, r_oh,
                                                     theta, +1, a1, b1, c1);
    cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(r_ol, theta_lol, r_oh,
                                                     theta, -1, a2, b2, c2);
    cpu::Frame w(5);
    w.span(kAtom_x)[0] = 0;
    w.span(kAtom_y)[0] = 0;
    w.span(kAtom_z)[0] = 0;
    w.span(kAtom_x)[1] = r_oh * s;
    w.span(kAtom_y)[1] = 0;
    w.span(kAtom_z)[1] = r_oh * c;
    w.span(kAtom_x)[2] = -r_oh * s;
    w.span(kAtom_y)[2] = 0;
    w.span(kAtom_z)[2] = r_oh * c;
    for (int i = 0; i < 5; ++i) {
        w.span(kAtom_tag)[i] = i + 1;
        w.span(kAtom_type)[i] = (i == 0) ? 0 : (i <= 2 ? 1 : 2);
        w.span(kAtom_mass)[i] = 0;
    }
    w.box = cpu::Box::cubic(20.0);
    w.clear_tag_index();
    w.rebuild_tag_index(0, w.nlocal());
    cpu::md::VirtualSites tip5p;
    tip5p.add_out_of_plane3(4, 1, 2, 3, a1, b1, c1);
    tip5p.add_out_of_plane3(5, 1, 2, 3, a2, b2, c2);
    tip5p.place(w);
    auto rOL = [&](int i) {
        const double dx = w.span(kAtom_x)[i] - w.span(kAtom_x)[0];
        const double dy = w.span(kAtom_y)[i] - w.span(kAtom_y)[0];
        const double dz = w.span(kAtom_z)[i] - w.span(kAtom_z)[0];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    const double rol1 = rOL(3), rol2 = rOL(4);
    std::printf("TIP5P r_OL = %.16e, %.16e (expect %.2f)\n", rol1, rol2, r_ol);
    if (!(std::fabs(rol1 - r_ol) <= 1e-8)
        || !(std::fabs(rol2 - r_ol) <= 1e-8)) {
        std::printf("FAIL: TIP5P r_OL\n");
        return 1;
    }
    double v1x = w.span(kAtom_x)[3], v1y = w.span(kAtom_y)[3],
           v1z = w.span(kAtom_z)[3];
    double v2x = w.span(kAtom_x)[4], v2y = w.span(kAtom_y)[4],
           v2z = w.span(kAtom_z)[4];
    const double cosang = (v1x * v2x + v1y * v2y + v1z * v2z) / (rol1 * rol2);
    const double ang = std::acos(std::max(-1.0, std::min(1.0, cosang)));
    std::printf("TIP5P theta_LOL = %.16e rad (expect %.16e)\n", ang, theta_lol);
    if (!(std::fabs(ang - theta_lol) <= 1e-8)) {
        std::printf("FAIL: TIP5P theta_LOL\n");
        return 1;
    }

    std::printf("PASS: virtual-site-multi-topology vs LAMMPS offline pin + "
                "TIP5P geometry\n");
    return 0;
}
