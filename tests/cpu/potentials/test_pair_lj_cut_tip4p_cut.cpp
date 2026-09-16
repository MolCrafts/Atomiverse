/// @file tests/cpu/potentials/test_pair_lj_cut_tip4p_cut.cpp
/// @brief Unit: TIP4P M-site geometry + dimer energy/force vs offline LAMMPS pin.
///
/// Pin: tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/
/// (units metal, pair_style lj/cut/tip4p/cut, pair_modify shift yes).
/// No live LAMMPS in ctest.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"

using namespace ATV;

namespace {

// Classic Jorgensen TIP4P geometry / charges.
constexpr double kROH = 0.9572;
constexpr double kThetaDeg = 104.52;
constexpr double kQDist = 0.15;
constexpr double kEpsKcal = 0.1550;
constexpr double kSigma = 3.15365;
constexpr double kCut = 10.0;
constexpr double kQO = -1.04;
constexpr double kQH = 0.52;

// Offline pin — LAMMPS 30 Mar 2026 develop, units metal, shift yes.
// See tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/log.metal
constexpr double kPinE = 0.52924228894955327451; // eV
constexpr double kPinFx[6] = {
    -0.094382238074752045076, -1.0593571566097745862, -0.057823257196248978307,
    0.094382238074751600987,  1.0593571566097748082,  0.057823257196249200351};
constexpr double kPinFz[6] = {
    0.33187666441899288383, -0.31522207906906996033, -0.016654585349923187176,
    0.33187666441899305037, -0.31522207906906962727, -0.01665458534992322534};

double theta_rad() { return kThetaDeg * units::kDegreeInRad; }

double eps_ev() { return cpu::kcal_per_mol_to_ev(kEpsKcal); }

cpu::PairLJCutTip4pCut make_tip4p() {
    return cpu::PairLJCutTip4pCut(/*type_O=*/0, /*type_H=*/1, kQDist, kROH,
                                  theta_rad(), eps_ev(), kSigma, kCut, kCut)
        .set_special_bonds(0.0, 0.0, 0.0);
}

/// Two waters matching oracle data.tip4p_dimer_metal (types 0=O, 1=H).
cpu::Frame tip4p_dimer() {
    const double half = 0.5 * theta_rad();
    const double s = std::sin(half), c = std::cos(half);
    // water1 at origin; water2 O at 2.8 Å, rotated π about z
    const double ox2 = 2.8;
    cpu::Frame f(6);
    // O1
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    // H1, H2
    f.span(kAtom_x)[1] = kROH * s;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = kROH * c;
    f.span(kAtom_x)[2] = -kROH * s;
    f.span(kAtom_y)[2] = 0;
    f.span(kAtom_z)[2] = kROH * c;
    // O2
    f.span(kAtom_x)[3] = ox2;
    f.span(kAtom_y)[3] = 0;
    f.span(kAtom_z)[3] = 0;
    // H of water2 (phi = π)
    f.span(kAtom_x)[4] = ox2 - kROH * s;
    f.span(kAtom_y)[4] = 0;
    f.span(kAtom_z)[4] = kROH * c;
    f.span(kAtom_x)[5] = ox2 + kROH * s;
    f.span(kAtom_y)[5] = 0;
    f.span(kAtom_z)[5] = kROH * c;

    const int types[6] = {0, 1, 1, 0, 1, 1};
    const double mass[6] = {15.9994, 1.008, 1.008, 15.9994, 1.008, 1.008};
    const double q[6] = {kQO, kQH, kQH, kQO, kQH, kQH};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_tag)[i] = i + 1; // consecutive O,H1,H2
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
    return f;
}

void run_eval(cpu::Frame& f, cpu::PairLJCutTip4pCut pair) {
    cpu::Evaluate eval;
    eval.set_neighbor(
        cpu::md::NeighborList<>{pair.neighbor_cutoff(), 0.3, 1, 0, true});
    eval.set_evaluator(std::move(pair));
    eval.enable_timing(false);
    eval.build();
    eval.run(f);
}

} // namespace

TEST(PairLJCutTip4pCut, AlphaAndMSiteGeometry) {
    auto p = make_tip4p();
    // α = qdist / (cos(θ/2) r_OH)
    const double half = 0.5 * theta_rad();
    const double alpha_ref = kQDist / (std::cos(half) * kROH);
    EXPECT_NEAR(p.alpha(), alpha_ref, 1e-15);

    // Symmetric water: M at (0,0,qdist) when O at origin, H's at ±x, +z
    double xM, yM, zM;
    const double s = std::sin(half), c = std::cos(half);
    p.compute_newsite(0, 0, 0, kROH * s, 0, kROH * c, -kROH * s, 0, kROH * c,
                      xM, yM, zM);
    EXPECT_NEAR(xM, 0.0, 1e-12);
    EXPECT_NEAR(yM, 0.0, 1e-12);
    EXPECT_NEAR(zM, kQDist, 1e-10); // dual-gate geometry: OM = qdist
    const double om = std::sqrt(xM * xM + yM * yM + zM * zM);
    EXPECT_NEAR(om, kQDist, 1e-10);
}

TEST(PairLJCutTip4pCut, DimerEnergyMatchesLammpsPin) {
    cpu::Frame f = tip4p_dimer();
    run_eval(f, make_tip4p());
    const double E = f.meta.get(kEnergy);
    EXPECT_NEAR(E, kPinE, 1e-8) << "E=" << E;
}

TEST(PairLJCutTip4pCut, DimerForcesMatchLammpsPin) {
    cpu::Frame f = tip4p_dimer();
    run_eval(f, make_tip4p());
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(f.span(kAtom_fx)[i], kPinFx[i], 1e-6) << "fx[" << i << "]";
        EXPECT_NEAR(f.span(kAtom_fy)[i], 0.0, 1e-10) << "fy[" << i << "]";
        EXPECT_NEAR(f.span(kAtom_fz)[i], kPinFz[i], 1e-6) << "fz[" << i << "]";
    }
    // Newton III (total force)
    double fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < 6; ++i) {
        fx += f.span(kAtom_fx)[i];
        fy += f.span(kAtom_fy)[i];
        fz += f.span(kAtom_fz)[i];
    }
    EXPECT_NEAR(fx, 0.0, 1e-10);
    EXPECT_NEAR(fy, 0.0, 1e-10);
    EXPECT_NEAR(fz, 0.0, 1e-10);
}

TEST(PairLJCutTip4pCut, RejectsBadTheta) {
    EXPECT_THROW(
        cpu::PairLJCutTip4pCut(0, 1, 0.15, 0.9572, 0.0, 0.01, 3.0, 10.0),
        std::invalid_argument);
}
