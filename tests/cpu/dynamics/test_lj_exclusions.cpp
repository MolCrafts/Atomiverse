/// @file tests/cpu/md/test_lj_exclusions.cpp
/// @brief LJ::set_special_bonds() correctness on a single C20H42 chain.
///
/// In the all-trans alkane geometry every 1-2 (bond) distance is 1.529 Å,
/// every 1-3 (across an angle) distance is roughly 2.55 Å and every 1-4
/// (across a dihedral) is roughly 3.85 Å. With OPLS-AA UA σ = 4.01 Å the
/// 1-2 and 1-3 separations sit deep on the repulsive wall of LJ — without
/// exclusions the energy and the per-atom forces blow up. With LAMMPS-style
/// `special_bonds 0 0 0` the bonded pairs are skipped and the remaining
/// LJ contribution comes only from chain-internal 1-5+ neighbours.

#include <gtest/gtest.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

inline constexpr double kBondR0 = 1.53;
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0;

cpu::Frame make_all_trans_alkane(int n_carbons, double box_len) {
    cpu::Frame frame(n_carbons);
    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);
    const double x0 = (box_len - (n_carbons - 1) * dx_step) / 2.0;
    const double y0 = box_len / 2.0;
    const double z0 = box_len / 2.0;
    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x0 + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    const int n_b = n_carbons - 1;
    const int n_a = n_carbons - 2;
    const int n_d = n_carbons - 3;
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_b; ++k) {
        frame.add_bond(k, k + 1, 0);
    }
    for (int k = 0; k < n_a; ++k) {
        frame.add_angle(k, k + 1, k + 2, 0);
    }
    for (int k = 0; k < n_d; ++k) {
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    }
    frame.build_special_list();
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

double evaluate_lj_only(cpu::Frame& frame, double eps, double sig,
                        double cutoff, double s12, double s13, double s14) {
    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    cpu::LaunchContext lctx{ctx, frame, 0, 0};

    cpu::md::NeighborList<> nlist(cutoff);
    nlist.build(bctx);
    nlist.launch(lctx);

    cpu::PairLJCut lj(eps, sig, cutoff);
    lj.set_special_bonds(s12, s13, s14);
    lj.build(bctx);

    const int n = frame.nlocal();
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }
    frame.meta.get(ATV::kEnergy) = 0;
    lj.launch(lctx);
    lj.destroy({ctx, frame});
    return frame.meta.get(ATV::kEnergy);
}

} // namespace

// ── 1-2/1-3/1-4 exclusions remove the bonded contribution entirely ──

TEST(LjExclusions, SingleChainExclusionsDropEnergy) {
    auto frame_a = make_all_trans_alkane(20, 50.0);
    auto frame_b = make_all_trans_alkane(20, 50.0);

    // OPLS-AA UA CH2: σ = 4.01 Å, ε = 0.118 kcal/mol = 0.00512 eV.
    const double eps = 0.00512;
    const double sig = 4.01;
    const double cutoff = 10.5;

    const double E_with_excl =
        evaluate_lj_only(frame_a, eps, sig, cutoff, 0.0, 0.0, 0.0);
    const double E_no_excl =
        evaluate_lj_only(frame_b, eps, sig, cutoff, 1.0, 1.0, 1.0);

    // Sanity: bonded-pair LJ at r = 1.529 Å with σ = 4.01 → r/σ ≈ 0.38 →
    // (σ/r)^12 ≈ 1.7e8 → V per pair ~ +2 keV. With 19 such bonds, the
    // un-excluded LJ energy is in the +1e4 eV range. The excluded version
    // sees only 1-5+ pairs and stays modest.
    EXPECT_GT(E_no_excl, 1e3)
        << "without exclusions, 1-2 LJ should be enormous";
    EXPECT_LT(std::abs(E_with_excl), 1.0)
        << "with full bonded exclusions, only 1-5+ LJ contributes";
    EXPECT_LT(E_with_excl, E_no_excl);
}

// ── Toggle: a 4-atom dihedral with bond-only exclusion ───────────────

TEST(LjExclusions, BondExclusionRemovesOne12PairOnly) {
    cpu::Frame frame_a(4);
    cpu::Frame frame_b(4);
    for (auto* f : {&frame_a, &frame_b}) {
        for (int i = 0; i < 4; ++i) {
            (*f).span(ATV::kAtom_x)[i] = 1.5 * i;
            (*f).span(ATV::kAtom_y)[i] = 0;
            (*f).span(ATV::kAtom_z)[i] = 0;
            (*f).span(ATV::kAtom_mass)[i] = 14.0;
            (*f).span(ATV::kAtom_type)[i] = 0;
        }
        (*f).clear_tag_index();
        (*f).rebuild_tag_index(0, (*f).nlocal());
        (*f).add_bond(0, 1, 0);
        (*f).build_special_list();
        (*f).box = ATV::cpu::Box::cubic(40.0);
    }

    // Excluding only 1-2 should drop only the 0-1 pair from LJ; pairs (0,2),
    // (0,3), (1,2), (1,3), (2,3) still contribute.
    const double eps = 0.00512;
    const double sig = 4.01;
    const double cutoff = 10.5;

    const double E_excl_12 =
        evaluate_lj_only(frame_a, eps, sig, cutoff, 0.0, 1.0, 1.0);
    const double E_no_excl =
        evaluate_lj_only(frame_b, eps, sig, cutoff, 1.0, 1.0, 1.0);

    EXPECT_GT(E_no_excl, E_excl_12)
        << "removing the 0-1 LJ pair must lower the total LJ energy";
}
