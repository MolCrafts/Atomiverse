/// @file tests/cpu/md/test_pe_multi_chain.cpp
/// @brief Unit surface for multi-molecule bonded topology + LJ specials.
///
/// Multi-step NVE conservation is a **regression**
/// (`regressions/md/pe_bonded_nve`). This file only exercises one force
/// launch: net force ≈ 0 (Newton 3) and finite LJ energy with special-bonds
/// exclusions on 8 independent C20 chains.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

inline constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
inline constexpr double kBondK_kcal = 268.0;
inline constexpr double kBondR0 = 1.53;
inline constexpr double kAngleK_kcal = 60.0;
inline constexpr double kAngleTheta0 = 109.5 * units::pi / 180.0;

inline double bond_K_eV() { return kBondK_kcal * kKcalPerMolToEv; }
inline double angle_K_eV() { return kAngleK_kcal * kKcalPerMolToEv; }
inline std::vector<double> dihedral_Cn_eV() {
    return {
        1.73 * kKcalPerMolToEv,
        -4.49 * kKcalPerMolToEv,
        0.776 * kKcalPerMolToEv,
        6.99 * kKcalPerMolToEv,
        0.0,
    };
}

struct MultiChainConfig {
    int n_chains_y = 2;
    int n_chains_z = 4;
    int n_carbons = 20;
    double chain_dy = 12.0;
    double chain_dz = 12.0;
};

cpu::Frame make_multi_chain_box(const MultiChainConfig& cfg) {
    const int n_chains = cfg.n_chains_y * cfg.n_chains_z;
    const int n_atoms = n_chains * cfg.n_carbons;
    cpu::Frame frame(n_atoms);

    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);

    const double Lx = (cfg.n_carbons - 1) * dx_step + 6.0;
    const double Ly = cfg.n_chains_y * cfg.chain_dy;
    const double Lz = cfg.n_chains_z * cfg.chain_dz;

    int idx = 0;
    for (int cy = 0; cy < cfg.n_chains_y; ++cy) {
        for (int cz = 0; cz < cfg.n_chains_z; ++cz) {
            const double y0 = (cy + 0.5) * cfg.chain_dy;
            const double z0 = (cz + 0.5) * cfg.chain_dz;
            for (int i = 0; i < cfg.n_carbons; ++i) {
                frame.span(ATV::kAtom_x)[idx] = 3.0 + i * dx_step;
                frame.span(ATV::kAtom_y)[idx] =
                    y0 + (i % 2 == 0 ? 0.0 : dy_step);
                frame.span(ATV::kAtom_z)[idx] = z0;
                frame.span(ATV::kAtom_mass)[idx] = 14.027;
                frame.span(ATV::kAtom_type)[idx] = 0;
                ++idx;
            }
        }
    }

    const int n_bonds_per = cfg.n_carbons - 1;
    const int n_angles_per = cfg.n_carbons - 2;
    const int n_dihedrals_per = cfg.n_carbons - 3;

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int c = 0; c < n_chains; ++c) {
        const int base = c * cfg.n_carbons;
        for (int k = 0; k < n_bonds_per; ++k)
            frame.add_bond(base + k, base + k + 1, 0);
        for (int k = 0; k < n_angles_per; ++k)
            frame.add_angle(base + k, base + k + 1, base + k + 2, 0);
        for (int k = 0; k < n_dihedrals_per; ++k)
            frame.add_dihedral(base + k, base + k + 1, base + k + 2,
                               base + k + 3, 0);
    }
    frame.build_special_list();
    frame.box = ATV::cpu::Box::orthorhombic(Lx, Ly, Lz);
    return frame;
}

void zero_forces(cpu::Frame& f) {
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(ATV::kAtom_fx)[i] = 0;
        f.span(ATV::kAtom_fy)[i] = 0;
        f.span(ATV::kAtom_fz)[i] = 0;
    }
    f.meta.get(ATV::kEnergy) = 0;
}

} // namespace

// ── One force launch: Newton's 3rd on bonded multi-molecule topology ──

TEST(PeMultiChain, BondedOnlyNetForceOneLaunch) {
    MultiChainConfig cfg;
    auto frame = make_multi_chain_box(cfg);

    cpu::BondHarmonic bond{{bond_K_eV()}, {kBondR0}};
    cpu::AngleHarmonic angle{{angle_K_eV()}, {kAngleTheta0}};
    cpu::DihedralMultiHarmonic dih{{dihedral_Cn_eV()}};

    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    bond.build(bctx);
    angle.build(bctx);
    dih.build(bctx);

    zero_forces(frame);
    cpu::LaunchContext lctx{ctx, frame, 0, 0};
    bond.launch(lctx);
    angle.launch(lctx);
    dih.launch(lctx);

    double sum_fx = 0, sum_fy = 0, sum_fz = 0;
    for (int i = 0; i < frame.nlocal(); ++i) {
        sum_fx += frame.span(ATV::kAtom_fx)[i];
        sum_fy += frame.span(ATV::kAtom_fy)[i];
        sum_fz += frame.span(ATV::kAtom_fz)[i];
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_fx)[i]));
    }
    EXPECT_NEAR(sum_fx, 0.0, 1e-9);
    EXPECT_NEAR(sum_fy, 0.0, 1e-9);
    EXPECT_NEAR(sum_fz, 0.0, 1e-9);
    EXPECT_TRUE(std::isfinite(frame.meta.get(ATV::kEnergy)));

    bond.destroy({ctx, frame});
    angle.destroy({ctx, frame});
    dih.destroy({ctx, frame});
}

// ── One force launch: LJ + specials stay finite on multi-chain box ──
// Mirrors tests/cpu/md/test_lj_exclusions.cpp launch order.

TEST(PeMultiChain, BondedPlusLjSpecialsFiniteOneLaunch) {
    MultiChainConfig cfg;
    auto frame = make_multi_chain_box(cfg);

    const double lj_eps = 0.00512;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;

    cpu::Context ctx;
    cpu::BuildContext bctx{ctx, frame};
    cpu::LaunchContext lctx{ctx, frame, 0, 0};

    cpu::md::NeighborList<> nlist(lj_cut);
    nlist.build(bctx);
    nlist.launch(lctx);

    cpu::PairLJCut lj(lj_eps, lj_sig, lj_cut);
    lj.set_special_bonds(0.0, 0.0, 0.0);
    lj.build(bctx);

    zero_forces(frame);
    lj.launch(lctx);

    EXPECT_FALSE(std::isnan(frame.meta.get(ATV::kEnergy)))
        << "LJ + specials produced NaN energy";
    EXPECT_LT(std::abs(frame.meta.get(ATV::kEnergy)), 1e6)
        << "energy " << frame.meta.get(ATV::kEnergy)
        << " is wildly large — LJ exclusions likely broken";
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_fx)[i]))
            << "atom " << i << " fx went NaN";
    }

    lj.destroy({ctx, frame});
    nlist.destroy({ctx, frame});
}
