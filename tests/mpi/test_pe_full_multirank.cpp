/// @file tests/mpi/test_pe_full_multirank.cpp
/// @brief Phase 0.3 final / Phase B.2 -- multi-rank PE with bonds, angles,
///        dihedrals AND LJ + special-bonds exclusions, on a chain that
///        spans the rank boundary.
///
/// This is the first multi-rank test that exercises ALL the moving parts
/// of the PE force pipeline together:
///   * tag-based bonded force resolution across rank boundary
///   * LJ ghosts populated by HaloSend
///   * tag-based special-bonds exclusions on multi-rank ghosts
///
/// Without correct exclusions the bonded pairs (r ≈ 1.5–4 Å, well below
/// σ = 4.01 Å) would create LJ repulsion in the +1e3 eV range and blow
/// up immediately. Passing the test means LJ correctly skips the bonded
/// 1-2 / 1-3 / 1-4 pairs even when the partner atom is a halo ghost on
/// another rank.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
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
        1.73 * kKcalPerMolToEv,  // C0  (MSU paper, Tschopp 2010)
        -4.49 * kKcalPerMolToEv, // C1
        0.776 * kKcalPerMolToEv, // C2
        6.99 * kKcalPerMolToEv,  // C3
        0.0,                     // C4
    };
}

cpu::Frame make_chain_along_x(int n_carbons, double box_x, double box_yz) {
    cpu::Frame frame(n_carbons);
    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);
    const double chain_len = (n_carbons - 1) * dx_step;
    const double x_start = (box_x - chain_len) / 2.0;
    const double y0 = box_yz / 2.0;
    const double z0 = box_yz / 2.0;
    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x_start + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_carbons - 1; ++k)
        frame.add_bond(k, k + 1, 0);
    for (int k = 0; k < n_carbons - 2; ++k)
        frame.add_angle(k, k + 1, k + 2, 0);
    for (int k = 0; k < n_carbons - 3; ++k)
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    frame.build_special_list();
    frame.simbox = ATV::cpu::Box::orthorhombic(box_x, box_yz, box_yz);
    return frame;
}

} // namespace

TEST(PeFullMultiRank, ChainCrossingWithLjExclusions) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank specific test; got world_size=" << world_size;
    }

    // Box must satisfy slab > 2 * LJ_cutoff (= 21 Å for cutoff 10.5) so an
    // atom isn't shipped to the same neighbour through both ±x faces (the
    // dims=2 self-doubling problem documented in test_mpi_lj_nve).
    auto frame = make_chain_along_x(20, 50.0, 30.0);

    // OPLS-AA UA CH2: σ = 4.01 Å, ε = 0.118 kcal/mol.
    // r_cut = 10.5 Å, halo cutoff matches.
    const double lj_eps = 0.00512;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{lj_cut}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{lj_cut}, -10)
        .add(cpu::PairLJCut{lj_eps, lj_sig, lj_cut}.set_special_bonds(0.0, 0.0,
                                                                      0.0))
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 200);

    const double e1 = frame.meta.get(ATV::kEnergy)
                      + frame.meta.get(ATV::kKinetic); // already global
    EXPECT_FALSE(std::isnan(e1)) << "global E went NaN -- LJ likely saw a "
                                    "bonded ghost partner that exclusions "
                                    "failed to suppress";
    EXPECT_LT(std::abs(e1), 1e3)
        << "global energy " << e1
        << " is wildly large -- bonded pair was not excluded from LJ";

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x went NaN";
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
