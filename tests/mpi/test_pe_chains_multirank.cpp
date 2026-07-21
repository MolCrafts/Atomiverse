/// @file tests/mpi/test_pe_chains_multirank.cpp
/// @brief Phase 0.3 MVP -- multi-rank PE bonded forces.
///
/// Two PE chains (C20H42 each) placed entirely inside two adjacent ranks'
/// slabs along x, so no bond/angle/dihedral straddles the slab boundary
/// and the partition-time topology filter (msu-pe-validation Phase 0.3
/// MVP) can keep every owned tuple in its rank's local subset. Bonds
/// using ghost atoms are deferred to the Phase 0.3-extended milestone.
///
/// Drives a short bonded-only NVE run and asserts the global energy stays
/// finite (no NaN, no runaway). Cross-validation against the single-rank
/// run on the same chains is left to a downstream commit because Phase 2
/// has no momentum-zero initialisation across ranks.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

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

// Build a frame with two C20H42 chains, each sitting fully inside one
// half of the box along x. With 2 MPI ranks and dims = (2,1,1) the
// partition assigns chain 0 (centred at x = box_x/4) to rank 0 and chain 1
// (centred at x = 3·box_x/4) to rank 1.
cpu::Frame make_two_chain_box(double box_x, double box_yz) {
    const int n_carbons = 20;
    const int n_atoms = 2 * n_carbons;
    cpu::Frame frame(n_atoms);

    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dy_step = kBondR0 * std::cos(half_apex); // along y now
    const double dz_step = kBondR0 * std::sin(half_apex); // small z zigzag

    auto place_chain = [&](int base, double cx) {
        // Chain laid along y, centred at (cx, box_yz/2, box_yz/2).
        const double y0 = (box_yz - (n_carbons - 1) * dy_step) / 2.0;
        for (int i = 0; i < n_carbons; ++i) {
            frame.span(ATV::kAtom_x)[base + i] = cx;
            frame.span(ATV::kAtom_y)[base + i] = y0 + i * dy_step;
            frame.span(ATV::kAtom_z)[base + i] =
                box_yz / 2.0 + (i % 2 == 0 ? 0.0 : dz_step);
            frame.span(ATV::kAtom_mass)[base + i] = 14.027;
            frame.span(ATV::kAtom_type)[base + i] = 0;
        }
    };
    place_chain(0, box_x * 0.25); // owned by rank 0 (x in [0, box_x/2))
    place_chain(n_carbons,
                box_x * 0.75); // owned by rank 1 (x in [box_x/2, box_x))

    // Tiny opposite-sign kick per chain so we exercise the integrator
    // without driving atoms across the slab boundary in 200 steps.
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < n_carbons; ++i) {
            const double sign = (i % 2 == 0) ? +1.0 : -1.0;
            frame.span(ATV::kAtom_vz)[c * n_carbons + i] = sign * 1e-3;
        }
    }

    // Pre-allocate to TOTAL size for both chains, then fill all entries.
    // (xt::xtensor::resize wipes the existing data on grow, so per-chain
    // append doesn't work without manually preserving previous writes.)
    const int n_chains = 2;
    const int n_b_per = n_carbons - 1;
    const int n_a_per = n_carbons - 2;
    const int n_d_per = n_carbons - 3;
    const int n_b_tot = n_chains * n_b_per;
    const int n_a_tot = n_chains * n_a_per;
    const int n_d_tot = n_chains * n_d_per;

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int c = 0; c < n_chains; ++c) {
        const int base = c * n_carbons;
        for (int k = 0; k < n_b_per; ++k)
            frame.add_bond(base + k, base + k + 1, 0);
        for (int k = 0; k < n_a_per; ++k)
            frame.add_angle(base + k, base + k + 1, base + k + 2, 0);
        for (int k = 0; k < n_d_per; ++k)
            frame.add_dihedral(base + k, base + k + 1, base + k + 2,
                               base + k + 3, 0);
    }
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::orthorhombic(box_x, box_yz, box_yz);
    return frame;
}

double total_energy_global(const cpu::Frame& frame) {
    // Reduce already wrote the global PE/KE back into frame, so
    // every rank has the same value here.
    return frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
}

} // namespace

TEST(PeChainsMultiRank, BondedOnlyNveStable) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank specific test; got world_size=" << world_size;
    }

    auto frame = make_two_chain_box(40.0, 30.0);

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 200);

    const double e_global = total_energy_global(frame);
    EXPECT_FALSE(std::isnan(e_global)) << "global energy went NaN";
    EXPECT_LT(std::abs(e_global), 1e6)
        << "global energy " << e_global
        << " is wildly large -- topology filter or bonded kernel is broken";

    // Per-rank: owners must remain finite.
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
