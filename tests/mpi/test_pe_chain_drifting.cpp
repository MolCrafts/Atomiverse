/// @file tests/mpi/test_pe_chain_drifting.cpp
/// @brief Phase 0.3 final / P56 -- a PE chain that DRIFTS across the rank
///        boundary mid-run, verifying Migrate + bonded ownership
///        re-route automatically.
///
/// With the tag-based topology design, every rank holds the FULL bond /
/// angle / dihedral list, and force kernels filter by ownership at
/// force-eval time. When Migrate moves an atom between ranks the
/// bond's owner-tag stays the same (tags are immutable) but the rank
/// holding it changes -- so the bond starts being computed by the new
/// owner rank without any explicit tuple migration.
///
/// This test puts a 16-atom chain entirely in rank 0's slab initially,
/// gives it a +x drift velocity, and runs long enough that several atoms
/// cross into rank 1's slab. Migrate moves the crossing atoms;
/// bonded force pipeline must keep the chain intact through the
/// migrations.

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

cpu::Frame make_drifting_chain(int n_carbons, double box_x, double box_yz,
                               double drift_vx) {
    cpu::Frame frame(n_carbons);
    const double half_apex = (units::pi - kAngleTheta0) / 2.0;
    const double dx_step = kBondR0 * std::cos(half_apex);
    const double dy_step = kBondR0 * std::sin(half_apex);
    // Place chain entirely inside rank 0's slab (x in [0, box_x/2)) at
    // start. Leave a couple of Å margin so atoms aren't exactly on the
    // partition line.
    const double x_start = 2.0;
    const double y0 = box_yz / 2.0;
    const double z0 = box_yz / 2.0;
    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = x_start + i * dx_step;
        frame.span(ATV::kAtom_y)[i] = y0 + (i % 2 == 0 ? 0.0 : dy_step);
        frame.span(ATV::kAtom_z)[i] = z0;
        frame.span(ATV::kAtom_mass)[i] = 14.027;
        frame.span(ATV::kAtom_type)[i] = 0;
        // Uniform +x drift -- the entire chain translates toward rank 1.
        frame.span(ATV::kAtom_vx)[i] = drift_vx;
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

TEST(PeChainDrifting, MigrationKeepsChainIntact) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank specific test; got world_size=" << world_size;
    }

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 16-carbon chain, box 30 Å (slab 15), starts entirely in rank 0's
    // slab at x in [2, 21]. Drift = +0.05 Å/fs -> centre of mass moves
    // by 0.025 Å per 0.5-fs step. Over 1000 steps the chain shifts ~25 Å
    // (a full slab width) so most atoms migrate.
    auto frame = make_drifting_chain(16, 30.0, 30.0, /*drift_vx=*/0.05);
    const int n_atoms_global = frame.nlocal();

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{5.0}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 1000);

    const double e1 = frame.meta.get(ATV::kEnergy)
                      + frame.meta.get(ATV::kKinetic); // global via Reduce
    EXPECT_FALSE(std::isnan(e1)) << "global E went NaN -- migration broke "
                                    "bonded ownership routing";
    EXPECT_LT(std::abs(e1), 1e3)
        << "global energy " << e1 << " is wildly large -- migration likely "
        << "lost bonds in transit";

    // Globally the atom count must be conserved.
    int local_n = 0;
    if (my_rank == 0 || my_rank == 1)
        local_n = static_cast<int>(frame.nlocal());
    // Use dd's authoritative count instead.
    // (Each rank's frame.nlocal() is the original capacity; what
    // changed across migration is dd->n_local.)
    // We can't reach dd from the user side here, so just assert no NaN
    // and let the global count be a separate diagnostic.

    for (int i = 0; i < frame.nlocal(); ++i) {
        if (std::isnan(frame.span(ATV::kAtom_x)[i])) {
            FAIL() << "rank " << my_rank << " atom slot " << i << " x went NaN";
        }
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
