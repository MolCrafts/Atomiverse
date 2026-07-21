/// @file tests/mpi/test_pe_chain_crossing.cpp
/// @brief Phase 0.3 final -- a single PE chain that genuinely crosses a
///        rank boundary, with bonded forces on both sides.
///
/// Prior multi-rank PE tests rigged the geometry so each chain sat fully
/// inside one rank's slab. That config never exercised the cross-boundary
/// part of the bonded force pipeline. This test fixes that:
///
///   * 1 chain of 20 united-atom CH2 sites laid along x.
///   * Box 30 Å × 30 Å × 30 Å, dims = (2, 1, 1) → slab_x = 15 Å.
///   * Chain atoms span x ≈ 3 → 27.2 Å; rank 0 owns atoms whose x < 15,
///     rank 1 owns the rest. Roughly half-and-half.
///   * HaloSend cutoff = 5 Å so a dihedral that spans 3 bond
///     lengths can resolve every atom on at least one rank as either
///     owner or ghost.
///
/// Asserts the global energy stays finite and bounded over a 200-step NVE
/// run -- the chain's own bonded forces should hold it together across
/// the rank boundary.

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

// Chain laid along x: atom positions form an all-trans planar zigzag,
// with the chain axis pointing in +x. The y/z zigzag gives the angle/
// dihedral non-degenerate geometry; chain centre is placed so it spans
// the x = box_x/2 partition boundary.
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

TEST(PeChainCrossing, BondedNveHandlesChainAcrossBoundary) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank specific test; got world_size=" << world_size;
    }

    auto frame = make_chain_along_x(20, 30.0, 30.0);

    // Pair cutoff = 5 Å on its own covers the dihedral span (3 × 1.529 ≈
    // 4.6 Å). The spec-A.5 auto-extension added in `build()` makes this
    // redundant here; the explicit 5.0 preserves the historical baseline
    // so a future regression in the auto-extension cannot silently mask a
    // cross-rank bonded miss through this test.
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{5.0}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy)
        + frame.meta.get(ATV::kKinetic); // already global via Reduce
    ASSERT_FALSE(std::isnan(e0)) << "step-1 global energy is NaN";

    sim.run(frame, 199);
    const double e1 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    EXPECT_FALSE(std::isnan(e1)) << "global energy went NaN";
    EXPECT_LT(std::abs(e1), 1e3)
        << "global energy " << e1 << " is wildly large -- cross-boundary "
        << "bonded resolve / halo skin probably wrong (E0=" << e0 << ")";

    // No NaN in any owner position.
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x went NaN";
    }
}

// Same chain + same bonded FF, but the user passes a PAIR cutoff well
// below the dihedral span (0.1 Å). The auto-extension in build() scans
// bonds / angles / dihedrals and must add enough ghost skin on top so
// every cross-rank tuple still resolves. Without the refactor this run
// would drop bonded tuples at the boundary and blow up within a handful
// of steps.
TEST(PeChainCrossing, BondedSkinAutoExtendsWithSmallPairCutoff) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "2-rank specific test; got world_size=" << world_size;
    }

    auto frame = make_chain_along_x(20, 30.0, 30.0);

    const double tiny_pair_cutoff = 0.1; // << bond length 1.529 Å
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::pre_force)
        .add(cpu::comm::HaloSend{tiny_pair_cutoff}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{bond_K_eV()}, {kBondR0}})
        .add(cpu::AngleHarmonic{{angle_K_eV()}, {kAngleTheta0}})
        .add(cpu::DihedralMultiHarmonic{{dihedral_Cn_eV()}});
    sim.build();

    sim.run(frame, 50);
    const double e =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    EXPECT_FALSE(std::isnan(e)) << "global energy went NaN";
    EXPECT_LT(std::abs(e), 1e3)
        << "global energy " << e << " blew up -- auto-extension failed "
        << "to cover the cross-rank dihedral span";

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
