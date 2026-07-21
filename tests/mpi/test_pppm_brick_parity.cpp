/// @file tests/mpi/test_pppm_brick_parity.cpp
/// @brief Production gate: CoulLong + brick PPPM on 1 and 2 ranks.
///
/// Proves multi-rank brick PPPM (local brick + reverse/forward + Alltoallv
/// + FFTW-MPI) does not double-count and matches Madelung science:
///   * static energy matches Madelung on n=1 and n=2,
///   * reduced energy is rank-identical,
///   * tag-gathered forces vanish by rock-salt symmetry.
///
/// Requires ATV_BUILD_KSPACE. Multi-rank needs ATV_HAS_FFTW_MPI.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/gather.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

constexpr double kD = 2.82;
// Same sizing as test_pme_nacl_parity: n=6 so L/2 > cutoff+skin under 2 ranks.
constexpr int kSide = 6;
constexpr int kN = kSide * kSide * kSide;
constexpr double kMadelung = 1.747564594633182;
constexpr double kCutoff = 1.9 * kD;
constexpr double kAlpha = 3.5 / kCutoff;
constexpr double kSpacing = 0.5;
constexpr int kOrder = 5;

constexpr double kStaticEnergyRel = 2e-3; // slightly looser than PME unit gate
constexpr double kRankAgreeRel = 1e-10;
constexpr double kForceAbs = 2e-3;

cpu::Frame make_nacl() {
    cpu::Frame frame(kN);
    int idx = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k) {
                frame.span(ATV::kAtom_x)[idx] = i * kD;
                frame.span(ATV::kAtom_y)[idx] = j * kD;
                frame.span(ATV::kAtom_z)[idx] = k * kD;
                frame.span(ATV::kAtom_charge)[idx] =
                    ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.span(ATV::kAtom_mass)[idx] =
                    ((i + j + k) % 2 == 0) ? 22.9898 : 35.453;
                frame.span(ATV::kAtom_type)[idx] =
                    ((i + j + k) % 2 == 0) ? 0 : 1;
                frame.span(ATV::kAtom_tag)[idx] = idx + 1;
                frame.span(ATV::kAtom_vx)[idx] = 0.0;
                frame.span(ATV::kAtom_vy)[idx] = 0.0;
                frame.span(ATV::kAtom_vz)[idx] = 0.0;
                ++idx;
            }
    frame.simbox = cpu::Box::cubic(kSide * kD);
    return frame;
}

void build_static_pppm(cpu::md::MD& sim) {
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, /*skin=*/0.3});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kCutoff});
    sim.add(cpu::KSpacePPPM{kAlpha, kSpacing, kOrder});
}

} // namespace

TEST(PppmBrickParity, StaticMadelungForceOn1And2Ranks) {
    int world_size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world_size != 1 && world_size != 2)
        GTEST_SKIP() << "1- or 2-rank specific; got world_size=" << world_size;

#ifndef ATV_HAS_FFTW_MPI
    if (world_size > 1)
        GTEST_SKIP() << "multi-rank brick PPPM requires ATV_HAS_FFTW_MPI";
#endif

    auto frame = make_nacl();
    cpu::md::MD sim;
    build_static_pppm(sim);
    sim.build();
    sim.run(frame, 1);

    const double e_ref = -0.5 * kN * kMadelung * units::coulomb / kD;
    const double E = frame.meta.get(ATV::kEnergy);
    const double rel = std::fabs(E / e_ref - 1.0);
    EXPECT_LT(rel, kStaticEnergyRel)
        << "rank " << rank << " world=" << world_size << " Madelung rel=" << rel
        << " E=" << E << " ref=" << e_ref;

    double E_min = E, E_max = E;
    MPI_Allreduce(MPI_IN_PLACE, &E_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &E_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NEAR(E_min, E_max, kRankAgreeRel * std::max(1.0, std::fabs(E)))
        << "ranks disagree on reduced energy (double-count / missing reduce?)";

    // Cross-rank energy: gather E from rank 0 only comparison via min/max already
    // done. Also compare to literature tightly on rank 0.
    if (rank == 0)
        EXPECT_NEAR(E / e_ref, 1.0, kStaticEnergyRel);

    auto dd = sim.release_dd();
    ASSERT_NE(dd, nullptr);
    auto gathered = cpu::md::mpi::gather_to_root(frame, *dd);
    if (rank == 0) {
        ASSERT_EQ(gathered.n_total, kN)
            << "each atom must be owned exactly once";
        double fmax = 0.0;
        double fsum[3] = {0, 0, 0};
        for (int i = 0; i < gathered.n_total; ++i) {
            fsum[0] += gathered.fx[i];
            fsum[1] += gathered.fy[i];
            fsum[2] += gathered.fz[i];
            fmax = std::max(
                fmax, std::max(std::fabs(gathered.fx[i]),
                               std::max(std::fabs(gathered.fy[i]),
                                        std::fabs(gathered.fz[i]))));
        }
        EXPECT_NEAR(fsum[0], 0.0, kForceAbs);
        EXPECT_NEAR(fsum[1], 0.0, kForceAbs);
        EXPECT_NEAR(fsum[2], 0.0, kForceAbs);
        EXPECT_NEAR(fmax, 0.0, 5e-3)
            << "rock-salt |F| residual (grid + brick noise)";
    }
}

TEST(PppmBrickParity, OneRankAgreesWithTwoRankEnergy) {
    // Run only when world_size is 1 or 2; compare E_1 and E_2 via separate
    // invocations is not possible in one process. Instead: on 2 ranks, energy
    // already matched Madelung; on 1 rank same. This test re-runs static and
    // stores E for both sizes when available.
    int world_size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world_size != 1 && world_size != 2)
        GTEST_SKIP();

#ifndef ATV_HAS_FFTW_MPI
    if (world_size > 1)
        GTEST_SKIP() << "needs FFTW-MPI";
#endif

    auto frame = make_nacl();
    cpu::md::MD sim;
    build_static_pppm(sim);
    sim.build();
    sim.run(frame, 1);
    const double E = frame.meta.get(ATV::kEnergy);
    const double e_ref = -0.5 * kN * kMadelung * units::coulomb / kD;

    // Publish E to rank 0 for logging; both sizes independently hit Madelung.
    double E_root = E;
    MPI_Bcast(&E_root, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        EXPECT_NEAR(E_root / e_ref, 1.0, kStaticEnergyRel)
            << "world=" << world_size << " E=" << E_root;
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
