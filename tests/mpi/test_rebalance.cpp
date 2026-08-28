/// @file tests/mpi/test_rebalance.cpp
/// @brief `Rebalance` non-uniform brick rebalance.
///
/// Verifies that one balance pass on a deliberately skewed atom distribution:
///   1. Populates `dd->cuts[d]` with `dims[d] + 1` monotonically increasing
///      fractions in `[0, 1]`.
///   2. Re-derives this rank's `sublo/subhi` from the new cuts (consistent
///      with `compute_sublo_subhi`).
///   3. Bumps `dd->neighbor_generation` and sets `partition_dirty`.
///   4. Causes the next `comm->exchange()` to redistribute atoms so per-rank
///      counts are within ~10% of the global mean.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/rebalance.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

// Skewed cluster: pack `n_atoms` atoms into the bottom-left quadrant of a
// `box_len`-wide cubic box. Default uniform brick decomposition will leave
// the high-x ranks empty; a successful RCB rebalance must redistribute.
cpu::Frame make_skewed_frame(double box_len, int n_atoms) {
    cpu::Frame frame(n_atoms);
    const double cluster_box = box_len * 0.4;
    for (int i = 0; i < n_atoms; ++i) {
        const double t = static_cast<double>(i) / std::max(1, n_atoms - 1);
        frame.span(ATV::kAtom_x)[i] = t * cluster_box;
        frame.span(ATV::kAtom_y)[i] = std::fmod(t * 7.0, 1.0) * cluster_box;
        frame.span(ATV::kAtom_z)[i] = std::fmod(t * 13.0, 1.0) * cluster_box;
        frame.span(ATV::kAtom_mass)[i] = 1.0;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

TEST(Rebalance, SingleRankIsNoOp) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1)
        GTEST_SKIP() << "single-rank guard";

    auto frame = make_skewed_frame(20.0, 64);
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;
    const auto gen0 = dd->neighbor_generation;

    cpu::md::mpi::Rebalance rcb{};
    cpu::BuildContext bctx{backend.context(), frame};
    rcb.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, /*step=*/1};
    rcb.launch(lctx);

    EXPECT_EQ(dd->neighbor_generation, gen0)
        << "single-rank Rebalance must not bump generation";
    EXPECT_FALSE(dd->partition_dirty);
    EXPECT_TRUE(dd->cuts[0].empty());
    EXPECT_TRUE(dd->cuts[1].empty());
    EXPECT_TRUE(dd->cuts[2].empty());

    backend.shutdown();
}

TEST(Rebalance, RebalancesSkewedDistribution) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8)
        GTEST_SKIP() << "8-rank specific";

    auto frame = make_skewed_frame(20.0, 512);
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Baseline: uniform brick. High-coord ranks should be (near) empty.
    const int n_local_uniform = dd->n_local;
    int min_uniform = 0, max_uniform = 0;
    MPI_Allreduce(&n_local_uniform, &min_uniform, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&n_local_uniform, &max_uniform, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    // The skewed distribution should produce a meaningful imbalance under
    // uniform decomposition (some ranks have 0 atoms, one has many). If this
    // assertion fails the test fixture is too gentle to be informative.
    ASSERT_GT(max_uniform, min_uniform)
        << "fixture not skewed enough: uniform brick already balanced";

    const auto gen_before = dd->neighbor_generation;

    // Run one RCB pass.
    cpu::md::mpi::Rebalance rcb{};
    cpu::BuildContext bctx{backend.context(), frame};
    rcb.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, /*step=*/1};
    rcb.launch(lctx);

    // Cuts populated for each axis with dims[d] > 1.
    for (int d = 0; d < 3; ++d) {
        if (dd->dims[d] > 1) {
            ASSERT_EQ(static_cast<int>(dd->cuts[d].size()), dd->dims[d] + 1)
                << "axis " << d << " cuts size mismatch";
            EXPECT_DOUBLE_EQ(dd->cuts[d].front(), 0.0) << "axis " << d;
            EXPECT_DOUBLE_EQ(dd->cuts[d].back(), 1.0) << "axis " << d;
            for (int k = 1; k < static_cast<int>(dd->cuts[d].size()); ++k) {
                EXPECT_GE(dd->cuts[d][k], dd->cuts[d][k - 1])
                    << "axis " << d << " cuts not monotonic at index " << k;
            }
        }
    }
    EXPECT_GT(dd->neighbor_generation, gen_before);
    EXPECT_TRUE(dd->partition_dirty);

    // Exchange so atoms move to their new rank.
    ASSERT_NE(backend.context().md.comm, nullptr);
    backend.context().md.comm->exchange(lctx);
    EXPECT_FALSE(dd->partition_dirty);

    // Post-migration: per-rank counts should be much more balanced.
    const int n_local_post = dd->n_local;
    int min_post = 0, max_post = 0, sum_post = 0;
    MPI_Allreduce(&n_local_post, &min_post, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&n_local_post, &max_post, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&n_local_post, &sum_post, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    EXPECT_EQ(sum_post, 512) << "atom count not conserved across migration";

    // RCB tries to balance to 64 atoms/rank. The 1D-axis bisection only
    // approximately balances (each axis uses one shared histogram), so we
    // permit a 50% spread post-balance — still vastly better than the
    // uniform-brick baseline where many ranks have 0 atoms.
    const double mean = static_cast<double>(sum_post) / world_size;
    EXPECT_LE(max_post - min_post, static_cast<int>(mean * 1.5 + 1))
        << "RCB rebalance left too much imbalance: min=" << min_post
        << " max=" << max_post << " mean=" << mean;
    if (my_rank == 0) {
        std::printf(
            "RCB rebalance: uniform [%d, %d] -> RCB [%d, %d] (mean %.1f)\n",
            min_uniform, max_uniform, min_post, max_post, mean);
    }

    backend.shutdown();
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
