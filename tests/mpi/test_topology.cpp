/// @file tests/mpi/test_topology.cpp
/// @brief Phase 2.1 topology + spatial-partition tests.
///
/// Launched under `mpirun -n N`. Runs at N=1, 2, 4, 8 per CTest config.
/// Verifies:
///   - Cartesian topology is built (cart_comm non-null on N > 1).
///   - dims factorizes world size, coords are within [0, dims[d]).
///   - All six face_neighbors resolve to live ranks under PBC.
///   - Partitioned `n_local` sums to the original global atom count.
///   - Every atom tag ends up on exactly one rank (no double-count, no drop).
///   - Owned atom positions actually lie in this rank's [sublo, subhi) slab.

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

// Deterministic uniformly-distributed positions in [0, box_len)^3.
cpu::Frame make_uniform_grid(int n_per_side, double box_len) {
    const int n = n_per_side * n_per_side * n_per_side;
    cpu::Frame frame(n);
    const double step = box_len / static_cast<double>(n_per_side);
    int idx = 0;
    for (int ix = 0; ix < n_per_side; ++ix)
        for (int iy = 0; iy < n_per_side; ++iy)
            for (int iz = 0; iz < n_per_side; ++iz) {
                frame.span(ATV::kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * step;
                frame.span(ATV::kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * step;
                frame.span(ATV::kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * step;
                frame.span(ATV::kAtom_mass)[idx] = 1.0;
                ++idx;
            }
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// ── Cartesian topology: dims factorizes world, cart_comm non-null ────────

TEST(MpiTopology, CartCommAndDimsFactorWorld) {
    cpu::Runtime backend;
    backend.init();

    auto& ctx = backend.context();
    auto* dd = ctx.md.dd;
    ASSERT_NE(dd, nullptr);

    const int world_size = ctx.mpi_size;
    const int product = dd->dims[0] * dd->dims[1] * dd->dims[2];
    EXPECT_EQ(product, world_size);

    if (world_size > 1) {
        EXPECT_NE(dd->cart_comm, MPI_COMM_NULL);
        for (int d = 0; d < 3; ++d) {
            EXPECT_GE(dd->coords[d], 0);
            EXPECT_LT(dd->coords[d], dd->dims[d]);
        }
    }

    backend.shutdown();
}

// ── Face neighbors under PBC should all be valid ranks ──────────────────

TEST(MpiTopology, FaceNeighborsValidUnderPBC) {
    cpu::Runtime backend;
    backend.init();

    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);

    if (backend.context().mpi_size > 1) {
        // With periodic topology on all axes, every face neighbor must be a
        // live rank; MPI_PROC_NULL would indicate a non-periodic gap.
        for (int f = 0; f < 6; ++f) {
            EXPECT_NE(dd->face_neighbors[f], MPI_PROC_NULL)
                << "face " << f << " has MPI_PROC_NULL under periodic PBC";
            EXPECT_GE(dd->face_neighbors[f], 0);
            EXPECT_LT(dd->face_neighbors[f], backend.context().mpi_size);
        }
    }

    backend.shutdown();
}

// ── Partition: n_local sums to n_total, no double-count, no drop ────────

TEST(MpiTopology, PartitionConservesTotalAtomCount) {
    auto frame = make_uniform_grid(4, 10.0); // 64 atoms, box 10x10x10
    const int n_total = frame.nlocal();

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame); // runs partition_atoms on multi-rank

    auto* dd = backend.context().md.dd;
    const int n_local = dd->n_local;

    int global_n = 0;
    MPI_Allreduce(&n_local, &global_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(global_n, n_total)
        << "sum of n_local across ranks (" << global_n
        << ") does not match total atom count (" << n_total << ")";

    backend.shutdown();
}

// ── Every tag lives on exactly one rank ─────────────────────────────────

TEST(MpiTopology, PartitionAssignsEveryTagOnce) {
    auto frame = make_uniform_grid(4, 10.0);
    const int n_total = frame.nlocal();

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;

    // Each rank reports which tags it owns.
    std::vector<int> tag_counts(static_cast<std::size_t>(n_total), 0);
    for (int i = 0; i < dd->n_local; ++i) {
        const std::int64_t t = frame.span(ATV::kAtom_tag)[i];
        if (t >= 0 && t < n_total) {
            tag_counts[static_cast<std::size_t>(t)] = 1;
        }
    }
    std::vector<int> global_counts(static_cast<std::size_t>(n_total), 0);
    MPI_Allreduce(tag_counts.data(), global_counts.data(), n_total, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    for (int t = 0; t < n_total; ++t) {
        EXPECT_EQ(global_counts[static_cast<std::size_t>(t)], 1)
            << "tag " << t << " is owned by "
            << global_counts[static_cast<std::size_t>(t)]
            << " ranks (expected 1)";
    }

    backend.shutdown();
}

// ── Owned atoms actually lie in this rank's [sublo, subhi) slab ────────

TEST(MpiTopology, OwnedAtomsInSlab) {
    auto frame = make_uniform_grid(4, 10.0);

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;

    // For a uniform-grid initial placement, all atom positions are in
    // [0, box_len); no PBC-wrap is needed, so sublo <= pos < subhi must hold.
    for (int i = 0; i < dd->n_local; ++i) {
        const double x = frame.span(ATV::kAtom_x)[i];
        const double y = frame.span(ATV::kAtom_y)[i];
        const double z = frame.span(ATV::kAtom_z)[i];
        EXPECT_GE(x, dd->sublo[0]) << "atom " << i << " x out of slab";
        EXPECT_LT(x, dd->subhi[0]) << "atom " << i << " x out of slab";
        EXPECT_GE(y, dd->sublo[1]) << "atom " << i << " y out of slab";
        EXPECT_LT(y, dd->subhi[1]) << "atom " << i << " y out of slab";
        EXPECT_GE(z, dd->sublo[2]) << "atom " << i << " z out of slab";
        EXPECT_LT(z, dd->subhi[2]) << "atom " << i << " z out of slab";
    }

    backend.shutdown();
}

// ── Custom main that initializes MPI before/after GoogleTest ────────────
//
// OpenMPI 5.0.x on macOS occasionally segfaults inside its own BTL
// teardown during `MPI_Finalize` even after the user code has cleanly
// completed all communication. This can produce a non-zero exit code for
// an otherwise passing test; CTest invocations of MPI tests in this tree
// use `PASS_REGULAR_EXPRESSION="\\[  PASSED  \\]"` to read the GoogleTest
// summary line instead of the prterun exit code.

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
