/// @file tests/mpi/test_halo_refresh_charges.cpp
/// @brief Multi-rank correctness of `comm::HaloRefresh{Field::charges}`.
///
/// The constant-potential gap this pins (map §1/§6 P0): a charge solver
/// rewrites OWNED charges after the step's forward halo, so ghost charges go
/// stale mid-step. `HaloRefresh{charges}` re-sends the owner `charge` column
/// into the existing ghost slots. Checks (2 ranks):
///   - after HaloSend (full rebuild / pack_border), ghost q equals the owner
///     value (charge rides kColBorder, not the per-step kColComm payload);
///   - after mutating owned q (simulated Q solve) the ghosts are stale;
///   - after HaloRefresh{charges}, every ghost q equals the owner's new
///     value EXACTLY (transport is memcpy — bitwise);
///   - the charge refresh does not touch ghost positions;
///   - single rank: strict no-op.
/// Owner values are a deterministic function of the atom tag, so no
/// cross-rank lookup is needed to know the expected ghost value.

#include <gtest/gtest.h>

#include <mpi.h>

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo_exchange.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

constexpr TagInt kTagBase = TagInt{1};

double q_initial(TagInt tag) { return 0.01 * static_cast<double>(tag); }
double q_solved(TagInt tag) {
    return -0.5 * q_initial(tag) + 0.123; // "the Q solve" — any new value
}

cpu::Frame make_uniform_grid(int n_per_side, double box_len) {
    const int n = n_per_side * n_per_side * n_per_side;
    cpu::Frame frame(n);
    const double step = box_len / static_cast<double>(n_per_side);
    int idx = 0;
    for (int ix = 0; ix < n_per_side; ++ix)
        for (int iy = 0; iy < n_per_side; ++iy)
            for (int iz = 0; iz < n_per_side; ++iz) {
                frame.span(kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * step;
                frame.span(kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * step;
                frame.span(kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * step;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 1;
                const TagInt tag = kTagBase + TagInt(idx);
                frame.span(kAtom_tag)[idx] = tag;
                frame.span(kAtom_charge)[idx] =
                    static_cast<StorageReal>(q_initial(tag));
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    return frame;
}

} // namespace

TEST(HaloRefreshCharges, GhostChargesMatchOwnerAfterRefresh) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP() << "multi-rank transport test; skipped on single rank";

    // 64 atoms on a 4x4x4 grid in a 10 Å box; cutoff 2.0 Å captures the
    // inner-face atoms of each slab (same geometry as test_halo_exchange).
    auto frame = make_uniform_grid(4, 10.0);

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;

    cpu::comm::HaloSend fwd{2.0};
    cpu::BuildContext bctx{backend.context(), frame};
    fwd.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    fwd.launch(lctx);
    ASSERT_GT(dd->n_ghost, 0) << "no ghosts: the test would be vacuous";

    // Baseline: the first HaloSend is a full rebuild, so pack_border ships
    // charge (kColBorder). Per-step kColComm is x/y/z only.
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const TagInt tag = frame.span(kAtom_tag)[row];
        EXPECT_EQ(static_cast<double>(frame.span(kAtom_charge)[row]),
                  q_initial(tag));
    }

    // "Charge solve": rewrite OWNED charges only — ghosts now stale.
    for (int i = 0; i < dd->n_local; ++i) {
        const TagInt tag = frame.span(kAtom_tag)[i];
        frame.span(kAtom_charge)[i] = static_cast<StorageReal>(q_solved(tag));
    }
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const TagInt tag = frame.span(kAtom_tag)[row];
        EXPECT_NE(static_cast<double>(frame.span(kAtom_charge)[row]),
                  q_solved(tag))
            << "ghost row " << row << " already updated? test is vacuous";
    }

    // Record ghost positions: the charge refresh must not touch them.
    std::vector<double> gx(static_cast<std::size_t>(dd->n_ghost));
    std::vector<double> gz(static_cast<std::size_t>(dd->n_ghost));
    for (int g = 0; g < dd->n_ghost; ++g) {
        gx[static_cast<std::size_t>(g)] =
            static_cast<double>(frame.span(kAtom_x)[dd->n_local + g]);
        gz[static_cast<std::size_t>(g)] =
            static_cast<double>(frame.span(kAtom_z)[dd->n_local + g]);
    }

    cpu::comm::HaloRefresh refresh{cpu::comm::HaloRefresh::Field::charges};
    refresh.build(bctx);
    refresh.launch(lctx);

    // Transport is memcpy: ghost q must equal the owner's value EXACTLY.
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const TagInt tag = frame.span(kAtom_tag)[row];
        EXPECT_EQ(static_cast<double>(frame.span(kAtom_charge)[row]),
                  q_solved(tag))
            << "stale ghost charge at row " << row;
        EXPECT_EQ(static_cast<double>(frame.span(kAtom_x)[row]),
                  gx[static_cast<std::size_t>(g)])
            << "charge refresh touched ghost x at row " << row;
        EXPECT_EQ(static_cast<double>(frame.span(kAtom_z)[row]),
                  gz[static_cast<std::size_t>(g)])
            << "charge refresh touched ghost z at row " << row;
    }

    backend.shutdown();
}

TEST(HaloRefreshCharges, SingleRankNoOp) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1)
        GTEST_SKIP() << "single-rank no-op baseline";

    auto frame = make_uniform_grid(2, 10.0);
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    cpu::comm::HaloRefresh refresh{cpu::comm::HaloRefresh::Field::charges};
    cpu::BuildContext bctx{backend.context(), frame};
    refresh.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    EXPECT_NO_THROW(refresh.launch(lctx));
    for (int i = 0; i < frame.nlocal(); ++i) {
        const TagInt tag = frame.span(kAtom_tag)[i];
        EXPECT_EQ(static_cast<double>(frame.span(kAtom_charge)[i]),
                  q_initial(tag));
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
