/// @file tests/mpi/test_comm_brick.cpp
/// @brief MPI unit tests for `cpu::comm::CommBrick` 8-atom brick goldens.
///
/// Hard-coded: 8 atoms, cubic box L=20 Å, shell 6 Å, ATV_FORCE_DIMS=2,1,1.
/// n1: n_ghost==0. n2: n_ghost==8; reverse sum fx==8 eV/Å; exchange keeps 8
/// owners. No MD.run.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

struct ForcedDims {
    explicit ForcedDims(const char* grid) {
        ::setenv("ATV_FORCE_DIMS", grid, 1);
    }
    ~ForcedDims() { ::unsetenv("ATV_FORCE_DIMS"); }
};

cpu::Frame make_8_atom_cube() {
    cpu::Frame frame(8);
    int idx = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz) {
                frame.span(kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * 10.0;
                frame.span(kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * 10.0;
                frame.span(kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * 10.0;
                frame.span(kAtom_mass)[idx] = 1.0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

TEST(TestCommBrick, N1CutoffSixHasNoGhosts) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1)
        GTEST_SKIP() << "n1 golden";

    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    comm->set_ghost_cutoff(6.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    EXPECT_EQ(backend.context().md.dd->n_ghost, 0);
    comm->set_ghost_cutoff(0.0);
    comm->borders(lctx);
    EXPECT_EQ(backend.context().md.dd->n_ghost, 0);
    backend.shutdown();
}

TEST(TestCommBrick, N2BrickGoldens) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 2)
        GTEST_SKIP() << "n2 ATV_FORCE_DIMS=2,1,1 golden";

    ForcedDims dims("2,1,1");
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);
    EXPECT_EQ(dd->dims[0], 2);
    EXPECT_EQ(dd->dims[1], 1);
    EXPECT_EQ(dd->dims[2], 1);

    comm->set_ghost_cutoff(6.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    EXPECT_EQ(dd->n_ghost, 8);

    for (int i = 0; i < dd->n_local; ++i)
        frame.span(kAtom_fx)[i] = 0.0;
    for (int g = 0; g < dd->n_ghost; ++g)
        frame.span(kAtom_fx)[dd->n_local + g] = 1.0;
    comm->reverse_comm(lctx);
    double sum_fx = 0.0;
    for (int i = 0; i < dd->n_local; ++i)
        sum_fx += static_cast<double>(frame.span(kAtom_fx)[i]);
    EXPECT_NEAR(sum_fx, 8.0, 1e-12);

    comm->exchange(lctx);
    int n_local = dd->n_local;
    int n_owners = 0;
    MPI_Allreduce(&n_local, &n_owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(n_owners, 8);

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
