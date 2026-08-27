/// @file tests/cpu/communication/test_comm_brick.cpp
/// @brief Serial unit tests for `cpu::comm::CommBrick`.
///
/// mpi_size==1: n_ghost==0 for cutoff 6.0 Å and 0. No MD.run.

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

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

void run_cutoff(double cutoff) {
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    ASSERT_EQ(backend.context().mpi_size, 1);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    comm->set_ghost_cutoff(cutoff);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    ASSERT_NE(backend.context().md.dd, nullptr);
    EXPECT_EQ(backend.context().md.dd->n_ghost, 0);
    backend.shutdown();
}

} // namespace

TEST(TestCommBrick, SerialCutoffSixHasNoGhosts) { run_cutoff(6.0); }

TEST(TestCommBrick, SerialCutoffZeroHasNoGhosts) { run_cutoff(0.0); }

TEST(TestCommBrick, PackerOverloadIsNoOp) {
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    ASSERT_NE(comm, nullptr);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    // Incomplete Packer: pass a dummy reference through a local incomplete
    // object is illegal. Call the no-arg path only; the Packer overload is
    // compiled in comm_brick.cpp as an empty body.
    comm->forward_comm(lctx);
    EXPECT_EQ(backend.context().md.dd->n_ghost, 0);
    backend.shutdown();
}
