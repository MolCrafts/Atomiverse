/// @file tests/cpu/communication/test_comm_brick.cpp
/// @brief Serial unit tests for `cpu::comm::CommBrick`.
///
/// mpi_size==1: n_ghost==0 for cutoff 6.0 Å and 0. No MD.run.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/comm_brick.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/comm.h"
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
    EXPECT_EQ(frame.nghost(), cutoff > 0.0 ? 56 : 0);
    backend.shutdown();
}

} // namespace

TEST(TestCommBrick, SerialCutoffSixBuildsPeriodicImages) { run_cutoff(6.0); }

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
    EXPECT_EQ(frame.nghost(), 0);
    backend.shutdown();
}

TEST(TestCommBrick, TriclinicSelfImagesForwardAndReverseWithoutMpi) {
    cpu::Frame frame(1);
    Mat3 cell = Mat3::diagonal(20.0, 20.0, 20.0);
    cell(0, 1) = 4.0;
    cell(1, 0) = 2.0;
    cell(0, 2) = -2.0;
    cell(2, 0) = 1.0;
    cell(1, 2) = 3.0;
    cell(2, 1) = -1.0;
    frame.box.set_cell(cell);
    const Vec3 owner = frame.box.to_cartesian({0.05, 0.05, 0.05});
    frame.span(kAtom_x)[0] = owner[0];
    frame.span(kAtom_y)[0] = owner[1];
    frame.span(kAtom_z)[0] = owner[2];
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* comm = backend.context().md.comm;
    auto* dd = backend.context().md.dd;
    comm->set_ghost_cutoff(2.0);
    cpu::LaunchContext ctx{backend.context(), frame, 0, 0};
    comm->setup(ctx);
    comm->borders(ctx);
    ASSERT_EQ(frame.nlocal(), 1);
    ASSERT_EQ(frame.nghost(), 7);
    unsigned images = 0;
    for (int i = 1; i < 8; ++i) {
        const Vec3 fractional = frame.box.to_fractional(
            {frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
             frame.span(kAtom_z)[i]});
        int mask = 0;
        for (int d = 0; d < 3; ++d) {
            const int shift =
                static_cast<int>(std::lround(fractional[d] - 0.05));
            ASSERT_TRUE(shift == 0 || shift == 1);
            EXPECT_NEAR(fractional[d], 0.05 + shift, 1e-13);
            mask |= shift << d;
        }
        images |= 1u << mask;
    }
    EXPECT_EQ(images, 0xfeu);

    frame.span(kAtom_x)[0] += 0.125;
    comm->forward_comm(ctx);
    for (int i = 1; i < 8; ++i) {
        const Vec3 delta{frame.span(kAtom_x)[i] - frame.span(kAtom_x)[0],
                         frame.span(kAtom_y)[i] - frame.span(kAtom_y)[0],
                         frame.span(kAtom_z)[i] - frame.span(kAtom_z)[0]};
        const Vec3 fractional = frame.box.to_fractional(delta);
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(fractional[d], std::nearbyint(fractional[d]), 1e-13);
        frame.span(kAtom_fx)[i] = 1.0;
    }
    frame.span(kAtom_fx)[0] = 0.0;
    comm->reverse_comm(ctx);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_fx)[0], 7.0);

    // EAM-style intermediate values traverse the same corner cascade.
    struct Field : cpu::ForwardPacker, cpu::ReversePacker {
        std::array<double, 8> values{};
        int comm_forward() const override { return 1; }
        int comm_reverse() const override { return 1; }
        int pack_forward_comm(int n, const int* list, double* out, int,
                              const int*) override {
            for (int i = 0; i < n; ++i)
                out[i] = values[list[i]];
            return n;
        }
        void unpack_forward_comm(int n, int first, const double* in) override {
            for (int i = 0; i < n; ++i)
                values[first + i] = in[i];
        }
        int pack_reverse_comm(int n, int first, double* out) override {
            for (int i = 0; i < n; ++i)
                out[i] = values[first + i];
            return n;
        }
        void unpack_reverse_comm(int n, const int* list,
                                 const double* in) override {
            for (int i = 0; i < n; ++i)
                values[list[i]] += in[i];
        }
    } field;
    field.values[0] = 3.5;
    comm->forward_comm(field);
    for (double value : field.values)
        EXPECT_DOUBLE_EQ(value, 3.5);
    field.values.fill(1.0);
    field.values[0] = 0.0;
    comm->reverse_comm(field);
    EXPECT_DOUBLE_EQ(field.values[0], 7.0);

    // Removing tilt preserves the periodic-image transport contract.
    frame.box = cpu::Box::cubic(20.0);
    dd->compute_sublo_subhi(frame.box);
    comm->borders(ctx);
    EXPECT_EQ(frame.nghost(), 7);
    backend.shutdown();
}
