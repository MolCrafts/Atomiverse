#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/md/init.cuh"
#include "atomiverse/facade/stage.h"

using namespace ATV;

TEST(Frame, HostCreate) {
    cpu::Frame frame(100);
    EXPECT_EQ(frame.nlocal(), 100);
    EXPECT_GE(frame.capacity(), 100);
    EXPECT_EQ(static_cast<int>(frame.span(kAtom_x).size()), frame.capacity());
    EXPECT_EQ(static_cast<int>(frame.span(kAtom_mass).size()),
              frame.capacity());
}

TEST(Frame, Mat3Box) {
    cpu::Frame frame(2);
    frame.simbox = cpu::Box::cubic(10.0);
    EXPECT_DOUBLE_EQ(frame.simbox.length(0), 10.0);
    EXPECT_DOUBLE_EQ(frame.simbox.length(1), 10.0);
    EXPECT_DOUBLE_EQ(frame.simbox.length(2), 10.0);
}

TEST(Frame, ToDeviceAndBack) {
    cpu::Frame frame(4);
    auto x = frame.span(kAtom_x);
    auto mass = frame.span(kAtom_mass);
    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;
    x[3] = 4.0;
    for (int i = 0; i < 4; ++i)
        mass[i] = 1.0;
    frame.simbox = cpu::Box::cubic(10.0);

    auto d_frame = cpu::to_cuda(frame);
    EXPECT_NE(d_frame.atoms.d_size, nullptr);

    for (int i = 0; i < 4; ++i)
        x[i] = 0.0;
    cuda::to_cpu(d_frame, frame);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[0], 1.0);
    EXPECT_DOUBLE_EQ(frame.span(kAtom_x)[3], 4.0);

    cuda::destroy(d_frame);
}

TEST(Frame, TopologyRoundTrip) {
    cpu::Frame frame(4);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto mass = frame.span(kAtom_mass);
    for (int i = 0; i < 4; ++i) {
        x[i] = 1.53 * i;
        y[i] = 0.0;
        z[i] = 0.0;
        mass[i] = 14.0;
    }
    frame.simbox = cpu::Box::cubic(20.0);

    // Per-atom topology (LAMMPS layout): the owner atom holds each
    // bond/angle/dihedral; cpu::to_cuda() re-flattens to device tables.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_bond(0, 1, 0);
    frame.add_bond(1, 2, 0);
    frame.add_bond(2, 3, 0);
    frame.add_angle(0, 1, 2, 0);
    frame.add_angle(1, 2, 3, 0);
    frame.add_dihedral(0, 1, 2, 3, 0);

    auto d_frame = cpu::to_cuda(frame);

    EXPECT_NE(d_frame.bonds.atomi, nullptr);
    EXPECT_NE(d_frame.bonds.n_bonds, nullptr);
    EXPECT_NE(d_frame.angles.atomi, nullptr);
    EXPECT_NE(d_frame.angles.n_angles, nullptr);
    EXPECT_NE(d_frame.dihedrals.atomi, nullptr);
    EXPECT_NE(d_frame.dihedrals.n_dihedrals, nullptr);
    EXPECT_NE(d_frame.scalars, nullptr);

    cuda::destroy(d_frame);
}
