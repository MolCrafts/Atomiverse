#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

TEST(Frame, HostCreate) {
    cpu::Frame frame(100);
    EXPECT_EQ(frame.nlocal(), 100);
    EXPECT_GE(frame.capacity(), 100);
    // The atoms block (x, mass, ... columns) is allocated to `capacity` rows.
    EXPECT_EQ(frame.block("atoms").rows(), frame.capacity());
}

TEST(Frame, Mat3Box) {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(10.0);
    EXPECT_DOUBLE_EQ(frame.box.length(0), 10.0);
    EXPECT_DOUBLE_EQ(frame.box.length(1), 10.0);
    EXPECT_DOUBLE_EQ(frame.box.length(2), 10.0);
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
    frame.box = cpu::Box::cubic(10.0);

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
    frame.box = cpu::Box::cubic(20.0);

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

// to_cpu reconstructs the physical coordinate rather than exporting only the
// force-position high part. This case contains sub-FP32 motion at a large offset.
TEST(Frame, ExportPreservesMotionPrecision) {
    cpu::Frame frame(3);
    frame.box = cpu::Box::cubic(1000.123456789);
    const double x[] = {512.0 + 1e-7, 10.0 / 3.0, -3.141592653589793};
    const double v[] = {1e-10, -0.123456789012345, 1.0 / 7.0};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = x[2 - i];
        frame.span(kAtom_z)[i] = -x[i];
        frame.span(kAtom_vx)[i] = v[i];
        frame.span(kAtom_mass)[i] = 12.011;
        frame.span(kAtom_type)[i] = i + 6;
        frame.span(kAtom_tag)[i] = 100 + 7 * i;
    }
    auto device = cpu::to_cuda(frame);
    if constexpr (cuda::kMixedPrecision) {
        EXPECT_NE(device.atoms.x_correction, nullptr);
        EXPECT_NE(device.atoms.y_correction, nullptr);
        EXPECT_NE(device.atoms.z_correction, nullptr);
    } else {
        EXPECT_EQ(device.atoms.x_correction, nullptr);
    }
    cpu::Frame exported(3);
    cuda::to_cpu(device, exported);
    for (int i = 0; i < 3; ++i) {
        const double xtol = sizeof(cuda::motion_t) == 8 ? 2e-13 : 4e-5;
        const double vtol = sizeof(cuda::motion_t) == 8 ? 1e-16 : 1e-8;
        EXPECT_NEAR(exported.span(kAtom_x)[i], x[i], xtol);
        EXPECT_NEAR(exported.span(kAtom_vx)[i], v[i], vtol);
        EXPECT_NEAR(exported.span(kAtom_mass)[i], 12.011,
                    sizeof(cuda::motion_t) == 8 ? 1e-14 : 1e-6);
        EXPECT_EQ(exported.span(kAtom_type)[i], i + 6);
        EXPECT_EQ(exported.span(kAtom_tag)[i], 100 + 7 * i);
    }
    cuda::destroy(device);
}

TEST(Frame, EmptyTransfer) {
    cpu::Frame frame(0);
    frame.box = cpu::Box::cubic(10);
    auto device = cpu::to_cuda(frame);
    EXPECT_EQ(device.n_atoms(), 0);
    EXPECT_EQ(device.atoms.x, nullptr);
    cuda::to_cpu(device, frame);
    cuda::destroy(device);
}
