/// @file tests/cuda/dynamics/test_fix_efield.cu
/// @brief cuda::md::FixEfield on a charged pair: F = qE, U = −Σ q E·r^u with
///        the triclinic unwrap, against the analytic values and the CPU class.
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/fix_efield.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

// Forces are 64-bit fixed point; positions are FP32 in the mixed/single
// modes, which bounds the unwrapped energy.
constexpr double kForceTol = 1e-9;
constexpr double kEnergyTol =
    std::is_same_v<ATV::cuda::pos_t, float> ? 1e-4 : 1e-11;

// Columns are lattice vectors: a = (20,0,0), b = (3,18,0), c = (2,-1.5,16).
Mat3 tilted_cell() {
    Mat3 cell = Mat3::diagonal(20.0, 18.0, 16.0);
    cell(0, 1) = 3.0;
    cell(0, 2) = 2.0;
    cell(1, 2) = -1.5;
    return cell;
}

cpu::Frame charged_pair() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 1.0;
    f.span(kAtom_y)[0] = 0.5;
    f.span(kAtom_z)[0] = 0.25;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 1.0;
    f.span(kAtom_z)[1] = 2.0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -0.5;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box(tilted_cell());
    // Atom 0 image (0, +1, -1); atom 1 image (2, 0, 0).
    f.span(kAtom_image)(0, 1) = static_cast<ATV::ImageInt>(1);
    f.span(kAtom_image)(0, 2) = static_cast<ATV::ImageInt>(-1);
    f.span(kAtom_image)(1, 0) = static_cast<ATV::ImageInt>(2);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kStress).fill(0);
    return f;
}

struct DeviceInput {
    ATV::cuda::Frame frame;
    ATV::cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        ATV::cuda::geom_t cell[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[3 * i + j] = source.box.cell()(i, j);
        context.host_box.assign_cell(cell);
    }
    ~DeviceInput() { ATV::cuda::destroy(frame); }
};

constexpr double kEx = 0.1, kEy = -0.2, kEz = 0.05;

} // namespace

TEST(CudaFixEfield, ChargedPairForceIsQTimesE) {
    cpu::Frame host = charged_pair();
    DeviceInput input(host);
    ATV::cuda::md::FixEfield field(kEx, kEy, kEz);
    field.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    const double q[2] = {1.0, -0.5};
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(host.span(kAtom_fx)[i], q[i] * kEx, kForceTol);
        EXPECT_NEAR(host.span(kAtom_fy)[i], q[i] * kEy, kForceTol);
        EXPECT_NEAR(host.span(kAtom_fz)[i], q[i] * kEz, kForceTol);
    }
    // No virial (LAMMPS fix_modify virial is off by default).
    for (int c = 0; c < 6; ++c)
        EXPECT_EQ(host.meta.get(kStress)[c], 0.0);
}

TEST(CudaFixEfield, EnergyUsesTriclinicUnwrapAndMatchesCpu) {
    cpu::Frame host = charged_pair();
    DeviceInput input(host);
    ATV::cuda::md::FixEfield field(kEx, kEy, kEz);
    field.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);

    // r0^u = (1,0.5,0.25) + b - c = (2, 20, -15.75);
    // r1^u = (3,1,2) + 2a = (43, 1, 2).
    const double u0 = -1.0 * (kEx * 2.0 + kEy * 20.0 + kEz * -15.75);
    const double u1 = 0.5 * (kEx * 43.0 + kEy * 1.0 + kEz * 2.0);
    EXPECT_NEAR(host.meta.get(kEnergy), u0 + u1, kEnergyTol);

    cpu::Frame reference = charged_pair();
    cpu::Context cpu_ctx;
    cpu::md::FixEfield cpu_field(kEx, kEy, kEz);
    cpu_field.launch({cpu_ctx, reference, 0, 0});
    EXPECT_NEAR(reference.meta.get(kEnergy), u0 + u1, 1e-12);
    EXPECT_NEAR(host.meta.get(kEnergy), reference.meta.get(kEnergy),
                kEnergyTol);
}

TEST(CudaFixEfield, RequestsGateForceAndEnergyIndependently) {
    cpu::Frame host = charged_pair();
    DeviceInput input(host);
    ATV::cuda::md::FixEfield field(kEx, kEy, kEz);
    ATV::cuda::LaunchContext ctx{input.context, input.frame, 0, 0};
    ctx.forces_requested = false;
    ctx.energy_requested = false;
    field.launch(ctx);
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(input.frame, host);
    EXPECT_EQ(host.meta.get(kEnergy), 0.0);
    EXPECT_EQ(host.span(kAtom_fx)[0], 0.0);
    // A field change must change the captured-launch key.
    const auto epoch = field.graph_epoch();
    field.Ez = 0.06;
    EXPECT_NE(field.graph_epoch(), epoch);
}
