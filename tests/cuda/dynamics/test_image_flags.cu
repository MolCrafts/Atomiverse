/// @file tests/cuda/dynamics/test_image_flags.cu
/// @brief CUDA wraps keep r + H·image (LAMMPS Domain::pbc), and a tilting
///        integrator keeps the plan on general-cell kernel dispatch.
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/driver/backend.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

constexpr double kTol = std::is_same_v<ATV::cuda::pos_t, float> ? 1e-5 : 1e-12;

double unwrapped(const cpu::Frame& f, int i, int a) {
    double u = a == 0   ? f.span(kAtom_x)[i]
               : a == 1 ? f.span(kAtom_y)[i]
                        : f.span(kAtom_z)[i];
    for (int b = 0; b < 3; ++b)
        u +=
            f.box.cell()(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
    return u;
}

// One velocity-Verlet drift with zero force: r ← r + v dt, then the wrap.
void drift_once(cpu::Frame& host) {
    ATV::cuda::Frame frame = cpu::to_cuda(host);
    ATV::cuda::Context context;
    ATV::cuda::geom_t cell[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell[3 * i + j] = host.box.cell()(i, j);
    context.host_box.assign_cell(cell);
    ATV::cuda::md::VelocityVerlet vv(1.0);
    vv.launch({context, frame, slot_of(md::Phase::initial_integrate), 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    ATV::cuda::to_cpu(frame, host);
    ATV::cuda::destroy(frame);
}

cpu::Frame movers(const cpu::Box& box) {
    cpu::Frame f(2);
    // Row 0 leaves through +x, row 1 through -y and +z.
    f.span(kAtom_x)[0] = 9.95;
    f.span(kAtom_y)[0] = 4.0;
    f.span(kAtom_z)[0] = 3.0;
    f.span(kAtom_vx)[0] = 0.1;
    f.span(kAtom_x)[1] = 5.0;
    f.span(kAtom_y)[1] = 0.02;
    f.span(kAtom_z)[1] = 7.95;
    f.span(kAtom_vy)[1] = -0.1;
    f.span(kAtom_vz)[1] = 0.1;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = box;
    return f;
}

} // namespace

TEST(CudaImageFlags, VelocityVerletWrapCountsOrthorhombicImages) {
    cpu::Frame host = movers(cpu::Box::orthorhombic(10.0, 9.0, 8.0));
    double expected[2][3];
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            expected[i][a] = unwrapped(host, i, a)
                             + (a == 0   ? host.span(kAtom_vx)[i]
                                : a == 1 ? host.span(kAtom_vy)[i]
                                         : host.span(kAtom_vz)[i]);
    drift_once(host);
    EXPECT_EQ(host.span(kAtom_image)(0, 0), 1);
    EXPECT_EQ(host.span(kAtom_image)(1, 1), -1);
    EXPECT_EQ(host.span(kAtom_image)(1, 2), 1);
    EXPECT_NEAR(host.span(kAtom_x)[0], 0.05, kTol);
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            EXPECT_NEAR(unwrapped(host, i, a), expected[i][a], kTol)
                << "row " << i << " axis " << a;
}

TEST(CudaImageFlags, VelocityVerletWrapCountsTriclinicImages) {
    Mat3 cell = Mat3::diagonal(10.0, 9.0, 8.0);
    cell(0, 1) = 2.0;  // xy
    cell(1, 2) = -1.0; // yz
    cpu::Frame host = movers(cpu::Box(cell));
    double expected[2][3];
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            expected[i][a] = unwrapped(host, i, a)
                             + (a == 0   ? host.span(kAtom_vx)[i]
                                : a == 1 ? host.span(kAtom_vy)[i]
                                         : host.span(kAtom_vz)[i]);
    drift_once(host);
    int moved = 0;
    for (int i = 0; i < 2; ++i)
        for (int b = 0; b < 3; ++b)
            moved += std::abs(int(host.span(kAtom_image)(i, b)));
    EXPECT_GT(moved, 0);
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            EXPECT_NEAR(unwrapped(host, i, a), expected[i][a], kTol)
                << "row " << i << " axis " << a;
}

TEST(CudaImageFlags, ShearingSllodKeepsGeneralCellDispatch) {
    // An orthorhombic cell that an xy SLLOD run will tilt: the host-side
    // geometry specialization must be the general one from the first step.
    for (int direction : {0, 5}) {
        cpu::Frame host(4);
        for (int i = 0; i < 4; ++i) {
            host.span(kAtom_x)[i] = 1.0 + i;
            host.span(kAtom_mass)[i] = 1.0;
        }
        host.box = cpu::Box::cubic(10.0);
        ATV::cuda::md::NvtSllod sllod(300.0, 100.0, 1.0, 1e-5, direction);
        PlanBuilder<ATV::cuda::Component> builder;
        builder.add({slot_of(md::Phase::initial_integrate), 0, &sllod});
        auto plan = builder.build(ATV::slot_count<md::Phase>());
        ATV::cuda::Backend backend;
        backend.init();
        backend.load(plan);
        backend.materialize(host);
        EXPECT_EQ(backend.context().host_box.ortho, direction < 3)
            << "direction " << direction;
        backend.shutdown();
    }
}
