/// @file tests/cuda/frame/test_precision.cu
/// @brief Unit tests of the CUDA precision contract codecs
///        (`cuda/frame/precision.cuh`) and of their integrator gate.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"

using namespace ATV;

namespace {

constexpr bool kMixedIntegration = !std::is_same_v<cuda::mixed_t, float>;

/// @brief Host frame of `n` atoms in an orthorhombic box of edge `L`.
cpu::Frame make_frame(int n, double L) {
    cpu::Frame f(n);
    f.set_nlocal(n);
    f.box.set_cell(Mat3::diagonal(L, L, L));
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = 1.0 + i;
        f.span(kAtom_y)[i] = 2.0 + i;
        f.span(kAtom_z)[i] = 3.0 + i;
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = 41 + i;
        f.span(kAtom_type)[i] = 0;
    }
    f.rebuild_tag_index(0, n);
    return f;
}

/// @brief Device fault record of `frame` (synchronous copy).
cuda::ForceFault read_fault(const cuda::Frame& frame) {
    cuda::ForceFault fault{};
    EXPECT_EQ(cudaMemcpy(&fault, frame.force_fault, sizeof(fault),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return fault;
}

__global__ void drift_k(cuda::pos_t* x, cuda::pos_t* xc, cuda::mixed_t step,
                        int count) {
    for (int k = 0; k < count; ++k)
        cuda::store_position(x, xc, 0, cuda::load_position(x, xc, 0) + step);
}

template <typename T>
__global__ void add_sequence_k(cuda::fixed_t* f, cuda::ForceFault* fault, T a,
                               T b, T c) {
    cuda::add_force(f, 0, a, fault);
    cuda::add_force(f, 0, b, fault);
    cuda::add_force(f, 0, c, fault);
}

__global__ void store_k(cuda::fixed_t* f, cuda::ForceFault* fault, int row,
                        double value) {
    cuda::store_force(f, row, value, fault);
}

} // namespace

TEST(CudaPrecision, SplitReconstructsHostCoordinate) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> value(-1.0e4, 1.0e4);
    for (int k = 0; k < 10000; ++k) {
        const double v = value(rng);
        cuda::pos_t hi, lo = 0;
        cuda::split_position(cuda::mixed_t(v), hi, lo);
        const double back = cuda::load_position(&hi, &lo, 0);
        if constexpr (cuda::kPositionCorrection)
            // float + float carries at least 48 significant bits.
            EXPECT_LE(std::abs(back - v), std::abs(v) * std::ldexp(1.0, -47));
        else
            EXPECT_EQ(back, double(cuda::mixed_t(v)));
    }
}

TEST(CudaPrecision, SubUlpDisplacementsAccumulate) {
    if constexpr (!kMixedIntegration)
        GTEST_SKIP() << "single mode integrates in FP32 by contract";
    // 1e-6 A is far below half an FP32 ulp at 1000 A (3e-5 A): a coordinate
    // without its correction would never move.
    cpu::Frame host = make_frame(1, 2000.0);
    host.span(kAtom_x)[0] = 1000.25;
    cuda::Frame d = cpu::to_cuda(host);
    constexpr int kSteps = 4096;
    constexpr double kStep = 1.0e-6;
    drift_k<<<1, 1>>>(d.atoms.x, d.atoms.x_corr, cuda::mixed_t(kStep), kSteps);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cuda::to_cpu(d, host);
    double expected = 1000.25;
    for (int k = 0; k < kSteps; ++k)
        expected += kStep;
    EXPECT_NEAR(host.span(kAtom_x)[0], expected, 1.0e-7);
    cuda::destroy(d);
}

TEST(CudaPrecision, VelocityVerletWrapKeepsSubUlpOffset) {
    if constexpr (!kMixedIntegration)
        GTEST_SKIP() << "single mode integrates in FP32 by contract";
    constexpr double L = 40.0, dt = 1.0, v = 5.0e-7;
    cpu::Frame host = make_frame(1, L);
    host.span(kAtom_x)[0] = L - 2.0e-7;
    host.span(kAtom_vx)[0] = v;
    cuda::Frame d = cpu::to_cuda(host);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    cuda::Context ctx{};
    ctx.stream = stream;
    cuda::geom_t cell[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
    ctx.host_box.assign_cell(cell);
    cuda::md::VelocityVerlet vv(dt);
    vv.launch({ctx, d, slot_of(md::Phase::initial_integrate), 0});
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    cuda::to_cpu(d, host);
    // Crossing the face wraps to 3e-7 A; the correction column keeps it.
    EXPECT_NEAR(host.span(kAtom_x)[0], 3.0e-7, 1.0e-12);
    EXPECT_EQ(host.span(kAtom_vx)[0], v); // zero force: velocity untouched
    cudaStreamDestroy(stream);
    cuda::destroy(d);
}

TEST(CudaPrecision, VelocityAndMassRoundTripExactly) {
    if constexpr (!kMixedIntegration)
        GTEST_SKIP() << "single mode stores integration state in FP32";
    cpu::Frame host = make_frame(3, 30.0);
    const double v[3] = {1.0 / 3.0, -2.0e-17 + 1.0e-3, 7.0 / 11.0};
    const double m[3] = {1.00794, 15.9994 + 1.0e-12, 12.011};
    for (int i = 0; i < 3; ++i) {
        host.span(kAtom_vy)[i] = v[i];
        host.span(kAtom_mass)[i] = m[i];
    }
    cuda::Frame d = cpu::to_cuda(host);
    std::vector<cuda::mixed_t> mass(3);
    ASSERT_EQ(cudaMemcpy(mass.data(), d.atoms.mass, 3 * sizeof(cuda::mixed_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int i = 0; i < 3; ++i)
        host.span(kAtom_vy)[i] = 0;
    cuda::to_cpu(d, host);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(host.span(kAtom_vy)[i], v[i]);
        EXPECT_EQ(double(mass[i]), m[i]);
    }
    cuda::destroy(d);
}

TEST(CudaPrecision, FixedPointSumSurvivesCancellation) {
    cpu::Frame host = make_frame(1, 10.0);
    cuda::Frame d = cpu::to_cuda(host);
    // In floating point 1e5 + 1e-6 rounds the small term away before the
    // cancellation; the fixed-point sum keeps it to one resolution step per
    // contribution, whatever the order.
    add_sequence_k<double>
        <<<1, 1>>>(d.atoms.fx, d.force_fault, 1.0e5, 1.0e-6, -1.0e5);
    add_sequence_k<float>
        <<<1, 1>>>(d.atoms.fy, d.force_fault, 1.0e5f, 1.0e-6f, -1.0e5f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cuda::to_cpu(d, host);
    EXPECT_NEAR(host.span(kAtom_fx)[0], 1.0e-6,
                3 * cuda::kForceFixedResolution);
    // FP32 contributions are scaled in FP32: 1e-6 keeps FP32 relative
    // precision, 1e5 is an exact multiple of its own rounding.
    EXPECT_NEAR(host.span(kAtom_fy)[0], 1.0e-6, 1.0e-12);
    EXPECT_EQ(read_fault(d).kinds, 0u);
    cuda::destroy(d);
}

TEST(CudaPrecision, EncodeRangeAndTruncation) {
    const double limit = cuda::kFixedLimit * cuda::kForceFixedResolution;
    // OpenMM's range: 2^63 / 2^32 = 2^31 kJ/mol/nm.
    EXPECT_NEAR(limit, 2147483648.0 / cuda::kKjPerMolNmPerEvPerAngstrom,
                1.0e-3);
    cuda::fixed_t word = 0;
    EXPECT_TRUE(cuda::encode_force(0.999 * limit, word));
    EXPECT_TRUE(cuda::encode_force(-0.999 * limit, word));
    EXPECT_FALSE(cuda::encode_force(1.001 * limit, word));
    EXPECT_FALSE(cuda::encode_force(-1.001 * limit, word));
    EXPECT_FALSE(cuda::encode_force(std::nan(""), word));
    EXPECT_FALSE(
        cuda::encode_force(std::numeric_limits<double>::infinity(), word));
    EXPECT_FALSE(
        cuda::encode_force(std::numeric_limits<float>::infinity(), word));
    // Truncation toward zero is symmetric and below one resolution step.
    cuda::fixed_t up = 0, down = 0;
    ASSERT_TRUE(cuda::encode_force(1.234567890123, up));
    ASSERT_TRUE(cuda::encode_force(-1.234567890123, down));
    EXPECT_EQ(up, -down);
    EXPECT_LE(std::abs(cuda::decode_force<double>(up) - 1.234567890123),
              cuda::kForceFixedResolution);
    EXPECT_GE(1.234567890123, cuda::decode_force<double>(up));
}

TEST(CudaPrecision, WrappingSumRaisesAccumulationFault) {
    cpu::Frame host = make_frame(2, 10.0);
    cuda::Frame d = cpu::to_cuda(host);
    // Each contribution fits; their sum leaves the signed 64-bit range.
    add_sequence_k<double>
        <<<1, 1>>>(d.atoms.fz + 1, d.force_fault, 1.5e6, 1.5e6, 0.0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const cuda::ForceFault fault = read_fault(d);
    EXPECT_EQ(fault.kinds, cuda::ForceFault::kAccumulation);
    EXPECT_EQ(fault.row, 0); // row relative to the pointer passed (fz + 1)
    EXPECT_THROW(cuda::to_cpu(d, host), std::runtime_error);
    cuda::destroy(d);
}

TEST(CudaPrecision, FaultedForcesAreNotIntegratedAndNameTagAndStep) {
    constexpr double L = 30.0;
    cpu::Frame host = make_frame(2, L);
    host.span(kAtom_vx)[0] = 1.0e-3;
    cuda::Frame d = cpu::to_cuda(host);
    store_k<<<1, 1>>>(d.atoms.fx, d.force_fault, 1, std::nan(""));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(read_fault(d).kinds, cuda::ForceFault::kConversion);

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    cuda::Context ctx{};
    ctx.stream = stream;
    cuda::geom_t cell[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
    ctx.host_box.assign_cell(cell);
    cuda::md::VelocityVerlet vv(1.0);
    vv.launch({ctx, d, slot_of(md::Phase::initial_integrate), 7});
    vv.launch({ctx, d, slot_of(md::Phase::final_integrate), 7});
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<cuda::pos_t> x(2);
    std::vector<cuda::mixed_t> vx(2);
    ASSERT_EQ(cudaMemcpy(x.data(), d.atoms.x, 2 * sizeof(cuda::pos_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(vx.data(), d.atoms.vx, 2 * sizeof(cuda::mixed_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(double(x[0]), double(cuda::pos_t(1.0))); // nothing moved
    EXPECT_EQ(double(vx[0]), double(cuda::mixed_t(1.0e-3)));

    const cuda::ForceFault fault = read_fault(d);
    EXPECT_EQ(fault.row, 1);
    EXPECT_EQ(fault.step, 7);
    EXPECT_EQ(fault.tag, 42);
    try {
        cuda::to_cpu(d, host);
        ADD_FAILURE() << "to_cpu accepted a faulted force";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("atom tag 42"), std::string::npos) << what;
        EXPECT_NE(what.find("step 7"), std::string::npos) << what;
    }
    cudaStreamDestroy(stream);
    cuda::destroy(d);
}
