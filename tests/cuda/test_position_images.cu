/// @file cuda/test_position_images.cu
/// @brief Direct coordinate wrapping and optimizer rollback preserve images.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#endif
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
namespace gpu = ATV::cuda;
constexpr double kTolerance = sizeof(gpu::motion_t) == 8 ? 5e-13 : 3e-6;

struct State {
    ATV::cpu::Frame host;
    gpu::Frame device;
    gpu::Context runtime;
    ATV::LoopMetrics metrics;

    explicit State(int n) : host(n) {
        host.box = ATV::cpu::Box::cubic(10);
        for (int i = 0; i < n; ++i)
            host.span(ATV::kAtom_mass)[i] = 1;
        device = ATV::cpu::to_cuda(host);
        runtime.loop.metrics = &metrics;
        for (int d = 0; d < 3; ++d)
            runtime.host_motion_box.cell[4 * d] = 10;
        runtime.host_motion_box.refresh();
        runtime.host_box = gpu::box_cast<gpu::real_t>(runtime.host_motion_box);
    }
    ~State() { gpu::destroy(device); }

    void upload(const std::array<double, 3>* positions,
                const std::array<std::int8_t, 3>* images) {
        for (int i = 0; i < host.nlocal(); ++i) {
            host.span(ATV::kAtom_x)[i] = positions[i][0];
            host.span(ATV::kAtom_y)[i] = positions[i][1];
            host.span(ATV::kAtom_z)[i] = positions[i][2];
        }
        gpu::destroy(device);
        device = ATV::cpu::to_cuda(host);
        ATV_CUDA_CHECK(cudaMemcpy(device.atoms.image, images,
                                  3 * host.nlocal() * sizeof(std::int8_t),
                                  cudaMemcpyHostToDevice));
    }

    void download() {
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        gpu::to_cpu(device, host);
    }
};

__global__ void wrap(gpu::Frame::Atoms atoms, gpu::MotionBox box, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        gpu::wrap_position(atoms.x, atoms.y, atoms.z, atoms.x_correction,
                           atoms.y_correction, atoms.z_correction, atoms.image,
                           i, box,
                           gpu::load_position(atoms.x, atoms.x_correction, i),
                           gpu::load_position(atoms.y, atoms.y_correction, i),
                           gpu::load_position(atoms.z, atoms.z_correction, i));
}

void expect_state(State& state, int i, const std::array<double, 3>& position,
                  const std::array<int, 3>& image) {
    EXPECT_NEAR(state.host.span(ATV::kAtom_x)[i], position[0], kTolerance);
    EXPECT_NEAR(state.host.span(ATV::kAtom_y)[i], position[1], kTolerance);
    EXPECT_NEAR(state.host.span(ATV::kAtom_z)[i], position[2], kTolerance);
    std::array<std::int8_t, 3> actual;
    ATV_CUDA_CHECK(cudaMemcpy(actual.data(), state.device.atoms.image + 3 * i,
                              sizeof(actual), cudaMemcpyDeviceToHost));
    for (int d = 0; d < 3; ++d)
        EXPECT_EQ(int(actual[d]), image[d]);
}
} // namespace

TEST(PositionImages, WrapHandlesFacesMultipleCrossingsAndNonperiodicAxes) {
    State state(5);
    const double below =
        double(std::nextafter(gpu::motion_t(10), gpu::motion_t(0)));
    const std::array<double, 3> positions[] = {
        {10, -25, 30},
        {-0.25, 15, -20.5},
        {31, -0.5, -1},
        {below, 10, 0},
        {-double(std::numeric_limits<gpu::motion_t>::epsilon()), 0, 0}};
    const std::array<std::int8_t, 3> images[] = {
        {2, -7, 3}, {-4, 8, 5}, {0, 127, 0}, {127, -128, -128}, {0, 0, 0}};
    state.upload(positions, images);
    state.runtime.host_motion_box.periodic[1] = false;
    wrap<<<1, 32>>>(state.device.atoms, state.runtime.host_motion_box, 5);
    ATV_CUDA_CHECK(cudaGetLastError());
    // Wrapping already-wrapped physical positions must not count twice.
    wrap<<<1, 32>>>(state.device.atoms, state.runtime.host_motion_box, 5);
    ATV_CUDA_CHECK(cudaGetLastError());
    state.download();
    expect_state(state, 0, {0, -25, 0}, {3, -7, 6});
    expect_state(state, 1, {9.75, 15, 9.5}, {-5, 8, 2});
    expect_state(state, 2, {1, -0.5, 9}, {3, 127, -1});
    expect_state(state, 3, {below, 10, 0}, {127, -128, -128});
    expect_state(state, 4, {below, 0, 0}, {-1, 0, 0});
}

TEST(PositionImages, WrapTriclinicCountsLatticeCrossings) {
    State state(1);
    // Cell columns are lattice vectors. Fractional (1.25, -.5, 2.75)
    // maps to (14.25, 3.25, 33); wrapped (.25, .5, .75) maps to (4.25, 7.25, 9).
    const std::array<double, 3> positions[] = {{14.25, 3.25, 33}};
    const std::array<std::int8_t, 3> images[] = {{4, -2, 0}};
    state.upload(positions, images);
    auto& box = state.runtime.host_motion_box;
    box.cell[0] = 10;
    box.cell[1] = 2;
    box.cell[2] = 1;
    box.cell[3] = 0;
    box.cell[4] = 10;
    box.cell[5] = 3;
    box.cell[6] = 0;
    box.cell[7] = 0;
    box.cell[8] = 12;
    box.refresh();
    wrap<<<1, 32>>>(state.device.atoms, box, 1);
    ATV_CUDA_CHECK(cudaGetLastError());
    state.download();
    expect_state(state, 0, {4.25, 7.25, 9}, {5, -3, 2});
    state.upload(positions, images);
    box.periodic[1] = false;
    wrap<<<1, 32>>>(state.device.atoms, box, 1);
    ATV_CUDA_CHECK(cudaGetLastError());
    state.download();
    expect_state(state, 0, {2.25, -2.75, 9}, {5, -2, 2});
}

TEST(PositionImages, OverflowAndMissingStorageFailBeforeStateMutation) {
    gpu::MotionBox box;
    box.cell[0] = box.cell[4] = box.cell[8] = 10;
    box.refresh();
    gpu::real_t x = 2, y = 3, z = 4, xc = 0, yc = 0, zc = 0;
    std::int8_t image[] = {127, 0, 0};
    EXPECT_THROW(
        gpu::wrap_position(&x, &y, &z, &xc, &yc, &zc, image, 0, box, 10, 3, 4),
        std::overflow_error);
    EXPECT_EQ(int(image[0]), 127);
    EXPECT_EQ(x, 2);
    image[0] = -128;
    EXPECT_THROW(gpu::wrap_position(&x, &y, &z, &xc, &yc, &zc, image, 0, box,
                                    -0.25, 3, 4),
                 std::overflow_error);
    EXPECT_EQ(int(image[0]), -128);
    EXPECT_THROW(gpu::wrap_position(&x, &y, &z, &xc, &yc, &zc, nullptr, 0, box,
                                    20, 3, 4),
                 std::overflow_error);
    EXPECT_THROW(gpu::wrap_position(
                     &x, &y, &z, &xc, &yc, &zc, image, 0, box,
                     std::numeric_limits<gpu::motion_t>::infinity(), 3, 4),
                 std::overflow_error);
}

#ifdef ATV_HAS_MINIMIZER
TEST(PositionImages, LbfgsRollbackRestoresOriginImagesBeforeContractedTrial) {
    State state(1);
    const std::array<double, 3> positions[] = {{9.925, 1, 1}};
    const std::array<std::int8_t, 3> images[] = {{4, -3, 2}};
    state.upload(positions, images);
    const gpu::real_t force = 1;
    ATV_CUDA_CHECK(cudaMemcpy(state.device.atoms.fx, &force, sizeof(force),
                              cudaMemcpyHostToDevice));
    gpu::minimizer::LBFGS optimizer(2, 0.1);
    optimizer.line_search(1e-4, 0.5, 1e-8);
    optimizer.build({state.runtime, state.device});
    const auto search = ATV::slot_of(ATV::minimizer::Phase::search_direction);
    const auto check = ATV::slot_of(ATV::minimizer::Phase::convergence_check);
    optimizer.launch({state.runtime, state.device, search, 0});
    state.download();
    expect_state(state, 0, {0.025, 1, 1}, {5, -3, 2});
    const double rejected_energy = 1;
    ATV_CUDA_CHECK(cudaMemcpy(&state.device.scalars->energy, &rejected_energy,
                              sizeof(rejected_energy), cudaMemcpyHostToDevice));
    optimizer.launch({state.runtime, state.device, check, 0});
    optimizer.launch({state.runtime, state.device, search, 1});
    state.download();
    expect_state(state, 0, {9.975, 1, 1}, {4, -3, 2});
    // Reject again: the same original image must be restored, rather than
    // accumulating either of the previous trial's crossings.
    optimizer.launch({state.runtime, state.device, search, 2});
    state.download();
    expect_state(state, 0, {9.95, 1, 1}, {4, -3, 2});
    optimizer.destroy({state.runtime, state.device});
}
#endif
