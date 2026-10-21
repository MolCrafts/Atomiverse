/// @brief XyzRecorder::freeze_slot captures coordinates and their own cell.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/record/recorder.cuh"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>

namespace {
class SnapshotRecorder : public ATV::cuda::XyzRecorder {
  public:
    using XyzRecorder::d2h_slot;
    using XyzRecorder::freeze_slot;
    using XyzRecorder::on_build;
    using XyzRecorder::on_data_ready;
    using XyzRecorder::on_destroy;
    using XyzRecorder::XyzRecorder;
};
} // namespace

TEST(XyzRecorder, FreezeIncludesResidualAndOriginalCell) {
    constexpr char filename[] = "test-cuda-xyz-snapshot.xyz";
    constexpr double length = 512.000000123;
    constexpr double position = 100.0 + 1e-8;
    ATV::cpu::Frame host(1);
    host.box = ATV::cpu::Box::cubic(length);
    host.span(ATV::kAtom_x)[0] = position;
    host.span(ATV::kAtom_type)[0] = 6;
    auto device = ATV::cpu::to_cuda(host);
    cudaStream_t compute, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&compute));
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    ATV::cuda::Context runtime;
    runtime.stream = compute;
    runtime.defer_stream = defer;
    SnapshotRecorder recorder{filename};
    recorder.on_build({runtime, device});
    ATV::cuda::LaunchContext ctx{runtime, device, 0, 7};
    recorder.freeze_slot(ctx, 0);
    ATV_CUDA_CHECK(cudaStreamSynchronize(compute));
    // A later step changes both cell caches and the live device cell before
    // the callback runs. The captured step must still carry its original cell.
    ATV::cuda::prec_t later_cell[9] = {1024, 0, 0, 0, 1024, 0, 0, 0, 1024};
    ATV_CUDA_CHECK(cudaMemcpy(device.box, later_cell, sizeof(later_cell),
                              cudaMemcpyHostToDevice));
    for (int d = 0; d < 3; ++d) {
        runtime.host_box.cell[d * 4] = 1024;
        runtime.host_motion_box.cell[d * 4] = 1024;
    }
    recorder.d2h_slot(ctx, 0);
    ATV_CUDA_CHECK(cudaStreamSynchronize(defer));
    recorder.on_data_ready(ctx, 0);
    recorder.on_destroy({runtime, device});
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "1");
    std::getline(file, line);
    const auto start = line.find("Lattice=\"");
    ASSERT_NE(start, std::string::npos);
    std::istringstream cell(line.substr(start + 9));
    double written_length = 0;
    cell >> written_length;
    EXPECT_NEAR(written_length, length,
                sizeof(ATV::cuda::prec_t) == 8 ? 1e-10 : 4e-5);
    std::getline(file, line);
    std::istringstream atom(line);
    std::string species;
    double x = 0, y = 0, z = 0;
    atom >> species >> x >> y >> z;
    EXPECT_NEAR(x, position, sizeof(ATV::cuda::motion_t) == 8 ? 1e-11 : 2e-5);
    EXPECT_EQ(species, "C");
    file.close();
    std::remove(filename);
    ATV_CUDA_CHECK(cudaStreamDestroy(compute));
    ATV_CUDA_CHECK(cudaStreamDestroy(defer));
    ATV::cuda::destroy(device);
}
