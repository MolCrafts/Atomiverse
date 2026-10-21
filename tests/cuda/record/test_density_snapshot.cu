/// @file cuda/record/test_density_snapshot.cu
/// @brief DensityRecorder::launch freezes matching geometry and density slots.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/position.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/record/density_recorder.cuh"
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gpu = ATV::cuda;

__global__ void shift_x(gpu::Frame::Atoms atoms, gpu::motion_t distance) {
    const int i = threadIdx.x;
    if (i < 2)
        gpu::store_position(atoms.x, atoms.x_correction, i,
                            gpu::load_position(atoms.x, atoms.x_correction, i)
                                + distance);
}

struct Directory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / ("atv-density-snapshot-"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct Snapshot {
    double origin_x;
    int atomic_numbers[2];
    double atom_x[2];
    std::vector<double> values;
};

Snapshot read_snapshot(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("missing density snapshot: " + path.string());
    std::string line;
    std::getline(input, line);
    std::getline(input, line);
    Snapshot snapshot{};
    int count;
    double y, z;
    input >> count >> snapshot.origin_x >> y >> z;
    if (count != 2)
        throw std::runtime_error("unexpected density atom count");
    for (int d = 0; d < 3; ++d) {
        double x;
        input >> count >> x >> y >> z;
    }
    for (int i = 0; i < 2; ++i) {
        double charge;
        input >> snapshot.atomic_numbers[i] >> charge >> snapshot.atom_x[i] >> y
            >> z;
    }
    double value;
    while (input >> value)
        snapshot.values.push_back(value);
    if (snapshot.values.size() != 3 * 4 * 5)
        throw std::runtime_error("incomplete density snapshot");
    return snapshot;
}
} // namespace

TEST(DensitySnapshot, LaunchFreezesResidualCoordinatesAndDensityTogether) {
    ATV::cpu::Frame host(2);
    for (int i = 0; i < 2; ++i) {
        host.span(ATV::kAtom_x)[i] = 100.0 + 1e-8 + 0.74 * i;
        host.span(ATV::kAtom_type)[i] = 1;
    }
    const double initial = host.span(ATV::kAtom_x)[0];
    auto device = ATV::cpu::to_cuda(host);
    cudaStream_t main, defer;
    ATV_CUDA_CHECK(cudaStreamCreate(&main));
    ATV_CUDA_CHECK(cudaStreamCreate(&defer));
    gpu::Context runtime;
    runtime.stream = main;
    runtime.defer_stream = defer;
    // Building the source only supplies a matrix allocation to this recorder
    // unit. There is no SCF iteration, force evaluation, or MD driver.
    gpu::scf::SCF source(1, 0.1, 1e-8, 1);
    source.build({runtime, device});
    Directory output;
    const auto prefix = output.path / "density.cube";
    gpu::DensityRecorder recorder(prefix.string(), source, 3, 4, 5, 1.0);
    recorder.build({runtime, device});
    const gpu::prec_t first[] = {1, 0, 0, 1};
    const gpu::prec_t second[] = {2, 0, 0, 2};
    const gpu::prec_t later[] = {3, 0, 0, 3};
    const auto matrix = source.workspace().P;
    ATV_CUDA_CHECK(cudaMemcpyAsync(matrix, first, sizeof(first),
                                   cudaMemcpyHostToDevice, main));
    recorder.launch({runtime, device, 0, 0});

    constexpr double shift = 2e-8;
    shift_x<<<1, 2, 0, main>>>(device.atoms, gpu::motion_t(shift));
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpyAsync(matrix, second, sizeof(second),
                                   cudaMemcpyHostToDevice, main));
    recorder.launch({runtime, device, 0, 1});

    // Change every live input again before draining the recorded snapshots.
    shift_x<<<1, 2, 0, main>>>(device.atoms, gpu::motion_t(shift));
    ATV_CUDA_CHECK(cudaGetLastError());
    ATV_CUDA_CHECK(cudaMemcpyAsync(matrix, later, sizeof(later),
                                   cudaMemcpyHostToDevice, main));
    const int changed_species[] = {8, 8};
    ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.type, changed_species,
                                   sizeof(changed_species),
                                   cudaMemcpyHostToDevice, main));
    recorder.destroy({runtime, device});
    ATV_CUDA_CHECK(cudaStreamSynchronize(main));

    const auto a = read_snapshot(output.path / "density.step-0.cube");
    const auto b = read_snapshot(output.path / "density.step-1.cube");
    constexpr double factor = ATV::units::kAngstromInBohr;
    const double start = double(gpu::motion_t(initial));
    const double next = double(gpu::motion_t(start) + gpu::motion_t(shift));
    EXPECT_NEAR(a.atom_x[0], double(ATV::StorageReal(start)) * factor, 1e-12);
    EXPECT_NEAR(b.atom_x[0], double(ATV::StorageReal(next)) * factor, 1e-12);
    EXPECT_NEAR(b.origin_x - a.origin_x, (next - start) * factor, 1e-12);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(a.atomic_numbers[i], 1);
        EXPECT_EQ(b.atomic_numbers[i], 1);
    }
    const double first_sum =
        std::accumulate(a.values.begin(), a.values.end(), 0.0);
    const double second_sum =
        std::accumulate(b.values.begin(), b.values.end(), 0.0);
    EXPECT_GT(first_sum, 0.0);
    EXPECT_NEAR(second_sum, 2 * first_sum, 5e-5 * first_sum);
    source.destroy({runtime, device});
    gpu::destroy(device);
    ATV_CUDA_CHECK(cudaStreamDestroy(defer));
    ATV_CUDA_CHECK(cudaStreamDestroy(main));
}
