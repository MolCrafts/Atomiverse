/// @file benchmarks/cuda/bench_multi_gpu_force.cu
/// @brief Manual forward-halo plus combined LJ/bonded force benchmark.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/communication/multi_gpu.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ATV;
namespace {
cpu::Frame make_grid(double L) {
    const int s = 4, n = s * s * s;
    cpu::Frame f(n);
    f.box.set_cell(Mat3::cubic(L));
    double sp = L / s;
    int idx = 0;
    for (int ix = 0; ix < s; ++ix)
        for (int iy = 0; iy < s; ++iy)
            for (int iz = 0; iz < s; ++iz) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * sp;
                f.span(kAtom_y)[idx] = (iy + 0.5) * sp;
                f.span(kAtom_z)[idx] = (iz + 0.5) * sp;
                f.span(kAtom_vx)[idx] = 0.01 * idx;
                f.span(kAtom_mass)[idx] = 1.0;
                f.span(kAtom_type)[idx] = 0;
                f.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    return f;
}

cpu::Frame make_bonded_grid(double L) {
    cpu::Frame f = make_grid(L);
    const int s = 4;
    // add_bond/add_angle resolve tags through the tag→local map; build it first.
    f.rebuild_tag_index(0, f.nlocal());
    auto tag_of = [&](int ix, int iy, int iz) {
        return (ix * s + iy) * s + iz;
    };
    for (int iy = 0; iy < s; ++iy)
        for (int iz = 0; iz < s; ++iz) {
            for (int ix = 0; ix < s - 1; ++ix)
                f.add_bond(tag_of(ix, iy, iz), tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 1; ++ix)
                f.add_angle(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                            tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 2; ++ix)
                f.add_dihedral(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                               tag_of(ix + 1, iy, iz), tag_of(ix + 2, iy, iz),
                               0);
        }
    f.build_special_list();
    return f;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc > 3)
        throw std::invalid_argument(
            "usage: bench_multi_gpu_force [iterations=100] [warmup=10]");
    const int iterations = argc > 1 ? std::stoi(argv[1]) : 100;
    const int warmup = argc > 2 ? std::stoi(argv[2]) : 10;
    if (iterations < 1 || warmup < 0)
        throw std::invalid_argument(
            "positive iterations and nonnegative warmup required");
    int visible = 0;
    ATV_CUDA_CHECK(cudaGetDeviceCount(&visible));
    if (visible < 2)
        throw std::runtime_error(
            "bench_multi_gpu_force requires at least 2 visible GPUs");

    ATV::cuda::md::MultiGpu mg(2);
    auto g = make_bonded_grid(12.0);
    const std::vector<double> K{100.0}, r0{2.0}, aK{50.0}, aT{109.5};
    const std::vector<std::array<double, 5>> dC{{1.0, 0.5, -0.3, 0.2, 0.1}};
    mg.partition(g);
    mg.setup_force_engine(5.0, 0.1, 2.0);
    mg.setup_bonded(K, r0, aK, aT, dC);
    mg.enable_special_bonds();
    mg.reneighbor();
    for (int i = 0; i < warmup; ++i) { // warmup
        mg.forward_comm();
        mg.compute_forces();
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        mg.forward_comm();   // per-step position halo (forward, packed)
        mg.compute_forces(); // LJ + bonded + reverse halo (packed)
    }
    auto t1 = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    if (!std::isfinite(us) || us <= 0)
        throw std::runtime_error(
            "non-finite or non-positive benchmark duration");
    std::printf("forward_comm+compute_forces us_per_iteration=%.9g gpus=%d "
                "iterations=%d warmup=%d\n",
                us, mg.count(), iterations, warmup);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_multi_gpu_force: %s\n", e.what());
    return 1;
}
