/// @file benchmarks/cuda/bench_nonbonded_style.cu
/// @brief Single-GPU LJ NVE wall timing for the OpenMM cluster MD path.
///
/// Usage:
///   ./bin/bench_nonbonded_style [N_or_n_side] [n_steps] [warmup]
///
/// If the first argument is ≥ 1000, it is treated as **atom count N** and a
/// cubic lattice is built with n_side = ceil(N^{1/3}) (actual N = n_side³ ≥
/// requested). If < 1000, it is n_side (legacy).
///
/// Prints ms/step, M atoms·step/s, and energy. Dual-style A/B removed.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

static cpu::Frame make_lj(int n_side, double density) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 18;
                frame.span(kAtom_vx)[idx] = 0.01 * ((ix + iy + iz) % 3 - 1);
                frame.span(kAtom_vy)[idx] = 0.01 * ((ix + 2 * iy) % 3 - 1);
                frame.span(kAtom_vz)[idx] = 0.01 * ((2 * ix + iz) % 3 - 1);
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

static int side_for_n(long long want_n) {
    int s = static_cast<int>(std::ceil(std::cbrt(static_cast<double>(want_n))));
    if (s < 2)
        s = 2;
    return s;
}

int main(int argc, char** argv) {
    long long arg1 = (argc > 1) ? std::atoll(argv[1]) : 10000;
    int steps = (argc > 2) ? std::atoi(argv[2]) : 50;
    int warm = (argc > 3) ? std::atoi(argv[3]) : 10;
    if (steps < 5)
        steps = 5;
    if (warm < 1)
        warm = 1;

    // ≥1000 → atom count; <1000 → n_side (legacy small lattices).
    int n_side = (arg1 >= 1000) ? side_for_n(arg1) : static_cast<int>(arg1);
    if (n_side < 2)
        n_side = 2;
    const long long N = static_cast<long long>(n_side) * n_side * n_side;

    auto frame = make_lj(n_side, 0.8);
    const double cutoff = 2.5;
    const double skin = 0.3;

    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff, skin});
    sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, cutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{0.005});
    sim.build();
    sim.run(frame, warm);
    auto t0 = std::chrono::steady_clock::now();
    sim.run(frame, steps);
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / steps;
    // throughput: million atom-steps per second
    double matom_step_s = (static_cast<double>(N) / ms) * 1e-3;
    std::printf("path=openmm_cluster n_side=%d N=%lld steps=%d  "
                "ms/step=%.4f  Matom*step/s=%.3f  energy=%.6f\n",
                n_side, N, steps, ms, matom_step_s,
                static_cast<double>(frame.meta.get(kEnergy)));
    return 0;
}
