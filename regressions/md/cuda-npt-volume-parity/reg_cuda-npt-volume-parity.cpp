/// @file regressions/md/cuda-npt-volume-parity/reg_cuda-npt-volume-parity.cpp
/// @brief Single-step fp64 parity: cuda::md::NoseHooverNPT vs CPU twin.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

cpu::Frame make_lj_cube(int n_side) {
    const int n = n_side * n_side * n_side;
    const double density = 0.8;
    const double box_len = std::cbrt(double(n) / density);
    const double spacing = box_len / double(n_side);
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
                frame.span(kAtom_vx)[idx] = 0.02 * ((ix + iy + iz) % 3 - 1);
                frame.span(kAtom_vy)[idx] = 0.02 * ((ix + 2 * iy) % 3 - 1);
                frame.span(kAtom_vz)[idx] = 0.02 * ((2 * ix + iz) % 3 - 1);
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

void one_step_parity() {
    constexpr double kDt = 0.005;
    constexpr double kCut = 2.5;
    constexpr double kSkin = 0.3;
    // n_side=6 → L≈6.46 Å, L/2 > r_c=2.5 (n=8 at ρ=0.8 is L≈2.15, illegal).
    auto cpu_f = make_lj_cube(6);
    auto cuda_f = make_lj_cube(6);
    const double V0 = cpu_f.box.volume();

    {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
        sim.set_evaluator(cpu::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cpu::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.set_observer(cpu::md::Virial{});
        sim.build();
        sim.run(cpu_f, 1);
    }
    {
        cuda::md::MD sim;
        sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
        sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cuda::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.build();
        sim.run(cuda_f, 1);
    }

    const double Vc = cpu_f.box.volume();
    const double Vg = cuda_f.box.volume();
    if (!std::isfinite(Vc))
        throw std::runtime_error("non-finite CPU volume");
    if (!std::isfinite(Vg))
        throw std::runtime_error("non-finite CUDA volume");
    if (!(std::abs(Vc - V0) > 1e-12))
        throw std::runtime_error("CPU barostat did not move the cell");
    // One MTK step, identical start, fp64. Not bit-exact (tile vs SIMD
    // virial). 1e-8 relative is still far tighter than a dead barostat.
    std::printf("one-step CPU=%.16g CUDA=%.16g relative-error=%.6g\n", Vc, Vg,
                std::abs(Vg - Vc) / std::max(1.0, std::abs(Vc)));
    if (!(std::abs(Vg - Vc) <= 1e-8 * std::max(1.0, std::abs(Vc))))
        throw std::runtime_error("one-step volume parity failed");
}

void two_step_parity() {
    // Both backends prime the initial pair virial before step 1. Step 2
    // also checks that subsequent force passes continue to publish stress.
    constexpr double kDt = 0.005;
    constexpr double kCut = 2.5;
    constexpr double kSkin = 0.3;
    auto cpu_f = make_lj_cube(6);
    auto cuda_f = make_lj_cube(6);

    {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
        sim.set_evaluator(cpu::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cpu::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.set_observer(cpu::md::Virial{});
        sim.build();
        sim.run(cpu_f, 2);
    }
    {
        cuda::md::MD sim;
        sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
        sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cuda::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.build();
        sim.run(cuda_f, 2);
    }

    const double Vc = cpu_f.box.volume();
    const double Vg = cuda_f.box.volume();
    if (!std::isfinite(Vc))
        throw std::runtime_error("non-finite CPU volume");
    if (!std::isfinite(Vg))
        throw std::runtime_error("non-finite CUDA volume");
    std::printf("two-step CPU=%.16g CUDA=%.16g relative-error=%.6g\n", Vc, Vg,
                std::abs(Vg - Vc) / std::max(1.0, std::abs(Vc)));
    if (!(std::abs(Vg - Vc) <= 1e-6 * std::max(1.0, std::abs(Vc))))
        throw std::runtime_error("two-step volume parity failed");
}

int main() try {
    one_step_parity();
    two_step_parity();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-npt-volume-parity: %s\n", e.what());
    return 1;
}
