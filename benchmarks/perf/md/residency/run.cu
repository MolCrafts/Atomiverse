/// @file benchmarks/perf/md/residency/run.cu
/// @brief Steady-state CUDA MD cases for the no-D2H audit and for bitwise
///        before/after trajectory comparison (iron law 10).
///
/// Usage:
///   md-residency --case lj-nve|lj-nh|lj-npt|ion-pme --out FILE
///                [--steps N] [--warm N] [--side N] [--profile]
///
/// The case runs `warm` steps, then `steps` more inside
/// cudaProfilerStart/Stop when `--profile` is given, so an Nsight capture
/// with `--capture-range=cudaProfilerApi` holds only steady steps plus the
/// end-of-run retrieve. FILE receives the final positions, velocities,
/// potential and kinetic energy as hexadecimal floats (exact comparison).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"

using namespace ATV;

namespace {

// Argon-like LJ liquid started from a simple cubic lattice at 2x the target
// temperature: it melts within the warm-up, so the steady interval rebuilds.
constexpr double kEps = 0.0104, kSigma = 3.4, kCutoff = 8.5, kDt = 2.0;
constexpr double kTemperature = 120.0, kMass = 39.948;

cpu::Frame make_liquid(int n_side, double density, bool charged) {
    const int n = n_side * n_side * n_side;
    const double box = std::cbrt(double(n) / density);
    const double spacing = box / double(n_side);
    cpu::Frame frame(n);
    std::mt19937_64 rng(20261029);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double sigma_v = std::sqrt(units::kB_ev_per_k * 2.0 * kTemperature
                                     / (units::mass_v2_to_energy * kMass));
    double p[3] = {};
    int i = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz, ++i) {
                frame.span(kAtom_x)[i] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[i] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[i] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[i] = kMass;
                frame.span(kAtom_type)[i] = 0;
                frame.span(kAtom_tag)[i] = i;
                frame.span(kAtom_charge)[i] =
                    charged ? ((ix + iy + iz) % 2 ? 0.5 : -0.5) : 0.0;
                const double v[3] = {normal(rng) * sigma_v,
                                     normal(rng) * sigma_v,
                                     normal(rng) * sigma_v};
                frame.span(kAtom_vx)[i] = v[0];
                frame.span(kAtom_vy)[i] = v[1];
                frame.span(kAtom_vz)[i] = v[2];
                for (int d = 0; d < 3; ++d)
                    p[d] += v[d];
            }
    for (int k = 0; k < n; ++k) { // equal masses: remove the COM velocity
        frame.span(kAtom_vx)[k] -= p[0] / n;
        frame.span(kAtom_vy)[k] -= p[1] / n;
        frame.span(kAtom_vz)[k] -= p[2] / n;
    }
    frame.box = cpu::Box::cubic(box);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    return frame;
}

void write_state(const std::string& path, const cpu::Frame& f, int rebuilds) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::hexfloat << "energy " << f.meta.get(kEnergy) << "\nkinetic "
        << f.meta.get(kKinetic) << "\nrebuilds " << std::dec << rebuilds
        << std::hexfloat << '\n';
    for (int i = 0; i < f.nlocal(); ++i)
        out << f.span(kAtom_x)[i] << ' ' << f.span(kAtom_y)[i] << ' '
            << f.span(kAtom_z)[i] << ' ' << f.span(kAtom_vx)[i] << ' '
            << f.span(kAtom_vy)[i] << ' ' << f.span(kAtom_vz)[i] << '\n';
}

} // namespace

int main(int argc, char** argv) try {
    std::string name, out;
    int steps = 400, warm = 200, side = 16;
    bool profile = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        auto value = [&]() -> std::string {
            if (a + 1 >= argc)
                throw std::invalid_argument(arg + " needs a value");
            return argv[++a];
        };
        if (arg == "--case")
            name = value();
        else if (arg == "--out")
            out = value();
        else if (arg == "--steps")
            steps = std::stoi(value());
        else if (arg == "--warm")
            warm = std::stoi(value());
        else if (arg == "--side")
            side = std::stoi(value());
        else if (arg == "--profile")
            profile = true;
        else
            throw std::invalid_argument("unknown argument " + arg);
    }
    if (out.empty() || steps < 1 || warm < 1)
        throw std::invalid_argument("--case, --out and positive steps needed");

    const bool charged = name == "ion-pme";
    auto frame = make_liquid(side, 0.0177, charged); // side^3 atoms
    cuda::NeighborList neighbor{kCutoff};
    cuda::md::MD sim;
    sim.set_neighbor_ref(neighbor);
    if (name == "lj-nve" || name == "lj-nh" || name == "lj-npt") {
        sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
    } else if (charged) {
        constexpr double alpha = 0.3;
        sim.set_evaluator(
            cuda::PairLJCutCoulLong{{kEps}, {kSigma}, alpha, kCutoff});
        sim.on(md::Phase::force).add(cuda::KSpacePME{alpha, 1.0, 5}, 100);
    } else {
        throw std::invalid_argument("unknown case " + name);
    }
    if (name == "lj-npt")
        sim.set_integrator(cuda::md::NoseHooverNPT{
            kTemperature, 100 * kDt, units::bar_to_ev_per_a3, 1000 * kDt, kDt});
    else
        sim.set_integrator(cuda::md::VelocityVerlet{kDt});
    if (name == "lj-nh")
        sim.set_thermostat(cuda::md::NoseHoover{kTemperature, 100 * kDt, kDt});
    sim.build();

    sim.run(frame, warm);
    if (profile)
        cudaProfilerStart();
    const auto start = std::chrono::steady_clock::now();
    sim.run(frame, steps); // ends with the retrieve's synchronization
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    if (profile)
        cudaProfilerStop();
    write_state(out, frame, neighbor.rebuild_count());
    std::cout << "case=" << name << " atoms=" << frame.nlocal()
              << " steps=" << warm + steps
              << " rebuilds=" << neighbor.rebuild_count()
              << " energy=" << frame.meta.get(kEnergy)
              << " us_per_step=" << us / steps << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
