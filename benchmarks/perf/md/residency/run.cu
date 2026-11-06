/// @file benchmarks/perf/md/residency/run.cu
/// @brief Steady-state CUDA MD cases for the no-D2H audit and for bitwise
///        before/after trajectory comparison (iron law 10).
///
/// Usage:
///   md-residency --case lj-nve|lj-nh|lj-npt|lj-langevin|lj-brownian|
///                       lj-deform|lj-nvelimit|lj-sllod|ion-pme|min-lbfgs|
///                       min-lbfgs-ls|water-settle|ethane-rattle
///                       --out FILE
///                [--steps N] [--warm N] [--side N] [--profile]
///
/// The case runs `warm` steps, then `steps` more inside
/// cudaProfilerStart/Stop when `--profile` is given, so an Nsight capture
/// with `--capture-range=cudaProfilerApi` holds only steady steps plus the
/// end-of-run retrieve. `water-settle` (rigid TIP3P through SETTLE) and
/// `ethane-rattle` (OPLS-like ethane, C-H bonds through RATTLE) run PME
/// electrostatics at a 2 fs step with `--side` molecules per box edge.
/// FILE receives the final positions, velocities,
/// potential and kinetic energy as hexadecimal floats (exact comparison).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/fix_rattle.cuh"
#include "atomiverse/cuda/dynamics/fix_settle.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/dynamics/nve_limit.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "atomiverse/cuda/potentials/periodic_torsion.cuh"

using namespace ATV;

namespace {

// Argon-like LJ liquid started from a simple cubic lattice at 2x the target
// temperature: it melts within the warm-up, so the steady interval rebuilds.
constexpr double kEps = 0.0104, kSigma = 3.4, kCutoff = 8.5, kDt = 2.0;
// The classical engine works in LJ units (kB = 1, mvv2e = 1): the target
// temperature is the energy kB·120 K of the eV-valued well depth.
constexpr double kMass = 39.948;
const double kTemperature = units::kB_ev_per_k * 120.0;

cpu::Frame make_liquid(int n_side, double density, bool charged) {
    const int n = n_side * n_side * n_side;
    const double box = std::cbrt(double(n) / density);
    const double spacing = box / double(n_side);
    cpu::Frame frame(n);
    std::mt19937_64 rng(20261029);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double sigma_v = std::sqrt(2.0 * kTemperature / kMass);
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

// ── Constrained molecular liquids (2 fs step, PME) ───────────────────────

/// One femtosecond in the engine's eV·Å·amu time unit (10.18 fs).
constexpr double kFs = 1.0 / 10.180505;
/// LJ-unit charge of one elementary charge (qqr2e = 1).
const double kQ = std::sqrt(14.399645);
constexpr double kMolCutoff = 9.0, kMolAlpha = 0.3;
const double kMolTemperature = units::kB_ev_per_k * 300.0;
constexpr double kDOH = 0.9572, kDHH = 1.5139;
constexpr double kCC = 1.529, kCH = 1.09, kHCC = 1.9321;

std::array<double, 3> rotate(std::array<double, 3> v, double a, double b,
                             double c) {
    v = {v[0], std::cos(a) * v[1] - std::sin(a) * v[2],
         std::sin(a) * v[1] + std::cos(a) * v[2]};
    v = {std::cos(b) * v[0] + std::sin(b) * v[2], v[1],
         -std::sin(b) * v[0] + std::cos(b) * v[2]};
    return {std::cos(c) * v[0] - std::sin(c) * v[1],
            std::sin(c) * v[0] + std::cos(c) * v[1], v[2]};
}

/// Place `side`^3 copies of a rigid molecule (`body`, centre-of-mass frame)
/// on a cubic lattice with jittered orientations and Maxwell velocities at
/// 300 K; tags equal rows. The constraints project the velocities on the
/// first step.
cpu::Frame molecular_lattice(int side, double spacing,
                             const std::vector<std::array<double, 3>>& body,
                             const std::vector<double>& mass,
                             const std::vector<int>& type,
                             const std::vector<double>& charge, double tilt) {
    const int per = static_cast<int>(body.size());
    const int n = side * side * side * per;
    const double L = side * spacing;
    cpu::Frame f(n);
    std::mt19937_64 rng(20261106);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> u(-tilt, tilt);
    double p[3] = {}, m_sum = 0.0;
    int i = 0;
    for (int ix = 0; ix < side; ++ix)
        for (int iy = 0; iy < side; ++iy)
            for (int iz = 0; iz < side; ++iz) {
                const double a = u(rng), b = u(rng), c = u(rng);
                const double com[3] = {(ix + 0.5) * spacing,
                                       (iy + 0.5) * spacing,
                                       (iz + 0.5) * spacing};
                for (int s = 0; s < per; ++s, ++i) {
                    const auto r =
                        rotate(body[static_cast<std::size_t>(s)], a, b, c);
                    const double m = mass[static_cast<std::size_t>(s)];
                    const double sv = std::sqrt(kMolTemperature / m);
                    const double v[3] = {normal(rng) * sv, normal(rng) * sv,
                                         normal(rng) * sv};
                    f.span(kAtom_x)[i] = std::fmod(com[0] + r[0] + L, L);
                    f.span(kAtom_y)[i] = std::fmod(com[1] + r[1] + L, L);
                    f.span(kAtom_z)[i] = std::fmod(com[2] + r[2] + L, L);
                    f.span(kAtom_vx)[i] = v[0];
                    f.span(kAtom_vy)[i] = v[1];
                    f.span(kAtom_vz)[i] = v[2];
                    f.span(kAtom_mass)[i] = m;
                    f.span(kAtom_type)[i] = type[static_cast<std::size_t>(s)];
                    f.span(kAtom_tag)[i] = i;
                    f.span(kAtom_charge)[i] =
                        charge[static_cast<std::size_t>(s)] * kQ;
                    for (int d = 0; d < 3; ++d)
                        p[d] += m * v[d];
                    m_sum += m;
                }
            }
    for (int k = 0; k < n; ++k) {
        f.span(kAtom_vx)[k] -= p[0] / m_sum;
        f.span(kAtom_vy)[k] -= p[1] / m_sum;
        f.span(kAtom_vz)[k] -= p[2] / m_sum;
    }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    return f;
}

/// Rigid TIP3P water at ~1 g/cm³ (3.104 Å lattice); the O-H bonds and the
/// H-O-H angle exist for the exclusions only.
cpu::Frame make_water(int side) {
    const double half = 0.5 * kDHH, h = std::sqrt(kDOH * kDOH - half * half);
    const double m_o = 15.9994, m_h = 1.008, yo = 2 * m_h * h / (m_o + 2 * m_h);
    auto f = molecular_lattice(
        side, 3.104,
        {{0.0, yo, 0.0}, {-half, yo - h, 0.0}, {half, yo - h, 0.0}},
        {m_o, m_h, m_h}, {0, 1, 1}, {-0.834, 0.417, 0.417}, 0.3);
    for (int k = 0; k < f.nlocal() / 3; ++k) {
        f.add_bond(3 * k, 3 * k + 1, 0);
        f.add_bond(3 * k, 3 * k + 2, 0);
        f.add_angle(3 * k + 1, 3 * k, 3 * k + 2, 0);
    }
    f.build_special_list();
    return f;
}

/// Staggered OPLS-like ethane at the liquid density (4.5 Å lattice).
cpu::Frame make_ethane(int side) {
    const double tp = M_PI - kHCC;
    std::vector<std::array<double, 3>> body(8);
    body[0] = {-0.5 * kCC, 0.0, 0.0};
    body[1] = {0.5 * kCC, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        const double p1 = 2.0 * M_PI * k / 3.0, p2 = p1 + M_PI / 3.0;
        body[2 + k] = {-0.5 * kCC - kCH * std::cos(tp),
                       kCH * std::sin(tp) * std::cos(p1),
                       kCH * std::sin(tp) * std::sin(p1)};
        body[5 + k] = {0.5 * kCC + kCH * std::cos(tp),
                       kCH * std::sin(tp) * std::cos(p2),
                       kCH * std::sin(tp) * std::sin(p2)};
    }
    std::vector<double> mass(8, 1.008), charge(8, 0.06);
    std::vector<int> type(8, 1);
    for (int c = 0; c < 2; ++c) {
        mass[static_cast<std::size_t>(c)] = 12.011;
        charge[static_cast<std::size_t>(c)] = -0.18;
        type[static_cast<std::size_t>(c)] = 0;
    }
    auto f = molecular_lattice(side, 4.5, body, mass, type, charge, M_PI);
    for (int m = 0; m < f.nlocal() / 8; ++m) {
        const int b = 8 * m;
        f.add_bond(b, b + 1, 0);
        for (int h = 0; h < 3; ++h) {
            f.add_bond(b, b + 2 + h, 1);
            f.add_bond(b + 1, b + 5 + h, 1);
            f.add_angle(b + 2 + h, b, b + 1, 0);
            f.add_angle(b + 5 + h, b + 1, b, 0);
            for (int g = h + 1; g < 3; ++g) {
                f.add_angle(b + 2 + h, b, b + 2 + g, 1);
                f.add_angle(b + 5 + h, b + 1, b + 5 + g, 1);
            }
            for (int g = 0; g < 3; ++g)
                f.add_dihedral(b + 2 + h, b, b + 1, b + 5 + g, 0);
        }
    }
    f.build_special_list();
    return f;
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

    static const std::set<std::string> kCases = {
        "lj-nve",      "lj-nh",     "lj-npt",       "lj-langevin",
        "lj-brownian", "lj-deform", "lj-nvelimit",  "lj-sllod",
        "ion-pme",     "min-lbfgs", "water-settle", "ethane-rattle",
        "min-lbfgs-ls"};
    if (!kCases.count(name))
        throw std::invalid_argument("unknown case " + name);
    if (name == "min-lbfgs" || name == "min-lbfgs-ls") {
        // Per-iteration cost of the CUDA L-BFGS loop (forces included) from
        // the melted-lattice start: two fresh runs of `warm` and
        // `warm + steps` iterations, so setup cancels in the difference.
        // min-lbfgs-ls adds the bounded Armijo line search.
        auto timed = [&](int budget, cpu::Frame& frame) {
            cuda::minimizer::Minimize min;
            min.set_neighbor(cuda::NeighborList{kCutoff});
            min.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
            cuda::minimizer::LBFGS lbfgs{5, 0.01, 10};
            if (name == "min-lbfgs-ls")
                lbfgs.line_search();
            min.set_optimizer(std::move(lbfgs));
            min.build();
            const auto start = std::chrono::steady_clock::now();
            const auto iters = min.run(frame, budget);
            return std::make_pair(iters,
                                  std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
        };
        // The lattice start is a stationary point: displace every row.
        auto displaced = [&] {
            auto f = make_liquid(side, 0.0177, false);
            std::mt19937_64 rng(7);
            std::uniform_real_distribution<double> u(-0.4, 0.4);
            for (int i = 0; i < f.nlocal(); ++i) {
                f.span(kAtom_x)[i] += u(rng);
                f.span(kAtom_y)[i] += u(rng);
                f.span(kAtom_z)[i] += u(rng);
            }
            return f;
        };
        auto warm_frame = displaced();
        timed(2, warm_frame); // context, modules and graph capture warm-up
        auto short_frame = displaced();
        auto frame = displaced();
        const auto [i1, t1] = timed(warm, short_frame);
        const auto [i2, t2] = timed(warm + steps, frame);
        write_state(out, frame, 0);
        std::cout << "case=" << name << " atoms=" << frame.nlocal()
                  << " iters=" << i1 << "/" << i2
                  << " energy=" << frame.meta.get(kEnergy) << " us_per_step="
                  << (t2 - t1) / double(std::max<long long>(1, i2 - i1))
                  << '\n';
        return 0;
    }
    const bool water = name == "water-settle";
    const bool ethane = name == "ethane-rattle";
    const bool charged = name == "ion-pme";
    auto frame = water    ? make_water(side)
                 : ethane ? make_ethane(side)
                          : make_liquid(side, 0.0177, charged); // side^3 atoms
    cuda::NeighborList neighbor{water || ethane ? kMolCutoff : kCutoff};
    cuda::md::MD sim;
    sim.set_neighbor_ref(neighbor);
    const bool lj = name.rfind("lj-", 0) == 0;
    if (water || ethane) {
        const double dt = 2.0 * kFs;
        sim.set_integrator(cuda::md::VelocityVerlet{dt});
        sim.on(md::Phase::force).add(cuda::KSpacePME{kMolAlpha, 1.0, 5}, 100);
        if (water) {
            sim.set_evaluator(cuda::PairLJCutCoulLong{
                {{0, 0, 0.00660, 3.1507}, {1, 1, 0.0, 1.0}, {0, 1, 0.0, 1.0}},
                kMolAlpha,
                kMolCutoff});
            cuda::md::FixSettle settle{dt, kDOH, kDHH};
            for (int k = 0; k < frame.nlocal() / 3; ++k)
                settle.add_water(3 * k, 3 * k + 1, 3 * k + 2);
            sim.set_observer(std::move(settle));
        } else {
            sim.set_evaluator(cuda::PairLJCutCoulLong{{{0, 0, 0.002862, 3.5},
                                                       {1, 1, 0.001301, 2.5},
                                                       {0, 1, 0.001930, 2.958}},
                                                      kMolAlpha,
                                                      kMolCutoff});
            sim.set_evaluator(cuda::BondHarmonic{{11.62, 14.74}, {kCC, kCH}});
            sim.set_evaluator(
                cuda::AngleHarmonic{{1.626, 1.431}, {kHCC, 1.8814}});
            sim.set_evaluator(cuda::DihedralFourier{{{{0.0065, 3, 0.0}}}});
            cuda::md::FixRattle rattle{dt};
            rattle.constrain_bond(1, kCH);
            sim.set_observer(std::move(rattle));
        }
    } else if (lj) {
        sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
    } else {
        constexpr double alpha = 0.3;
        sim.set_evaluator(
            cuda::PairLJCutCoulLong{{{0, 0, kEps, kSigma}}, alpha, kCutoff});
        sim.on(md::Phase::force).add(cuda::KSpacePME{alpha, 1.0, 5}, 100);
    }
    if (water || ethane) {
        // Integrator and constraints are mounted with the force field above.
    } else if (name == "lj-npt") {
        sim.set_integrator(cuda::md::NoseHooverNPT{
            kTemperature, 100 * kDt, units::bar_to_ev_per_a3, 1000 * kDt, kDt});
    } else if (name == "lj-brownian") {
        sim.set_integrator(cuda::md::Brownian{kDt, 1e-3, kTemperature, 5});
    } else if (name == "lj-nvelimit") {
        sim.set_integrator(cuda::md::NVELimit{kDt, 0.05});
    } else if (name == "lj-sllod") {
        sim.set_integrator(cuda::md::NvtSllod{kTemperature, 100 * kDt, kDt,
                                              1e-6, md::kFlowXY, 3});
    } else {
        sim.set_integrator(cuda::md::VelocityVerlet{kDt});
    }
    if (name == "lj-nh")
        sim.set_thermostat(cuda::md::NoseHoover{kTemperature, 100 * kDt, kDt});
    if (name == "lj-langevin")
        sim.set_thermostat(cuda::md::FixLangevin{kTemperature, kTemperature,
                                                 100 * kDt, kDt, 5u});
    if (name == "lj-deform")
        sim.on(md::Phase::initial_integrate)
            .add(cuda::md::Deform{cuda::md::Deform::Mode::erate, 1e-7, kDt, 0},
                 200);
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
