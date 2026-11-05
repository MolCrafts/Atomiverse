/// @file tests/cuda/dynamics/test_md_parity.cu
/// @brief CPU <-> CUDA parity of every classical MD component through the
///        public drivers, one case per component:
///        - one step: forces, energy, kinetic energy, virial, positions,
///          velocities and cell against the same-settings CPU run;
///        - a 50-step trajectory against the CPU run;
///        - the replayed step graphs against the eager launch path
///          (`enable_timing()` keeps every step eager) for the same CUDA
///          run, which catches host state baked into a captured launch.
///        Components without a deterministic CPU twin (stochastic noise
///        from a different generator) run the graph-vs-eager check only.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/deform.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_fourier.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cuda/dynamics/barostat.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/dynamics/nve_limit.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observe/stress.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "atomiverse/cuda/potentials/periodic_torsion.cuh"
#include "atomiverse/cuda/record/pressure_series.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cuda/longrange/pme.cuh"
#endif

using namespace ATV;

namespace {

constexpr bool kDouble = std::is_same_v<cuda::force_t, double>
                         && std::is_same_v<cuda::pos_t, double>;

/// Relative tolerances of one case (scaled by 1 + |reference|).
struct Tolerance {
    double step; ///< After one step (summation order only in FP64).
    double traj; ///< After the trajectory (order noise grows with time).
};
constexpr Tolerance kExact{kDouble ? 1e-9 : 1e-4, kDouble ? 1e-7 : 1e-2};
constexpr Tolerance kMesh{kDouble ? 1e-6 : 1e-3, kDouble ? 1e-5 : 1e-2};
/// Graphed vs eager: the same kernels in the same order; only FP64 atomic
/// accumulation order may differ between two runs.
constexpr double kReplay = kDouble ? 1e-10 : 1e-4;
constexpr int kTrajectorySteps = 50;

// ── Systems ───────────────────────────────────────────────────────────────

enum class System { lj, ions, chains };

constexpr double kLjEps = 0.0104, kLjSigma = 3.4, kLjCut = 6.5, kLjSkin = 1.0;
constexpr double kIonAlpha = 0.34;
// Bonded neighbors (1.3-1.9 Å) are not excluded from LJ: sigma stays below
// the bond lengths, as in the per-term energy test.
constexpr double kChainEps = 0.004, kChainSigma = 1.1, kChainCut = 4.0;

/// 4x4x4 jittered lattice, 3.9 Å spacing, small thermal velocities with
/// zero total momentum. `ions` alternates +-0.5 e on two LJ types.
cpu::Frame lattice(bool ions) {
    constexpr int side = 4, n = side * side * side;
    constexpr double a = 3.9, L = side * a;
    cpu::Frame f(n);
    double p[3] = {};
    int i = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++i) {
                const double j = 0.12 * std::sin(1.7 * i + 0.3);
                f.span(kAtom_x)[i] = (x + 0.5) * a + j;
                f.span(kAtom_y)[i] = (y + 0.5) * a - 0.7 * j;
                f.span(kAtom_z)[i] = (z + 0.5) * a + 0.4 * j;
                f.span(kAtom_vx)[i] = 2e-3 * std::sin(1.3 * i);
                f.span(kAtom_vy)[i] = 2e-3 * std::cos(0.7 * i + 0.2);
                f.span(kAtom_vz)[i] = 2e-3 * std::sin(2.1 * i + 0.4);
                p[0] += f.span(kAtom_vx)[i];
                p[1] += f.span(kAtom_vy)[i];
                p[2] += f.span(kAtom_vz)[i];
                f.span(kAtom_mass)[i] = 39.948;
                f.span(kAtom_tag)[i] = i + 1;
                f.span(kAtom_type)[i] = ions ? (x + y + z) % 2 : 0;
                f.span(kAtom_charge)[i] =
                    ions ? ((x + y + z) % 2 ? -0.5 : 0.5) : 0.0;
            }
    for (int k = 0; k < n; ++k) {
        f.span(kAtom_vx)[k] -= p[0] / n;
        f.span(kAtom_vy)[k] -= p[1] / n;
        f.span(kAtom_vz)[k] -= p[2] / n;
    }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    f.build_special_list();
    return f;
}

/// Eight six-bead kinked chains (two bond / angle / dihedral / improper
/// types) in a 16 Å box; chain 3 crosses the x face.
cpu::Frame chains() {
    constexpr int kChains = 8, kPer = 6, n = kChains * kPer;
    constexpr double L = 16.0;
    cpu::Frame f(n);
    for (int c = 0; c < kChains; ++c)
        for (int k = 0; k < kPer; ++k) {
            const int i = c * kPer + k;
            const double x =
                1.0 + 1.25 * k + (c == 3 ? 9.5 : 0.0) + 4.0 * (c % 2);
            f.span(kAtom_x)[i] = std::fmod(x, L);
            f.span(kAtom_y)[i] = 1.5 + 3.8 * (c / 2) + 0.75 * (k % 2);
            f.span(kAtom_z)[i] =
                2.0 + 7.0 * (c % 2) + 0.15 * std::sin(1.9 * k + c);
            f.span(kAtom_vx)[i] = 3e-3 * std::sin(1.1 * i);
            f.span(kAtom_vy)[i] = 3e-3 * std::cos(0.9 * i);
            f.span(kAtom_vz)[i] = 3e-3 * std::sin(0.5 * i + 1.0);
            f.span(kAtom_mass)[i] = 12.011;
            // Topology refers to tags; the CUDA transfer resolves tag = row.
            f.span(kAtom_tag)[i] = i;
            f.span(kAtom_type)[i] = 0;
            f.span(kAtom_charge)[i] = 0.0;
        }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int c = 0; c < kChains; ++c) {
        const int b = c * kPer;
        for (int k = 0; k + 1 < kPer; ++k)
            f.add_bond(b + k, b + k + 1, k % 2);
        for (int k = 0; k + 2 < kPer; ++k)
            f.add_angle(b + k, b + k + 1, b + k + 2, k % 2);
        for (int k = 0; k + 3 < kPer; ++k)
            f.add_dihedral(b + k, b + k + 1, b + k + 2, b + k + 3, k % 2);
        // Out-of-plane terms around beads 1 and 3 (owner = second atom).
        f.add_improper(b + 0, b + 1, b + 2, b + 3, 0);
        f.add_improper(b + 2, b + 3, b + 4, b + 5, 1);
    }
    f.build_special_list();
    return f;
}

cpu::Frame make_system(System s) {
    switch (s) {
    case System::lj:
        return lattice(false);
    case System::ions:
        return lattice(true);
    case System::chains:
        return chains();
    }
    return lattice(false);
}

std::vector<LJTypeCoeff> ion_lj() {
    return {{0, 0, 0.010, 3.2}, {1, 1, 0.006, 3.6}, {0, 1, 0.008, 3.4}};
}

// Near the built geometry (bonds 1.46-1.5 Å, angles ~2.07 rad), so 50 steps
// stay close to equilibrium.
const std::vector<double> kBondK = {6.0, 4.0}, kBondR0 = {1.45, 1.5};
const std::vector<double> kFeneK = {0.05, 0.04}, kFeneR0 = {2.4, 2.6};
const std::vector<double> kCosineK = {0.05, 0.03};
const std::vector<double> kAngleK = {1.2, 0.9}, kTheta0 = {2.0, 2.2};
const std::vector<std::vector<double>> kDihCpu = {
    {0.03, -0.02, 0.04, 0.01, -0.005}, {0.02, 0.01, -0.03, 0.004, 0.002}};
const std::vector<std::array<double, 5>> kDihCuda = {
    {0.03, -0.02, 0.04, 0.01, -0.005}, {0.02, 0.01, -0.03, 0.004, 0.002}};
// Amber-like multi-term torsions (n up to 4, nonzero phases) and cvff
// impropers (d = -1 / +1, n = 2 / 3).
const std::vector<std::vector<TorsionTerm>> kFourier = {
    {{0.02, 1, 0.0}, {0.015, 2, 3.14159265358979}, {0.004, 3, 0.3}},
    {{0.01, 4, 1.1}, {0.006, 2, -0.4}}};
const std::vector<double> kCharmmK = {0.03, 0.012};
const std::vector<int> kCharmmN = {3, 2};
const std::vector<double> kCharmmDelta = {0.0, 3.14159265358979};
const std::vector<double> kCvffK = {0.05, 0.02};
const std::vector<int> kCvffD = {-1, 1};
const std::vector<int> kCvffN = {2, 3};

// ── Cases ─────────────────────────────────────────────────────────────────

struct Case {
    std::string name;
    System system;
    /// Empty: the component has no deterministic CPU twin.
    std::function<void(cpu::md::MD&)> cpu;
    std::function<void(cuda::md::MD&)> gpu;
    Tolerance tol = kExact;
};

double cut_of(System s) { return s == System::chains ? kChainCut : kLjCut; }

void mount_cpu_forces(cpu::md::MD& sim, System s) {
    sim.set_neighbor(cpu::md::NeighborList<>{cut_of(s), kLjSkin, 1, 0, true});
    sim.on(md::Phase::post_force).add(cpu::md::Virial{}, -50);
    switch (s) {
    case System::lj:
        sim.set_evaluator(cpu::PairLJCut{kLjEps, kLjSigma, kLjCut});
        break;
    case System::ions:
        sim.set_evaluator(cpu::PairLJCutCoulLong{ion_lj(), kIonAlpha, kLjCut});
        break;
    case System::chains:
        sim.set_evaluator(cpu::PairLJCut{kChainEps, kChainSigma, kChainCut});
        break;
    }
}

void mount_cuda_forces(cuda::md::MD& sim, System s) {
    sim.set_neighbor(cuda::NeighborList{cut_of(s), kLjSkin});
    sim.on(md::Phase::post_force).add(cuda::Virial{}, -50);
    switch (s) {
    case System::lj:
        sim.set_evaluator(cuda::PairLJCut{kLjEps, kLjSigma, kLjCut});
        break;
    case System::ions:
        sim.set_evaluator(cuda::PairLJCutCoulLong{ion_lj(), kIonAlpha, kLjCut});
        break;
    case System::chains:
        sim.set_evaluator(cuda::PairLJCut{kChainEps, kChainSigma, kChainCut});
        break;
    }
}

constexpr double kDt = 2.0;
// The classical engine works in LJ units (kB = 1): thermostat targets are
// energies, on the scale of the LJ well depth (~0.4-1 eps).
constexpr double kT = 0.005, kTHot = 0.0075, kTRamp = 0.01;

std::vector<Case> cases() {
    std::vector<Case> out;
    out.push_back(
        {"VelocityVerletLJ", System::lj,
         [](cpu::md::MD& s) { s.set_integrator(cpu::md::VelocityVerlet{kDt}); },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt});
         }});
    out.push_back({"LJCutCoulLongEwald", System::ions,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::KSpaceEwald{kIonAlpha, 1e-8});
                       s.set_integrator(cpu::md::VelocityVerlet{1.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::KSpaceEwald{kIonAlpha, 1e-8});
                       s.set_integrator(cuda::md::VelocityVerlet{1.0});
                   },
                   kMesh});
#ifdef ATV_HAS_KSPACE
    out.push_back({"LJCutCoulLongPME", System::ions,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::KSpacePME{kIonAlpha, 0.6, 5});
                       s.set_integrator(cpu::md::VelocityVerlet{1.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::KSpacePME{kIonAlpha, 0.6, 5});
                       s.set_integrator(cuda::md::VelocityVerlet{1.0});
                   },
                   kMesh});
#endif
    out.push_back({"BondAngleDihedralHarmonic", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cpu::AngleHarmonic{kAngleK, kTheta0});
                       s.set_evaluator(cpu::DihedralMultiHarmonic{kDihCpu});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cuda::AngleHarmonic{kAngleK, kTheta0});
                       s.set_evaluator(cuda::DihedralMultiHarmonic{kDihCuda});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back(
        {"DihedralFourierImproperCvff", System::chains,
         [](cpu::md::MD& s) {
             s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
             s.set_evaluator(cpu::DihedralFourier{kFourier});
             s.set_evaluator(cpu::ImproperCvff{kCvffK, kCvffD, kCvffN});
             s.set_integrator(cpu::md::VelocityVerlet{0.5});
         },
         [](cuda::md::MD& s) {
             s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
             s.set_evaluator(cuda::DihedralFourier{kFourier});
             s.set_evaluator(cuda::ImproperCvff{kCvffK, kCvffD, kCvffN});
             s.set_integrator(cuda::md::VelocityVerlet{0.5});
         }});
    out.push_back({"DihedralCharmm", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cpu::DihedralCharmm{kCharmmK, kCharmmN,
                                                           kCharmmDelta});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cuda::DihedralCharmm{kCharmmK, kCharmmN,
                                                            kCharmmDelta});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back({"BondFeneAngleCosine", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondFene{kFeneK, kFeneR0});
                       s.set_evaluator(cpu::AngleCosine{kCosineK});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondFene{kFeneK, kFeneR0});
                       s.set_evaluator(cuda::AngleCosine{kCosineK});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back({"NoseHoover", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.set_thermostat(cpu::md::NoseHoover{kT, 40.0, kDt});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_thermostat(cuda::md::NoseHoover{kT, 40.0, kDt});
                   }});
    out.push_back({"NoseHooverNPT", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::NoseHooverNPT{
                           kT, 40.0, 50 * units::bar_to_ev_per_a3, 200.0, kDt});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::NoseHooverNPT{
                           kT, 40.0, 50 * units::bar_to_ev_per_a3, 200.0, kDt});
                   }});
    // A temperature ramp over the run span (host-side target today).
    out.push_back(
        {"NoseHooverNPTRamp", System::lj,
         [](cpu::md::MD& s) {
             cpu::md::NoseHooverNPT npt{
                 kT,   40.0, 0.0, 200.0, kDt, 0.0, 3, 3, {true, true, true},
                 kTHot};
             npt.set_run_span(0, kTrajectorySteps);
             s.set_integrator(std::move(npt));
         },
         [](cuda::md::MD& s) {
             cuda::md::NoseHooverNPT npt{
                 kT,   40.0, 0.0, 200.0, kDt, 0.0, 3, 3, {true, true, true},
                 kTHot};
             npt.set_run_span(0, kTrajectorySteps);
             s.set_integrator(std::move(npt));
         }});
    out.push_back({"BerendsenBarostat", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.set_barostat(cpu::md::Barostat{
                           100 * units::bar_to_ev_per_a3, 100.0, kDt, 10.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_barostat(cuda::md::Barostat{
                           100 * units::bar_to_ev_per_a3, 100.0, kDt, 10.0});
                   }});
    // xmax below the fastest atom's |v| dt: the limiter engages.
    out.push_back(
        {"NVELimit", System::lj,
         [](cpu::md::MD& s) { s.set_integrator(cpu::md::NVELimit{kDt, 2e-3}); },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::NVELimit{kDt, 2e-3});
         }});
    out.push_back({"NvtSllod", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::NvtSllod{kT, 40.0, kDt, 2e-4,
                                                          md::kFlowXY, 3});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::NvtSllod{kT, 40.0, kDt, 2e-4,
                                                           md::kFlowXY, 3});
                   }});
    out.push_back({"Deform", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.on(md::Phase::initial_integrate)
                           .add(cpu::md::Deform{cpu::md::Deform::Mode::erate,
                                                2e-5, kDt, 1},
                                200);
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.on(md::Phase::initial_integrate)
                           .add(cuda::md::Deform{cuda::md::Deform::Mode::erate,
                                                 2e-5, kDt, 1},
                                200);
                   }});
    // T = 0: drag only, so the CPU twin is deterministic.
    out.push_back(
        {"LangevinDrag", System::lj,
         [](cpu::md::MD& s) {
             s.set_integrator(cpu::md::VelocityVerlet{kDt});
             s.set_thermostat(cpu::md::FixLangevin{0.0, 0.0, 80.0, kDt, 7u});
         },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt});
             s.set_thermostat(cuda::md::FixLangevin{0.0, 0.0, 80.0, kDt, 7u});
         }});
    // Noise keyed on (seed, tag, step): only graph-vs-eager is meaningful.
    out.push_back({"LangevinNoise", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_thermostat(
                           cuda::md::FixLangevin{kT, kT, 80.0, kDt, 7u});
                   }});
    out.push_back({"LangevinNoiseRamp", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       cuda::md::FixLangevin langevin{kT, kTRamp, 80.0, kDt,
                                                      7u};
                       langevin.set_run_span(0, kTrajectorySteps);
                       s.set_thermostat(std::move(langevin));
                   }});
    out.push_back({"Brownian", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::Brownian{kDt, 1e-4, kT, 11u});
                   }});
    return out;
}

// ── Runs and comparisons ──────────────────────────────────────────────────

cpu::Frame run_cpu(const Case& c, int steps) {
    cpu::Frame f = make_system(c.system);
    cpu::md::MD sim;
    mount_cpu_forces(sim, c.system);
    c.cpu(sim);
    sim.build();
    sim.run(f, steps);
    return f;
}

cpu::Frame run_cuda(const Case& c, int steps, bool eager) {
    cpu::Frame f = make_system(c.system);
    cuda::md::MD sim;
    mount_cuda_forces(sim, c.system);
    c.gpu(sim);
    if (eager)
        sim.enable_timing();
    sim.build();
    sim.run(f, steps);
    return f;
}

struct Row {
    std::array<double, 3> unwrapped, v, f;
};

std::map<std::int64_t, Row> rows(const cpu::Frame& f) {
    std::map<std::int64_t, Row> out;
    const auto& h = f.box.cell();
    for (int i = 0; i < f.nlocal(); ++i) {
        Row r;
        const double x[3] = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                             f.span(kAtom_z)[i]};
        for (int a = 0; a < 3; ++a) {
            r.unwrapped[a] = x[a];
            for (int b = 0; b < 3; ++b)
                r.unwrapped[a] +=
                    h(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
        }
        r.v = {f.span(kAtom_vx)[i], f.span(kAtom_vy)[i], f.span(kAtom_vz)[i]};
        r.f = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i], f.span(kAtom_fz)[i]};
        out[f.span(kAtom_tag)[i]] = r;
    }
    return out;
}

double scale_of(const std::map<std::int64_t, Row>& rows,
                std::array<double, 3> Row::* field) {
    double m = 0.0;
    for (const auto& [tag, r] : rows)
        for (double x : r.*field)
            m = std::max(m, std::abs(x));
    return m;
}

/// Every compared quantity within `tol * (1 + scale)`, where the scale of a
/// per-atom field is its largest reference magnitude.
void expect_same_state(const cpu::Frame& got, const cpu::Frame& want,
                       double tol, bool forces) {
    const auto g = rows(got), w = rows(want);
    ASSERT_EQ(g.size(), w.size());
    const double sx = 1.0 + scale_of(w, &Row::unwrapped);
    const double sv = 1.0 + scale_of(w, &Row::v);
    const double sf = 1.0 + scale_of(w, &Row::f);
    int reported = 0;
    for (const auto& [tag, wr] : w) {
        const auto it = g.find(tag);
        ASSERT_NE(it, g.end()) << "tag " << tag;
        const Row& gr = it->second;
        for (int a = 0; a < 3; ++a) {
            const bool ok =
                std::abs(gr.unwrapped[a] - wr.unwrapped[a]) <= tol * sx
                && std::abs(gr.v[a] - wr.v[a]) <= tol * sv
                && (!forces || std::abs(gr.f[a] - wr.f[a]) <= tol * sf);
            if (!ok && reported++ < 4) {
                ADD_FAILURE() << "tag " << tag << " axis " << a << ": x "
                              << gr.unwrapped[a] << " vs " << wr.unwrapped[a]
                              << ", v " << gr.v[a] << " vs " << wr.v[a]
                              << ", f " << gr.f[a] << " vs " << wr.f[a];
            }
        }
    }
    EXPECT_EQ(reported, 0) << "atoms out of tolerance";
    auto near = [&](double a, double b, const char* what) {
        EXPECT_NEAR(a, b, tol * (1.0 + std::abs(b))) << what;
    };
    near(got.meta.get(kEnergy), want.meta.get(kEnergy), "energy");
    near(got.meta.get(kKinetic), want.meta.get(kKinetic), "kinetic");
    const auto& sg = got.meta.get(kStress);
    const auto& sw = want.meta.get(kStress);
    double ss = 0.0;
    for (int k = 0; k < 6; ++k)
        ss = std::max(ss, std::abs(sw[k]));
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(sg[k], sw[k], tol * (1.0 + ss)) << "virial " << k;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_NEAR(got.box.cell()(a, b), want.box.cell()(a, b),
                        tol * (1.0 + std::abs(want.box.cell()(a, b))))
                << "cell(" << a << "," << b << ")";
}

class CudaMdParity : public testing::TestWithParam<Case> {};

} // namespace

TEST_P(CudaMdParity, OneStepMatchesCpu) {
    const Case& c = GetParam();
    if (!c.cpu)
        GTEST_SKIP() << c.name << " has no deterministic CPU twin";
    const cpu::Frame want = run_cpu(c, 1);
    const cpu::Frame got = run_cuda(c, 1, /*eager=*/false);
    ASSERT_GT(std::abs(want.meta.get(kEnergy)), 1e-8) << "vacuous energy";
    expect_same_state(got, want, c.tol.step, /*forces=*/true);
}

TEST_P(CudaMdParity, TrajectoryMatchesCpu) {
    const Case& c = GetParam();
    if (!c.cpu)
        GTEST_SKIP() << c.name << " has no deterministic CPU twin";
    const cpu::Frame want = run_cpu(c, kTrajectorySteps);
    const cpu::Frame got = run_cuda(c, kTrajectorySteps, /*eager=*/false);
    expect_same_state(got, want, c.tol.traj, /*forces=*/true);
}

TEST_P(CudaMdParity, GraphedStepsMatchEagerSteps) {
    const Case& c = GetParam();
    const cpu::Frame eager = run_cuda(c, kTrajectorySteps, /*eager=*/true);
    const cpu::Frame graphed = run_cuda(c, kTrajectorySteps, /*eager=*/false);
    // A replay that froze the first step's host state would leave the
    // trajectory far from the eager one; make sure the run moved at all.
    const cpu::Frame start = make_system(c.system);
    double moved = 0.0;
    const auto s = rows(start), e = rows(eager);
    for (const auto& [tag, r] : s)
        for (int a = 0; a < 3; ++a)
            moved = std::max(moved,
                             std::abs(e.at(tag).unwrapped[a] - r.unwrapped[a]));
    ASSERT_GT(moved, 1e-6) << "vacuous: the eager run did not move";
    expect_same_state(graphed, eager, kReplay, /*forces=*/true);
}

INSTANTIATE_TEST_SUITE_P(Components, CudaMdParity, testing::ValuesIn(cases()),
                         [](const testing::TestParamInfo<Case>& info) {
                             return info.param.name;
                         });

// ── PressureSeries: rows of the replayed run equal the eager run's ────────

namespace {

std::vector<std::vector<double>> series_rows(bool eager) {
    const std::string path = testing::TempDir() + "atv_parity_series_"
                             + (eager ? "eager" : "graphed") + ".dat";
    {
        cpu::Frame f = lattice(false);
        cuda::md::MD sim;
        mount_cuda_forces(sim, System::lj);
        sim.set_integrator(
            cuda::md::NvtSllod{kT, 40.0, kDt, 2e-4, md::kFlowXY, 1});
        sim.on(md::Phase::end_of_step)
            .add(cuda::md::PressureSeries{path, 5, kLjEps, md::kFlowXY}, 20);
        if (eager)
            sim.enable_timing();
        sim.build();
        sim.run(f, kTrajectorySteps);
    }
    std::vector<std::vector<double>> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream row(line);
        std::vector<double> values;
        double v;
        while (row >> v)
            values.push_back(v);
        out.push_back(values);
    }
    std::remove(path.c_str());
    return out;
}

} // namespace

TEST(CudaMdParitySeries, PressureSeriesGraphedRowsMatchEagerRows) {
    const auto eager = series_rows(true);
    const auto graphed = series_rows(false);
    ASSERT_EQ(eager.size(), std::size_t(kTrajectorySteps / 5 + 1));
    ASSERT_EQ(graphed.size(), eager.size());
    for (std::size_t r = 0; r < eager.size(); ++r) {
        ASSERT_EQ(graphed[r].size(), eager[r].size()) << "row " << r;
        for (std::size_t k = 0; k < eager[r].size(); ++k)
            EXPECT_NEAR(graphed[r][k], eager[r][k],
                        1e-8 * (1.0 + std::abs(eager[r][k])))
                << "row " << r << " column " << k;
    }
}
