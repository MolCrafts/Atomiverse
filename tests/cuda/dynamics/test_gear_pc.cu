/// @file tests/cuda/dynamics/test_gear_pc.cu
/// @brief `cuda::md::GearPC` on a driven Tosi–Fumi NaCl melt, with the CPU
///        integrator as oracle:
///        - the constrained temperature of each Gaussian thermostat,
///          measured from the returned frame with its own degrees of
///          freedom, is held to round-off under a strong field (and is not
///          held without the thermostat);
///        - the trajectory and the thermostat diagnostics of every
///          thermostat follow the CPU run;
///        - a run split into chunks, and the replayed step graphs, give the
///          trajectory of one eager run (the Nordsieck history and the
///          thermostat scalars live on the device across launches).
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/dynamics/gauss_thermostat.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cuda/dynamics/gear_pc.cuh"

#include "nacl_parity.h"

using namespace ATV;
using cpu::md::ConfigurationalThermostat;
using cpu::md::GaussKind;
using cpu::md::GaussThermostat;
using nacl_parity::Atom;
using nacl_parity::Terms;

namespace {

constexpr double kDt = 0.001;  // [ps]
constexpr double kField = 0.5; // [V/Å] along x
constexpr double kT0 = 1500.0; // [K]
constexpr double kQ = 20.0;    // [Å²/ps²]
constexpr int kSteps = 40;
/// Trajectory tolerance after `kSteps`, relative to the largest reference
/// magnitude of each field (FP64: summation order and the fixed-point
/// force grid, amplified by the dynamics).
constexpr double kTraj = nacl_parity::kDouble ? 1e-9 : 1e-2;
/// Two CUDA runs of the same kernels (atomic accumulation order only).
constexpr double kReplay = nacl_parity::kDouble ? 1e-10 : 1e-4;
const Terms kDriven{true, true, true, kField};

/// The start of every run: the melt at 1800 K moving as a whole, so that
/// the barycentric mean a thermostat subtracts is not zero.
cpu::Frame drifting_melt() {
    cpu::Frame f = nacl_fixture::nacl(1800.0, 5);
    const double drift[3] = {2.0, -1.5, 1.0}; // [Å/ps]
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_vx)[i] += drift[0];
        f.span(kAtom_vy)[i] += drift[1];
        f.span(kAtom_vz)[i] += drift[2];
    }
    return f;
}

/// What a run mounts as its integrator.
struct Scheme {
    std::string name;
    std::function<cpu::md::GearPC()> on_cpu;
    std::function<cuda::md::GearPC()> on_cuda;
};

template <class Thermostat>
Scheme scheme(std::string name, Thermostat t) {
    return {std::move(name), [t] { return cpu::md::GearPC{kDt, t}; },
            [t] { return cuda::md::GearPC{kDt, t}; }};
}

std::vector<Scheme> schemes() {
    return {{"NVE", [] { return cpu::md::GearPC{kDt}; },
             [] { return cuda::md::GearPC{kDt}; }},
            scheme("GaussKinetic", GaussThermostat{GaussKind::Kinetic, kT0, 0}),
            scheme("GaussFlow", GaussThermostat{GaussKind::Flow, kT0, 0}),
            scheme("GaussTransverse",
                   GaussThermostat{GaussKind::Transverse, kT0, 0}),
            scheme("Configurational", ConfigurationalThermostat{kT0, kQ})};
}

struct GearRun {
    cpu::Frame frame;
    double drift = 0, alpha = 0, xi = 0, t_c1 = 0;
};

template <class Gear>
void read_diagnostics(const Gear& gear, GearRun& out) {
    out.drift = gear.constraint_drift();
    out.alpha = gear.gauss_multiplier();
    out.xi = gear.xi();
    out.t_c1 = gear.configurational_temperature();
}

GearRun run_cpu(const Scheme& s) {
    GearRun out{drifting_melt()};
    cpu::md::GearPC gear = s.on_cpu();
    cpu::md::MD sim;
    nacl_parity::mount(sim, kDriven);
    sim.set_integrator_ref(gear);
    sim.build();
    sim.run(out.frame, kSteps);
    read_diagnostics(gear, out);
    return out;
}

/// `chunks` consecutive run() calls of kSteps / chunks steps each.
GearRun run_cuda(const Scheme& s, int chunks = 1, bool eager = false) {
    GearRun out{drifting_melt()};
    cuda::md::GearPC gear = s.on_cuda();
    cuda::md::MD sim;
    nacl_parity::mount(sim, kDriven);
    sim.set_integrator_ref(gear);
    if (eager)
        sim.enable_timing();
    sim.build();
    for (int c = 0; c < chunks; ++c)
        sim.run(out.frame, kSteps / chunks);
    read_diagnostics(gear, out);
    return out;
}

void expect_same_trajectory(const GearRun& got, const GearRun& want,
                            double tol) {
    const auto g = nacl_parity::atoms(got.frame);
    const auto w = nacl_parity::atoms(want.frame);
    ASSERT_EQ(g.size(), w.size());
    EXPECT_LT(nacl_parity::worst(g, w, &Atom::unwrapped, "position"), tol);
    EXPECT_LT(nacl_parity::worst(g, w, &Atom::v, "velocity"), tol);
    EXPECT_LT(nacl_parity::worst(g, w, &Atom::f, "force"), tol);
}

/// Mass-weighted mean of one velocity component over the rows of `type`
/// (-1: all rows).
double group_mean(const cpu::Frame& f, const std::vector<double>& v, int type) {
    double m = 0, p = 0;
    for (int i = 0; i < f.nlocal(); ++i)
        if (type < 0 || f.span(kAtom_type)[i] == type) {
            m += f.span(kAtom_mass)[i];
            p += f.span(kAtom_mass)[i] * v[i];
        }
    return p / m;
}

/// The temperature a Gaussian thermostat of `kind` (axis x) constrains,
/// from the frame alone [K]: barycentric peculiar velocities with 3N − 3
/// degrees of freedom, x relative to the species flow with 3N − 4 (two
/// species), or y and z only with 2N − 2.
double constrained_temperature(const cpu::Frame& f, GaussKind kind) {
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    const int n = f.nlocal();
    std::vector<double> v[3];
    for (int i = 0; i < n; ++i) {
        v[0].push_back(f.span(kAtom_vx)[i]);
        v[1].push_back(f.span(kAtom_vy)[i]);
        v[2].push_back(f.span(kAtom_vz)[i]);
    }
    double sum = 0;
    for (int d = 0; d < 3; ++d) {
        if (d == 0 && kind == GaussKind::Transverse)
            continue;
        const bool by_species = d == 0 && kind == GaussKind::Flow;
        const double mean[2] = {group_mean(f, v[d], by_species ? 0 : -1),
                                group_mean(f, v[d], by_species ? 1 : -1)};
        for (int i = 0; i < n; ++i) {
            const double w = v[d][i] - mean[f.span(kAtom_type)[i]];
            sum += f.span(kAtom_mass)[i] * w * w;
        }
    }
    const double dof = kind == GaussKind::Kinetic ? 3 * n - 3
                       : kind == GaussKind::Flow  ? 3 * n - 4
                                                  : 2 * n - 2;
    return sum * metal.mvv2e / (metal.boltz * dof);
}

class CudaGearPC : public testing::TestWithParam<Scheme> {};
class CudaGearPCGauss : public testing::TestWithParam<GaussKind> {};

std::string kind_name(GaussKind kind) {
    return kind == GaussKind::Kinetic ? "Kinetic"
           : kind == GaussKind::Flow  ? "Flow"
                                      : "Transverse";
}

} // namespace

TEST_P(CudaGearPCGauss, HoldsTheConstrainedTemperatureUnderAStrongField) {
    const GaussKind kind = GetParam();
    const Scheme thermostatted = scheme("gauss", GaussThermostat{kind, kT0, 0});
    // Chunked: the constraint holds at every returned frame.
    cpu::Frame frame = drifting_melt();
    cuda::md::GearPC gear = thermostatted.on_cuda();
    cuda::md::MD sim;
    nacl_parity::mount(sim, kDriven);
    sim.set_integrator_ref(gear);
    sim.build();
    double worst = 0;
    for (int chunk = 0; chunk < 8; ++chunk) {
        sim.run(frame, kSteps / 8);
        worst = std::max(
            worst, std::abs(constrained_temperature(frame, kind) / kT0 - 1));
    }
    EXPECT_LT(worst, 1e-12);
    // The measurement can fail: the same frame quantity of the
    // unthermostatted run under the same field is nowhere near T0.
    const GearRun free = run_cuda(schemes()[0]);
    EXPECT_GT(std::abs(constrained_temperature(free.frame, kind) / kT0 - 1),
              1e-3);
    // The projection removes integrator truncation error only (the CPU
    // test bounds it by 1e-6 after the start-up); zero would mean the
    // corrector was never measured.
    EXPECT_GT(gear.constraint_drift(), 0.0);
    EXPECT_LT(gear.constraint_drift(), 1e-3);
}

INSTANTIATE_TEST_SUITE_P(Kinds, CudaGearPCGauss,
                         testing::Values(GaussKind::Kinetic, GaussKind::Flow,
                                         GaussKind::Transverse),
                         [](const auto& info) {
                             return kind_name(info.param);
                         });

TEST_P(CudaGearPC, TrajectoryMatchesCpu) {
    const Scheme& s = GetParam();
    const GearRun want = run_cpu(s);
    const GearRun got = run_cuda(s);
    // The run moved: a frozen pair of runs would agree trivially.
    const auto start = nacl_parity::atoms(drifting_melt());
    ASSERT_GT(nacl_parity::worst(nacl_parity::atoms(want.frame), start,
                                 &Atom::unwrapped, "start"),
              1e-4);
    expect_same_trajectory(got, want, kTraj);
    EXPECT_NEAR(got.alpha, want.alpha, kTraj * (std::abs(want.alpha) + 1e-12));
    EXPECT_NEAR(got.xi, want.xi, kTraj * (std::abs(want.xi) + 1e-12));
    EXPECT_NEAR(got.t_c1, want.t_c1, kTraj * (std::abs(want.t_c1) + 1e-12));
    // The drift is a difference of nearly equal sums: same magnitude only.
    EXPECT_NEAR(got.drift, want.drift, 1e-3 * want.drift + 1e-13);
}

TEST_P(CudaGearPC, ChunkedRunMatchesOneRun) {
    const Scheme& s = GetParam();
    expect_same_trajectory(run_cuda(s, /*chunks=*/4), run_cuda(s), kReplay);
}

TEST_P(CudaGearPC, GraphedStepsMatchEagerSteps) {
    const Scheme& s = GetParam();
    expect_same_trajectory(run_cuda(s, 1, /*eager=*/false),
                           run_cuda(s, 1, /*eager=*/true), kReplay);
}

INSTANTIATE_TEST_SUITE_P(Thermostats, CudaGearPC, testing::ValuesIn(schemes()),
                         [](const testing::TestParamInfo<Scheme>& info) {
                             return info.param.name;
                         });

// The thermostats act: each thermostatted trajectory leaves the NVE one,
// and the configurational one reports a nonzero ξ and a physical T_C1.
TEST(CudaGearPCThermostats, EachOneChangesTheTrajectory) {
    const auto all = schemes();
    const GearRun nve = run_cuda(all[0]);
    const auto reference = nacl_parity::atoms(nve.frame);
    for (std::size_t k = 1; k < all.size(); ++k) {
        const GearRun run = run_cuda(all[k]);
        EXPECT_GT(nacl_parity::worst(nacl_parity::atoms(run.frame), reference,
                                     &Atom::v, "velocity"),
                  1e-4)
            << all[k].name;
        if (all[k].name == "Configurational") {
            EXPECT_NE(run.xi, 0.0);
            EXPECT_GT(run.t_c1, 100.0);
            EXPECT_LT(run.t_c1, 10000.0);
        } else {
            EXPECT_NE(run.alpha, 0.0) << all[k].name;
        }
    }
}
