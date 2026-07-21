/// @file tests/cpu/md/test_step_timing.cpp
/// @brief Timer subsystem: component classifier + opt-in capture through the
/// Backend's TimerRegistry (ac-001, cpu-md-lammps-perf-assessment).
///
/// Single-rank, bonded-only (no MPI halo) so it is safe under bare ctest.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"

using namespace ATV;
using cpu::md::StepTiming;

namespace {

inline constexpr double kKcalToEv = 1.0 / 23.060547830619026;

cpu::Frame make_chain(int n, double box_len) {
    cpu::Frame frame(n);
    const double theta0 = 109.5 * units::pi / 180.0;
    const double half_apex = (units::pi - theta0) / 2.0;
    const double dx = 1.53 * std::cos(half_apex);
    const double dy = 1.53 * std::sin(half_apex);
    const double x0 = (box_len - (n - 1) * dx) / 2.0;
    for (int i = 0; i < n; ++i) {
        frame.span(ATV::kAtom_x)[i] = x0 + i * dx;
        frame.span(ATV::kAtom_y)[i] = box_len / 2.0 + (i % 2 ? dy : 0.0);
        frame.span(ATV::kAtom_z)[i] = box_len / 2.0;
        frame.span(ATV::kAtom_mass)[i] = 14.027;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n - 1; ++k)
        frame.add_bond(k, k + 1, 0);
    for (int k = 0; k < n - 2; ++k)
        frame.add_angle(k, k + 1, k + 2, 0);
    for (int k = 0; k < n - 3; ++k)
        frame.add_dihedral(k, k + 1, k + 2, k + 3, 0);
    frame.build_special_list();
    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

void configure_sim(cpu::md::MD& sim) {
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{268.0 * kKcalToEv}, {1.53}})
        .add(
            cpu::AngleHarmonic{{60.0 * kKcalToEv}, {109.5 * units::pi / 180.0}})
        .add(cpu::DihedralMultiHarmonic{
            {{1.73 * kKcalToEv, -4.49 * kKcalToEv, 0.776 * kKcalToEv,
              6.99 * kKcalToEv, 0.0}}});
}

} // namespace

// ── classify: LAMMPS-aligned sections ──────────────────────────────────
TEST(StepTiming, ClassifyKnownComponentNames) {
    EXPECT_EQ(StepTiming::classify("lj"), StepTiming::Pair);
    EXPECT_EQ(StepTiming::classify("lj/cut/coul/long"), StepTiming::Pair);
    EXPECT_EQ(StepTiming::classify("coul/long"), StepTiming::Pair);
    EXPECT_EQ(StepTiming::classify("morse"), StepTiming::Pair);
    EXPECT_EQ(StepTiming::classify("harmonic_bond"), StepTiming::Bond);
    EXPECT_EQ(StepTiming::classify("harmonic_angle"), StepTiming::Bond);
    EXPECT_EQ(StepTiming::classify("multi_harmonic_dihedral"), StepTiming::Bond);
    EXPECT_EQ(StepTiming::classify("charmm_dihedral"), StepTiming::Bond);
    EXPECT_EQ(StepTiming::classify("pppm"), StepTiming::Kspace);
    EXPECT_EQ(StepTiming::classify("pme"), StepTiming::Kspace);
    EXPECT_EQ(StepTiming::classify("ewald"), StepTiming::Kspace);
    EXPECT_EQ(StepTiming::classify("neighbor_list"), StepTiming::Neigh);
    EXPECT_EQ(StepTiming::classify("halo_send"), StepTiming::Comm);
    EXPECT_EQ(StepTiming::classify("halo_recv"), StepTiming::Comm);
    EXPECT_EQ(StepTiming::classify("migrate"), StepTiming::Comm);
    EXPECT_EQ(StepTiming::classify("rebalance"), StepTiming::Comm);
    EXPECT_EQ(StepTiming::classify("reduce"), StepTiming::Comm);
    EXPECT_EQ(StepTiming::classify("thermo"), StepTiming::Output);
    EXPECT_EQ(StepTiming::classify("velocity_verlet"), StepTiming::Modify);
    EXPECT_EQ(StepTiming::classify("nose_hoover_npt"), StepTiming::Modify);
    EXPECT_EQ(StepTiming::classify("fix_langevin"), StepTiming::Modify);
    EXPECT_EQ(StepTiming::classify("kinetic_energy"), StepTiming::Modify);
}

// ── always-on: timing is ENABLED by default so every driver emits the
// built-in end-of-run summary (LAMMPS-style). (Was OptInDisabledByDefault
// before the summary became a mandatory built-in driver feature.)
TEST(StepTiming, OnByDefault) {
    auto frame = make_chain(50, 100.0);
    cpu::md::MD sim;
    configure_sim(sim);
    sim.build();
    sim.run(frame, 20);
    const TimerRegistry* reg = sim.timing();
    ASSERT_NE(reg, nullptr);
    EXPECT_TRUE(reg->enabled());
    EXPECT_EQ(reg->steps(), 20u);
}

// ── opt-out still works: enable_timing(false) leaves the registry empty ──
TEST(StepTiming, OptOutLeavesRegistryEmpty) {
    auto frame = make_chain(50, 100.0);
    cpu::md::MD sim;
    configure_sim(sim);
    sim.enable_timing(false);
    sim.build();
    sim.run(frame, 20);
    const TimerRegistry* reg = sim.timing();
    if (reg)
        EXPECT_EQ(reg->steps(), 0u);
}

// The wall-clock-fraction capture gate (per-category sum within ±5% of the
// step-loop wall) is a PERFORMANCE assertion and lives in
// benchmarks/md/pe/validation/bench_timing_capture.cpp — unit tests assert
// code correctness only (classifier mapping + on/off plumbing above).
