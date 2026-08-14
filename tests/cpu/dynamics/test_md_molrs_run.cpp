/// @file tests/cpu/dynamics/test_md_molrs_run.cpp
/// @brief `cpu::md::MD::run(molrs::FrameRef&, int)` — the molrs overload's
///        documented contract: **each call is an independent simulation**.
///
/// One concern, one test. The `cpu::Frame&` overload is the CONTINUABLE one
/// (`run(burn_in); run(sample); …` extends a single trajectory with a
/// cumulative `ctx.step`); the `molrs::FrameRef&` overload is explicitly NOT
/// (`cpu/dynamics/md.h`, @see MD::run). So two consecutive `run(ref, 3)` calls
/// must both hand the components the step indices 0, 1, 2 — never 0…2 then 3…5.
///
/// **Why this can silently regress — the address-reuse mechanism.**
/// Continuation is keyed on the raw frame address: `Driver::begin_run_`
/// (cpu/driver/driver.h) tears the live run down only when
/// `frame_ptr_ != &frame`. The molrs overload materializes its working
/// `cpu::Frame` as a **stack local** that dies when the call returns, so
/// `frame_ptr_` is left dangling — and the next call, issued from the same call
/// site at the same stack depth, typically places its own local in the *same*
/// stack slot. The pointer then compares EQUAL to a frame that no longer
/// exists: `live_` stays true, `setup_pass()` is skipped, and `step_counter_`
/// keeps counting, so call #2 silently runs steps 3, 4, 5 of call #1's
/// trajectory. Independence must be STRUCTURAL (the overload tears its own run
/// down before returning), never an accident of stack layout.
///
/// Both `run(ref, …)` calls are therefore issued from ONE call expression (the
/// loop below), so the two working frames land in the same stack slot
/// deterministically — this test must not pass merely because the compiler
/// happened to give call #2 a different address.
///
/// The observable is `kStep` frame meta, which `cpu::to_molrs` serializes into
/// the molrs frame's `step` entry as the call returns. A minimal probe
/// component stamps it at `end_of_step`, exactly as the production consumer
/// does (`cpu::XyzRecorder`, cpu/record/xyz_recorder.cpp: `ctx.frame.meta.get(
/// kStep) = ctx.step`) — inlined here so this driver-contract test owns no
/// recorder cadence, no trajectory file and no I/O of any kind.
///
/// Deterministic: literal geometry, zero initial velocities, no RNG, no
/// InitVelocity, serial, no filesystem.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

constexpr int kN = 4;            // atoms
constexpr int kSteps = 3;        // steps per run(ref, …) call
constexpr double kEps = 0.0104;  // [eV]
constexpr double kSigma = 3.4;   // [Å]
constexpr double kCutoff = 6.0;  // [Å]
constexpr double kSkin = 0.3;    // [Å]
constexpr double kDt = 1.0;      // [fs]
constexpr double kSpacing = 3.8; // [Å]

/// @brief Publish the iteration identity into `kStep` frame meta, so the step
///        index travels through `cpu::to_molrs` into the molrs frame.
///
/// The one line `cpu::XyzRecorder::launch` runs before it writes a frame
/// (cpu/record/xyz_recorder.cpp), with the cadence gate and the file removed.
/// Pipeline glue, so it inherits `Component` directly.
class StepStamp final : public cpu::Component {
  public:
    std::string name() const override { return "step_stamp"; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +10)};
    }
    void launch(const cpu::LaunchContext& ctx) override {
        ctx.frame.meta.get(ATV::kStep) = static_cast<std::int64_t>(ctx.step);
    }
};

/// @brief Four argon atoms on a line in a 20 Å cubic box, at rest.
///
/// Built through the molrs C++ API the way `tests/cpu/test_materialize.cpp`
/// builds its fixtures: the schema-v2 required columns (x/y/z/atomic_number)
/// plus mass (optional at the boundary, but a zero mass makes velocity-Verlet
/// divide by zero) and tags.
::rust::Box<molrs::FrameRef> make_argon_frame() {
    auto fr = molrs::frame_new();

    std::vector<double> x(kN), y(kN), z(kN), mass(kN);
    std::vector<std::uint32_t> atomic_number(kN);
    std::vector<std::int32_t> tag(kN);
    for (int i = 0; i < kN; ++i) {
        const auto k = static_cast<std::size_t>(i);
        x[k] = 8.0 + kSpacing * static_cast<double>(i);
        y[k] = 10.0;
        z[k] = 10.0;
        mass[k] = 39.948;       // [amu]
        atomic_number[k] = 18u; // argon
        tag[k] = i + 1;
    }

    molrs::frame_set_column_f64(
        *fr, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));
    molrs::frame_set_column_f64(
        *fr, "atoms", "mass",
        ::rust::Slice<const double>(mass.data(), mass.size()));
    molrs::frame_set_column_u32(
        *fr, "atoms", "atomic_number",
        ::rust::Slice<const std::uint32_t>(atomic_number.data(),
                                           atomic_number.size()));
    molrs::frame_set_column_i32(
        *fr, "atoms", "tag",
        ::rust::Slice<const std::int32_t>(tag.data(), tag.size()));

    const std::vector<double> h = {20.0, 0.0, 0.0, 0.0, 20.0,
                                   0.0,  0.0, 0.0, 20.0};
    molrs::frame_set_box(*fr, ::rust::Slice<const double>(h.data(), h.size()));
    return fr;
}

/// @brief The `step` typed-meta entry of a molrs frame [step index].
///
/// `kStep` is `MetaKey<std::int64_t>{"step", /*serializable=*/true}`
/// (core/frame_schema.h), so it crosses the boundary as an `I64` meta entry.
std::int64_t molrs_step(const molrs::FrameRef& frame) {
    for (const molrs::MetaEntry& entry : molrs::frame_meta_entries(frame)) {
        if (std::string(entry.key.data(), entry.key.size()) != "step")
            continue;
        EXPECT_EQ(entry.dtype, molrs::MetaType::I64);
        return entry.i64_value;
    }
    ADD_FAILURE() << "molrs frame carries no 'step' meta entry — the probe "
                     "stamps kStep every end_of_step and to_molrs serializes "
                     "it back";
    return -1;
}

} // namespace

TEST(MDMolrsRun, MolrsRunsAreIndependentSimulations) {
    auto fr = make_argon_frame();

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.on(ATV::md::Phase::end_of_step).add(StepStamp{}, +10);
    sim.build();

    // ONE call expression, executed twice: both working frames occupy the same
    // stack slot, which is exactly the aliasing that makes the continuation
    // check in begin_run_() mistake call #1 for a continuation of call #0.
    for (int call = 0; call < 2; ++call) {
        SCOPED_TRACE("run(molrs::FrameRef&, " + std::to_string(kSteps)
                     + ") call #" + std::to_string(call));
        ASSERT_EQ(sim.run(*fr, kSteps), kSteps);
        EXPECT_EQ(molrs_step(*fr), std::int64_t{kSteps - 1})
            << "every run(molrs::FrameRef&, n) call is an independent "
               "simulation: the last step index must be n-1 = "
            << kSteps - 1
            << " on EVERY call. A larger value means the run was continued — "
               "begin_run_() compared the new working frame's address against "
               "a dangling frame_ptr_ left behind by the previous call's "
               "destroyed stack local, skipped setup_pass() and kept "
               "step_counter_ running.";
    }
}
