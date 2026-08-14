/// @file tests/cpu/test_task_materializer.cpp
/// @brief Unit tests for `cpu::task::TaskMaterializer` — the rank-agnostic
///        rebuilder that turns a TaskSpec into a runnable pipeline.
///
/// Spec: .claude/specs/mpi-launcher-embedded-python-02-codec.md. The fidelity
/// contract under test: the driver materialized from a TaskSpec, consumed via
/// the single task-kind-agnostic `run(frame)` entry, is THE SAME object a user
/// would hand-build — so a 5-step LJ MD run and an AM1 single point must agree
/// BITWISE (EXPECT_EQ, no tolerance) with directly-built `cpu::md::MD` /
/// `cpu::Evaluate` pipelines on identically constructed frames. Serial,
/// sub-second, no MPI, no I/O.

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#endif
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#ifdef ATV_HAS_IO
#include "atomiverse/cpu/record/xyz_recorder.h"
#endif
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/task/task_materializer.h"
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;
using namespace ATV::cpu::task;

namespace {

// The bitwise-equality claims below compare two runs of the SAME pipeline
// configuration inside one process. Under OpenMP (> 1 thread) the force
// reduction order varies between runs, so bitwise equality only holds
// single-threaded — pin it before any parallel region fires (the
// test_evaluate_driver / test_scf_lindep convention; setenv would be too
// late, libgomp snapshots OMP_NUM_THREADS in its ELF constructor). This
// does not weaken the fidelity contract (materializer ≡ hand-built
// driver); it only removes scheduler noise from the comparison.
#ifdef _OPENMP
[[maybe_unused]] const bool kOmpPinned = [] {
    omp_set_num_threads(1);
    return true;
}();
#endif

// ── LJ fixture ──────────────────────────────────────────────────────────

// Argon-like LJ parameters; kSpacing ≈ 2^(1/6)·σ so the cluster starts near
// the LJ minimum and 5 steps at 1 fs stay tame.
constexpr double kEps = 0.0104;  // [eV]
constexpr double kSigma = 3.4;   // [Å]
constexpr double kCutoff = 6.0;  // [Å]
constexpr double kSkin = 0.3;    // [Å]
constexpr double kDt = 1.0;      // [fs]
constexpr double kSpacing = 3.8; // [Å]
constexpr double kTinit = 10.0;  // [K]
constexpr unsigned kSeed = 42u;
constexpr int kNSteps = 5;

// Electrostatics. α is the Ewald splitting parameter [1/Å]: it must be
// IDENTICAL on the real-space (`CoulLong`) and reciprocal-space
// (`KSpacePME`) halves or the driver refuses to build, so both the TaskSpec
// entries and the hand-built twin read it from here. 3.5/r_c is the LAMMPS
// rule of thumb: erfc(3.5) ≈ 7e-7, i.e. the screened real-space pair term
// is negligible by the cutoff (regressions/md/npt_pme_nacl/REFERENCE.md).
constexpr double kAlpha = 3.5 / kCutoff; // [1/Å]
constexpr double kPmeSpacing = 1.0;      // [Å] target PME grid spacing
constexpr int kPmeOrder = 6;             // B-spline interpolation order

/// @brief Deterministic 10-atom LJ cluster (two rows of five along x) in a
///        30 Å cubic box — literals only, so two constructions are bitwise
///        identical (the test_bcast_frame reference-frame pattern).
cpu::Frame make_lj_frame() {
    constexpr int n = 10;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        const int col = i % 5;
        const int row = i / 5;
        frame.span(ATV::kAtom_x)[i] = 8.0 + kSpacing * static_cast<double>(col);
        frame.span(ATV::kAtom_y)[i] =
            10.0 + kSpacing * static_cast<double>(row);
        frame.span(ATV::kAtom_z)[i] = 15.0;
        frame.span(ATV::kAtom_mass)[i] = 39.948;
        frame.span(ATV::kAtom_type)[i] = 18;
    }
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

/// @brief `make_lj_frame()` with alternating ±1 e on the charge column.
///
/// Ten atoms, so the cell is net neutral — which is what the Ewald split
/// assumes (a charged cell would be silently neutralized by the PME
/// background and the comparison would be about that, not electrostatics).
/// Literals only, so two constructions stay bitwise identical.
cpu::Frame make_charged_lj_frame() {
    cpu::Frame frame = make_lj_frame();
    for (int i = 0; i < frame.nlocal(); ++i)
        frame.span(ATV::kAtom_charge)[i] = (i % 2 == 0) ? 1.0 : -1.0;
    return frame;
}

/// @brief LJ cluster plus a two-atom harmonic bond (atoms 0–1, bond type 0).
///        Tag map populated first so `add_bond` can resolve global tags,
///        exactly as in tests/mpi/test_bcast_frame.cpp / test_pe_single_chain.
cpu::Frame make_bonded_lj_frame() {
    cpu::Frame frame = make_lj_frame();
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_bond(/*ti=*/0, /*tj=*/1, /*type=*/0);
    frame.build_special_list();
    return frame;
}

/// @brief Bitwise frame agreement: scalars.energy and every atoms.x/y/z.
void expect_frames_bitwise_equal(const cpu::Frame& got,
                                 const cpu::Frame& want) {
    EXPECT_EQ(got.meta.get(ATV::kEnergy), want.meta.get(ATV::kEnergy));
    ASSERT_EQ(got.nlocal(), want.nlocal());
    const int n = want.nlocal();
    for (int i = 0; i < n; ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        EXPECT_EQ(got.span(ATV::kAtom_x)[i], want.span(ATV::kAtom_x)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_y)[i], want.span(ATV::kAtom_y)[i]);
        EXPECT_EQ(got.span(ATV::kAtom_z)[i], want.span(ATV::kAtom_z)[i]);
    }
}

std::vector<std::vector<std::string>>
plan_signature(const Plan<cpu::Component>& plan) {
    std::vector<std::vector<std::string>> result;
    result.reserve(plan.slots.size());
    for (const auto& slot : plan.slots) {
        auto& names = result.emplace_back();
        names.reserve(slot.ops.size());
        for (const auto& op : slot.ops)
            names.push_back(op.component->name());
    }
    return result;
}

// ── (a) MD equivalence: LJ + NeighborList + VelocityVerlet + InitVelocity ─

TEST(TaskMaterializer, MdLjRunMatchesHandBuiltDriverBitwise) {
    // TaskSpec mirror of the hand-built registration below, entry for entry.
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_lj_frame();
    TaskMaterializer mat(spec);
    EXPECT_EQ(mat.kind(), TaskKind::MD);
    mat.run(materialized_frame);

    // Hand-built driver with identical ctor args and identical registration
    // calls (the minimal working LJ wiring of tests/cpu/test_md_no_alloc.cpp
    // and tests/mpi/test_lj_nve.cpp).
    cpu::Frame direct_frame = make_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

// ── (b) MD variant: + BondHarmonic (explicit) + NoseHoover (default) ─────

TEST(TaskMaterializer, MdBondedThermostattedRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(MDComponentEntry{BondHarmonicParams{{2.0}, {kSpacing}},
                                       MDBinding{ATV::md::Phase::force, 1}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(
        MDComponentEntry{NoseHooverParams{kTinit, 100.0, kDt}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.on(ATV::md::Phase::force).add(cpu::BondHarmonic{{2.0}, {kSpacing}}, 1);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_thermostat(cpu::md::NoseHoover{kTinit, 100.0, kDt});
    sim.build();
    sim.run(direct_frame, kNSteps);

    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

TEST(TaskMaterializer, MdLangevinRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    // T_start == T_stop, so the ramp span the materializer installs
    // (`set_run_span(0, n_steps)`) is inert — T_live = T_start + delta·0 —
    // and the hand-built twin needs no matching call. Give this entry a
    // real ramp and the twin must gain `lang.set_run_span(0, kNSteps)`.
    entries.push_back(MDComponentEntry{
        FixLangevinParams{kTinit, kTinit, 100.0, kDt, kSeed}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_thermostat(cpu::md::FixLangevin{kTinit, kTinit, 100.0, kDt, kSeed});
    sim.build();
    sim.run(direct_frame, kNSteps);

    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

TEST(TaskMaterializer, MdNptRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    // The knob tail is spelled out with the *C++ constructor defaults*
    // ({} would value-initialize couple_axis to {false,false,false} and
    // T_stop to 0.0, neither of which is the hand-built twin's behaviour).
    // T_stop < 0 folds to T_target, so the materializer's set_run_span is
    // inert here and the twin needs none.
    entries.push_back(MDComponentEntry{NoseHooverNptParams{kTinit,
                                                           100.0,
                                                           0.0,
                                                           1000.0,
                                                           kDt,
                                                           0.0,
                                                           3,
                                                           3,
                                                           {true, true, true},
                                                           -1.0},
                                       std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_integrator(cpu::md::NoseHooverNPT{kTinit, 100.0, 0.0, 1000.0, kDt});
    sim.build();
    sim.run(direct_frame, kNSteps);

    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

// The NPT tail knobs (drag, chain lengths, per-axis couple, T_stop ramp)
// must reach the constructor — every one is set to a non-default value, and
// the ramp span the materializer installs (`set_run_span(0, n_steps)`) is
// live because T_stop != T_target.
TEST(TaskMaterializer, MdNptKnobsRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(MDComponentEntry{NoseHooverNptParams{kTinit,
                                                           100.0,
                                                           0.0,
                                                           1000.0,
                                                           kDt,
                                                           0.5,
                                                           4,
                                                           4,
                                                           {true, true, false},
                                                           50.0},
                                       std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    cpu::md::NoseHooverNPT nh{
        kTinit, 100.0, 0.0, 1000.0, kDt, 0.5, 4, 4, {true, true, false}, 50.0};
    nh.set_run_span(0, kNSteps);
    sim.set_integrator(std::move(nh));
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

// ParrinelloRahmanNPT is an Integrator (it replaces VelocityVerlet, like
// NoseHooverNPT). Note the constructor order T, P, tau_T, tau_P, dt — it
// differs from NoseHooverNPT's T, tau_T, P, tau_P, dt, so a swapped
// materializer arm would show up as a bitwise mismatch here.
TEST(TaskMaterializer, MdPrNptRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(MDComponentEntry{
        ParrinelloRahmanNptParams{kTinit, 0.0, 100.0, 1000.0, kDt},
        std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    // No ramp on PR-NPT — no set_run_span on either side.
    sim.set_integrator(
        cpu::md::ParrinelloRahmanNPT{kTinit, 0.0, 100.0, 1000.0, kDt});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

// The Berendsen barostat is a Barostat kind: it rides alongside a plain
// VelocityVerlet integrator via `set_barostat`, unlike the NPT integrators
// above which replace it.
TEST(TaskMaterializer, MdBerendsenRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(MDComponentEntry{
        BerendsenBarostatParams{0.0, 1000.0, kDt, 2.0}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_bonded_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_bonded_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_barostat(cpu::md::Barostat{0.0, 1000.0, kDt, 2.0});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

// ── (b′) Electrostatics through the plural categorical evaluator slot ────
//
// Every Coulomb component is materialized categorically (`std::nullopt`
// binding → `set_evaluator`), which binds at the MD eval phase order 0 —
// the same slot PairLJCut already occupies. That is deliberate: the
// evaluator slot is plural, so the Coulomb style JOINS the pair style
// instead of replacing it. Equal-order ops keep registration order
// (`Plan` stable_sort), so the hand-built twin registers in exactly the
// entry order of the spec — a different order sums the force
// contributions in a different sequence and the bitwise claim would fail
// for a reason that has nothing to do with fidelity.

TEST(TaskMaterializer, MdCoulCutRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{PairCoulCutParams{kCutoff}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_charged_lj_frame();
    TaskMaterializer mat(spec);
    EXPECT_EQ(mat.kind(), TaskKind::MD);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_charged_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_evaluator(cpu::PairCoulCut{kCutoff});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}

#ifdef ATV_HAS_KSPACE
// CoulLong + KSpacePME is ONE physical model split across two components,
// and the driver enforces that: exactly one real-space participant, exactly
// one reciprocal-space participant, matching α. Both halves therefore have
// to survive the wire round the same way — dropping either in the
// materializer is a build error, and mis-ordering them is a bitwise
// mismatch here.
TEST(TaskMaterializer, MdCoulLongPmeRunMatchesHandBuiltDriverBitwise) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{CoulLongParams{kAlpha, kCutoff}, std::nullopt});
    entries.push_back(MDComponentEntry{
        KSpacePmeParams{kAlpha, kPmeSpacing, kPmeOrder}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_charged_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_charged_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_evaluator(cpu::CoulLong{kAlpha, kCutoff});
    // KSpacePME is move-only (it owns FFTW plans), so the prvalue goes
    // straight into the owning setter — no named local to move from.
    sim.set_evaluator(cpu::KSpacePME{kAlpha, kPmeSpacing, kPmeOrder});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);
}
#endif // ATV_HAS_KSPACE

// The matched-pair rule is the driver's, not the materializer's, but the
// materializer is what a launcher payload reaches it through: a spec that
// names only the real-space half must fail loudly at build time (the
// TaskMaterializer constructor calls `MD::build()`), never run half an
// electrostatic model. CoulLong alone needs no ATV_HAS_KSPACE guard — it
// is a plain potential, built whenever POTENTIALS is.
TEST(TaskMaterializer, MdCoulLongWithoutReciprocalThrowsOnBuild) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{CoulLongParams{kAlpha, kCutoff}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    try {
        TaskMaterializer mat(spec);
        cpu::Frame frame = make_charged_lj_frame();
        (void)mat.run(frame);
        FAIL() << "a real-space-only electrostatic spec was accepted";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what())
                      .find("requires exactly one reciprocal-space partner"),
                  std::string::npos)
            << err.what();
    }
}

// ── (b″) Recorder through the categorical recorder slot ─────────────────

#ifdef ATV_HAS_IO
/// @brief Per-frame atom lines of an XYZ trajectory ("symbol x y z" each).
///
/// The comment line is deliberately dropped: it carries typed meta rendered
/// from a molrs hash map, so its field ORDER is not reproducible between two
/// runs even when the physics is bit-identical. The count line and the atom
/// lines are.
std::vector<std::vector<std::string>>
read_xyz_atom_lines(const std::string& path) {
    std::ifstream is(path);
    if (!is)
        throw std::runtime_error("read_xyz_atom_lines: cannot open " + path);
    std::vector<std::vector<std::string>> frames;
    std::string line;
    while (std::getline(is, line)) {
        int n = 0;
        std::istringstream head(line);
        if (!(head >> n))
            throw std::runtime_error("read_xyz_atom_lines: bad count line: '"
                                     + line + "'");
        if (!std::getline(is, line)) // comment line
            throw std::runtime_error("read_xyz_atom_lines: truncated frame");
        auto& atoms = frames.emplace_back();
        atoms.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (!std::getline(is, line))
                throw std::runtime_error(
                    "read_xyz_atom_lines: truncated at atom "
                    + std::to_string(i));
            atoms.push_back(line);
        }
    }
    return frames;
}

// An XyzRecorder entry rides the categorical (`std::nullopt`) recorder slot —
// `set_record`, which binds it at md::Phase::end_of_step. The recorder is
// FrameImmutable, so the two runs' PHYSICS must stay bit-identical (that is
// the fidelity claim), and their two trajectory FILES must agree frame for
// frame and atom line for atom line: same cadence, same geometry, one file
// per path.
TEST(TaskMaterializer, MdXyzRecorderRunMatchesHandBuiltDriverBitwise) {
    const auto path_a =
        (std::filesystem::temp_directory_path() / "atv_task_xyz_spec.xyz")
            .string();
    const auto path_b =
        (std::filesystem::temp_directory_path() / "atv_task_xyz_direct.xyz")
            .string();
    std::error_code ec;
    std::filesystem::remove(path_a, ec);
    std::filesystem::remove(path_b, ec);

    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{XyzRecorderParams{path_a, 2}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    cpu::Frame materialized_frame = make_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(materialized_frame);

    cpu::Frame direct_frame = make_lj_frame();
    cpu::md::MD sim;
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    sim.set_observer(cpu::md::InitVelocity{kTinit, kSeed});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_record(cpu::XyzRecorder{path_b, 2});
    sim.build();
    sim.run(direct_frame, kNSteps);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";
    expect_frames_bitwise_equal(materialized_frame, direct_frame);

    ASSERT_TRUE(std::filesystem::exists(path_a));
    ASSERT_TRUE(std::filesystem::exists(path_b));
    const auto frames_a = read_xyz_atom_lines(path_a);
    const auto frames_b = read_xyz_atom_lines(path_b);
    // kNSteps = 5 → ctx.step ∈ {0…4}; the freq-2 gate opens on 0, 2, 4.
    EXPECT_EQ(frames_a.size(), 3u);
    ASSERT_EQ(frames_a.size(), frames_b.size());
    for (std::size_t f = 0; f < frames_a.size(); ++f) {
        SCOPED_TRACE("frame " + std::to_string(f));
        EXPECT_EQ(frames_a[f], frames_b[f]);
    }

    std::filesystem::remove(path_a, ec);
    std::filesystem::remove(path_b, ec);
}
#else
// Gating contract (py-recorder-xyz ac-001): the params struct and the wire ID
// are unconditional, so an IO-less build still DECODES a payload naming the
// recorder — and must then refuse it loudly at materialize time rather than
// running a trajectory-less MD. Same shape as `throw_kspace_not_built`.
TEST(TaskMaterializer, MdXyzRecorderWithoutIoBuildThrows) {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{XyzRecorderParams{"traj.xyz", 2}, std::nullopt});
    const TaskSpec spec(MDTask{std::move(entries), kNSteps});

    try {
        TaskMaterializer mat(spec);
        cpu::Frame frame = make_lj_frame();
        mat.run(frame);
        FAIL() << "an XyzRecorder payload was accepted in an IO-less build";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what()).find("ATV_BUILD_IO"),
                  std::string::npos)
            << err.what();
    }
}
#endif // ATV_HAS_IO

// ── (c) SinglePoint AM1 equivalence on H2 ───────────────────────────────

/// @brief Fixed-geometry H2 (0.74 Å, 20 Å box) — the exact fixture shape of
///        tests/cpu/semiempirical/test_am1_scaffold.cpp
///        `RunConvergesClosedShellH2AgainstReferenceChargeOracle`.
cpu::Frame make_h2_frame() {
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1; // Z = 1 (hydrogen)
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

TEST(TaskMaterializer, SinglePointAm1H2MatchesHandBuiltEvaluateBitwise) {
    // AM1EvaluatorParams mirroring the NDDOConfig defaults, spelled out so
    // the two pipelines demonstrably share one configuration.
    AM1EvaluatorParams params{};
    params.charge = 0;
    params.multiplicity = 1;
    params.max_iter = 200;
    params.density_tol = 1.0e-8;
    params.energy_tol = 1.0e-10;
    const TaskSpec spec(SinglePointTask{params});

    cpu::Frame materialized_frame = make_h2_frame();
    TaskMaterializer mat(spec);
    EXPECT_EQ(mat.kind(), TaskKind::SinglePoint);
    mat.run(materialized_frame);

    // Directly-built single point: Evaluate + AM1Evaluator (same args as
    // AM1EvaluatorParams / constructor — no intermediate Config bag).
    cpu::Frame direct_frame = make_h2_frame();
    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate)
        .add(cpu::semiempirical::AM1Evaluator{0, 1, 200, 1.0e-8, 1.0e-10});
    eval.build();
    eval.run(direct_frame);

    ASSERT_NE(direct_frame.meta.get(ATV::kEnergy), 0.0)
        << "AM1 H2 single point produced zero energy — the equivalence "
           "check would be vacuous";
    EXPECT_EQ(materialized_frame.meta.get(ATV::kEnergy),
              direct_frame.meta.get(ATV::kEnergy));
    ASSERT_EQ(materialized_frame.nlocal(), direct_frame.nlocal());
    for (int i = 0; i < direct_frame.nlocal(); ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        EXPECT_EQ(materialized_frame.span(ATV::kAtom_charge)[i],
                  direct_frame.span(ATV::kAtom_charge)[i]);
    }
}

// ── (d) kind() accessor ─────────────────────────────────────────────────

TEST(TaskMaterializer, MinimizePlanMatchesDirectTypedConstruction) {
    const ForceConvergedParams convergence{0.02, 0.01, 0.003, 0.002, 2e-6};
    std::vector<MinimizeComponentEntry> entries;
    entries.push_back(MinimizeComponentEntry{
        NeighborListParams{kCutoff, kSkin},
        MinimizeBinding{ATV::minimizer::Phase::force, -10}});
    entries.push_back(MinimizeComponentEntry{
        PairLJCutParams{kEps, kSigma, kCutoff},
        MinimizeBinding{ATV::minimizer::Phase::force, 0}});
    entries.push_back(
        MinimizeComponentEntry{LBFGSParams{7, 0.02, 12}, std::nullopt});
    const TaskSpec spec(MinimizeTask{std::move(entries), 11, convergence});

    TaskMaterializer materialized(spec);
    EXPECT_EQ(materialized.kind(), TaskKind::Minimize);

    cpu::minimizer::Minimize direct;
    direct.add_criterion(cpu::ForceConverged{
        convergence.ftol, convergence.rms_ftol, convergence.max_dtol,
        convergence.rms_dtol, convergence.etol});
    direct.on(ATV::minimizer::Phase::force)
        .add(cpu::md::NeighborList<>{kCutoff, kSkin}, -10)
        .add(cpu::PairLJCut{kEps, kSigma, kCutoff}, 0);
    // An unbound (`std::nullopt`) optimizer entry is materialized through the
    // categorical setter, so the hand-built mirror uses it too.
    direct.set_optimizer(cpu::minimizer::LBFGS{7, 0.02, 12});
    direct.build();

    EXPECT_EQ(plan_signature(materialized.plan()),
              plan_signature(direct.plan()));
}

TEST(TaskMaterializer, KindReportsTaskKindOfTheSpec) {
    // Minimal VALID MD composition: MD::build() (fired by the materializer
    // ctor) demands an Integrator plus a Force-capable component.
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    const TaskSpec md_spec(MDTask{std::move(entries), 1});
    EXPECT_EQ(TaskMaterializer(md_spec).kind(), TaskKind::MD);

    const TaskSpec sp_spec(SinglePointTask{AM1EvaluatorParams{}});
    EXPECT_EQ(TaskMaterializer(sp_spec).kind(), TaskKind::SinglePoint);

    std::vector<MinimizeComponentEntry> min_entries;
    min_entries.push_back(MinimizeComponentEntry{
        NeighborListParams{kCutoff, kSkin},
        MinimizeBinding{ATV::minimizer::Phase::force, -10}});
    min_entries.push_back(MinimizeComponentEntry{
        PairLJCutParams{kEps, kSigma, kCutoff},
        MinimizeBinding{ATV::minimizer::Phase::force, 0}});
    min_entries.push_back(MinimizeComponentEntry{LBFGSParams{}, std::nullopt});
    const TaskSpec min_spec(MinimizeTask{std::move(min_entries), 1, {}});
    EXPECT_EQ(TaskMaterializer(min_spec).kind(), TaskKind::Minimize);
}

} // namespace
