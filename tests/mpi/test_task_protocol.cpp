/// @file tests/mpi/test_task_protocol.cpp
/// @brief cpu::task RUN_TASK/SHUTDOWN protocol tests — TaskDispatcher on
///        rank 0, WorkerLoop on ranks > 0 (mpi-launcher-embedded-python-03).
///
/// Launched under `mpirun -n N`. Runs at N=1, 2, 4 per CTest config.
///
/// Session shape: every TEST is one complete "handshake → task(s) →
/// SHUTDOWN" session, coordinated by rank branches INSIDE the test body —
/// rank 0 constructs a TaskDispatcher and dispatches, ranks > 0 construct a
/// WorkerLoop and block in run() until rank 0 broadcasts SHUTDOWN. Every
/// collective is entered by every rank in the same order (lockstep), so no
/// scenario can deadlock a passing binary; the version-mismatch and
/// bad-TaskSpec scenarios pin exactly the lockstep-preserving error paths.
///
/// Reference tiering for the physics equivalence (a, c, e): the protocol
/// result must equal the SAME TaskSpec run directly (no protocol) through
/// TaskMaterializer on every rank — under MPI the direct run decomposes the
/// identical full-replica frame exactly like the protocol's post-broadcast
/// path, so typed Frame energy metadata (globally reduced by mpi::Reduce
/// at end_of_step) is comparable on rank 0 WITHOUT a second gather. Per-atom
/// trajectories are only whole on a single rank, so:
///   - every N:  energy relative 1e-10; result-frame shape (n atoms, the
///     full tag set in ascending order); passive columns (mass, type,
///     charge, image, box) exact against the fixture — this pins the
///     charge+image gather extension of this chain link;
///   - N == 1:   additionally positions/velocities per atom within 1e-10
///     (absolute) against the whole direct frame. NOT bitwise: the protocol
///     path is a different code path with identical physics.
/// OpenMP is pinned to 1 thread (test_task_materializer convention) so the
/// comparison sees no scheduler noise.
///
/// Verifies (spec .claude/specs/mpi-launcher-embedded-python-03-protocol.md):
///   - (a) dispatched ~10-atom LJ MD task ≡ direct run          [ac-002/003]
///   - (b) clean SHUTDOWN releases every worker, exit 0          [ac-004]
///   - (c) two sequential tasks in one session, worker loop-back [ac-005]
///   - (d) asymmetric version mismatch → every rank throws
///         std::runtime_error naming both versions, no deadlock  [ac-006]
///   - (e) materialize-failing TaskSpec → dispatch() throws on
///         rank 0, workers survive, next valid task succeeds     [ac-007]
///   - (f) Minimize uses the same result/gather protocol shape
///   - wire-shape constants + wrong-side validating ctors        [ac-001]
///
/// Not testable here: the MPI-uninitialized WorkerLoop ctor throw (this
/// binary's main initializes MPI before any test runs) — covered by the
/// non-MPI-build compile path, as in test_bcast_frame.cpp.

#include <gtest/gtest.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/task/task_materializer.h"
#include "atomiverse/cpu/task/task_protocol.h"
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;
using namespace ATV::cpu::task;

namespace {

// Pin OpenMP to one thread before any parallel region fires, so the
// direct-vs-protocol comparison sees identical force-reduction order
// (the test_task_materializer.cpp convention; setenv would be too late).
#ifdef _OPENMP
[[maybe_unused]] const bool kOmpPinned = [] {
    omp_set_num_threads(1);
    return true;
}();
#endif

int world_size() {
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

// ── LJ fixture (the test_task_materializer.cpp cluster, plus nonzero
//    charge/image so the gather's passive columns are actually pinned) ──

constexpr double kEps = 0.0104;  // [eV]
constexpr double kSigma = 3.4;   // [Å]
constexpr double kCutoff = 6.0;  // [Å]
constexpr double kSkin = 0.3;    // [Å]
constexpr double kDt = 1.0;      // [fs]
constexpr double kSpacing = 3.8; // [Å] ≈ 2^(1/6)·σ, near the LJ minimum
constexpr double kTinit = 10.0;  // [K]
constexpr unsigned kSeed = 42u;
constexpr int kNAtoms = 10;
constexpr int kNSteps = 5;

/// @brief Expected charge of fixture atom `i` [e] — nonzero, distinct, inert
///        under LJ MD (no coulomb style registered), so it must gather back
///        bit-exactly.
StorageReal fixture_charge(int i) {
    return StorageReal((i % 2 == 0 ? 1.0 : -1.0) * (0.1 + 0.01 * i));
}

/// @brief Expected image flag of fixture atom `i` on axis `d` — nonzero and
///        untouched by this run (all atoms stay > cutoff inside the 30 Å
///        box for 5 steps at 10 K, so nothing wraps).
std::int8_t fixture_image(int i, int d) {
    return static_cast<std::int8_t>((d + 1) * (i - 4));
}

/// @brief Deterministic 10-atom LJ cluster (two rows of five along x) in a
///        30 Å cubic box — literals only, so every construction on every
///        rank is bitwise identical. Tags stay at the grow() identity
///        default (tag(i) == i), giving a known full tag set {0..9}.
cpu::Frame make_lj_frame() {
    cpu::Frame frame(kNAtoms);
    for (int i = 0; i < kNAtoms; ++i) {
        const int col = i % 5;
        const int row = i / 5;
        frame.span(ATV::kAtom_x)[i] = 8.0 + kSpacing * static_cast<double>(col);
        frame.span(ATV::kAtom_y)[i] =
            10.0 + kSpacing * static_cast<double>(row);
        frame.span(ATV::kAtom_z)[i] = 15.0;
        frame.span(ATV::kAtom_mass)[i] = 39.948;
        frame.span(ATV::kAtom_type)[i] = 18;
        frame.span(ATV::kAtom_charge)[i] = fixture_charge(i);
        for (int d = 0; d < 3; ++d)
            frame.span(ATV::kAtom_image)(i, d) = fixture_image(i, d);
    }
    frame.box = ATV::cpu::Box::cubic(30.0);
    return frame;
}

/// @brief The valid MD TaskSpec every happy-path scenario dispatches:
///        NeighborList + PairLJCut explicit at force (distinct orders),
///        VelocityVerlet + InitVelocity via default bindings, 5 steps.
TaskSpec make_lj_spec() {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{kCutoff, kSkin},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{kEps, kSigma, kCutoff},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{kTinit, kSeed}, std::nullopt});
    return TaskSpec(MDTask{std::move(entries), kNSteps});
}

/// @brief Minimal typed Minimize task used only to exercise orchestration:
///        one iteration, then return control metadata and gather owner rows.
TaskSpec make_minimize_spec() {
    std::vector<MinimizeComponentEntry> entries;
    entries.push_back(MinimizeComponentEntry{
        NeighborListParams{kCutoff, kSkin},
        MinimizeBinding{ATV::minimizer::Phase::force, -10}});
    entries.push_back(MinimizeComponentEntry{
        PairLJCutParams{kEps, kSigma, kCutoff},
        MinimizeBinding{ATV::minimizer::Phase::force, 0}});
    entries.push_back(
        MinimizeComponentEntry{LBFGSParams{5, 0.01, 1}, std::nullopt});
    return TaskSpec(MinimizeTask{std::move(entries), 1, {}});
}

/// @brief Run `spec` directly (no protocol) on a fresh fixture frame. A
///        COLLECTIVE over every rank when world > 1 (the materializer's
///        Backend decomposes MPI_COMM_WORLD exactly like the protocol's
///        post-broadcast execution path) — every rank must call this at the
///        same point. After it returns, `scalars` holds the globally
///        reduced values on EVERY rank.
cpu::Frame direct_reference_run(const TaskSpec& spec) {
    cpu::Frame frame = make_lj_frame();
    TaskMaterializer mat(spec);
    mat.run(frame);
    return frame;
}

/// @brief |got - want| <= 1e-10 · |want| (relative; want must be nonzero —
///        callers assert non-vacuity separately).
void expect_energy_close(double got, double want) {
    EXPECT_LE(std::abs(got - want), 1e-10 * std::abs(want))
        << "got " << got << " want " << want;
}

/// @brief Rank-0 result-frame contract: n atoms, full tag set {0..9} in
///        ascending order (tag-ordered gather), passive columns (mass,
///        type, charge, image) exact against the fixture, box intact, and
///        energy within 1e-10 relative of the direct reference.
void expect_result_frame_contract(const cpu::Frame& result,
                                  const cpu::Frame& direct) {
    ASSERT_EQ(result.nlocal(), kNAtoms);
    EXPECT_EQ(result.nghost(), 0);

    for (int i = 0; i < kNAtoms; ++i) {
        SCOPED_TRACE("atom row " + std::to_string(i));
        // Identity tags + tag-ordered contract ⇒ row i carries tag i.
        ASSERT_EQ(result.span(ATV::kAtom_tag)[i], TagInt(i));
        EXPECT_EQ(result.span(ATV::kAtom_mass)[i], StorageReal(39.948));
        EXPECT_EQ(result.span(ATV::kAtom_type)[i], 18);
        // Passive columns ride the gather exactly (the charge+image gather
        // extension of this chain link; LJ MD never touches either).
        EXPECT_EQ(result.span(ATV::kAtom_charge)[i], fixture_charge(i));
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(result.span(ATV::kAtom_image)(i, d), fixture_image(i, d))
                << "image axis " << d;
    }

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(result.box.cell()(r, c), (r == c) ? 30.0 : 0.0)
                << "box(" << r << ", " << c << ")";

    // Globally reduced scalars, comparable on rank 0 without a second
    // gather (mpi::Reduce wrote the global values onto every rank's
    // direct-run frame at end_of_step).
    expect_energy_close(result.meta.get(ATV::kEnergy),
                        direct.meta.get(ATV::kEnergy));
    expect_energy_close(result.meta.get(ATV::kKinetic),
                        direct.meta.get(ATV::kKinetic));

    // Forces are part of the result schema; 5 steps near the LJ minimum
    // must leave SOME nonzero component (presence check — per-atom force
    // comparison at N > 1 would need a second gather).
    bool any_force = false;
    for (int i = 0; i < kNAtoms && !any_force; ++i)
        any_force = result.span(ATV::kAtom_fx)[i] != 0.0
                    || result.span(ATV::kAtom_fy)[i] != 0.0
                    || result.span(ATV::kAtom_fz)[i] != 0.0;
    EXPECT_TRUE(any_force) << "gathered result carries no forces";
}

/// @brief N == 1 deep comparison: the whole direct frame is on this rank,
///        so positions/velocities compare per atom within 1e-10 absolute
///        (same physics, different code path — NOT bitwise by contract).
void expect_trajectory_close_serial(const cpu::Frame& result,
                                    const cpu::Frame& direct) {
    ASSERT_EQ(result.nlocal(), direct.nlocal());
    for (int i = 0; i < direct.nlocal(); ++i) {
        SCOPED_TRACE("atom " + std::to_string(i));
        EXPECT_NEAR(result.span(ATV::kAtom_x)[i], direct.span(ATV::kAtom_x)[i],
                    1e-10);
        EXPECT_NEAR(result.span(ATV::kAtom_y)[i], direct.span(ATV::kAtom_y)[i],
                    1e-10);
        EXPECT_NEAR(result.span(ATV::kAtom_z)[i], direct.span(ATV::kAtom_z)[i],
                    1e-10);
        EXPECT_NEAR(result.span(ATV::kAtom_vx)[i],
                    direct.span(ATV::kAtom_vx)[i], 1e-10);
        EXPECT_NEAR(result.span(ATV::kAtom_vy)[i],
                    direct.span(ATV::kAtom_vy)[i], 1e-10);
        EXPECT_NEAR(result.span(ATV::kAtom_vz)[i],
                    direct.span(ATV::kAtom_vz)[i], 1e-10);
    }
}

} // namespace

// ── Wire shape: constants and header layout are the protocol ABI ────────────
//
// Same assertions on every rank, no collectives. Command starts at 1 so a
// zero-initialized buffer is never a legal command; the header is one
// MPI_BYTE Bcast, so it must be trivially copyable.

TEST(MpiTaskProtocol, WireConstantsAreStable) {
    EXPECT_EQ(static_cast<std::int32_t>(Command::RUN_TASK), 1);
    EXPECT_EQ(static_cast<std::int32_t>(Command::SHUTDOWN), 2);
    EXPECT_EQ(kProtocolVersion, 2);

    static_assert(std::is_same_v<std::underlying_type_t<Command>, std::int32_t>,
                  "Command must be int32 on the wire");
    static_assert(std::is_trivially_copyable_v<CommandHeader>,
                  "CommandHeader is broadcast as MPI_BYTE");
    static_assert(
        std::is_same_v<decltype(CommandHeader::version), std::int32_t>);
    static_assert(
        std::is_same_v<decltype(CommandHeader::command), std::int32_t>);
    static_assert(
        std::is_same_v<decltype(CommandHeader::payload_bytes), std::int64_t>);
}

// ── Validating ctors: wrong side throws locally, BEFORE any collective ──────
//
// Every rank checks its own wrong-side construction: rank 0 may not be a
// WorkerLoop, ranks > 0 may not be a TaskDispatcher, nobody may pass
// MPI_COMM_NULL to WorkerLoop. The rank/null checks are local (they must
// precede the handshake, or a throwing ctor would desync the collective),
// so no rank enters a collective here and lockstep is trivially preserved.

TEST(MpiTaskProtocol, WrongSideConstructionThrowsWithoutCollectives) {
    if (world_rank() == 0) {
        EXPECT_THROW(WorkerLoop(MPI_COMM_WORLD), std::runtime_error)
            << "WorkerLoop on rank 0 must throw";
    } else {
        EXPECT_THROW(TaskDispatcher(MPI_COMM_WORLD), std::runtime_error)
            << "TaskDispatcher off rank 0 must throw";
    }
    EXPECT_THROW(WorkerLoop(MPI_COMM_NULL), std::runtime_error)
        << "WorkerLoop with a null communicator must throw";
}

// ── (a) Dispatch ≡ direct run: ac-002 (N=2/4) and ac-003 (N=1) ──────────────

TEST(MpiTaskProtocol, DispatchedLjMdTaskMatchesDirectRun) {
    const TaskSpec spec = make_lj_spec();

    // Reference first, OUTSIDE the session (collective across all ranks).
    const cpu::Frame direct = direct_reference_run(spec);
    ASSERT_NE(direct.meta.get(ATV::kEnergy), 0.0)
        << "fixture produced zero LJ energy — the equivalence check would "
           "be vacuous";

    if (world_rank() == 0) {
        TaskDispatcher dispatcher(MPI_COMM_WORLD);
        DispatchResult dispatched = dispatcher.dispatch(spec, make_lj_frame());
        const cpu::Frame& result = dispatched.frame;

        EXPECT_EQ(dispatched.control.iterations, kNSteps);
        EXPECT_FALSE(dispatched.control.converged);
        EXPECT_TRUE(dispatched.control.stop_reason.empty());

        expect_result_frame_contract(result, direct);
        if (world_size() == 1)
            expect_trajectory_close_serial(result, direct);

        dispatcher.shutdown();
    } else {
        WorkerLoop worker(MPI_COMM_WORLD);
        EXPECT_NO_THROW(
            worker.run()); // one RUN_TASK, then SHUTDOWN releases it
    }
}

// ── (b) Clean SHUTDOWN: ac-004 ──────────────────────────────────────────────
//
// A zero-task session: constructing the pair completes the handshake, then
// SHUTDOWN must release every worker's run() and let every rank fall
// through to MPI_Finalize in main (exit 0 is the binary completing; no
// destructor may issue an MPI call, or finalize would blow up).

TEST(MpiTaskProtocol, ShutdownReleasesEveryWorkerCleanly) {
    if (world_rank() == 0) {
        TaskDispatcher dispatcher(MPI_COMM_WORLD);
        EXPECT_NO_THROW(dispatcher.shutdown());
    } else {
        WorkerLoop worker(MPI_COMM_WORLD);
        EXPECT_NO_THROW(worker.run());
    }
}

// ── (f) A second staged task reuses the protocol without MD erasure ────

TEST(MpiTaskProtocol, MinimizeReturnsControlAndGatheredFrame) {
    const TaskSpec spec = make_minimize_spec();

    if (world_rank() == 0) {
        TaskDispatcher dispatcher(MPI_COMM_WORLD);
        DispatchResult result = dispatcher.dispatch(spec, make_lj_frame());

        // Structural contract only: one typed loop iteration ran and the DD
        // owner rows were gathered through the same result envelope as MD.
        EXPECT_EQ(result.control.iterations, 1);
        ASSERT_EQ(result.frame.nlocal(), kNAtoms);
        for (int i = 0; i < kNAtoms; ++i)
            EXPECT_EQ(result.frame.span(ATV::kAtom_tag)[i], TagInt(i));

        dispatcher.shutdown();
    } else {
        WorkerLoop worker(MPI_COMM_WORLD);
        EXPECT_NO_THROW(worker.run());
    }
}

// ── (c) Two sequential tasks in one session: ac-005 ─────────────────────────
//
// Workers must loop back to the blocking Bcast after the first RUN_TASK.
// Each dispatch starts from a fresh copy of the same initial frame with the
// same seed, so both must match the direct reference — and, having executed
// the identical code path twice in one process (OpenMP pinned to 1 thread,
// fixed MPI reduction topology), they must also agree with each other
// bitwise.

TEST(MpiTaskProtocol, TwoSequentialTasksInOneSessionBothMatch) {
    const TaskSpec spec = make_lj_spec();

    const cpu::Frame direct = direct_reference_run(spec);
    ASSERT_NE(direct.meta.get(ATV::kEnergy), 0.0);

    if (world_rank() == 0) {
        TaskDispatcher dispatcher(MPI_COMM_WORLD);
        DispatchResult first_result =
            dispatcher.dispatch(spec, make_lj_frame());
        DispatchResult second_result =
            dispatcher.dispatch(spec, make_lj_frame());
        const cpu::Frame& first = first_result.frame;
        const cpu::Frame& second = second_result.frame;

        expect_result_frame_contract(first, direct);
        expect_result_frame_contract(second, direct);

        // Same-process determinism: dispatch #2 is dispatch #1 replayed.
        EXPECT_EQ(first.meta.get(ATV::kEnergy), second.meta.get(ATV::kEnergy));
        ASSERT_EQ(first.nlocal(), second.nlocal());
        for (int i = 0; i < first.nlocal(); ++i) {
            SCOPED_TRACE("atom " + std::to_string(i));
            EXPECT_EQ(first.span(ATV::kAtom_x)[i],
                      second.span(ATV::kAtom_x)[i]);
            EXPECT_EQ(first.span(ATV::kAtom_y)[i],
                      second.span(ATV::kAtom_y)[i]);
            EXPECT_EQ(first.span(ATV::kAtom_z)[i],
                      second.span(ATV::kAtom_z)[i]);
            EXPECT_EQ(first.span(ATV::kAtom_vx)[i],
                      second.span(ATV::kAtom_vx)[i]);
            EXPECT_EQ(first.span(ATV::kAtom_vy)[i],
                      second.span(ATV::kAtom_vy)[i]);
            EXPECT_EQ(first.span(ATV::kAtom_vz)[i],
                      second.span(ATV::kAtom_vz)[i]);
        }

        dispatcher.shutdown();
    } else {
        WorkerLoop worker(MPI_COMM_WORLD);
        EXPECT_NO_THROW(worker.run()); // two RUN_TASKs, then SHUTDOWN
    }
}

// ── (d) Asymmetric version mismatch aborts lockstep: ac-006 ─────────────────
//
// Rank 1 constructs at kProtocolVersion + 1 (the test seam); every other
// rank at the default. The handshake's allreduce must surface the mismatch
// on EVERY rank — including ranks whose local version matches rank 0's —
// and every rank's ctor must throw std::runtime_error naming BOTH version
// numbers. The allreduce is the handshake's last collective, so throwing
// afterwards keeps lockstep: the test completing at all is the no-deadlock
// assertion.

TEST(MpiTaskProtocol, VersionMismatchThrowsOnEveryRankNamingBothVersions) {
    if (world_size() == 1)
        GTEST_SKIP() << "single-rank handshake is trivially self-consistent";

    const std::int32_t good = kProtocolVersion;
    const std::int32_t bad = kProtocolVersion + 1;

    bool threw = false;
    std::string message;
    try {
        if (world_rank() == 0) {
            TaskDispatcher dispatcher(MPI_COMM_WORLD, good);
        } else {
            WorkerLoop worker(MPI_COMM_WORLD, world_rank() == 1 ? bad : good);
        }
    } catch (const std::runtime_error& e) {
        threw = true;
        message = e.what();
    }

    EXPECT_TRUE(threw) << "rank " << world_rank()
                       << " must throw on the version mismatch";
    EXPECT_NE(message.find(std::to_string(good)), std::string::npos)
        << "message must name version " << good << ": \"" << message << "\"";
    EXPECT_NE(message.find(std::to_string(bad)), std::string::npos)
        << "message must name version " << bad << ": \"" << message << "\"";
}

// ── (e) Bad TaskSpec propagates, session survives: ac-007 ───────────────────
//
// The bad spec is VALID at TaskSpec construction (integrator only, n_steps
// fine) but fails materialization deterministically on every rank:
// MD::build's validate_components rejects a pipeline with no
// Capability::Force component. The setup-status allreduce must convert
// every rank's local failure into a synchronized skip of run/gather:
// dispatch() throws on rank 0, workers absorb the task-level failure and
// loop back, and the SAME session then executes a valid task correctly.

TEST(MpiTaskProtocol, MaterializeFailurePropagatesAndSessionSurvives) {
    std::vector<MDComponentEntry> bad_entries;
    bad_entries.push_back(
        MDComponentEntry{VelocityVerletParams{kDt}, std::nullopt});
    const TaskSpec bad_spec(MDTask{std::move(bad_entries), kNSteps});

    // Determinism prerequisite, same point on every rank: this spec fails
    // to materialize locally (the failure dispatch() must propagate).
    EXPECT_THROW(TaskMaterializer{bad_spec}, std::runtime_error);

    const TaskSpec good_spec = make_lj_spec();
    const cpu::Frame direct = direct_reference_run(good_spec);
    ASSERT_NE(direct.meta.get(ATV::kEnergy), 0.0);

    if (world_rank() == 0) {
        TaskDispatcher dispatcher(MPI_COMM_WORLD);

        EXPECT_THROW(dispatcher.dispatch(bad_spec, make_lj_frame()),
                     std::runtime_error);

        // The session survives the task-level failure: the very next
        // dispatch must produce the correct physics.
        DispatchResult result = dispatcher.dispatch(good_spec, make_lj_frame());
        expect_result_frame_contract(result.frame, direct);
        EXPECT_EQ(result.control.iterations, kNSteps);

        dispatcher.shutdown();
    } else {
        WorkerLoop worker(MPI_COMM_WORLD);
        // run() must NOT throw on the bad task (task-level failures loop
        // back; only protocol-level corruption is fatal) and must return
        // cleanly on SHUTDOWN after serving the valid task.
        EXPECT_NO_THROW(worker.run());
    }
}

// ── Custom main that initializes MPI before/after GoogleTest ────────────
//
// See test_topology.cpp for the OpenMPI 5.0.x teardown rationale (CTest
// reads the gtest summary via PASS_REGULAR_EXPRESSION, not the exit code).

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
