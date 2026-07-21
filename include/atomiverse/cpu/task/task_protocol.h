/// @file cpu/task/task_protocol.h
/// @brief The two-verb MPI task protocol: RUN_TASK / SHUTDOWN
///        (mpi-launcher-embedded-python-03).
///
/// Rank 0 (`TaskDispatcher`) makes ONE task-granular broadcast per submitted
/// task — a fixed `CommandHeader`, the TaskSpec bytes (02's codec), and the
/// Frame (01's `cpu::comm::bcast_frame`) — then EVERY rank (rank 0 included)
/// enters the same native compute path: `TaskMaterializer` (construction is
/// materialize + build) followed by its task-kind-agnostic `run(frame)`.
/// Worker ranks (`WorkerLoop`) block between tasks and never exit before
/// SHUTDOWN. There are exactly two verbs; per-object sync verbs
/// (CREATE_SYSTEM/…) are contractually out of scope.
///
/// **Lockstep discipline.** Every rank participates in every collective in
/// the same order. Local failures in unpack/materialize/build and in run are
/// caught, folded into a status code, and combined with `MPI_Allreduce(MAX)`
/// at two fixed points (post-setup, post-run) — any rank's failure makes
/// every rank skip the rest of the task in unison; `dispatch()` throws on
/// rank 0 and workers return to blocking (the session survives). Collectives
/// *inside* `build()`/`run()` stay lockstep because the same TaskSpec runs
/// the same deterministic code path on every rank; an asymmetric mid-run
/// collective failure is out of this protocol's recovery scope (same
/// boundary as today's native MPI runs).
///
/// **Version handshake.** Both constructors join one collective handshake
/// (root broadcasts its protocol version, every rank compares, the verdict
/// is all-reduced): a mismatch throws `std::runtime_error` on EVERY rank,
/// naming both versions. Dispatcher and workers must therefore construct in
/// lockstep. The header's `version` field is defence in depth.
///
/// **Results.** After an MD or Minimize task the owning decomposition is released from
/// the materialized driver (`test_eam_multirank` pattern) and the owners are
/// gathered to rank 0 (`gather_to_root` → `to_frame`, which carries
/// x/v/f/mass/charge/type/tag/image + cell); serializable Frame metadata needs no gather
/// (the Reduce component all-reduces global scalars onto every rank each
/// step). SinglePoint frames are full replicas — rank 0's own post-run frame
/// IS the result. Frame remains the only state currency; `TaskResult` is a
/// separate control-plane record (iterations, convergence, stop reason).
///
/// **Communicator.** v2 supports `comm == MPI_COMM_WORLD` (or a congruent
/// communicator with identical ranks): the Frame broadcast rides
/// `cpu::comm::bcast_frame`, which is WORLD-fixed by the
/// `scatter_atoms_from_root` precedent. Communicator parameterization is a
/// chain-01 boundary, not this layer's.
///
/// **Shutdown ordering.** Neither class owns its communicator (no dup, no
/// free) and destructors make no MPI calls; each task's driver is a local of
/// the execution path, so `Backend::shutdown()`/`release_comm()` complete
/// before the worker re-enters its blocking broadcast. The caller (the
/// launcher's `main`) owns `MPI_Finalize`.
///
/// @see cpu/task/task_spec.h / task_codec.h / task_materializer.h  (02)
/// @see cpu/comm/bcast_frame.h                                     (01)
#pragma once

#ifdef ATV_HAS_MPI

#include <mpi.h>

#include <cstdint>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/task/task_spec.h"

namespace ATV {
namespace cpu {
namespace task {

/// @brief Protocol verbs. Values start at 1 so a zero-initialized buffer is
///        never a valid command.
enum class Command : std::int32_t {
    RUN_TASK = 1, ///< One task follows: TaskSpec bytes + Frame broadcast.
    SHUTDOWN = 2, ///< Session over; WorkerLoop::run() returns.
};

/// @brief Protocol version; the handshake and every header carry it.
inline constexpr std::int32_t kProtocolVersion = 2;

/// @brief Fixed-size per-command header, broadcast as raw bytes.
struct CommandHeader {
    std::int32_t version = kProtocolVersion; ///< Defence in depth.
    std::int32_t command = 0;                ///< A `Command` value.
    std::int64_t payload_bytes = 0; ///< TaskSpec byte count (RUN_TASK).
};

/// @brief One dispatched task's state and control-plane result.
struct DispatchResult {
    cpu::Frame frame;
    TaskResult control;
};

/// @brief Worker-side blocking command loop (ranks != root).
class WorkerLoop {
  public:
    /// @brief Validating constructor; joins the collective version handshake.
    /// @param comm Communicator shared with the dispatcher (not owned; the
    ///             protocol neither dups nor frees it). Root is rank 0.
    /// @param version Protocol version to handshake with (test seam);
    ///                defaults to `kProtocolVersion`.
    /// @throws std::runtime_error if MPI is not initialized, `comm` is
    ///         MPI_COMM_NULL, this rank IS rank 0, or the handshake finds a
    ///         version mismatch on any rank (message names both versions).
    explicit WorkerLoop(MPI_Comm comm, std::int32_t version = kProtocolVersion);

    /// @brief Block on command broadcasts until SHUTDOWN.
    ///
    /// RUN_TASK executes the shared task path (task-level failure is folded
    /// into the status allreduces and does NOT exit the loop); SHUTDOWN
    /// returns cleanly. Protocol-level corruption (header version mismatch,
    /// unknown command) throws — the session is unrecoverable then.
    void run();

  private:
    MPI_Comm comm_;
    std::int32_t version_;
};

/// @brief Rank-0 dispatch endpoint: one broadcast per task, typed result back.
class TaskDispatcher {
  public:
    /// @brief Validating constructor; joins the collective version handshake.
    /// @param comm Communicator shared with the workers (not owned). Root is
    ///             rank 0 — this rank must BE rank 0.
    /// @param version Protocol version (test seam); defaults to
    ///                `kProtocolVersion`.
    /// @param do_handshake Join the collective version handshake (default).
    ///        The launcher's `main` constructs THE handshaking dispatcher
    ///        (pairing with every WorkerLoop ctor); the bindings' dispatch
    ///        seam — a separate copy of this library inside the extension
    ///        module — constructs with `handshake = false`, because the
    ///        workers are already blocked in their command loop and a
    ///        second handshake would misalign the collectives.
    /// @throws std::runtime_error if this rank is not rank 0 of `comm`, or
    ///         on handshake mismatch (every rank throws).
    explicit TaskDispatcher(MPI_Comm comm,
                            std::int32_t version = kProtocolVersion,
                            bool do_handshake = true);

    /// @brief Run one task on every rank; return state + loop control metadata.
    ///
    /// Broadcasts the spec + frame, joins the shared execution path, and
    /// (MD) gathers the owners back. At world size 1 this degrades to a
    /// purely local materialize + run with no collectives.
    ///
    /// @param spec  The validated task description.
    /// @param frame The full input system (owned rows `[0, nlocal)`;
    ///              positions [Angstrom], velocities [Angstrom/fs]).
    /// @return Result Frame plus iterations/convergence/stop reason. Distributed
    ///         Frames are tag-ordered owner gathers on rank 0.
    /// @throws std::runtime_error when any rank's setup or run failed (the
    ///         status allreduce carried the failure here in lockstep).
    DispatchResult dispatch(const TaskSpec& spec, cpu::Frame frame);

    /// @brief Broadcast SHUTDOWN; every WorkerLoop::run() returns.
    void shutdown();

  private:
    MPI_Comm comm_;
    std::int32_t version_;
};

} // namespace task
} // namespace cpu
} // namespace ATV

#endif // ATV_HAS_MPI
