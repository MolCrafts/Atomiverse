/// @file cpu/task/task_protocol.cpp
/// @brief RUN_TASK/SHUTDOWN command loop, handshake, and the shared
///        post-broadcast execution path (contract in task_protocol.h).

#include "atomiverse/cpu/task/task_protocol.h"

#ifdef ATV_HAS_MPI

#include <climits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/comm/bcast_frame.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/comm/gather.h"
#include "atomiverse/cpu/task/task_codec.h"
#include "atomiverse/cpu/task/task_materializer.h"

namespace ATV {
namespace cpu {
namespace task {

namespace {

/// Collective version handshake: MAX/MIN allreduce the local version so
/// EVERY rank learns both ends of a mismatch and can name them.
void handshake(MPI_Comm comm, std::int32_t version) {
    std::int32_t vmax = 0;
    std::int32_t vmin = 0;
    MPI_Allreduce(&version, &vmax, 1, MPI_INT32_T, MPI_MAX, comm);
    MPI_Allreduce(&version, &vmin, 1, MPI_INT32_T, MPI_MIN, comm);
    if (vmax != vmin)
        throw std::runtime_error(
            "task_protocol: protocol version mismatch across ranks (min "
            + std::to_string(vmin) + ", max " + std::to_string(vmax) + ")");
}

/// MAX-allreduce a local status code; any rank's failure fails all.
int combine_status(MPI_Comm comm, int local) {
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm);
    return global;
}

/// Stage codes `execute_task` returns; `dispatch` names them in its error
/// message — keep the two in sync through these constants.
constexpr int kFailDecode = 1;
constexpr int kFailSetup = 2;
constexpr int kFailRun = 3;
constexpr const char* kStageName[] = {"", "decode", "setup", "run"};

/// The shared post-broadcast execution path — exactly two callers
/// (TaskDispatcher::dispatch and WorkerLoop::run). Three lockstep stages,
/// each fenced by a status allreduce so no rank enters the next stage's
/// collectives after any rank failed the previous one:
///   1. unpack (purely local — a failure here must not reach build's
///      collectives),
///   2. materialize + build (collective-bearing, but deterministic: the
///      same bytes drive the same code path on every rank),
///   3. run.
/// Local exceptions are folded into the status; `err` carries this rank's
/// own message (empty when the failure was remote).
int execute_task(MPI_Comm comm, const std::vector<std::byte>& bytes,
                 cpu::Frame& frame, std::optional<TaskMaterializer>& mat,
                 TaskResult& result, std::string& err) {
    std::optional<TaskSpec> spec;
    int local = 0;
    try {
        spec.emplace(unpack(bytes.data(), bytes.size()));
    } catch (const std::exception& e) {
        local = 1;
        err = e.what();
    }
    if (combine_status(comm, local))
        return kFailDecode;

    try {
        mat.emplace(*spec);
    } catch (const std::exception& e) {
        local = 1;
        err = e.what();
    }
    if (combine_status(comm, local))
        return kFailSetup;

    try {
        result = mat->run(frame);
    } catch (const std::exception& e) {
        local = 1;
        err = e.what();
    }
    if (combine_status(comm, local))
        return kFailRun;

    return 0;
}

/// Post-success result step: MD tasks join the collective owner gather
/// (every rank calls this in lockstep); SinglePoint frames are full
/// replicas and need none. Returns the result frame on rank 0.
/// CONSUMES `frame` (moved-from afterwards) and the materializer's dd.
cpu::Frame assemble_result(MPI_Comm comm, TaskMaterializer& mat,
                           cpu::Frame& frame) {
    if (mat.kind() == TaskKind::SinglePoint)
        return std::move(frame);
    auto dd = mat.release_dd();
    if (!dd)
        throw std::runtime_error(
            "task_protocol: distributed task completed without an owner "
            "decomposition");
    const auto gathered = md::mpi::gather_to_root(frame, *dd, 0);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0)
        return cpu::Frame(0);
    cpu::Frame result = md::mpi::to_frame(gathered);
    // Global scalars were all-reduced onto every rank each step by the
    // Reduce component; rank 0's local copy is authoritative.
    result.meta.copy_serializable_from(frame.meta);
    return result;
}

void validate_common(MPI_Comm comm) {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited)
        throw std::runtime_error("task_protocol: MPI is not initialized");
    if (comm == MPI_COMM_NULL)
        throw std::runtime_error("task_protocol: communicator is "
                                 "MPI_COMM_NULL");
}

} // namespace

// ── WorkerLoop ────────────────────────────────────────────────────────────

WorkerLoop::WorkerLoop(MPI_Comm comm, std::int32_t version)
    : comm_(comm), version_(version) {
    validate_common(comm_);
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0)
        throw std::runtime_error(
            "task_protocol: WorkerLoop is for ranks != 0 (rank 0 owns the "
            "TaskDispatcher)");
    handshake(comm_, version_);
}

void WorkerLoop::run() {
    for (;;) {
        CommandHeader hdr{};
        MPI_Bcast(&hdr, sizeof(CommandHeader), MPI_BYTE, 0, comm_);
        if (hdr.version != version_)
            throw std::runtime_error(
                "task_protocol: corrupt command header (version "
                + std::to_string(hdr.version) + ", expected "
                + std::to_string(version_) + ")");
        if (hdr.command == static_cast<std::int32_t>(Command::SHUTDOWN))
            return;
        if (hdr.command != static_cast<std::int32_t>(Command::RUN_TASK))
            throw std::runtime_error("task_protocol: unknown command "
                                     + std::to_string(hdr.command));
        if (hdr.payload_bytes < 0 || hdr.payload_bytes > INT_MAX)
            throw std::runtime_error("task_protocol: implausible payload size "
                                     + std::to_string(hdr.payload_bytes));

        std::vector<std::byte> bytes(
            static_cast<std::size_t>(hdr.payload_bytes));
        MPI_Bcast(bytes.data(), static_cast<int>(hdr.payload_bytes), MPI_BYTE,
                  0, comm_);
        cpu::Frame frame(0);
        comm::bcast_frame(frame, 0);

        std::optional<TaskMaterializer> mat;
        TaskResult result;
        std::string err;
        const int status = execute_task(comm_, bytes, frame, mat, result, err);
        if (status == 0)
            (void)assemble_result(comm_, *mat, frame);
        else if (!err.empty())
            ATV_WARN("task_protocol: task {} failed on this rank: {}",
                     kStageName[status], err);
        // Task-level failure does not end the session: the status
        // allreduce already told rank 0, whose dispatch() throws; this
        // worker simply blocks for the next command.
    }
}

// ── TaskDispatcher ────────────────────────────────────────────────────────

TaskDispatcher::TaskDispatcher(MPI_Comm comm, std::int32_t version,
                               bool do_handshake)
    : comm_(comm), version_(version) {
    validate_common(comm_);
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    if (rank != 0)
        throw std::runtime_error(
            "task_protocol: TaskDispatcher must be constructed on rank 0");
    if (do_handshake)
        handshake(comm_, version_);
}

DispatchResult TaskDispatcher::dispatch(const TaskSpec& spec,
                                        cpu::Frame frame) {
    int world = 1;
    MPI_Comm_size(comm_, &world);
    if (world <= 1) {
        // Serial degrade: no workers, no collectives — materialize and run
        // locally (gather_to_root's own single-rank path is equivalent to
        // returning the frame as-is).
        TaskMaterializer mat(spec);
        TaskResult control = mat.run(frame);
        return DispatchResult{std::move(frame), std::move(control)};
    }

    std::vector<std::byte> bytes = pack(spec);
    if (bytes.size() > static_cast<std::size_t>(INT_MAX))
        throw std::runtime_error("task_protocol: TaskSpec payload of "
                                 + std::to_string(bytes.size())
                                 + " bytes exceeds the broadcast limit");
    CommandHeader hdr{};
    hdr.version = version_;
    hdr.command = static_cast<std::int32_t>(Command::RUN_TASK);
    hdr.payload_bytes = static_cast<std::int64_t>(bytes.size());
    MPI_Bcast(&hdr, sizeof(CommandHeader), MPI_BYTE, 0, comm_);
    MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE, 0, comm_);
    comm::bcast_frame(frame, 0);

    std::optional<TaskMaterializer> mat;
    TaskResult control;
    std::string err;
    const int status = execute_task(comm_, bytes, frame, mat, control, err);
    if (status != 0)
        throw std::runtime_error(
            "task_protocol: task " + std::string(kStageName[status]) + " failed"
            + (err.empty() ? " on a remote rank" : ": " + err));
    return DispatchResult{assemble_result(comm_, *mat, frame),
                          std::move(control)};
}

void TaskDispatcher::shutdown() {
    int world = 1;
    MPI_Comm_size(comm_, &world);
    if (world <= 1)
        return; // no workers to release
    CommandHeader hdr{};
    hdr.version = version_;
    hdr.command = static_cast<std::int32_t>(Command::SHUTDOWN);
    MPI_Bcast(&hdr, sizeof(CommandHeader), MPI_BYTE, 0, comm_);
}

} // namespace task
} // namespace cpu
} // namespace ATV

#endif // ATV_HAS_MPI
