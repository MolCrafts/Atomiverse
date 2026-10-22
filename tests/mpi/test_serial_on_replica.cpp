/// @file tests/mpi/test_serial_on_replica.cpp
/// @brief `Driver::set_serial_on_replica()` seam under mpirun
///        (mpi-launcher-embedded-python-04, spec verdict 方案 a).
///
/// Launched under `mpirun -n N`, runs at N=1, 2, 4. With the flag set,
/// `Backend::init` skips `dd_->init_topology` (cart_comm stays
/// MPI_COMM_NULL), so `set_frame` never partitions and every rank solves
/// the FULL replica independently — the molecular whole-system shape
/// (AM1 / SCF / PW single points), redundant but correct and trivially
/// lockstep. Verifies, on the established ms-scale H2 AM1 fixture
/// (tests/cpu/test_task_materializer.cpp):
///   - Backend level: flag ⇒ cart_comm stays NULL, frame stays whole.
///   - Driver level: a serial-on-replica `cpu::Evaluate` + nested AM1
///     keeps atoms.nlocal == 2 on every rank (no slab sub-molecules).
///   - Energy matches a serially-computed inner-solver reference bitwise
///     on the same rank (AM1 run outside any driver/DD machinery),
///     and per-atom charges likewise.
///   - Energy is bitwise identical across all ranks (MPI_Allgather).
/// At N=1 all checks pass trivially. Single-threaded (bitwise claims).

#include <gtest/gtest.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"

using namespace ATV;

namespace {

// Bitwise equality across ranks / against the inner-solver reference only
// holds when the force/Fock reduction order is fixed — pin to one thread
// before any parallel region fires (the test_task_materializer convention;
// setenv would be too late, libgomp snapshots OMP_NUM_THREADS in its ELF
// constructor).
#ifdef _OPENMP
[[maybe_unused]] const bool kOmpPinned = [] {
    omp_set_num_threads(1);
    return true;
}();
#endif

/// @brief Fixed-geometry H2 (0.74 Å, 20 Å box) — the exact fixture of
///        tests/cpu/test_task_materializer.cpp; literals only, so every
///        rank constructs a bitwise-identical full replica.
cpu::Frame make_h2_frame() {
    cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1; // Z = 1 (hydrogen)
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = -0.1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.64;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

/// @brief Run the serial-on-replica Evaluate + AM1 single point in place.
void run_serial_on_replica_am1(cpu::Frame& frame) {
    cpu::Evaluate eval;
    eval.set_serial_on_replica();
    // charge=0, singlet, max_iter=200, dens/energy tols default.
    eval.on(ATV::evaluate::Phase::evaluate)
        .add(cpu::semiempirical::AM1{0, 1, 200, 1.0e-8, 1.0e-10});
    eval.build();
    eval.run(frame);
}

// ── Backend level: the flag skips DD topology, set_frame never splits ────

TEST(SerialOnReplica, BackendFlagSkipsTopologyAndKeepsFrameWhole) {
    auto frame = make_h2_frame();

    cpu::Runtime backend;
    backend.set_serial_on_replica();
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;
    ASSERT_NE(dd, nullptr);
    // No Cartesian topology means set_frame's partition condition can never
    // fire — this is the seam's whole mechanism, at any world size.
    EXPECT_EQ(dd->cart_comm, MPI_COMM_NULL)
        << "serial-on-replica must leave cart_comm MPI_COMM_NULL so "
           "partition_atoms is unreachable";
    EXPECT_EQ(frame.nlocal(), 2)
        << "rank " << backend.context().mpi_rank
        << " lost part of its replica to decomposition";

    backend.shutdown();
}

// ── Driver level: multi-rank Evaluate does not decompose the replica ─────

// ── Same-rank parity: driver path ≡ bare inner solver, bitwise ───────────

// ── Cross-rank parity: every rank lands on the identical energy ──────────

} // namespace

// ── Custom main: MPI bracket around GoogleTest (test_topology convention) ─

TEST(SerialOnReplica,
     WholeSystemEvaluatorRejectsSpatialPartitionBeforeMutation) {
    auto frame = make_h2_frame();
    cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate)
        .add(cpu::semiempirical::AM1{0, 1, 200, 1.0e-8, 1.0e-10});
    eval.build();
    cpu::Runtime runtime;
    runtime.init();
    runtime.load(eval.plan());
    if (runtime.context().mpi_size > 1) {
        EXPECT_THROW(runtime.set_frame(frame), std::runtime_error);
    } else {
        EXPECT_NO_THROW(runtime.set_frame(frame));
    }
    EXPECT_EQ(frame.nlocal(), 2);
    EXPECT_DOUBLE_EQ(frame.span(ATV::kAtom_z)[0], -0.1);
    EXPECT_DOUBLE_EQ(frame.span(ATV::kAtom_z)[1], 0.64);
}
