/// @file tests/mpi/test_halo_exchange.cpp
/// @brief Phase 2.2 / 2.3 ghost halo correctness tests.
///
/// Multi-rank tests for `ctx.md.comm` (`CommBrick` borders / reverse).
/// Launched via `mpirun -n N`. Verifies:
///   - Forward populates `dd->n_ghost > 0`.
///   - Symmetry: my +x send count to my +x neighbor equals my +x neighbor's
///     -x send count to me (count exchange handshake is consistent).
///   - PBC shift: ghosts received from across a periodic-edge face land
///     at coordinates that, when wrapped back to `[0, L)`, match an actual
///     owner position somewhere in the neighbor's slab.
///   - Reverse symmetry: applying a known per-ghost force and reducing
///     back through `reverse_comm` adds exactly the right total to
///     each owner's force.
///
/// Extra-field `CommField` / `Endpoint` packing was removed in
/// lammps-comm-brick-04-packer; plugins use `ForwardPacker` instead.

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// @brief Publish a border/reverse schema naming a plugin runtime column.
///
/// Under the flag-driven wire a column travelled because it carried
/// `kColBorder`. Under the schema-driven wire it travels because something
/// asked for it (`comm-maint-09-schema`). A test that drives `CommBrick`
/// directly has no mounted component to declare on its behalf, so it plays
/// that part here — which is also the documentation of how a plugin gets a
/// column onto the wire.
/// Declares the base set plus whichever plugin columns THIS frame actually
/// carries — a component declares what it uses, and a test frame that never
/// inserted `halo_aux` must not put it on the wire.
ATV::GhostContract publish_wire(const cpu::Frame& frame) {
    ATV::GhostContract out;
    out.declare(ATV::base_fields::kEngine);
    if (frame.has("atoms", "halo_aux")) {
        // This test mutates it between `forward_comm` calls, so it must be
        // current every step. One declaration; the compile step decides that
        // means border + refresh.
        static constexpr std::array aux = {ATV::GhostField::read_each_step(
            "halo_aux", ColDType::Storage, 2, "test plugin")};
        out.declare(aux);
    }
    if (frame.has("atoms", "ghost_speed")) {
        // The DPD shape: one more SCALAR read at step freshness, which keeps
        // the whole wire field-major.
        static constexpr std::array speed = {ATV::GhostField::read_each_step(
            "ghost_speed", ColDType::Storage, 1, "test plugin")};
        out.declare(speed);
    }
    if (frame.has("atoms", "halo_reverse_aux")) {
        static constexpr std::array rev = {ATV::GhostField::reduce_to_owner(
            "halo_reverse_aux", ColDType::Acc, 2, "test plugin")};
        out.declare(rev);
    }
    return out;
}

const ColKey<StorageReal> kHaloAux{"atoms", "halo_aux", kDynamicFieldId,
                                   ColDType::Storage};
const ColKey<StorageReal> kHaloReverseAux{"atoms", "halo_reverse_aux",
                                          kDynamicFieldId, ColDType::Storage};
const ColKey<StorageReal> kGhostSpeed{"atoms", "ghost_speed", kDynamicFieldId,
                                      ColDType::Storage};

cpu::Frame make_uniform_grid(int n_per_side, double box_len) {
    const int n = n_per_side * n_per_side * n_per_side;
    cpu::Frame frame(n);
    const double step = box_len / static_cast<double>(n_per_side);
    int idx = 0;
    for (int ix = 0; ix < n_per_side; ++ix)
        for (int iy = 0; iy < n_per_side; ++iy)
            for (int iz = 0; iz < n_per_side; ++iz) {
                frame.span(ATV::kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * step;
                frame.span(ATV::kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * step;
                frame.span(ATV::kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * step;
                frame.span(ATV::kAtom_mass)[idx] = 1.0;
                ++idx;
            }
    frame.box = ATV::cpu::Box::cubic(box_len);
    return frame;
}

// Run init -> set_frame -> single forward exchange and return the dd state.
struct ForwardResult {
    int n_local;
    int n_ghost;
    std::array<int, 6> face_neighbors;
    int dims[3];
    int coords[3];
    double Lx, Ly, Lz;
    std::vector<double> ghost_x, ghost_y, ghost_z;
};

ForwardResult run_forward(double box_len, int n_per_side, double cutoff) {
    auto frame = make_uniform_grid(n_per_side, box_len);

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);

    auto* dd = backend.context().md.dd;
    cpu::Comm* comm = backend.context().md.comm;
    comm->set_ghost_cutoff(cutoff);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    ATV::GhostContract wires = publish_wire(frame);
    backend.context().md.ghost = &wires;
    comm->setup(lctx);
    comm->borders(lctx);

    ForwardResult r{};
    r.n_local = dd->n_local;
    r.n_ghost = dd->n_ghost;
    for (int f = 0; f < 6; ++f)
        r.face_neighbors[f] = dd->face_neighbors[f];
    for (int d = 0; d < 3; ++d) {
        r.dims[d] = dd->dims[d];
        r.coords[d] = dd->coords[d];
    }
    r.Lx = frame.box.cell()(0, 0);
    r.Ly = frame.box.cell()(1, 1);
    r.Lz = frame.box.cell()(2, 2);

    r.ghost_x.assign(static_cast<std::size_t>(dd->n_ghost), 0.0);
    r.ghost_y.assign(static_cast<std::size_t>(dd->n_ghost), 0.0);
    r.ghost_z.assign(static_cast<std::size_t>(dd->n_ghost), 0.0);
    for (int g = 0; g < dd->n_ghost; ++g) {
        r.ghost_x[g] = frame.span(ATV::kAtom_x)[dd->n_local + g];
        r.ghost_y[g] = frame.span(ATV::kAtom_y)[dd->n_local + g];
        r.ghost_z[g] = frame.span(ATV::kAtom_z)[dd->n_local + g];
    }

    backend.shutdown();
    return r;
}

} // namespace

// ── n_ghost > 0 when cutoff captures the neighbor slab ──────────────────

TEST(HaloForward, GhostsPopulatedAtModerateCutoff) {
    // 64 atoms on 4x4x4 grid in a 10 Å box. With 2x2x2 ranks, each slab is
    // 5 Å, so a cutoff of 2.0 Å captures the inner-face atoms.
    const auto r = run_forward(10.0, 4, 2.0);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1) {
        EXPECT_EQ(r.n_ghost, 0);
        return;
    }

    EXPECT_GT(r.n_ghost, 0)
        << "expected ghost atoms after Forward halo exchange at cutoff 2.0";
}

// ── per-swap nrecv consistent with the swap's send list ────────────────

TEST(HaloForward, RecvCountsConsistentWithSendLists) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1) {
        GTEST_SKIP() << "single-rank: no cross-rank symmetry to check";
    }

    auto frame = make_uniform_grid(4, 10.0);
#if defined(ATV_ID_BIGBIG)
    constexpr TagInt tag_base = TagInt{9007199254740993LL};
#else
    constexpr TagInt tag_base = TagInt{700000003};
#endif
    frame.block("atoms").insert(
        "halo_aux",
        cpu::ColumnSpec{ColDType::Storage, 2, ColGroup::Core, 0,
                        kColMigrate | kColExchange, kDynamicFieldId});
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_tag)[i] = tag_base + TagInt(i);
        frame.span(kHaloAux)(i, 0) = StorageReal(100 + i);
        frame.span(kHaloAux)(i, 1) = StorageReal(-200 - i);
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    cpu::Comm* comm = backend.context().md.comm;
    comm->set_ghost_cutoff(2.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    ATV::GhostContract wires = publish_wire(frame);
    backend.context().md.ghost = &wires;
    comm->setup(lctx);
    comm->borders(lctx);

    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const int source =
            static_cast<int>(frame.span(kAtom_tag)[row] - tag_base);
        ASSERT_GE(source, 0);
        ASSERT_LT(source, 64);
        EXPECT_EQ(frame.span(kHaloAux)(row, 0), StorageReal(100 + source));
        EXPECT_EQ(frame.span(kHaloAux)(row, 1), StorageReal(-200 - source));
    }

    for (int i = 0; i < dd->n_local; ++i) {
        frame.span(kHaloAux)(i, 0) += StorageReal(1000);
        frame.span(kHaloAux)(i, 1) -= StorageReal(1000);
    }
    comm->forward_comm(lctx); // steady comm plan, same ghost slots
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const int source =
            static_cast<int>(frame.span(kAtom_tag)[row] - tag_base);
        EXPECT_EQ(frame.span(kHaloAux)(row, 0), StorageReal(1100 + source));
        EXPECT_EQ(frame.span(kHaloAux)(row, 1), StorageReal(-1200 - source));
    }

    ASSERT_NE(backend.context().md.comm, nullptr);
    EXPECT_GT(dd->n_ghost, 0);

    backend.shutdown();
}

// ── a scalar step-freshness read rides the field-major refresh wire ───

/// The shape `PairDpd` produces: a scalar read at `each_step` freshness
/// alongside x/y/z, so the refresh wire is four scalars and stays
/// field-major. The 2-component `halo_aux` case above covers the
/// row-major arm; this covers the one production actually takes when a style
/// declares a neighbour velocity, and it covers it over real MPI rather than
/// in-process.
///
/// It also pins that presence is implied by the requirement, not declared
/// separately: the value is checked on the ghosts straight out of
/// `borders()`, before any `forward_comm` has run.
TEST(HaloForward, ScalarEveryStepFieldIsCreatedThenRefreshed) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP() << "single-rank: no ghosts to refresh";

    auto frame = make_uniform_grid(4, 10.0);
    constexpr TagInt tag_base = TagInt{1000};
    frame.block("atoms").insert(
        "ghost_speed",
        cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                        kColMigrate | kColExchange, kDynamicFieldId});
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_tag)[i] = tag_base + TagInt(i);
        frame.span(kGhostSpeed)[i] = StorageReal(7 + i);
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    cpu::Comm* comm = backend.context().md.comm;
    comm->set_ghost_cutoff(2.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    ATV::GhostContract wires = publish_wire(frame);
    backend.context().md.ghost = &wires;
    ASSERT_TRUE(wires.contains("ghost_speed"));
    ASSERT_EQ(wires.find("ghost_speed")->freshness, ATV::Freshness::step);
    comm->setup(lctx);
    comm->borders(lctx);
    ASSERT_GT(dd->n_ghost, 0);

    // Straight out of borders(): the creation wire carried it.
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const int src = static_cast<int>(frame.span(kAtom_tag)[row] - tag_base);
        ASSERT_GE(src, 0);
        ASSERT_LT(src, 64);
        EXPECT_EQ(frame.span(kGhostSpeed)[row], StorageReal(7 + src));
    }

    // Owners move on; the refresh pass carries the new values without a
    // rebuild — the guarantee a `stable` declaration would not give.
    for (int i = 0; i < dd->n_local; ++i)
        frame.span(kGhostSpeed)[i] += StorageReal(500);
    comm->forward_comm(lctx);
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const int src = static_cast<int>(frame.span(kAtom_tag)[row] - tag_base);
        EXPECT_EQ(frame.span(kGhostSpeed)[row], StorageReal(507 + src));
    }

    backend.shutdown();
}

// ── PBC-shifted ghosts correspond to real atoms after wrap ─────────────

TEST(HaloForward, GhostPositionsResolveToOwnersAfterWrap) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP();

    const auto r = run_forward(10.0, 4, 2.0);

    auto wrap = [](double q, double L) {
        double w = q - std::floor(q / L) * L;
        if (w < 0)
            w += L;
        if (w >= L)
            w -= L;
        return w;
    };

    const int step = static_cast<int>(10.0 / 4); // grid step is 2.5
    (void)step;
    const double dx = 10.0 / 4; // 2.5

    // Every ghost, after wrapping back to [0, L), must coincide with one
    // of the original 64 grid points (positions are k * 2.5 + 1.25).
    for (int g = 0; g < r.n_ghost; ++g) {
        const double xw = wrap(r.ghost_x[g], r.Lx);
        const double yw = wrap(r.ghost_y[g], r.Ly);
        const double zw = wrap(r.ghost_z[g], r.Lz);
        const double rx = std::round((xw - 0.5 * dx) / dx);
        const double ry = std::round((yw - 0.5 * dx) / dx);
        const double rz = std::round((zw - 0.5 * dx) / dx);
        EXPECT_NEAR(xw, rx * dx + 0.5 * dx, 1e-9)
            << "ghost " << g << " x mismatch";
        EXPECT_NEAR(yw, ry * dx + 0.5 * dx, 1e-9)
            << "ghost " << g << " y mismatch";
        EXPECT_NEAR(zw, rz * dx + 0.5 * dx, 1e-9)
            << "ghost " << g << " z mismatch";
    }
}

// ── Reverse: known ghost forces sum back to owners losslessly ──────────

TEST(HaloReverse, ReverseAddsGhostForcesBackToOwners) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP();

    auto frame = make_uniform_grid(4, 10.0);
    frame.block("atoms").insert(
        "halo_reverse_aux",
        cpu::ColumnSpec{ColDType::Storage, 1, ColGroup::Core, 0,
                        kColMigrate | kColExchange, kDynamicFieldId});
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;
    cpu::Comm* comm = backend.context().md.comm;
    comm->set_ghost_cutoff(2.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    ATV::GhostContract wires = publish_wire(frame);
    backend.context().md.ghost = &wires;
    comm->setup(lctx);
    comm->borders(lctx);

    // Plant a known force pattern on every atom (owners + ghosts).
    // Owner forces start at 1.0, 2.0, 3.0; ghost forces add 10.0 each.
    for (int i = 0; i < dd->n_local; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 1.0;
        frame.span(ATV::kAtom_fy)[i] = 2.0;
        frame.span(ATV::kAtom_fz)[i] = 3.0;
        frame.span(kHaloReverseAux)[i] = StorageReal(4.0);
    }
    for (int g = 0; g < dd->n_ghost; ++g) {
        frame.span(ATV::kAtom_fx)[dd->n_local + g] = 10.0;
        frame.span(ATV::kAtom_fy)[dd->n_local + g] = 10.0;
        frame.span(ATV::kAtom_fz)[dd->n_local + g] = 10.0;
        frame.span(kHaloReverseAux)[dd->n_local + g] = StorageReal(7.0);
    }

    comm->reverse_comm(lctx);

    // Reverse sums every ghost copy's force back to its owner. The forward
    // halo is a 3-axis cascade (x→y→z), so an owner near a corner is ghosted
    // on up to 7 neighbours at 2×2×2 (3 face + 3 edge + 1 corner) — a count
    // this rank's own swap send lists do NOT capture (the edge/corner copies
    // are forwarded by *other* ranks). The robust, topology-independent
    // invariant is global conservation: every ghost's planted force lands on
    // exactly one owner, so the global owner-force sum is
    //   Σ owner fx = base_x·N_owners + 10·N_ghosts   (likewise fy, fz),
    // with base = (1,2,3) on owners and 10 on every ghost.
    double local_f[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < dd->n_local; ++i) {
        local_f[0] += frame.span(ATV::kAtom_fx)[i];
        local_f[1] += frame.span(ATV::kAtom_fy)[i];
        local_f[2] += frame.span(ATV::kAtom_fz)[i];
        local_f[3] += frame.span(kHaloReverseAux)[i];
        // Per-owner sanity: the returned offset is a non-negative integer
        // multiple of 10 on every axis (no partial/duplicated ghost force).
        const double off = frame.span(ATV::kAtom_fx)[i] - 1.0;
        EXPECT_GE(off, -1e-9) << "owner " << i << " lost its base force";
        EXPECT_NEAR(off / 10.0, std::round(off / 10.0), 1e-9)
            << "owner " << i << " fx offset " << off
            << " is not a clean multiple of the 10.0 ghost force";
    }
    long long local_counts[2] = {dd->n_local, dd->n_ghost};
    double global_f[4] = {0.0, 0.0, 0.0, 0.0};
    long long global_counts[2] = {0, 0};
    MPI_Allreduce(local_f, global_f, 4, MPI_DOUBLE, MPI_SUM, dd->cart_comm);
    MPI_Allreduce(local_counts, global_counts, 2, MPI_LONG_LONG, MPI_SUM,
                  dd->cart_comm);
    const double no = static_cast<double>(global_counts[0]);
    const double ng = static_cast<double>(global_counts[1]);
    EXPECT_NEAR(global_f[0], 1.0 * no + 10.0 * ng, 1e-6) << "fx not conserved";
    EXPECT_NEAR(global_f[1], 2.0 * no + 10.0 * ng, 1e-6) << "fy not conserved";
    EXPECT_NEAR(global_f[2], 3.0 * no + 10.0 * ng, 1e-6) << "fz not conserved";
    EXPECT_NEAR(global_f[3], 4.0 * no + 7.0 * ng, 1e-6)
        << "runtime reverse column not conserved";

    backend.shutdown();
}

TEST(HaloForward, OneCommPerRuntime) {
    auto frame = make_uniform_grid(2, 10.0);
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    ASSERT_NE(backend.context().md.comm, nullptr);
    EXPECT_EQ(backend.context().md.comm, backend.context().md.comm);
    backend.shutdown();
}

// Extra-field CommField packing was removed (lammps-comm-brick-04-packer).

// See test_topology.cpp for the OpenMPI 5.0.x macOS teardown rationale.
// CTest reads the GoogleTest "[  PASSED  ]" summary line to determine
// pass/fail, bypassing any prterun exit-code wobble.
int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
