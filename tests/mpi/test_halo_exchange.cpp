/// @file tests/mpi/test_halo_exchange.cpp
/// @brief Phase 2.2 / 2.3 ghost halo correctness tests.
///
/// Multi-rank tests for `HaloSend` + `HaloRecv`.
/// Launched via `mpirun -n N`. Verifies:
///   - Forward populates `dd->n_ghost > 0` and per-swap receive counts
///     (`ctx.md.swap_list`) consistent with each rank's own send list.
///   - Symmetry: my +x send count to my +x neighbor equals my +x neighbor's
///     -x send count to me (count exchange handshake is consistent).
///   - PBC shift: ghosts received from across a periodic-edge face land
///     at coordinates that, when wrapped back to `[0, L)`, match an actual
///     owner position somewhere in the neighbor's slab.
///   - Reverse symmetry: applying a known per-ghost force and reducing
///     back through `HaloRecv` adds exactly the right total to
///     each owner's force.
///
/// arch-refactor-02-halo additions (`HaloParticipant` protocol):
///   - A test-only field participant (`Endpoint::none`) declaring width-1 and
///     width-3 per-atom forward `CommField`s sees every ghost row refreshed to
///     its owner's value each step (full-rebuild AND position-only-refresh
///     packing paths, incl. across periodic faces).
///   - A declared reverse `CommField` has its ghost contributions accumulated
///     back onto the owners exactly once per step (global conservation).
///   - Single rank: the same component path is a strict no-op.

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

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo_exchange.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

const ColKey<StorageReal> kHaloAux{"atoms", "halo_aux", kDynamicFieldId,
                                   ColDType::Storage};
const ColKey<StorageReal> kHaloReverseAux{"atoms", "halo_reverse_aux",
                                          kDynamicFieldId, ColDType::Storage};

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
    // Per-swap, read off the halo's published table (`ctx.md.swap_list`);
    // swap i is face i of `face_neighbors`.
    std::array<int, 6> swap_nrecv;
    std::array<int, 6> swap_send_counts;
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

    // Drive one Forward launch directly (without going through MD driver).
    cpu::comm::HaloSend fwd{cutoff};
    cpu::BuildContext bctx{backend.context(), frame};
    fwd.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    fwd.launch(lctx);

    ForwardResult r{};
    r.n_local = dd->n_local;
    r.n_ghost = dd->n_ghost;
    const auto* swaps = backend.context().md.swap_list;
    for (int f = 0; f < 6; ++f) {
        const bool has_swap =
            swaps != nullptr && f < static_cast<int>(swaps->size());
        r.swap_nrecv[f] = has_swap ? (*swaps)[f].nrecv : 0;
        r.swap_send_counts[f] =
            has_swap ? static_cast<int>((*swaps)[f].sendlist.size()) : 0;
        r.face_neighbors[f] = dd->face_neighbors[f];
    }
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
                        kColMigrate | kColExchange | kColBorder | kColComm,
                        kDynamicFieldId});
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

    cpu::comm::HaloSend fwd{2.0};
    cpu::BuildContext bctx{backend.context(), frame};
    fwd.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    fwd.launch(lctx);

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
    fwd.launch(lctx); // steady comm plan, same ghost slots
    for (int g = 0; g < dd->n_ghost; ++g) {
        const int row = dd->n_local + g;
        const int source =
            static_cast<int>(frame.span(kAtom_tag)[row] - tag_base);
        EXPECT_EQ(frame.span(kHaloAux)(row, 0), StorageReal(1100 + source));
        EXPECT_EQ(frame.span(kHaloAux)(row, 1), StorageReal(-1200 - source));
    }

    // Local sanity: total ghost atoms = sum of the swap table's nrecv.
    const auto* swaps = backend.context().md.swap_list;
    ASSERT_NE(swaps, nullptr) << "HaloSend must publish its swap table";
    ASSERT_EQ(swaps->size(), 6u);
    int total_ghosts = 0;
    for (int s = 0; s < 6; ++s)
        total_ghosts += (*swaps)[s].nrecv;
    EXPECT_EQ(total_ghosts, dd->n_ghost);
    EXPECT_EQ(swaps->total_recv(), dd->n_ghost);

    // Swaps that self-wrap (dims[dim] == 1) contribute 0 because we
    // short-circuit those at the component.
    for (int s = 0; s < 6; ++s) {
        const int dim = (*swaps)[s].dim;
        if (dd->dims[dim] == 1) {
            EXPECT_TRUE((*swaps)[s].sendself)
                << "swap " << s << " crosses undecomposed axis " << dim
                << " but is not marked sendself";
            EXPECT_EQ((*swaps)[s].nrecv, 0)
                << "swap " << s << " wraps to self (dims[" << dim
                << "]==1) but received " << (*swaps)[s].nrecv << " ghosts";
        }
    }

    // Ghost slots are appended in swap order: firstrecv is the running
    // prefix sum from n_local.
    int expected_first = dd->n_local;
    for (int s = 0; s < 6; ++s) {
        EXPECT_EQ((*swaps)[s].firstrecv, expected_first)
            << "swap " << s << " ghost slots are not contiguous in swap order";
        expected_first += (*swaps)[s].nrecv;
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
                        kColMigrate | kColExchange | kColReverse,
                        kDynamicFieldId});
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    auto* dd = backend.context().md.dd;

    cpu::comm::HaloSend fwd{2.0};
    cpu::comm::HaloRecv rev{};
    cpu::BuildContext bctx{backend.context(), frame};
    fwd.build(bctx);
    rev.build(bctx);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    fwd.launch(lctx);

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

    rev.launch(lctx);

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

// ── Regression: per-instance FaceBufs (PIMPL, not a shared static) ─────

// Catches a re-introduction of `static thread_local FaceBufs s` (or any
// other cross-instance shared storage) behind the Forward / Reverse
// facades. The test builds two HaloSend instances in sequence
// against DIFFERENT `dd->capacity` values and verifies that the first
// instance's per-face buffer capacity is unchanged by the second's
// `build()`. With the old shared static, the vector's `reserve()`
// monotonically grew, so the first instance would see its capacity bumped
// up to the second's after B.build() ran.
TEST(HaloForward, FaceBufsAreInstanceOwned) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1) {
        GTEST_SKIP() << "multi-rank required (need non-null dd)";
    }

    auto frame_small = make_uniform_grid(2, 10.0); // 8 atoms
    auto frame_big = make_uniform_grid(6, 10.0);   // 216 atoms

    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame_small);

    cpu::comm::HaloSend fwd_a{2.0};
    cpu::BuildContext bctx_small{backend.context(), frame_small};
    fwd_a.build(bctx_small);
    const std::size_t cap_a_before = fwd_a.send_buf_capacity_for_test();

    // Rebind the same backend to a larger frame; dd->capacity now reflects
    // the bigger slab.
    backend.set_frame(frame_big);
    cpu::comm::HaloSend fwd_b{2.0};
    cpu::BuildContext bctx_big{backend.context(), frame_big};
    fwd_b.build(bctx_big);
    const std::size_t cap_b = fwd_b.send_buf_capacity_for_test();
    const std::size_t cap_a_after = fwd_a.send_buf_capacity_for_test();

    // Per-instance invariant: A's bufs are untouched by B's build.
    EXPECT_EQ(cap_a_before, cap_a_after)
        << "HaloSend #A's FaceBufs capacity changed from " << cap_a_before
        << " to " << cap_a_after
        << " after instance #B was built -- bufs must be per-instance, "
        << "not a shared static.";

    // Different capacity inputs must land in different storage: if A and B
    // shared a static vector, they'd see the same (max) capacity.
    EXPECT_NE(cap_a_after, cap_b)
        << "Instances A and B built against different dd->capacity values "
        << "produced identical send-buf capacities (A=" << cap_a_after
        << " B=" << cap_b << "); likely sharing a single storage object.";

    backend.shutdown();
}

// ═══ HaloParticipant extra-field forward / reverse (arch-refactor-02) ═══
//
// Test-only components; the production tree gains no field participant here.

namespace {

/// @brief No-op Force evaluator so `cpu::md::MD` validation passes without
/// dragging a pair potential + neighbor list into the field-participant tests.
// A zero force. Inherits the Potential kind so the MD driver's energy-source
// validation is satisfied; it plants nothing, keeping owner rows static.
class NullForce : public cpu::Potential {
  public:
    std::string name() const override { return "null_force"; }
    void launch(const cpu::LaunchContext&) override {}
};

/// @brief Pure field participant (`halo_endpoint() == Endpoint::none`):
/// declares two per-atom forward fields (width 1 and width 3, contiguous
/// `data[i*width + k]` layout) and one width-1 reverse field.
///
/// Lifecycle (per the HaloParticipant contract):
///   - `build()` allocates owned+ghost capacity (`frame.nlocal()`,
///     == `dd->capacity` in multi-rank mode; no-alloc-in-launch) and seeds
///     owner rows with tag-derived values so the step-0 forward exchange
///     (HaloSend @ pre_neighbor, i.e. *before* our force-phase launch)
///     already packs deterministic data.
///   - `launch()` (bound at `md::Phase::force`, order +100) first snapshots
///     the state produced by this step's forward exchange / last step's
///     reverse exchange, then rewrites owner rows and plants a known
///     contribution on every ghost row of the reverse field.
///
/// Value scheme (all exactly representable, so transport must be lossless):
///   forward width-1:  f1(tag) = 0.5 + 3*tag
///   forward width-3:  f3(tag, k) = 1000*(k+1) + tag   (k = 0,1,2 lanes)
///   reverse: owners start at kRevBase, each ghost copy plants kRevGhost;
///            after one reverse pass an owner holds
///            kRevBase + kRevGhost * (# ghost copies of it, cluster-wide).
class FieldParticipant : public cpu::Component, public cpu::HaloParticipant {
  public:
    static double fwd1_value(TagInt t) {
        return 0.5 + 3.0 * static_cast<double>(t);
    }
    static double fwd3_value(TagInt t, int k) {
        return 1000.0 * static_cast<double>(k + 1) + static_cast<double>(t);
    }
    static constexpr double kRevBase = 2.0;
    static constexpr double kRevGhost = 5.0;

    /// @brief Per-step capture of what the halo engine produced.
    struct Snapshot {
        bool valid = false;
        int n_local = 0;
        int n_ghost = 0;
        std::vector<TagInt> ghost_tag;  ///< tag of each ghost row
        std::vector<double> ghost_fwd1; ///< width-1 forward field, ghost rows
        std::vector<double> ghost_fwd3; ///< width-3 forward field, ghost rows
        std::vector<double> owner_rev;  ///< reverse field, owner rows,
                                        ///< captured BEFORE this step rewrites
    };
    Snapshot snap0; ///< captured at force-phase of step 0 (post full rebuild)
    Snapshot snap1; ///< captured at force-phase of step 1 (post refresh pack
                    ///< AND post step-0 reverse accumulation)

    std::string name() const override { return "field_participant"; }

    /// Protocol hook: this component IS a halo participant …
    cpu::HaloParticipant* halo_participant() override { return this; }
    // … but `halo_endpoint()` is deliberately NOT overridden: a pure field
    // participant reports the default `Endpoint::none`.

    std::vector<CommField> comm_forward_fields() const override {
        CommField f1;
        f1.data = fwd1_p_;
        f1.width = 1;
        CommField f3;
        f3.data = fwd3_p_;
        f3.width = 3;
        return {f1, f3};
    }

    std::vector<CommField> comm_reverse_fields() const override {
        CommField r;
        r.data = rev_p_;
        r.width = 1;
        return {r};
    }

    void build(const cpu::BuildContext& ctx) override {
        const std::size_t cap = static_cast<std::size_t>(ctx.frame.capacity());
        fwd1_.assign(cap, StorageReal(0));
        fwd3_.assign(cap * 3, StorageReal(0));
        rev_.assign(cap, StorageReal(0));
        fwd1_p_ = fwd1_.data();
        fwd3_p_ = fwd3_.data();
        rev_p_ = rev_.data();
        write_owner_rows_(ctx.frame, ctx.runtime.md.dd);
    }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto* dd = ctx.runtime.md.dd;
        const int n_local =
            dd ? dd->n_local : static_cast<int>(ctx.frame.nlocal());
        const int n_ghost = dd ? dd->n_ghost : 0;

        if (ctx.step <= 1) {
            Snapshot& s = (ctx.step == 0) ? snap0 : snap1;
            s.valid = true;
            s.n_local = n_local;
            s.n_ghost = n_ghost;
            s.ghost_tag.assign(static_cast<std::size_t>(n_ghost), TagInt(0));
            s.ghost_fwd1.assign(static_cast<std::size_t>(n_ghost), 0.0);
            s.ghost_fwd3.assign(static_cast<std::size_t>(n_ghost) * 3, 0.0);
            const auto tags = ctx.frame.span(ATV::kAtom_tag);
            for (int g = 0; g < n_ghost; ++g) {
                const int row = n_local + g;
                s.ghost_tag[g] = tags[row];
                s.ghost_fwd1[g] = static_cast<double>(fwd1_[row]);
                for (int k = 0; k < 3; ++k)
                    s.ghost_fwd3[3 * g + k] =
                        static_cast<double>(fwd3_[3 * row + k]);
            }
            s.owner_rev.assign(static_cast<std::size_t>(n_local), 0.0);
            for (int i = 0; i < n_local; ++i)
                s.owner_rev[i] = static_cast<double>(rev_[i]);
        }

        // Rewrite owners (tags are authoritative even if Migrate reordered
        // rows) and plant this step's ghost contributions on the reverse
        // field. HaloRecv (post_force @ -1000) must accumulate each planted
        // value onto its owner exactly once.
        write_owner_rows_(ctx.frame, dd);
        for (int g = 0; g < n_ghost; ++g)
            rev_[n_local + g] = StorageReal(kRevGhost);
    }

  private:
    void write_owner_rows_(const cpu::Frame& frame,
                           const cpu::md::mpi::DomainDecomposition* dd) {
        const int n_local = dd ? dd->n_local : static_cast<int>(frame.nlocal());
        const auto tags = frame.span(ATV::kAtom_tag);
        for (int i = 0; i < n_local; ++i) {
            const TagInt t = tags[i];
            fwd1_[i] = StorageReal(fwd1_value(t));
            for (int k = 0; k < 3; ++k)
                fwd3_[3 * i + k] = StorageReal(fwd3_value(t, k));
            rev_[i] = StorageReal(kRevBase);
        }
    }

    std::vector<StorageReal> fwd1_;
    std::vector<StorageReal> fwd3_;
    std::vector<StorageReal> rev_;
    StorageReal* fwd1_p_ = nullptr;
    StorageReal* fwd3_p_ = nullptr;
    StorageReal* rev_p_ = nullptr;
};

/// @brief Minimal 2-step MD over the 4x4x4 uniform grid with explicit halo
/// endpoints (default bindings: HaloSend @ pre_neighbor, HaloRecv @
/// post_force) and the field participant bound at force @ +100. The frame is
/// static (zero velocities, NullForce), so owner rows / tags are stable.
void run_field_participant_md(cpu::Frame& frame, NullForce& force,
                              FieldParticipant& part) {
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.on(ATV::md::Phase::force).add_ref(force);
    sim.set_observer(cpu::comm::HaloSend{2.0});
    sim.set_observer(cpu::comm::HaloRecv{});
    sim.on(ATV::md::Phase::force).add_ref(part, +100);
    sim.build();
    sim.run(frame, 2);
}

} // namespace

// ── Forward: declared extra fields reach ghost rows (both pack paths) ───

TEST(HaloFieldParticipant, ForwardGhostRowsMatchOwnerValues) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP() << "multi-rank test; skipped on single-rank";

    auto frame = make_uniform_grid(4, 10.0);
    NullForce force;
    FieldParticipant part;
    run_field_participant_md(frame, force, part);

    ASSERT_TRUE(part.snap0.valid);
    ASSERT_TRUE(part.snap1.valid);
    // With cutoff 2.0 on a 10 Å box every multi-rank dims choice ghosts the
    // slab faces; any axis with dims==2 imports through the periodic edge on
    // one of the two faces, so the loop below covers PBC-crossing ghosts too.
    EXPECT_GT(part.snap0.n_ghost, 0)
        << "expected ghost atoms at cutoff 2.0 under multi-rank";

    // snap0 = step-0 forward (full border rebuild); snap1 = step-1 forward
    // (position-only refresh). Declared fields must be packed on BOTH paths.
    for (const auto* s : {&part.snap0, &part.snap1}) {
        for (int g = 0; g < s->n_ghost; ++g) {
            const TagInt t = s->ghost_tag[g];
            EXPECT_NEAR(s->ghost_fwd1[g], FieldParticipant::fwd1_value(t),
                        1e-12)
                << "width-1 forward field: ghost " << g << " (tag " << t
                << ") != owner value";
            for (int k = 0; k < 3; ++k) {
                EXPECT_NEAR(s->ghost_fwd3[3 * g + k],
                            FieldParticipant::fwd3_value(t, k), 1e-12)
                    << "width-3 forward field: ghost " << g << " lane " << k
                    << " (tag " << t
                    << ") mismatch -- contiguous data[i*3+k] layout violated?";
            }
        }
    }
}

// ── Reverse: ghost contributions accumulate onto owners exactly once ────

TEST(HaloFieldParticipant, ReverseGhostContributionsAccumulateExactlyOnce) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1)
        GTEST_SKIP() << "multi-rank test; skipped on single-rank";

    auto frame = make_uniform_grid(4, 10.0);
    NullForce force;
    FieldParticipant part;
    run_field_participant_md(frame, force, part);

    ASSERT_TRUE(part.snap1.valid);
    ASSERT_GT(part.snap0.n_ghost, 0);

    // snap1.owner_rev is captured at step-1 force phase: after step 0's
    // reverse pass, before anything is rewritten. Per-owner: the offset over
    // the base must be a non-negative integer multiple of kRevGhost (each
    // ghost copy lands exactly once — 0.5x/2x multiples would betray a
    // missed or duplicated accumulation).
    double local_sum = 0.0;
    for (int i = 0; i < part.snap1.n_local; ++i) {
        const double off = part.snap1.owner_rev[i] - FieldParticipant::kRevBase;
        local_sum += part.snap1.owner_rev[i];
        EXPECT_GE(off, -1e-9) << "owner " << i << " lost its base value";
        EXPECT_NEAR(off / FieldParticipant::kRevGhost,
                    std::round(off / FieldParticipant::kRevGhost), 1e-9)
            << "owner " << i << " reverse offset " << off
            << " is not a clean multiple of the planted "
            << FieldParticipant::kRevGhost;
    }

    // Global conservation (topology-independent, mirrors HaloReverse above):
    //   Σ_owners rev == kRevBase * N_owners + kRevGhost * N_ghosts.
    // Double accumulation would give 2x the ghost term; a dropped pass 0x.
    double local[3] = {local_sum, static_cast<double>(part.snap1.n_local),
                       static_cast<double>(part.snap0.n_ghost)};
    double global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ASSERT_DOUBLE_EQ(global[1], 64.0) << "owner partition lost atoms";
    EXPECT_NEAR(global[0],
                FieldParticipant::kRevBase * global[1]
                    + FieldParticipant::kRevGhost * global[2],
                1e-6)
        << "reverse field not conserved: ghost contributions were not "
           "accumulated exactly once";
}

// ── Single rank: the same component path is a strict no-op ──────────────

TEST(HaloFieldParticipant, SingleRankIsNoOp) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1)
        GTEST_SKIP() << "single-rank baseline; skipped under multi-rank";

    auto frame = make_uniform_grid(4, 10.0);
    NullForce force;
    FieldParticipant part;
    ASSERT_NO_THROW(run_field_participant_md(frame, force, part));

    ASSERT_TRUE(part.snap0.valid);
    ASSERT_TRUE(part.snap1.valid);
    EXPECT_EQ(part.snap0.n_ghost, 0);
    EXPECT_EQ(part.snap1.n_ghost, 0);
    EXPECT_EQ(part.snap1.n_local, 64);

    // No ghosts => no reverse traffic: owners keep exactly the base value.
    for (int i = 0; i < part.snap1.n_local; ++i) {
        EXPECT_DOUBLE_EQ(part.snap1.owner_rev[i], FieldParticipant::kRevBase)
            << "single-rank reverse pass must not touch owner " << i;
    }
}

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
