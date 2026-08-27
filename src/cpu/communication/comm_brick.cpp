/// @file src/cpu/communication/comm_brick.cpp
/// @brief `cpu::comm::CommBrick` — LAMMPS brick communicator.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/comm_brick.cpp` (`CommBrick::forward_comm` / `reverse_comm` /
/// `exchange` / `borders`, checkout `78a7ddf`, 2026-08-25). Structural
/// reimplementation over a private `comm::Halo` swap table — not a
/// verbatim dump. Public API is ATV-native (`cpu::Comm`).
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171

#include "atomiverse/cpu/communication/comm_brick.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/cpu/communication/atom_pack.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/communication/schema_mpi.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/dynamics/pbc.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace comm {

// ── Per-instance buffer storage (PIMPL) ──────────────────────────────
//
// `FaceBufs` holds only native byte arrays, request scratch and compiled
// plans, so the type is usable regardless of whether MPI is linked in; the
// send-/recv-force arrays simply stay empty on the !MPI path. Keeping the
// full definition here (rather than under `#ifdef ATV_HAS_MPI`) lets the
// non-MPI stubs and the destructors in both classes see a complete type.
//
// Everything is sized by the **swap count**, which is derived
// (`2 * sum_d maxneed_d`) rather than a fixed six.

namespace detail {

struct FaceBufs {
    std::vector<std::vector<std::byte>> send_pos;
    std::vector<std::vector<std::byte>> recv_pos;
    std::vector<std::vector<std::byte>> send_force;
    std::vector<std::vector<std::byte>> recv_force;
    /// Per-swap count-handshake scratch (`MPI_Isend`/`Irecv` need stable
    /// addresses for the duration of the exchange, so these cannot be
    /// stack locals of a helper).
    std::vector<int> send_count;
    std::vector<int> recv_count;
#ifdef ATV_HAS_MPI
    /// Request scratch: at most one send + one receive per swap in flight.
    std::vector<MPI_Request> requests;
#endif
    PackPlan border_plan;
    PackPlan comm_plan;
    PackPlan reverse_plan;

    /// @brief Size every per-swap array to `nswap`, preserving contents.
    void resize_swaps(std::size_t nswap) {
        send_pos.resize(nswap);
        recv_pos.resize(nswap);
        send_force.resize(nswap);
        recv_force.resize(nswap);
        send_count.resize(nswap, 0);
        recv_count.resize(nswap, 0);
#ifdef ATV_HAS_MPI
        requests.resize(2 * nswap);
#endif
    }
};

} // namespace detail

namespace {
std::size_t facebufs_bytes(const detail::FaceBufs* b) {
    if (!b)
        return 0;
    std::size_t n = 0;
    for (const auto& buf : b->send_pos)
        n += buf.size();
    for (const auto& buf : b->recv_pos)
        n += buf.size();
    for (const auto& buf : b->send_force)
        n += buf.size();
    for (const auto& buf : b->recv_force)
        n += buf.size();
    return n * sizeof(std::byte);
}
} // namespace

CommBrick::CommBrick() = default;
CommBrick::~CommBrick() = default;
CommBrick::CommBrick(CommBrick&&) noexcept = default;
CommBrick& CommBrick::operator=(CommBrick&&) noexcept = default;

std::size_t CommBrick::send_buf_capacity_for_test() const {
    return (bufs_ && !bufs_->send_pos.empty()) ? bufs_->send_pos[0].capacity()
                                               : 0;
}

std::size_t CommBrick::memory_bytes() const {
    return facebufs_bytes(bufs_.get());
}

void CommBrick::set_ghost_cutoff(double cutoff) { ghost_cutoff_ = cutoff; }

void CommBrick::remember_(const cpu::LaunchContext& ctx) {
    dd_ = ctx.runtime.md.dd;
    mpi_size_ = ctx.runtime.mpi_size;
}

#ifdef ATV_HAS_MPI

namespace {

static_assert(Swap::kNoPartner == MPI_PROC_NULL,
              "Swap::kNoPartner must mirror the transport's no-partner "
              "sentinel; Halo is MPI-free and cannot name MPI_PROC_NULL");

using detail::FaceBufs;

// Append owned atom `i`'s fixed-stride payload to the back of `buf`.
inline void pack_to(std::vector<std::byte>& buf, const Frame& frame,
                    const PackPlan& plan, int i, std::size_t payload_bytes) {
    const std::size_t off = buf.size();
    buf.resize(off + payload_bytes);
    std::byte* dst = buf.data() + off;
    std::memset(dst, 0, payload_bytes);
    (void)plan.pack_row(frame, i, dst, payload_bytes);
}

// Staged nearest-neighbor owner exchange along one Cartesian axis (LAMMPS
// comm_brick::exchange style). Owned atoms whose `dim` coordinate falls outside
// [sublo, subhi) are shipped to the lo/hi FACE NEIGHBOR; the loop forwards
// multi-subdomain movers (and the rare PBC wrap-around atom) one hop per round
// until no rank holds an in-flight atom. Uses only nearest-neighbor
// point-to-point + one single-int Allreduce per round — O(1) point-to-point +
// O(log P) reduce fabric state, versus MPI_Alltoallv's O(world) per-peer state
// that made AV's cross-node RSS blow up (3.16x LAMMPS at 1024 ranks). Positions
// are already wrapped into [0, L) by the integrator's wrap_orthorhombic, so a
// raw coord-vs-slab compare matches DomainDecomposition::owner_of exactly.
//
// Round shape, LAMMPS `CommBrick::exchange`: the first round is
// **unconditional** — LAMMPS runs exactly one per dim and never asks the world
// whether it should. The Allreduce here only decides whether a *further* hop is
// owed, which the single-pass LAMMPS version cannot express at all, and it
// tests received-and-still-out-of-slab rather than "anyone sent something": a
// rank that received only atoms that landed in-slab is done, which is the
// common case and is what keeps the usual dim at ONE Allreduce. Both count and
// payload exchanges are nonblocking pairs — the lo-going and hi-going streams
// have no ordering between them, and two blocking Sendrecvs in a row cost two
// serialized round trips where one Waitall costs one.
void exchange_along_dim(Frame& frame, md::mpi::DomainDecomposition* dd, int dim,
                        std::size_t payload_bytes, PackPlan& plan) {
    if (dd->dims[dim] <= 1)
        return; // not decomposed along this axis — nothing leaves via it
    const int lo_nbr = dd->face_neighbors[static_cast<std::size_t>(2 * dim)];
    const int hi_nbr =
        dd->face_neighbors[static_cast<std::size_t>(2 * dim + 1)];
    const double sublo = dd->sublo[dim];
    const double subhi = dd->subhi[dim];
    MPI_Comm comm = dd->cart_comm;
    const ColKey axis_key =
        (dim == 0) ? kAtom_x : ((dim == 1) ? kAtom_y : kAtom_z);

    std::vector<std::byte> send_lo, send_hi, recv_lo, recv_hi;
    while (true) {
        // Re-fetch each round: set_capacity below may have reallocated the SoA,
        // invalidating any span resolved in a previous round.
        const StorageReal* __restrict q = frame.span(axis_key);
        send_lo.clear();
        send_hi.clear();
        int keep = 0;
        for (int i = 0; i < dd->n_local; ++i) {
            const double qi = q[i];
            if (qi < sublo) {
                pack_to(send_lo, frame, plan, i,
                        payload_bytes); // exits −dim face
            } else if (qi >= subhi) {
                pack_to(send_hi, frame, plan, i,
                        payload_bytes); // exits +dim face
            } else {
                frame.copy_row(keep, i); // stays; compact to front
                ++keep;
            }
        }
        dd->n_local = keep;
        frame.set_nlocal(keep);

        const int n_send_lo = static_cast<int>(send_lo.size() / payload_bytes);
        const int n_send_hi = static_cast<int>(send_hi.size() / payload_bytes);

        // Two independent streams (distinct tags so they never collide even
        // when dims[dim]==2 and lo_nbr==hi_nbr): +dim atoms go to hi_nbr and
        // arrive from lo_nbr; −dim atoms go to lo_nbr and arrive from hi_nbr.
        int n_recv_lo = 0, n_recv_hi = 0; // +dim from lo, −dim from hi
        MPI_Request reqs[4];
        MPI_Isend(&n_send_hi, 1, MPI_INT, hi_nbr, 0, comm, &reqs[0]);
        MPI_Irecv(&n_recv_lo, 1, MPI_INT, lo_nbr, 0, comm, &reqs[1]);
        MPI_Isend(&n_send_lo, 1, MPI_INT, lo_nbr, 1, comm, &reqs[2]);
        MPI_Irecv(&n_recv_hi, 1, MPI_INT, hi_nbr, 1, comm, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // `resize` on a byte vector value-initializes the grown tail; the
        // payload overwrites all of it, so skip the zero pass.
        recv_lo.resize(static_cast<std::size_t>(n_recv_lo) * payload_bytes);
        recv_hi.resize(static_cast<std::size_t>(n_recv_hi) * payload_bytes);
        MPI_Isend(send_hi.data(),
                  checked_mpi_byte_count(send_hi.size(), "migrate/send_hi"),
                  MPI_BYTE, hi_nbr, 2, comm, &reqs[0]);
        MPI_Irecv(recv_lo.data(),
                  checked_mpi_byte_count(recv_lo.size(), "migrate/recv_lo"),
                  MPI_BYTE, lo_nbr, 2, comm, &reqs[1]);
        MPI_Isend(send_lo.data(),
                  checked_mpi_byte_count(send_lo.size(), "migrate/send_lo"),
                  MPI_BYTE, lo_nbr, 3, comm, &reqs[2]);
        MPI_Irecv(recv_hi.data(),
                  checked_mpi_byte_count(recv_hi.size(), "migrate/recv_hi"),
                  MPI_BYTE, hi_nbr, 3, comm, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        const int n_recv = n_recv_lo + n_recv_hi;
        const int n_needed = dd->n_local + n_recv;
        if (n_needed > frame.capacity()) {
            const std::int64_t requested =
                static_cast<std::int64_t>(n_needed) + n_needed / 2 + 1024;
            if (requested > std::numeric_limits<int>::max())
                throw std::overflow_error("migrate: capacity growth overflow");
            frame.set_capacity(static_cast<int>(requested));
            dd->capacity = frame.capacity();
            const std::uint64_t fingerprint = plan.fingerprint;
            plan = make_exchange_plan(frame);
            require_schema_fingerprint(plan, fingerprint, plan.version);
        }
        const int first_new = dd->n_local;
        int slot = dd->n_local;
        std::size_t o = 0;
        for (int k = 0; k < n_recv_lo; ++k, o += payload_bytes)
            (void)plan.unpack_row(frame, slot++, recv_lo.data() + o,
                                  payload_bytes);
        o = 0;
        for (int k = 0; k < n_recv_hi; ++k, o += payload_bytes)
            (void)plan.unpack_row(frame, slot++, recv_hi.data() + o,
                                  payload_bytes);
        dd->n_local = slot;
        frame.set_nlocal(slot);

        // Another hop is owed only for a row that just arrived and is still
        // outside this slab (multi-subdomain drift / PBC wrap-around). Every
        // pre-existing row was either shipped above or is a keeper, so this
        // scan over the new rows alone is the complete local condition.
        int pending = 0;
        const StorageReal* __restrict qn = frame.span(axis_key);
        for (int r = first_new; r < slot; ++r) {
            const double qr = qn[r];
            if (qr < sublo || qr >= subhi) {
                pending = 1;
                break;
            }
        }
        int global_pending = 0;
        MPI_Allreduce(&pending, &global_pending, 1, MPI_INT, MPI_MAX, comm);
        if (global_pending == 0)
            return;
    }
}

/// @brief Which exchange a halo message belongs to. One tag stride each, so
/// a table deeper than the old fixed six swaps cannot alias a count
/// handshake onto a payload.
enum class HaloTagKind : int { count = 0, forward = 1, reverse = 2 };

/// Swaps a single stride may address. A multi-hop table holds
/// `2 * sum_d maxneed_d` swaps — up to `2 * (dims - 1)` on a narrow axis —
/// so the stride is sized for a thousands-of-ranks axis while four strides
/// still fit the 32767 tag range MPI guarantees.
constexpr int kHaloTagStride = 4096;

/// @brief MPI tag for one swap's message of one kind.
/// @param kind Which of the four halo exchanges the message belongs to.
/// @param iswap Swap index in the published table.
/// @return Tag, unique across kinds and swaps.
/// @throws std::overflow_error if the table is deeper than one tag stride.
inline int halo_tag(HaloTagKind kind, int iswap) {
    if (iswap < 0 || iswap >= kHaloTagStride)
        throw std::overflow_error(
            "halo: swap table deeper than the MPI tag stride");
    return static_cast<int>(kind) * kHaloTagStride + iswap;
}

/// @brief Gather one swap's send list onto the wire.
///
/// The `full_rebuild` branch is gone: it used to pick `pack_border` vs
/// `pack_comm`, and the plan now carries its own kind. The caller passes
/// whichever plan the pass is running on and the wire follows.
void pack_forward_for_swap(const Frame& frame, const Swap& swap,
                           const PackPlan& plan,
                           std::vector<std::byte>& buffer) {
    const auto& list = swap.sendlist;
    const int n = static_cast<int>(list.size());
    const std::size_t wire_bytes = plan.bytes(n);
    buffer.resize(wire_bytes);
    double shift[3];
    swap.pbc_shift(frame.box, shift);
    (void)plan.pack(frame, list.data(), n, shift, buffer.data(), wire_bytes);
}

void prealloc_bufs(FaceBufs& b, int initial_cap, std::size_t border_row_bytes,
                   std::size_t comm_row_bytes, std::size_t reverse_row_bytes) {
    const std::size_t forward_row = std::max(border_row_bytes, comm_row_bytes);
    const std::size_t forward_bytes =
        checked_mul(static_cast<std::size_t>(initial_cap), forward_row,
                    "halo preallocate forward");
    const std::size_t reverse_bytes =
        checked_mul(static_cast<std::size_t>(initial_cap), reverse_row_bytes,
                    "halo preallocate reverse");
    for (auto& buf : b.send_pos)
        buf.reserve(forward_bytes);
    for (auto& buf : b.recv_pos)
        buf.reserve(forward_bytes);
    for (auto& buf : b.send_force)
        buf.reserve(reverse_bytes);
    for (auto& buf : b.recv_force)
        buf.reserve(reverse_bytes);
}

// Per-round helper for the forward path. Both the full rebuild and the
// position-only refresh walk the same rounds of the same table; what
// differs is whether send lists are rebuilt and whether the tag column is
// included in the packet. Encapsulating one round as a single function
// keeps the orchestrator readable.

/// @brief Drive one forwarding round — the `(lo, hi)` swap pair
/// `[begin, end)` of one hop of one axis — of the cascading forward halo.
///
/// Both swaps of the round go out concurrently (`Isend`/`Irecv` +
/// `Waitall`), never as two sequential exchanges: at `dims[d] == 2` the low
/// and high neighbour of the axis are the SAME rank, and a sequential pair
/// crosses its tags — each side blocks sending its first swap while waiting
/// to receive the second. It deadlocks, and it deadlocked.
///
/// `slot_in_out` is the running ghost-slot write cursor: on entry it is
/// the next row to write (`n_local` before the first round; after every
/// round it advances by the ghosts that round received). `frame_cap` is
/// the per-atom array capacity used for the overflow guard. When
/// `full_rebuild`, the function rescans each swap's own window
/// `[win_first[k], win_last[k])` for ghost candidates, packs positions+tag,
/// registers each received tag in `tag_to_local` / `ghost_tags`, and commits
/// each swap's `nrecv` / `firstrecv`. When `!full_rebuild`, send lists and
/// receive counts are replayed from the table and only positions get
/// refreshed.
///
/// @param win_first Per-swap inclusive lower scan bound, indexed by
///        `s - begin`; unused when `!full_rebuild`.
/// @param win_last Per-swap exclusive upper scan bound, same indexing.
inline void forward_round(Frame& frame, md::mpi::DomainDecomposition& dd,
                          Halo& swaps, FaceBufs& bufs, int begin, int end,
                          const int* win_first, const int* win_last,
                          bool full_rebuild, int& frame_cap, int my_rank,
                          int& slot_in_out, PackPlan& plan,
                          TimerRegistry* timer) {
    constexpr int kCommCat = static_cast<int>(md::StepTiming::Comm);
    // A silent swap (no partner, or the partner is this rank) ships and
    // receives nothing, so its send list stays as `setup` left it: empty.
    if (full_rebuild) {
        ScopedTimer scope(timer, "halo/brd_scan", -1, kCommCat);
        for (int s = begin; s < end; ++s)
            if (!swaps[s].is_silent())
                swaps.rebuild_sendlist(s, frame, win_first[s - begin],
                                       win_last[s - begin]);
    }

    for (int s = begin; s < end; ++s) {
        bufs.send_count[static_cast<std::size_t>(s)] =
            static_cast<int>(swaps[s].sendlist.size());
        bufs.recv_count[static_cast<std::size_t>(s)] =
            full_rebuild ? 0 : swaps[s].nrecv;
    }

    if (full_rebuild) {
        ScopedTimer scope(timer, "halo/brd_count", -1, kCommCat);
        int n_creqs = 0;
        for (int s = begin; s < end; ++s) {
            if (swaps[s].is_silent())
                continue;
            MPI_Isend(&bufs.send_count[static_cast<std::size_t>(s)], 1, MPI_INT,
                      swaps[s].sendproc, halo_tag(HaloTagKind::count, s),
                      dd.cart_comm,
                      &bufs.requests[static_cast<std::size_t>(n_creqs++)]);
            MPI_Irecv(
                &bufs.recv_count[static_cast<std::size_t>(s)], 1, MPI_INT,
                swaps[s].recvproc,
                halo_tag(HaloTagKind::count, Halo::mirror_swap(begin, end, s)),
                dd.cart_comm,
                &bufs.requests[static_cast<std::size_t>(n_creqs++)]);
        }
        if (n_creqs > 0)
            MPI_Waitall(n_creqs, bufs.requests.data(), MPI_STATUSES_IGNORE);
    }

    const std::size_t row_bytes = plan.bytes(1);
    {
        ScopedTimer scope(timer, "halo/brd_pack", -1, kCommCat);
        for (int s = begin; s < end; ++s) {
            auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
            auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
            if (swaps[s].is_silent())
                sbuf.clear();
            else
                pack_forward_for_swap(frame, swaps[s], plan, sbuf);
            const std::size_t nbytes =
                checked_mul(static_cast<std::size_t>(
                                bufs.recv_count[static_cast<std::size_t>(s)]),
                            row_bytes, "halo receive buffer");
            if (rbuf.size() != nbytes)
                rbuf.resize(nbytes);
        }
    }

    {
        ScopedTimer scope(timer, "halo/brd_wait", -1, kCommCat);
        int n_preqs = 0;
        for (int s = begin; s < end; ++s) {
            if (swaps[s].is_silent())
                continue;
            auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
            auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
            if (bufs.send_count[static_cast<std::size_t>(s)] > 0) {
                MPI_Isend(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo forward send"),
                    MPI_BYTE, swaps[s].sendproc,
                    halo_tag(HaloTagKind::forward, s), dd.cart_comm,
                    &bufs.requests[static_cast<std::size_t>(n_preqs++)]);
            }
            if (bufs.recv_count[static_cast<std::size_t>(s)] > 0) {
                MPI_Irecv(
                    rbuf.data(),
                    checked_mpi_byte_count(rbuf.size(), "halo forward receive"),
                    MPI_BYTE, swaps[s].recvproc,
                    halo_tag(HaloTagKind::forward,
                             Halo::mirror_swap(begin, end, s)),
                    dd.cart_comm,
                    &bufs.requests[static_cast<std::size_t>(n_preqs++)]);
            }
        }
        if (n_preqs > 0)
            MPI_Waitall(n_preqs, bufs.requests.data(), MPI_STATUSES_IGNORE);
    }

    if (full_rebuild) {
        // Capacity guard for this round. Ranks no longer hold the global-N
        // backing store (partition_atoms shrinks to owned + headroom to avoid
        // per-rank memory scaling with the total system size), so a thin
        // subdomain can legitimately need more ghost rows than the headroom —
        // and a multi-hop shell fills several rounds of them. Grow on demand
        // (explicit-copy realloc preserving the live rows) instead of
        // aborting; update the shared capacity mirrors and the by-ref
        // `frame_cap` so the remaining rounds see the new size.
        int new_ghosts = 0;
        for (int s = begin; s < end; ++s)
            new_ghosts += bufs.recv_count[static_cast<std::size_t>(s)];
        if (slot_in_out + new_ghosts > frame_cap) {
            const int needed = slot_in_out + new_ghosts;
            // 50% slack so a cascade of small overflows doesn't realloc
            // each round.
            const std::int64_t requested =
                static_cast<std::int64_t>(needed) + needed / 2 + 1024;
            if (requested > std::numeric_limits<int>::max())
                throw std::overflow_error("halo: capacity growth overflow");
            const std::uint64_t fingerprint = plan.fingerprint;
            frame.set_capacity(static_cast<int>(requested));
            frame_cap = frame.capacity();
            dd.capacity = frame.capacity();
            plan = make_border_plan(frame);
            require_schema_fingerprint(plan, fingerprint, plan.version);
            ATV_DEBUG("halo_exchange: rank {} grew ghost capacity to {} "
                      "(needed {}) at axis {} round {}",
                      my_rank, frame_cap, needed, swaps[begin].dim,
                      swaps[begin].round);
        }
    }

    // Unpack into ghost slots — for full rebuilds register tags too. The
    // unpack order ALSO defines the slot layout `CommBrick` mirrors, which
    // is why the rebuild commits it to the table (`commit_recv`) instead of
    // leaving each consumer to re-derive the prefix sum.
    //
    // Hoisted AFTER the capacity guard above (a `set_capacity` there would
    // invalidate any span resolved before it) and dominating both the
    // full_rebuild and refresh unpack loops below.
    ScopedTimer unpack_scope(timer, "halo/brd_unpack", -1, kCommCat);
    TagInt* __restrict sp_tag = frame.span(kAtom_tag).data();
    int slot = slot_in_out;
    // Ghost tag→slot tracking exists solely for bonded-tuple resolution
    // (ResolveView / select_owned_tuples). Pure pairwise systems never
    // query ghost tags, so skip the per-ghost hash inserts and the
    // companion ghost_tags log entirely — the map stays empty (~50 B per
    // ghost saved, plus the per-rebuild insert/erase churn).
    const bool track_tags = frame.has_bonded();
    for (int s = begin; s < end; ++s) {
        const int n = bufs.recv_count[static_cast<std::size_t>(s)];
        const auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        const std::size_t wire_bytes = plan.bytes(n);
        if (full_rebuild) {
            plan.unpack(frame, n, slot, rbuf.data(), wire_bytes);
            for (int k = 0; k < n; ++k) {
                const TagInt tag = sp_tag[slot + k];
                if (track_tags) {
                    dd.tag_to_local[tag] = slot + k;
                    dd.ghost_tags.push_back(tag);
                }
            }
            slot = swaps.commit_recv(s, n, slot);
        } else {
            // Replay: the table already knows where this swap's ghosts
            // live, and `firstrecv` is that same running prefix sum.
            const int first = swaps[s].firstrecv;
            plan.unpack(frame, n, first, rbuf.data(), wire_bytes);
            slot = first + n;
        }
    }
    slot_in_out = slot;
}

/// @brief Run the whole forward cascade over the published table.
///
/// Two nested walks, both in table order:
///
///   * **axis stages** — axis 0, then 1, then 2. Round 1 of an axis scans
///     `[0, slot)`, i.e. owners plus every ghost the earlier axes already
///     delivered, so a ghost gets *forwarded* to a later-axis neighbour and
///     the 12 edge / 8 corner neighbours are reached without ever addressing
///     them directly.
///   * **forwarding rounds inside an axis** — hop 1, then 2, … Round `r > 1`
///     of a face ships exactly what round `r - 1` delivered on the
///     **opposite** face of the same axis: the ghosts my high neighbour gave
///     me are what my low neighbour is missing at depth `r`, and vice versa.
///     That is the whole multi-hop shell — the same per-round shape
///     `exchange_along_dim` uses for owners
///     (`exchange_along_dim`, same file). Feeding a round the
///     owners again instead would re-ship them to the same rank at the same
///     periodic image and double-count every boundary pair.
///
/// Rounds are serialized (round `r` cannot be packed before round `r - 1`
/// arrived); the two faces of one round go out concurrently.
///
/// @param slot_in_out Ghost-slot write cursor; `n_local` on entry, one past
///        the last ghost row on return.
inline void forward_cascade(Frame& frame, md::mpi::DomainDecomposition& dd,
                            Halo& swaps, FaceBufs& bufs, bool full_rebuild,
                            int& frame_cap, int my_rank, int& slot_in_out,
                            PackPlan& plan, TimerRegistry* timer) {
    const int nswap = static_cast<int>(swaps.size());
    for (int begin = 0; begin < nswap;) {
        const int stop = swaps.stage_end(begin);
        const int sources_end = slot_in_out;
        // Slot range each face of the PREVIOUS round of this axis received.
        int prev_first[2] = {0, 0};
        int prev_last[2] = {0, 0};

        for (int rbegin = begin; rbegin < stop;) {
            const int rend = swaps.round_end(rbegin);
            const int width = rend - rbegin;

            int win_first[2] = {0, 0};
            int win_last[2] = {0, 0};
            Halo::round_scan_window(swaps[rbegin].round, width, sources_end,
                                    prev_first, prev_last, win_first, win_last);

            forward_round(frame, dd, swaps, bufs, rbegin, rend, win_first,
                          win_last, full_rebuild, frame_cap, my_rank,
                          slot_in_out, plan, timer);

            for (int k = 0; k < width; ++k) {
                prev_first[k] = swaps[rbegin + k].firstrecv;
                prev_last[k] =
                    swaps[rbegin + k].firstrecv + swaps[rbegin + k].nrecv;
            }
            rbegin = rend;
        }
        begin = stop;
    }
}

// Zero every schema-declared reverse accumulator on ghost rows. Runtime
// columns opt in with kColReverse and require no endpoint field-list edit.
void zero_ghost_forces(Frame& frame, const md::mpi::DomainDecomposition& dd,
                       PackPlan& reverse_plan) {
    const int lo = dd.n_local;
    const int hi = dd.n_local + dd.n_ghost;
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable) {
        reverse_plan = make_reverse_plan(frame);
    }
    if (hi <= lo)
        return;
    for (const ColumnRef& field : reverse_plan.fields) {
        std::byte* destination =
            field.mutable_bytes()
            + static_cast<std::size_t>(lo) * field.row_bytes();
        const std::size_t bytes =
            checked_mul(static_cast<std::size_t>(hi - lo), field.row_bytes(),
                        "halo reverse zero");
        if (bytes != 0)
            std::memset(destination, 0, bytes);
    }
}

} // namespace

void CommBrick::setup(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ghost_cutoff_ <= 0.0)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || ctx.runtime.mpi_size <= 1 || dd->cart_comm == MPI_COMM_NULL)
        return;

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);
    dd->ghost_shell_width = ghost_cutoff_;
    swaps_.setup_uniform(*dd, ctx.frame.box, ghost_cutoff_, my_rank);
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    bufs_->resize_swaps(swaps_.size());
    bufs_->border_plan = make_border_plan(ctx.frame);
    bufs_->comm_plan = make_comm_plan(ctx.frame);
    bufs_->reverse_plan = make_reverse_plan(ctx.frame);
    require_collective_plan(bufs_->border_plan, dd->cart_comm, "halo/border");
    require_collective_plan(bufs_->comm_plan, dd->cart_comm, "halo/comm");
    // Per-step / per-rebuild paths compare against these LOCALLY: the
    // schema is rank-uniform once the collective above has anchored it, so
    // a local mismatch is a hard local error, not a reason to put an
    // Allreduce barrier in front of every borders pass (measured ~15% of
    // the borders cost at 256 ranks, plus the skew it serializes).
    verified_border_fingerprint_ = bufs_->border_plan.fingerprint;
    verified_reverse_fingerprint_ = bufs_->reverse_plan.fingerprint;
    const int swap_cap = dd->capacity / 4 + 1024;
    prealloc_bufs(*bufs_, swap_cap, bufs_->border_plan.bytes(1),
                  bufs_->comm_plan.bytes(1), bufs_->reverse_plan.row_bytes());
}

void CommBrick::borders(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ctx.runtime.mpi_size <= 1 || ghost_cutoff_ <= 0.0)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    if (!bufs_)
        setup(ctx);
    if (!bufs_)
        return;

    Frame& frame = ctx.frame;
    FaceBufs& bufs = *bufs_;
    ++dd->atom_sort_step;

    PackPlan& forward_plan = bufs.border_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Border)
        forward_plan = make_border_plan(frame);

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);
    int frame_cap = frame.capacity();

    if (forward_plan.fingerprint != verified_border_fingerprint_)
        throw std::runtime_error(
            "halo/border: the Frame wire schema changed after Comm::setup "
            "anchored it collectively; rebuild the communicator (run "
            "setup again) instead of reneighbouring across a schema "
            "mutation.");
    if (dd->atom_sort_enabled && dd->atom_sort_nfreq > 0
        && dd->atom_sort_step >= dd->atom_sort_next) {
        double bs = dd->atom_sort_binsize;
        if (!(bs > 0.0))
            bs = 0.5
                 * (dd->atom_sort_neigh_cut > 0.0 ? dd->atom_sort_neigh_cut
                                                  : ghost_cutoff_);
        if (md::sort_owned_atoms(frame, dd->n_local, bs))
            md::notify_sorted(dd, frame, &swaps_, /*clear_ghosts=*/true);
        dd->atom_sort_next =
            dd->atom_sort_step
            + (dd->atom_sort_nfreq > 0 ? dd->atom_sort_nfreq : 1);
    }

    frame.set_nghost(0);
    dd->ghost_shell_width = ghost_cutoff_;
    swaps_.setup_uniform(*dd, frame.box, ghost_cutoff_, my_rank);
    bufs.resize_swaps(swaps_.size());

    {
        ATV_PROFILE_SCOPE(ctx, "halo/tagmap", -1,
                          static_cast<int>(md::StepTiming::Comm));
        for (auto t : dd->ghost_tags)
            dd->tag_to_local.erase(t);
        dd->ghost_tags.clear();
    }

    int slot = dd->n_local;
    forward_cascade(frame, *dd, swaps_, bufs, /*full_rebuild=*/true, frame_cap,
                    my_rank, slot, forward_plan, ctx.runtime.timer);
    dd->n_ghost = slot - dd->n_local;
    frame.set_nghost(dd->n_ghost);
    ++dd->halo_generation;
    zero_ghost_forces(frame, *dd, bufs.reverse_plan);
}

void CommBrick::forward_comm(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ctx.runtime.mpi_size <= 1 || ghost_cutoff_ <= 0.0)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    if (!bufs_ || swaps_.size() == 0)
        return;

    Frame& frame = ctx.frame;
    FaceBufs& bufs = *bufs_;
    ++dd->atom_sort_step;

    PackPlan& forward_plan = bufs.comm_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Comm)
        forward_plan = make_comm_plan(frame);

    // Replay: sizes and slot layout are already committed in the table, so
    // every receive is posted up front (one pass over all rounds), then the
    // rounds run pack → Isend → wait-this-round → unpack in table order.
    // Pre-posting removes the round-chained rendezvous: a partner's send
    // matches an already-standing receive instead of an unexpected-message
    // queue, and later-round receipts overlap earlier-round packing.
    const int nswap = static_cast<int>(swaps_.size());
    bufs.resize_swaps(static_cast<std::size_t>(nswap));
    MPI_Request* recv_reqs = bufs.requests.data();
    MPI_Request* send_reqs = bufs.requests.data() + nswap;
    const std::size_t row_bytes = forward_plan.bytes(1);

    for (int begin = 0; begin < nswap;) {
        const int end = swaps_.round_end(begin);
        for (int s = begin; s < end; ++s) {
            recv_reqs[s] = MPI_REQUEST_NULL;
            const Swap& swap = swaps_[s];
            if (swap.is_silent() || swap.nrecv == 0)
                continue;
            auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
            const std::size_t nbytes =
                checked_mul(static_cast<std::size_t>(swap.nrecv), row_bytes,
                            "halo refresh receive");
            if (rbuf.size() != nbytes)
                rbuf.resize(nbytes);
            MPI_Irecv(rbuf.data(),
                      checked_mpi_byte_count(nbytes, "halo refresh receive"),
                      MPI_BYTE, swap.recvproc,
                      halo_tag(HaloTagKind::forward,
                               Halo::mirror_swap(begin, end, s)),
                      dd->cart_comm, &recv_reqs[s]);
        }
        begin = end;
    }

    int n_sends = 0;
    for (int begin = 0; begin < nswap;) {
        const int end = swaps_.round_end(begin);
        {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.sendlist.empty())
                    continue;
                auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
                pack_forward_for_swap(frame, swap, forward_plan, sbuf);
                MPI_Isend(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo refresh send"),
                    MPI_BYTE, swap.sendproc, halo_tag(HaloTagKind::forward, s),
                    dd->cart_comm, &send_reqs[n_sends++]);
            }
        }
        if (end - begin > 0) {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_wait", -1,
                              static_cast<int>(md::StepTiming::Comm));
            MPI_Waitall(end - begin, recv_reqs + begin, MPI_STATUSES_IGNORE);
        }
        {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_unpack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.nrecv == 0)
                    continue;
                const auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
                forward_plan.unpack(frame, swap.nrecv, swap.firstrecv,
                                    rbuf.data(), rbuf.size());
            }
        }
        begin = end;
    }
    if (n_sends > 0)
        MPI_Waitall(n_sends, send_reqs, MPI_STATUSES_IGNORE);
    zero_ghost_forces(frame, *dd, bufs.reverse_plan);
}

void CommBrick::reverse_comm(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ctx.runtime.mpi_size <= 1)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    if (swaps_.size() == 0)
        return;
    Frame& frame = ctx.frame;
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    FaceBufs& bufs = *bufs_;

    PackPlan& reverse_plan = bufs.reverse_plan;
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable
        || reverse_plan.kind != PackKind::Reverse)
        reverse_plan = make_reverse_plan(frame);

    if (dd->halo_generation != last_checked_halo_generation_) {
        // Local guards only: the schema was anchored collectively in
        // setup(), and a broken ghost layout is a local fact — throwing
        // locally aborts the job with the same diagnostic, without an
        // Allreduce barrier per reneighbouring (which also serialized the
        // exchange/borders skew across all ranks).
        if (reverse_plan.fingerprint != verified_reverse_fingerprint_)
            throw std::runtime_error(
                "halo/reverse: the Frame wire schema changed after "
                "Comm::setup anchored it collectively; rebuild the "
                "communicator instead of mutating the schema mid-run.");
        if (swaps_.total_recv() != dd->n_ghost
            || dd->n_local + dd->n_ghost > frame.capacity())
            throw std::runtime_error(
                "halo reverse ghost layout disagrees with Frame capacity");
        last_checked_halo_generation_ = dd->halo_generation;
    }

    const int nswap = static_cast<int>(swaps_.size());
    if (bufs.send_force.size() != static_cast<std::size_t>(nswap)) {
        bufs.resize_swaps(static_cast<std::size_t>(nswap));
        prealloc_bufs(bufs, dd->capacity, 0, 0, reverse_plan.row_bytes());
    }

    // Same pre-posted-receive discipline as the forward replay: receive
    // sizes are the send-list lengths, known for every round up front.
    // Packing stays in reverse round order — a shallower round re-ships
    // ghost forces the deeper round just folded in.
    MPI_Request* recv_reqs = bufs.requests.data();
    MPI_Request* send_reqs = bufs.requests.data() + nswap;
    for (int end = nswap; end > 0;) {
        const int begin = swaps_.round_begin(end);
        for (int s = begin; s < end; ++s) {
            recv_reqs[s] = MPI_REQUEST_NULL;
            const Swap& swap = swaps_[s];
            if (swap.is_silent() || swap.sendlist.empty())
                continue;
            auto& rbuf = bufs.recv_force[static_cast<std::size_t>(s)];
            const std::size_t nbytes =
                reverse_plan.bytes(static_cast<int>(swap.sendlist.size()));
            if (rbuf.size() != nbytes)
                rbuf.resize(nbytes);
            MPI_Irecv(rbuf.data(),
                      checked_mpi_byte_count(nbytes, "halo reverse receive"),
                      MPI_BYTE, swap.sendproc,
                      halo_tag(HaloTagKind::reverse,
                               Halo::mirror_swap(begin, end, s)),
                      dd->cart_comm, &recv_reqs[s]);
        }
        end = begin;
    }

    int n_sends = 0;
    for (int end = nswap; end > 0;) {
        const int begin = swaps_.round_begin(end);
        {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.nrecv == 0)
                    continue;
                auto& sbuf = bufs.send_force[static_cast<std::size_t>(s)];
                reverse_plan.pack_rows(swap.firstrecv, swap.nrecv, sbuf);
                MPI_Isend(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo reverse send"),
                    MPI_BYTE, swap.recvproc, halo_tag(HaloTagKind::reverse, s),
                    dd->cart_comm, &send_reqs[n_sends++]);
            }
        }
        if (end - begin > 0) {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_wait", -1,
                              static_cast<int>(md::StepTiming::Comm));
            MPI_Waitall(end - begin, recv_reqs + begin, MPI_STATUSES_IGNORE);
        }
        {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_accum", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.sendlist.empty())
                    continue;
                reverse_plan.accumulate(
                    swap.sendlist,
                    bufs.recv_force[static_cast<std::size_t>(s)]);
            }
        }
        end = begin;
    }
    if (n_sends > 0)
        MPI_Waitall(n_sends, send_reqs, MPI_STATUSES_IGNORE);
}

void CommBrick::exchange(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ctx.runtime.mpi_size <= 1)
        return;
    if (ctx.runtime.opt.line_search_trial)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;

    Frame& frame = ctx.frame;

    // LAMMPS `Domain::pbc()` before `Comm::exchange()` (`src/verlet.cpp`
    // reneighbor block): owned coordinates are canonicalized into the box
    // BEFORE homing. An owned row stored at the wrong periodic image (an
    // unwrapped initial frame, or drift past the edge between
    // reneighbourings) would otherwise keep its off-image coordinate:
    // migration compares min-image, so it never corrects it, and the
    // extended-axis cell list — which by the dual-image convention never
    // wraps decomposed axes — then bins the row into a clamped edge cell
    // and computes every separation against it one box length off. The
    // pair silently vanishes by the distance test.
    md::wrap_orthorhombic(frame, dd->n_local);

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);

    PackPlan plan = make_exchange_plan(frame);
    if (!exchange_schema_anchored_) {
        // First migration of the run: anchor the wire schema collectively,
        // the one place this path is allowed to pay for it.
        require_collective_schema(plan, dd->cart_comm, "owner_exchange");
        verified_exchange_fingerprint_ = plan.fingerprint;
        exchange_schema_anchored_ = true;
    } else if (plan.fingerprint != verified_exchange_fingerprint_) {
        // Local guard, same discipline as the halo plans: the schema is
        // rank-uniform once anchored, so a mismatch is a local fact and
        // throwing locally aborts with the same diagnostic — without an
        // Allreduce barrier in front of every reneighbouring.
        throw std::runtime_error(
            "owner_exchange: the Frame wire schema changed after the first "
            "exchange anchored it collectively; rebuild the communicator "
            "instead of mutating the schema mid-run.");
    }

    int any_local = 0;
    {
        const StorageReal* __restrict sp_x = frame.span(kAtom_x);
        const StorageReal* __restrict sp_y = frame.span(kAtom_y);
        const StorageReal* __restrict sp_z = frame.span(kAtom_z);
        for (int i = 0; i < dd->n_local; ++i) {
            if (dd->owner_of(sp_x[i], sp_y[i], sp_z[i], frame.box) != my_rank) {
                any_local = 1;
                break;
            }
        }
    }

    // One reduction answers both questions this path used to ask separately:
    // does anyone have an atom to home, and how wide is the widest native
    // row on any rank. `MPI_MAX` over a 0/1 flag is the `MPI_LOR` it
    // replaces, and the row width has to be current — a rank whose topology
    // capacity grew must widen everyone's uniform payload stride — so it
    // rides here rather than being cached.
    const std::uint64_t probe[2] = {
        static_cast<std::uint64_t>(any_local),
        static_cast<std::uint64_t>(plan.max_row_bytes())};
    std::uint64_t reduced[2] = {};
    MPI_Allreduce(probe, reduced, 2, MPI_UINT64_T, MPI_MAX, dd->cart_comm);
    if (reduced[0] == 0)
        return;
    if (reduced[1]
        > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "owner_exchange: one native row exceeds MPI datatype count");
    const std::size_t payload_bytes = static_cast<std::size_t>(reduced[1]);

    frame.set_nghost(0);
    dd->n_ghost = 0;

    exchange_along_dim(frame, dd, 0, payload_bytes, plan);
    exchange_along_dim(frame, dd, 1, payload_bytes, plan);
    exchange_along_dim(frame, dd, 2, payload_bytes, plan);

    dd->n_ghost = 0;
    ++dd->neighbor_generation;

    dd->tag_to_local.clear();
    if (frame.has_bonded()) {
        dd->tag_to_local.reserve(static_cast<std::size_t>(dd->n_local));
        const TagInt* __restrict sp_tag = frame.span(kAtom_tag);
        for (int i = 0; i < dd->n_local; ++i) {
            dd->tag_to_local.emplace(sp_tag[i], i);
        }
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, dd->n_local);
    dd->ghost_tags.clear();
}

void CommBrick::forward_comm(Packer& packer) {
    // LAMMPS `CommBrick::forward_comm(Fix*)` (src/comm_brick.cpp:1078).
    if (mpi_size_ <= 1)
        return;
    if (!dd_ || dd_->cart_comm == MPI_COMM_NULL || swaps_.size() == 0)
        return;
    const int nsize = packer.comm_forward();
    if (nsize <= 0)
        return;

    int my_rank = 0;
    MPI_Comm_rank(dd_->cart_comm, &my_rank);
    const int nswap = static_cast<int>(swaps_.size());
    std::vector<double> sbuf;
    std::vector<double> rbuf;
    for (int iswap = 0; iswap < nswap; ++iswap) {
        const Swap& swap = swaps_[iswap];
        if (swap.is_silent())
            continue;
        const int n_send = static_cast<int>(swap.sendlist.size());
        const int n_recv = swap.nrecv;
        const int pbc_flag =
            (swap.pbc[0] != 0 || swap.pbc[1] != 0 || swap.pbc[2] != 0) ? 1 : 0;
        sbuf.resize(static_cast<std::size_t>(std::max(n_send * nsize, 1)));
        const int n = packer.pack_forward_comm(n_send, swap.sendlist.data(),
                                               sbuf.data(), pbc_flag, swap.pbc);
        rbuf.resize(static_cast<std::size_t>(std::max(n_recv * nsize, 1)));
        MPI_Request request = MPI_REQUEST_NULL;
        double* buf = sbuf.data();
        if (swap.sendproc != my_rank) {
            if (n_recv > 0)
                MPI_Irecv(rbuf.data(), n_recv * nsize, MPI_DOUBLE,
                          swap.recvproc, 0, dd_->cart_comm, &request);
            if (n_send > 0)
                MPI_Send(sbuf.data(), n, MPI_DOUBLE, swap.sendproc, 0,
                         dd_->cart_comm);
            if (n_recv > 0)
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            buf = rbuf.data();
        }
        packer.unpack_forward_comm(n_recv, swap.firstrecv, buf);
    }
}

#else // !ATV_HAS_MPI

void CommBrick::setup(const cpu::LaunchContext& /*ctx*/) {}
void CommBrick::forward_comm(const cpu::LaunchContext& /*ctx*/) {}
void CommBrick::reverse_comm(const cpu::LaunchContext& /*ctx*/) {}
void CommBrick::exchange(const cpu::LaunchContext& /*ctx*/) {}
void CommBrick::borders(const cpu::LaunchContext& /*ctx*/) {}
void CommBrick::forward_comm(Packer& /*packer*/) {}

#endif // ATV_HAS_MPI

} // namespace comm
} // namespace cpu
} // namespace ATV
