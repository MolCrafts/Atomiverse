/// @file src/cpu/communication/comm_brick.cpp
/// @brief `cpu::comm::CommBrick` — LAMMPS brick communicator.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/comm_brick.cpp` (`CommBrick::forward_comm` / `reverse_comm` /
/// `exchange` / `borders`, checkout `970d4ba`, 2026-08-27). Structural
/// reimplementation over a private `comm::Halo` swap table — not a
/// verbatim dump. Public API is ATV-native (`cpu::Comm`).
///
/// Extra-field reverse communication ported from `src/comm_brick.cpp`
/// `CommBrick::reverse_comm(Pair*, int)`, checkout
/// `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
///
/// Fractional slab selection and local periodic-image copies follow
/// LAMMPS src/comm_brick.cpp, checkout
/// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
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
#include <stdexcept>
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
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace comm {

// ── Per-instance buffer storage (PIMPL) ──────────────────────────────
//
// Sequential halo swaps share stream 0; owner migration uses two concurrent
// streams. Payload capacity belongs to the communicator, never to each swap.

namespace detail {

struct CommBuffers {
    std::array<std::vector<std::byte>, 2> send_pos;
    std::array<std::vector<std::byte>, 2> recv_pos;
    /// Extra-field forward/reverse packer paths: doubles, one
    /// shared buffer pair, sized at the end of `borders()`.
    std::vector<double> send_pack;
    std::vector<double> recv_pack;
    /// Migration count-handshake scratch: the two owner-exchange streams
    /// post their counts nonblocking (`exchange_round`), which needs
    /// addresses that outlive the call. The halo's own handshake is a
    /// blocking per-swap `Sendrecv` and uses stack locals.
    std::array<int, 2> send_count{};
#ifdef ATV_HAS_MPI
    /// Request scratch for `exchange_round`: at most one send + one receive
    /// per index in flight.
    std::array<MPI_Request, 4> requests;
#endif
    PackPlan border_plan;
    PackPlan comm_plan;
    PackPlan reverse_plan;
    PackPlan exchange_plan;

    /// @brief Reserve packer buffers for `nsize` doubles × `max_atoms`.
    ///        Capacity only grows; a steady-state interval does not
    ///        reallocate once this has been called with the interval max.
    void reserve_packer(int nsize, int max_atoms) {
        if (nsize <= 0 || max_atoms <= 0)
            return;
        const std::size_t cap = checked_mul(static_cast<std::size_t>(nsize),
                                            static_cast<std::size_t>(max_atoms),
                                            "halo packer reserve");
        send_pack.reserve(cap);
        recv_pack.reserve(cap);
    }

#ifdef ATV_HAS_MPI
    /// @brief One concurrent point-to-point round over `[begin, end)`.
    ///
    /// The owner-exchange (migration) transport: its two streams cross the
    /// two faces of one axis at once and are unordered with respect to each
    /// other, so they are posted together and distinguished by tag. The halo
    /// does not use this — its swaps are self-matched and sequential
    /// (`Swap::recvproc`).
    ///
    /// Buffers and requests are `this`. For each index `s` the recv and
    /// send callables may invoke `post(buf, nbytes, proc, tag)` zero or
    /// more times; a zero-byte or `MPI_PROC_NULL` post is a skip. After
    /// every index has filled, `Waitall`. MPI_BYTE throughout — a count
    /// handshake posts `sizeof(int)` bytes. No configuration struct.
    ///
    /// @param comm Cartesian communicator.
    /// @param begin Inclusive first index of this round.
    /// @param end Exclusive end index of this round.
    /// @param recv Called as `recv(s, post)` for each `s` in the interval.
    /// @param send Called as `send(s, post)` for each `s` in the interval.
    template <typename RecvFn, typename SendFn>
    void exchange_round(MPI_Comm comm, RecvFn&& recv, SendFn&& send) {
        int nreq = 0;
        auto post_recv = [&](void* buf, int nbytes, int proc, int tag) {
            if (nbytes <= 0 || proc == MPI_PROC_NULL
                || proc == Swap::kNoPartner)
                return;
            MPI_Irecv(buf, nbytes, MPI_BYTE, proc, tag, comm,
                      &requests[static_cast<std::size_t>(nreq++)]);
        };
        auto post_send = [&](void* buf, int nbytes, int proc, int tag) {
            if (nbytes <= 0 || proc == MPI_PROC_NULL
                || proc == Swap::kNoPartner)
                return;
            MPI_Isend(buf, nbytes, MPI_BYTE, proc, tag, comm,
                      &requests[static_cast<std::size_t>(nreq++)]);
        };
        for (int s = 0; s < 2; ++s) {
            recv(s, post_recv);
            send(s, post_send);
        }
        if (nreq > 0)
            MPI_Waitall(nreq, requests.data(), MPI_STATUSES_IGNORE);
    }
#endif
};

} // namespace detail

namespace {
std::size_t buffer_bytes(const detail::CommBuffers* b) {
    if (!b)
        return 0;
    std::size_t n = 0;
    for (const auto& buf : b->send_pos)
        n += buf.capacity();
    for (const auto& buf : b->recv_pos)
        n += buf.capacity();
    n += (b->send_pack.capacity() + b->recv_pack.capacity()) * sizeof(double);
    return n;
}
} // namespace

CommBrick::CommBrick(md::mpi::DomainDecomposition& domain) : dd_(&domain) {}
CommBrick::~CommBrick() = default;
CommBrick::CommBrick(CommBrick&&) noexcept = default;
CommBrick& CommBrick::operator=(CommBrick&&) noexcept = default;

std::size_t CommBrick::memory_bytes() const {
    return buffer_bytes(bufs_.get());
}

void CommBrick::set_ghost_cutoff(double cutoff) { ghost_cutoff_ = cutoff; }

namespace {

// The MPI-free halo sentinel is distinct from MPI_PROC_NULL on OpenMPI.
// Transport posting above accepts either marker without sending a request.

using detail::CommBuffers;

#ifdef ATV_HAS_MPI

// Append owned atom `i`'s fixed-stride payload to the back of `buf`.
inline void pack_to(std::vector<std::byte>& buf, const Frame& frame,
                    const PackPlan& plan, int i, std::size_t payload_bytes) {
    const std::size_t off = buf.size();
    buf.resize(off + payload_bytes); // new std::byte are value-initialized
    (void)plan.pack_row(frame, i, buf.data() + off, payload_bytes);
}

// Staged nearest-neighbor owner exchange along one Cartesian axis (LAMMPS
// comm_brick::exchange style). Owned atoms whose `dim` coordinate falls outside
// [sublo, subhi) are shipped to the lo/hi FACE NEIGHBOR; the loop forwards
// multi-subdomain movers (and the rare PBC wrap-around atom) one hop per round
// until no rank holds an in-flight atom. Uses only nearest-neighbor
// point-to-point + one single-int Allreduce per round — O(1) point-to-point +
// O(log P) reduce fabric state, versus MPI_Alltoallv's O(world) per-peer state
// that made AV's cross-node RSS blow up (3.16x LAMMPS at 1024 ranks). Positions
// are already wrapped into [0, L) by Domain::pbc() before exchange, so a
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
                        std::size_t payload_bytes, PackPlan& plan,
                        CommBuffers& bufs) {
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

    // Two streams occupy CommBuffers slots 0 (hi) and 1 (lo). Tags are 0 and 1
    // — the two concurrent messages, same reason the halo round uses the
    // swap index. Count Waitall finishes before payload, so both phases
    // reuse those tags (LAMMPS uses 0 because it is sequential).

    auto& send_hi = bufs.send_pos[0];
    auto& send_lo = bufs.send_pos[1];
    auto& recv_lo = bufs.recv_pos[0];
    auto& recv_hi = bufs.recv_pos[1];
    while (true) {
        // Re-fetch each round: set_capacity below may have reallocated the SoA,
        // invalidating any span resolved in a previous round.
        const StorageReal* __restrict q = frame.span(axis_key);
        const StorageReal* x = frame.span(kAtom_x);
        const StorageReal* y = frame.span(kAtom_y);
        const StorageReal* z = frame.span(kAtom_z);
        send_lo.clear();
        send_hi.clear();
        int keep = 0;
        for (int i = 0; i < frame.nlocal(); ++i) {
            double qi = frame.box.is_orthorhombic()
                            ? q[i] / frame.box.length(dim)
                            : frame.box.to_fractional({x[i], y[i], z[i]})[dim];
            qi -= std::floor(qi);
            if (qi >= 1.0)
                qi = 0.0;
            if (qi < sublo) {
                pack_to(send_lo, frame, plan, i,
                        payload_bytes); // exits −dim face
            } else if (qi >= subhi) {
                pack_to(send_hi, frame, plan, i,
                        payload_bytes); // exits +dim face
            } else {
                if (keep != i)
                    frame.copy_row(keep, i); // stays; compact to front
                ++keep;
            }
        }
        frame.set_nlocal(keep);

        const int n_send_lo = static_cast<int>(send_lo.size() / payload_bytes);
        const int n_send_hi = static_cast<int>(send_hi.size() / payload_bytes);

        // Two independent streams (distinct tags so they never collide even
        // when dims[dim]==2 and lo_nbr==hi_nbr): +dim atoms go to hi_nbr and
        // arrive from lo_nbr; −dim atoms go to lo_nbr and arrive from hi_nbr.
        int n_recv_lo = 0, n_recv_hi = 0; // +dim from lo, −dim from hi
        bufs.send_count[0] = n_send_hi;
        bufs.send_count[1] = n_send_lo;
        bufs.exchange_round(
            comm,
            [&](int s, auto post) {
                post(s == 0 ? &n_recv_lo : &n_recv_hi,
                     static_cast<int>(sizeof(int)), s == 0 ? lo_nbr : hi_nbr,
                     s);
            },
            [&](int s, auto post) {
                post(&bufs.send_count[static_cast<std::size_t>(s)],
                     static_cast<int>(sizeof(int)), s == 0 ? hi_nbr : lo_nbr,
                     s);
            });

        recv_lo.resize(static_cast<std::size_t>(n_recv_lo) * payload_bytes);
        recv_hi.resize(static_cast<std::size_t>(n_recv_hi) * payload_bytes);
        bufs.exchange_round(
            comm,
            [&](int s, auto post) {
                auto& rbuf = s == 0 ? recv_lo : recv_hi;
                post(rbuf.data(),
                     checked_mpi_byte_count(rbuf.size(), "migrate/recv"),
                     s == 0 ? lo_nbr : hi_nbr, s);
            },
            [&](int s, auto post) {
                auto& sbuf = s == 0 ? send_hi : send_lo;
                post(sbuf.data(),
                     checked_mpi_byte_count(sbuf.size(), "migrate/send"),
                     s == 0 ? hi_nbr : lo_nbr, s);
            });

        const std::int64_t n_needed_wide =
            static_cast<std::int64_t>(frame.nlocal()) + n_recv_lo + n_recv_hi;
        if (n_needed_wide > std::numeric_limits<int>::max())
            throw std::overflow_error(
                "migrate: owned row count exceeds INT_MAX");
        const int n_needed = static_cast<int>(n_needed_wide);
        if (n_needed > frame.capacity()) {
            const std::int64_t requested =
                static_cast<std::int64_t>(n_needed) + n_needed / 2 + 1024;
            if (requested > std::numeric_limits<int>::max())
                throw std::overflow_error("migrate: capacity growth overflow");
            frame.set_capacity(static_cast<int>(requested));
            const std::uint64_t fingerprint = plan.fingerprint;
            plan = make_exchange_plan(frame);
            require_schema_fingerprint(plan, fingerprint, plan.version);
        }
        const int first_new = frame.nlocal();
        int slot = frame.nlocal();
        std::size_t o = 0;
        for (int k = 0; k < n_recv_lo; ++k, o += payload_bytes)
            (void)plan.unpack_row(frame, slot++, recv_lo.data() + o,
                                  payload_bytes);
        o = 0;
        for (int k = 0; k < n_recv_hi; ++k, o += payload_bytes)
            (void)plan.unpack_row(frame, slot++, recv_hi.data() + o,
                                  payload_bytes);
        frame.set_nlocal(slot);

        // Another hop is owed only for a row that just arrived and is still
        // outside this slab (multi-subdomain drift / PBC wrap-around). Every
        // pre-existing row was either shipped above or is a keeper, so this
        // scan over the new rows alone is the complete local condition.
        int pending = 0;
        const StorageReal* __restrict qn = frame.span(axis_key);
        x = frame.span(kAtom_x);
        y = frame.span(kAtom_y);
        z = frame.span(kAtom_z);
        for (int r = first_new; r < slot; ++r) {
            double qr = frame.box.is_orthorhombic()
                            ? qn[r] / frame.box.length(dim)
                            : frame.box.to_fractional({x[r], y[r], z[r]})[dim];
            qr -= std::floor(qr);
            if (qr >= 1.0)
                qr = 0.0;
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

#endif // ATV_HAS_MPI

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

void prealloc_bufs(CommBuffers& b, int initial_cap,
                   std::size_t border_row_bytes, std::size_t comm_row_bytes,
                   std::size_t reverse_row_bytes) {
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
    b.send_pos[0].reserve(reverse_bytes);
    b.recv_pos[0].reserve(reverse_bytes);
}

// Per-swap helper for the border rebuild. The per-step position refresh is
// `CommBrick::forward_comm`, which replays this table instead of rebuilding
// it; encapsulating one swap as a single function keeps the orchestrator
// readable.

/// @brief Drive one swap of the cascading border rebuild.
///
/// LAMMPS `CommBrick::borders` shape, one swap at a time on tag 0: count
/// `Sendrecv`, standing `Irecv` from `recvproc`, pack, blocking `MPI_Send`
/// into `sendproc`, `Wait`, unpack. `Swap::recvproc` is the *opposite* face's
/// neighbour, so this rank's swap `s` is answered by the partner's swap `s`
/// — one message per pair per swap, which is what makes the literal tag 0
/// legal here, `dims[d] == 2` (both faces the same rank) included.
///
/// Rescans the round's window `[win_first, win_last)` for ghost candidates,
/// packs the border wire, registers each received tag in `tag_to_local` /
/// `ghost_tags`, and commits the swap's `nrecv` / `firstrecv`.
///
/// `slot_in_out` is the running ghost-slot write cursor: on entry it is
/// the next row to write (`n_local` before the first swap; after every
/// swap it advances by the ghosts that swap received). `frame_cap` is
/// the per-atom array capacity used for the overflow guard.
///
/// @param win_first Inclusive lower scan bound of this swap's forwarding
///        round.
/// @param win_last Exclusive upper scan bound, same.
inline void forward_swap(Frame& frame, md::mpi::DomainDecomposition& dd,
                         Halo& swaps, CommBuffers& bufs, int s, int win_first,
                         int win_last, int& frame_cap, int my_rank,
                         int& slot_in_out, PackPlan& plan,
                         const GhostContract& contract, TimerRegistry* timer) {
    constexpr int kCommCat = static_cast<int>(md::StepTiming::Comm);

    // A silent swap (no partner, or the partner is this rank) ships and
    // receives nothing, so its send list stays as `setup` left it: empty.
    // It still takes its place in the ghost-slot prefix sum.
    if (swaps[s].is_silent()) {
        slot_in_out = swaps.commit_recv(s, 0, slot_in_out);
        return;
    }

    {
        ScopedTimer scope(timer, "halo/brd_scan", -1, kCommCat);
        swaps.rebuild_sendlist(s, frame, win_first, win_last);
    }

    int n_recv = 0;
    auto& rbuf = bufs.recv_pos[0];
    if (swaps[s].sendself) {
        // LAMMPS local self-image path: identical wire and image shift, no MPI.
        n_recv = static_cast<int>(swaps[s].sendlist.size());
        pack_forward_for_swap(frame, swaps[s], plan, rbuf);
    } else {
#ifdef ATV_HAS_MPI
        {
            ScopedTimer scope(timer, "halo/brd_count", -1, kCommCat);
            int n_send = static_cast<int>(swaps[s].sendlist.size());
            MPI_Sendrecv(&n_send, 1, MPI_INT, swaps[s].sendproc, 0, &n_recv, 1,
                         MPI_INT, swaps[s].recvproc, 0, dd.cart_comm,
                         MPI_STATUS_IGNORE);
        }
        const std::size_t nbytes =
            checked_mul(static_cast<std::size_t>(n_recv), plan.bytes(1),
                        "halo receive buffer");
        rbuf.resize(nbytes);
        MPI_Request request = MPI_REQUEST_NULL;
        if (n_recv > 0)
            MPI_Irecv(
                rbuf.data(),
                checked_mpi_byte_count(rbuf.size(), "halo forward receive"),
                MPI_BYTE, swaps[s].recvproc, 0, dd.cart_comm, &request);
        {
            ScopedTimer scope(timer, "halo/brd_pack", -1, kCommCat);
            auto& sbuf = bufs.send_pos[0];
            pack_forward_for_swap(frame, swaps[s], plan, sbuf);
            if (!sbuf.empty())
                MPI_Send(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo forward send"),
                    MPI_BYTE, swaps[s].sendproc, 0, dd.cart_comm);
        }
        if (n_recv > 0) {
            ScopedTimer scope(timer, "halo/brd_wait", -1, kCommCat);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
#endif
    }

    const std::int64_t needed_wide =
        static_cast<std::int64_t>(slot_in_out) + n_recv;
    if (needed_wide > std::numeric_limits<int>::max())
        throw std::overflow_error("halo: ghost row count exceeds INT_MAX");
    const int needed = static_cast<int>(needed_wide);
    if (needed > frame_cap) {
        // Capacity guard. Ranks no longer hold the global-N backing store
        // (partition_atoms shrinks to owned + headroom to avoid per-rank
        // memory scaling with the total system size), so a thin subdomain
        // can legitimately need more ghost rows than the headroom — and a
        // multi-hop shell fills several rounds of them. Grow on demand
        // (explicit-copy realloc preserving the live rows) instead of
        // aborting; update the shared capacity mirrors and the by-ref
        // `frame_cap` so the remaining swaps see the new size.
        // 50% slack so a cascade of small overflows doesn't realloc each
        // swap.
        const std::int64_t requested =
            static_cast<std::int64_t>(needed) + needed / 2 + 1024;
        if (requested > std::numeric_limits<int>::max())
            throw std::overflow_error("halo: capacity growth overflow");
        const std::uint64_t fingerprint = plan.fingerprint;
        frame.set_capacity(static_cast<int>(requested));
        frame_cap = frame.capacity();
        plan = make_wire_plan(frame, plan.kind, contract);
        require_schema_fingerprint(plan, fingerprint, plan.version);
        ATV_DEBUG("halo_exchange: rank {} grew ghost capacity to {} "
                  "(needed {}) at axis {} round {}",
                  my_rank, frame_cap, needed, swaps[s].dim, swaps[s].round);
    }

    // Unpack into ghost slots, registering the tags. The unpack order ALSO
    // defines the slot layout `CommBrick` mirrors, which is why the rebuild
    // commits it to the table (`commit_recv`) instead of leaving each
    // consumer to re-derive the prefix sum.
    //
    // Span resolution sits AFTER the capacity guard above: a `set_capacity`
    // there would invalidate anything resolved before it.
    ScopedTimer unpack_scope(timer, "halo/brd_unpack", -1, kCommCat);
    const int slot = slot_in_out;
    plan.unpack(frame, n_recv, slot, rbuf.data(), plan.bytes(n_recv));
    // Ghost tag→slot tracking exists solely for bonded-tuple resolution
    // (ResolveView / select_owned_tuples). Pure pairwise systems never
    // query ghost tags, so skip the per-ghost hash inserts and the
    // companion ghost_tags log entirely — the map stays empty (~50 B per
    // ghost saved, plus the per-rebuild insert/erase churn).
    if (frame.has_bonded()) {
        const TagInt* __restrict sp_tag = frame.span(kAtom_tag).data();
        for (int k = 0; k < n_recv; ++k) {
            const TagInt tag = sp_tag[slot + k];
            // A self image must never replace its owned row in the tag map.
            // Likewise keep one canonical row for repeated periodic images.
            if (dd.tag_to_local.try_emplace(tag, slot + k).second)
                dd.ghost_tags.push_back(tag);
        }
    }
    slot_in_out = swaps.commit_recv(s, n_recv, slot);
}

/// @brief Run the whole border cascade over the published table.
///
/// Two nested walks, both in table order:
///
///   * **axis stages** — axis 0, then 1, then 2. Round 1 of an axis scans
///     `[0, slot)`, i.e. owners plus every ghost the earlier axes already
///     delivered, so a ghost gets *forwarded* to a later-axis neighbour and
///     the 12 edge / 8 corner neighbours are reached without ever addressing
///     them directly.
///   * **forwarding rounds inside an axis** — hop 1, then 2, … Round `r > 1`
///     offers exactly the ghost rows round `r - 1` delivered: LAMMPS
///     `CommBrick::borders`' `nfirst = nlast; nlast = nlocal + nghost`,
///     one window shared by the round's two faces. The `round > 1`
///     mid-plane band is what sorts that shared window between them — a
///     ghost that arrived across one face sits past the opposite edge of
///     this subdomain, so it fails its own face's band and passes the
///     other's. Feeding a round the owners again instead would re-ship them
///     to the same rank at the same periodic image and double-count every
///     boundary pair.
///
/// Rounds are serialized (round `r` cannot be packed before round `r - 1`
/// arrived); within a round the two swaps run one after the other, each a
/// self-matched tag-0 exchange.
///
/// @param slot_in_out Ghost-slot write cursor; `n_local` on entry, one past
///        the last ghost row on return.
inline void forward_cascade(Frame& frame, md::mpi::DomainDecomposition& dd,
                            Halo& swaps, CommBuffers& bufs, int& frame_cap,
                            int my_rank, int& slot_in_out, PackPlan& plan,
                            const GhostContract& contract,
                            TimerRegistry* timer) {
    const int nswap = static_cast<int>(swaps.size());
    for (int begin = 0; begin < nswap;) {
        const int stop = swaps.stage_end(begin);
        // Round 1 of the axis: owners plus every ghost the earlier axes
        // delivered. Each later round: exactly what the previous one added.
        int win_first = 0;
        int win_last = slot_in_out;

        for (int rbegin = begin; rbegin < stop;) {
            const int rend = swaps.round_end(rbegin);
            const int round_start = slot_in_out;
            for (int s = rbegin; s < rend; ++s)
                forward_swap(frame, dd, swaps, bufs, s, win_first, win_last,
                             frame_cap, my_rank, slot_in_out, plan, contract,
                             timer);
            win_first = round_start;
            win_last = slot_in_out;
            rbegin = rend;
        }
        begin = stop;
    }
}

// Zero every schema-declared reverse accumulator on ghost rows.
void zero_ghost_forces(Frame& frame, PackPlan& reverse_plan,
                       const GhostContract& contract) {
    const int lo = frame.nlocal();
    const int hi = frame.nlocal() + frame.nghost();
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable)
        reverse_plan = make_wire_plan(frame, PackKind::Reverse, contract);
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
    if (ghost_cutoff_ <= 0.0)
        return;
    auto* dd = dd_;
    if (!dd)
        return;
    dd->ghost_shell_width = ghost_cutoff_;
    if (!bufs_)
        bufs_ = std::make_unique<detail::CommBuffers>();

    // The wires are the compiled form of what the mounted components asked
    // for: `ctx.md.ghost` is the Driver's fold, and each pass compiles its
    // own plan off it — no per-wire copy to drift.
    // A Frame driven without a Driver fold (the MPI transport tests do this)
    // gets the base set — which is exactly what the fold produces when no
    // mounted component declares anything, so this is the same answer by a
    // shorter route, not a silent degradation.
    contract_.clear();
    if (ctx.runtime.md.ghost != nullptr)
        contract_ = *ctx.runtime.md.ghost;
    else
        contract_.declare(base_fields::kEngine);

    bufs_->border_plan = make_wire_plan(ctx.frame, PackKind::Border, contract_);
    bufs_->comm_plan = make_wire_plan(ctx.frame, PackKind::Comm, contract_);
    bufs_->reverse_plan =
        make_wire_plan(ctx.frame, PackKind::Reverse, contract_);
#ifdef ATV_HAS_MPI
    if (multi_rank_dd(ctx.runtime)) {
        require_collective_plan(bufs_->border_plan, dd->cart_comm,
                                "halo/border");
        require_collective_plan(bufs_->comm_plan, dd->cart_comm, "halo/comm");
        require_collective_plan(bufs_->reverse_plan, dd->cart_comm,
                                "halo/reverse");
    }
#endif
    // Per-step / per-rebuild paths compare against these LOCALLY: the
    // schema is rank-uniform once the collective above has anchored it, so
    // a local mismatch is a hard local error, not a reason to put an
    // Allreduce barrier in front of every borders pass (measured ~15% of
    // the borders cost at 256 ranks, plus the skew it serializes).
    verified_border_fingerprint_ = bufs_->border_plan.fingerprint;
    verified_reverse_fingerprint_ = bufs_->reverse_plan.fingerprint;
    const int swap_cap = ctx.frame.capacity() / 4 + 1024;
    prealloc_bufs(*bufs_, swap_cap, bufs_->border_plan.bytes(1),
                  bufs_->comm_plan.bytes(1), bufs_->reverse_plan.row_bytes());
    setup_done_ = true;
}

void CommBrick::borders(const cpu::LaunchContext& ctx) {
    if (ghost_cutoff_ <= 0.0) {
        ctx.frame.set_nghost(0);
        swaps_.invalidate();
        for (auto tag : dd_->ghost_tags)
            dd_->tag_to_local.erase(tag);
        dd_->ghost_tags.clear();
        dd_->ghost_shell_width = 0.0;
        ++dd_->halo_generation;
        return;
    }
    auto* dd = dd_;
    if (!dd)
        return;
    // `exchange()` and `reverse_comm()` also allocate `bufs_` on demand, so
    // its existence says nothing about whether the wire schemas were ever
    // cached. Key the one-time setup on the thing setup() actually produces.
    if (!setup_done_)
        setup(ctx);
    if (!setup_done_)
        return;

    Frame& frame = ctx.frame;
    CommBuffers& bufs = *bufs_;

    PackPlan& forward_plan = bufs.border_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Border)
        forward_plan = make_wire_plan(frame, PackKind::Border, contract_);

    int my_rank = 0;
#ifdef ATV_HAS_MPI
    if (multi_rank_dd(ctx.runtime))
        MPI_Comm_rank(dd->cart_comm, &my_rank);
#endif
    int frame_cap = frame.capacity();

    if (forward_plan.fingerprint != verified_border_fingerprint_)
        throw std::runtime_error(
            "halo/border: the Frame wire schema changed after setup "
            "anchored it; schema mutation mid-run is not supported.");

    frame.set_nghost(0);
    dd->ghost_shell_width = ghost_cutoff_;
    swaps_.setup_uniform(*dd, frame.box, ghost_cutoff_, my_rank);

    {
        ATV_PROFILE_SCOPE(ctx, "halo/tagmap", -1,
                          static_cast<int>(md::StepTiming::Comm));
        for (auto t : dd->ghost_tags)
            dd->tag_to_local.erase(t);
        dd->ghost_tags.clear();
    }

    int slot = frame.nlocal();
    forward_cascade(frame, *dd, swaps_, bufs, frame_cap, my_rank, slot,
                    forward_plan, contract_, ctx.runtime.timer);
    frame.set_nghost(slot - frame.nlocal());
    ++dd->halo_generation;
    zero_ghost_forces(frame, bufs.reverse_plan, contract_);
    int max_atoms = 1;
    for (std::size_t s = 0; s < swaps_.size(); ++s)
        max_atoms =
            std::max({max_atoms, static_cast<int>(swaps_[s].sendlist.size()),
                      swaps_[s].nrecv});
    bufs.reserve_packer(packer_nsize_, max_atoms);
}

void CommBrick::forward_comm(const cpu::LaunchContext& ctx) {
    if (ghost_cutoff_ <= 0.0)
        return;
    auto* dd = dd_;
    if (!dd)
        return;
    if (!bufs_ || swaps_.size() == 0)
        return;

    Frame& frame = ctx.frame;
    CommBuffers& bufs = *bufs_;

    PackPlan& forward_plan = bufs.comm_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Comm)
        forward_plan = make_wire_plan(frame, PackKind::Comm, contract_);

    const int nswap = static_cast<int>(swaps_.size());

#ifdef ATV_HAS_MPI
    const std::size_t row_bytes = forward_plan.bytes(1);
#endif

    // LAMMPS `CommBrick::forward_comm` (`src/comm_brick.cpp:485`), one swap
    // at a time on tag 0: standing receive, pack, blocking `MPI_Send`, wait
    // — no send-side `Isend`/`Waitall` (that Waitall was the measured Comm
    // wall at 256 ranks).
    //
    // Tag 0 is legal because `Swap::recvproc` is the OPPOSITE face's
    // neighbour: a swap ships one way and hears back from the other, so
    // both ranks' swap `s` are each other's partner and exactly one message
    // is in flight per pair per swap — at `dims[d] == 2`, where both faces
    // address the same rank, the two swaps are still separated by running
    // them in order.
    for (int s = 0; s < nswap; ++s) {
        const Swap& swap = swaps_[s];
        if (swap.is_silent())
            continue;
        auto& rbuf = bufs.recv_pos[0];
        if (swap.sendself) {
            pack_forward_for_swap(frame, swap, forward_plan, rbuf);
            forward_plan.unpack(frame, swap.nrecv, swap.firstrecv, rbuf.data(),
                                rbuf.size());
            continue;
        }
#ifdef ATV_HAS_MPI
        MPI_Request request = MPI_REQUEST_NULL;
        if (swap.nrecv > 0) {
            const std::size_t nbytes =
                checked_mul(static_cast<std::size_t>(swap.nrecv), row_bytes,
                            "halo refresh receive");
            if (rbuf.size() != nbytes)
                rbuf.resize(nbytes);
            MPI_Irecv(rbuf.data(),
                      checked_mpi_byte_count(nbytes, "halo refresh receive"),
                      MPI_BYTE, swap.recvproc, 0, dd->cart_comm, &request);
        }
        if (!swap.sendlist.empty()) {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            auto& sbuf = bufs.send_pos[0];
            pack_forward_for_swap(frame, swap, forward_plan, sbuf);
            MPI_Send(sbuf.data(),
                     checked_mpi_byte_count(sbuf.size(), "halo refresh send"),
                     MPI_BYTE, swap.sendproc, 0, dd->cart_comm);
        }
        if (swap.nrecv > 0) {
            {
                ATV_PROFILE_SCOPE(ctx, "halo/fwd_wait", -1,
                                  static_cast<int>(md::StepTiming::Comm));
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_unpack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            forward_plan.unpack(frame, swap.nrecv, swap.firstrecv, rbuf.data(),
                                rbuf.size());
        }
#endif
    }
    zero_ghost_forces(frame, bufs.reverse_plan, contract_);
}

void CommBrick::reverse_comm(const cpu::LaunchContext& ctx) {
    auto* dd = dd_;
    if (!dd)
        return;
    if (swaps_.size() == 0)
        return;
    Frame& frame = ctx.frame;
    if (!bufs_)
        bufs_ = std::make_unique<detail::CommBuffers>();
    CommBuffers& bufs = *bufs_;

    PackPlan& reverse_plan = bufs.reverse_plan;
    if (!reverse_plan.valid_for(frame) || !reverse_plan.writable
        || reverse_plan.kind != PackKind::Reverse)
        reverse_plan = make_wire_plan(frame, PackKind::Reverse, contract_);

    if (dd->halo_generation != last_checked_halo_generation_) {
        // Local guards only: the schema was anchored collectively in
        // setup(), and a broken ghost layout is a local fact — throwing
        // locally aborts the job with the same diagnostic, without an
        // Allreduce barrier per reneighbouring (which also serialized the
        // exchange/borders skew across all ranks).
        if (reverse_plan.fingerprint != verified_reverse_fingerprint_)
            throw std::runtime_error(
                "halo/reverse: the Frame wire schema changed after setup "
                "anchored it; schema mutation mid-run is not supported.");
        if (swaps_.total_recv() != frame.nghost()
            || frame.nlocal() + frame.nghost() > frame.capacity())
            throw std::runtime_error(
                "halo reverse ghost layout disagrees with Frame capacity: "
                "swap recv rows "
                + std::to_string(swaps_.total_recv()) + ", frame nghost "
                + std::to_string(frame.nghost()) + ", nlocal "
                + std::to_string(frame.nlocal()) + ", capacity "
                + std::to_string(frame.capacity()));
        last_checked_halo_generation_ = dd->halo_generation;
    }

    const int nswap = static_cast<int>(swaps_.size());

    // Mirror of `forward_comm` (LAMMPS `CommBrick::reverse_comm`,
    // `src/comm_brick.cpp:545`): walk the table backwards one swap at a
    // time, standing receive from `sendproc` — whoever holds the owners I
    // gave ghosts to — while the ghost accumulators go back to `recvproc`.
    // Deepest hop first: a shallower swap re-ships ghost forces the deeper
    // one just folded in. Tag 0, for the same self-matching reason as the
    // forward pass.
    for (int s = nswap - 1; s >= 0; --s) {
        const Swap& swap = swaps_[s];
        if (swap.is_silent())
            continue;
        auto& rbuf = bufs.recv_pos[0];
        if (swap.sendself) {
            reverse_plan.pack_rows(swap.firstrecv, swap.nrecv, rbuf);
            reverse_plan.accumulate(swap.sendlist, rbuf);
            continue;
        }
#ifdef ATV_HAS_MPI
        MPI_Request request = MPI_REQUEST_NULL;
        if (!swap.sendlist.empty()) {
            const std::size_t nbytes =
                reverse_plan.bytes(static_cast<int>(swap.sendlist.size()));
            if (rbuf.size() != nbytes)
                rbuf.resize(nbytes);
            MPI_Irecv(rbuf.data(),
                      checked_mpi_byte_count(nbytes, "halo reverse receive"),
                      MPI_BYTE, swap.sendproc, 0, dd->cart_comm, &request);
        }
        if (swap.nrecv != 0) {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            auto& sbuf = bufs.send_pos[0];
            reverse_plan.pack_rows(swap.firstrecv, swap.nrecv, sbuf);
            MPI_Send(sbuf.data(),
                     checked_mpi_byte_count(sbuf.size(), "halo reverse send"),
                     MPI_BYTE, swap.recvproc, 0, dd->cart_comm);
        }
        if (!swap.sendlist.empty()) {
            {
                ATV_PROFILE_SCOPE(ctx, "halo/rev_wait", -1,
                                  static_cast<int>(md::StepTiming::Comm));
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
            ATV_PROFILE_SCOPE(ctx, "halo/rev_accum", -1,
                              static_cast<int>(md::StepTiming::Comm));
            reverse_plan.accumulate(swap.sendlist, rbuf);
        }
#endif
    }
}

void CommBrick::exchange(const cpu::LaunchContext& ctx) {
#ifdef ATV_HAS_MPI
    if (ctx.runtime.mpi_size <= 1)
        return;
    if (ctx.runtime.opt.line_search_trial)
        return;
    auto* dd = dd_;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;

    Frame& frame = ctx.frame;

    // Owned rows are already inside the box: the driver runs LAMMPS
    // `Domain::pbc()` before calling exchange (Runtime::comm_rebuild_).
    // Migration compares min-image, so it could not correct an off-image row
    // itself, and the extended-axis cell list would compute every separation
    // against it one box length off.

    if (!bufs_)
        bufs_ = std::make_unique<detail::CommBuffers>();
    PackPlan& plan = bufs_->exchange_plan;
    if (!plan.valid_for(frame) || plan.kind != PackKind::Exchange)
        plan = make_exchange_plan(frame);
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
            "exchange anchored it; schema mutation mid-run is not supported.");
    }

    int any_local = 0;
    {
        // Same predicate `exchange_along_dim` uses (slab edges), not a
        // 3-D `owner_of` that wraps and binary-searches cuts again.
        const StorageReal* __restrict sp_x = frame.span(kAtom_x);
        const StorageReal* __restrict sp_y = frame.span(kAtom_y);
        const StorageReal* __restrict sp_z = frame.span(kAtom_z);
        for (int i = 0; i < frame.nlocal(); ++i) {
            Vec3 q = frame.box.to_fractional({sp_x[i], sp_y[i], sp_z[i]});
            {
                for (int d = 0; d < 3; ++d) {
                    q[d] -= std::floor(q[d]);
                    if (q[d] >= 1.0)
                        q[d] = 0.0;
                }
            }
            if (q[0] < dd->sublo[0] || q[0] >= dd->subhi[0]
                || q[1] < dd->sublo[1] || q[1] >= dd->subhi[1]
                || q[2] < dd->sublo[2] || q[2] >= dd->subhi[2]) {
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
    dd->partition_dirty = false;
    if (reduced[0] == 0)
        return;
    if (reduced[1]
        > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "owner_exchange: one native row exceeds MPI datatype count");
    const std::size_t payload_bytes = static_cast<std::size_t>(reduced[1]);

    frame.set_nghost(0);

    exchange_along_dim(frame, dd, 0, payload_bytes, plan, *bufs_);
    exchange_along_dim(frame, dd, 1, payload_bytes, plan, *bufs_);
    exchange_along_dim(frame, dd, 2, payload_bytes, plan, *bufs_);

    ++dd->neighbor_generation;

    dd->tag_to_local.clear();
    if (frame.has_bonded()) {
        dd->tag_to_local.reserve(static_cast<std::size_t>(frame.nlocal()));
        const TagInt* __restrict sp_tag = frame.span(kAtom_tag);
        for (int i = 0; i < frame.nlocal(); ++i) {
            dd->tag_to_local.emplace(sp_tag[i], i);
        }
    }
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    dd->ghost_tags.clear();
#endif
}

void CommBrick::forward_comm(Packer& packer) {
    // LAMMPS `CommBrick::forward_comm(Fix*)` (src/comm_brick.cpp:1078):
    // sequential per swap, tag 0 — the same self-matched exchange the
    // position halo runs (see `Swap::recvproc`).
    if (!dd_ || swaps_.size() == 0)
        return;
    const int nsize = packer.comm_forward();
    if (nsize <= 0)
        return;
    packer_nsize_ = std::max(packer_nsize_, nsize);
    if (!bufs_)
        bufs_ = std::make_unique<detail::CommBuffers>();
    CommBuffers& bufs = *bufs_;
    const int nswap = static_cast<int>(swaps_.size());

    int max_atoms = 1;
    for (int s = 0; s < nswap; ++s)
        max_atoms =
            std::max({max_atoms, static_cast<int>(swaps_[s].sendlist.size()),
                      swaps_[s].nrecv});
    bufs.reserve_packer(packer_nsize_, max_atoms);

    for (int s = 0; s < nswap; ++s) {
        const Swap& swap = swaps_[s];
        if (swap.is_silent())
            continue;
        auto& sbuf = bufs.send_pack;
        auto& rbuf = bufs.recv_pack;
        const int n_send = static_cast<int>(swap.sendlist.size());
        const std::size_t send_bytes = packer_bytes(n_send, nsize);
        const std::size_t recv_bytes = packer_bytes(swap.nrecv, nsize);

        sbuf.resize(std::max(send_bytes / sizeof(double), std::size_t{1}));
        rbuf.resize(std::max(recv_bytes / sizeof(double), std::size_t{1}));
        const int pbc_flag =
            (swap.pbc[0] != 0 || swap.pbc[1] != 0 || swap.pbc[2] != 0) ? 1 : 0;
        const int packed = packer.pack_forward_comm(
            n_send, swap.sendlist.data(), sbuf.data(), pbc_flag, swap.pbc);
        if (packed < 0
            || static_cast<std::size_t>(packed) != send_bytes / sizeof(double))
            throw std::runtime_error(
                "forward packer returned an inconsistent field count");
        if (swap.sendself) {
            packer.unpack_forward_comm(swap.nrecv, swap.firstrecv, sbuf.data());
            continue;
        }
#ifdef ATV_HAS_MPI
        const int send_count =
            checked_mpi_byte_count(send_bytes, "packer send");
        const int recv_count =
            checked_mpi_byte_count(recv_bytes, "packer recv");
        MPI_Request request = MPI_REQUEST_NULL;
        if (swap.nrecv > 0)
            MPI_Irecv(rbuf.data(), recv_count, MPI_BYTE, swap.recvproc, 0,
                      dd_->cart_comm, &request);
        if (n_send > 0)
            MPI_Send(sbuf.data(), send_count, MPI_BYTE, swap.sendproc, 0,
                     dd_->cart_comm);

        if (swap.nrecv > 0) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            packer.unpack_forward_comm(swap.nrecv, swap.firstrecv, rbuf.data());
        }
#endif
    }
}

void CommBrick::reverse_comm(ReversePacker& packer) {
    // LAMMPS reverse_comm(Pair*): undo each halo hop in reverse order,
    // accumulating onto its sendlist before an earlier hop packs those rows.
    if (!dd_ || swaps_.size() == 0)
        return;
    const int nsize = packer.comm_reverse();
    if (nsize <= 0)
        return;
    packer_nsize_ = std::max(packer_nsize_, nsize);
    if (!bufs_)
        bufs_ = std::make_unique<detail::CommBuffers>();
    CommBuffers& bufs = *bufs_;
    const int nswap = static_cast<int>(swaps_.size());

    int max_atoms = 1;
    for (int s = 0; s < nswap; ++s)
        max_atoms =
            std::max({max_atoms, static_cast<int>(swaps_[s].sendlist.size()),
                      swaps_[s].nrecv});
    bufs.reserve_packer(packer_nsize_, max_atoms);

    for (int s = nswap - 1; s >= 0; --s) {
        const Swap& swap = swaps_[s];
        if (swap.is_silent())
            continue;
        const int n_return = static_cast<int>(swap.sendlist.size());
        auto& sbuf = bufs.send_pack;
        auto& rbuf = bufs.recv_pack;
        const std::size_t send_bytes = packer_bytes(swap.nrecv, nsize);
        const std::size_t recv_bytes = packer_bytes(n_return, nsize);

        sbuf.resize(std::max(send_bytes / sizeof(double), std::size_t{1}));
        rbuf.resize(std::max(recv_bytes / sizeof(double), std::size_t{1}));
        const int packed =
            packer.pack_reverse_comm(swap.nrecv, swap.firstrecv, sbuf.data());
        if (packed < 0
            || static_cast<std::size_t>(packed) != send_bytes / sizeof(double))
            throw std::runtime_error(
                "reverse packer returned an inconsistent field count");
        if (swap.sendself) {
            packer.unpack_reverse_comm(n_return, swap.sendlist.data(),
                                       sbuf.data());
            continue;
        }
#ifdef ATV_HAS_MPI
        const int send_count =
            checked_mpi_byte_count(send_bytes, "reverse packer send");
        const int recv_count =
            checked_mpi_byte_count(recv_bytes, "reverse packer recv");
        MPI_Request request = MPI_REQUEST_NULL;
        if (n_return > 0)
            MPI_Irecv(rbuf.data(), recv_count, MPI_BYTE, swap.sendproc, 0,
                      dd_->cart_comm, &request);
        if (swap.nrecv > 0)
            MPI_Send(sbuf.data(), send_count, MPI_BYTE, swap.recvproc, 0,
                     dd_->cart_comm);
        if (n_return > 0) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            packer.unpack_reverse_comm(n_return, swap.sendlist.data(),
                                       rbuf.data());
        }
#endif
    }
}

} // namespace comm
} // namespace cpu
} // namespace ATV
