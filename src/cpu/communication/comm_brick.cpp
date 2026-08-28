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
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/dynamics/atom_sort.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/pbc.h"

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
    /// Extra-field packer path (`forward_comm(Packer&)`): doubles, one
    /// pair of buffers per swap, sized once at the end of `borders()`.
    std::vector<std::vector<double>> send_pack;
    std::vector<std::vector<double>> recv_pack;
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
    PackPlan exchange_plan;

    /// @brief Size every per-swap array to `nswap`, preserving contents.
    void resize_swaps(std::size_t nswap) {
        send_pos.resize(nswap);
        recv_pos.resize(nswap);
        send_force.resize(nswap);
        recv_force.resize(nswap);
        send_pack.resize(nswap);
        recv_pack.resize(nswap);
        send_count.resize(nswap, 0);
        recv_count.resize(nswap, 0);
#ifdef ATV_HAS_MPI
        requests.resize(2 * nswap);
#endif
    }

    /// @brief Reserve packer buffers for `nsize` doubles × `max_atoms`.
    ///        Capacity only grows; a steady-state interval does not
    ///        reallocate once this has been called with the interval max.
    void reserve_packer(int nsize, int max_atoms) {
        if (nsize <= 0 || max_atoms <= 0)
            return;
        const std::size_t cap = checked_mul(static_cast<std::size_t>(nsize),
                                            static_cast<std::size_t>(max_atoms),
                                            "halo packer reserve");
        for (auto& b : send_pack)
            if (b.capacity() < cap)
                b.reserve(cap);
        for (auto& b : recv_pack)
            if (b.capacity() < cap)
                b.reserve(cap);
    }

#ifdef ATV_HAS_MPI
    /// @brief One concurrent point-to-point round over `[begin, end)`.
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
    void exchange_round(MPI_Comm comm, int begin, int end, RecvFn&& recv,
                        SendFn&& send) {
        const int width = std::max(0, end - begin);
        const std::size_t need = 2 * static_cast<std::size_t>(width);
        if (requests.size() < need)
            requests.resize(need);
        int nreq = 0;
        auto post_recv = [&](void* buf, int nbytes, int proc, int tag) {
            if (nbytes <= 0 || proc == MPI_PROC_NULL)
                return;
            MPI_Irecv(buf, nbytes, MPI_BYTE, proc, tag, comm,
                      &requests[static_cast<std::size_t>(nreq++)]);
        };
        auto post_send = [&](void* buf, int nbytes, int proc, int tag) {
            if (nbytes <= 0 || proc == MPI_PROC_NULL)
                return;
            MPI_Isend(buf, nbytes, MPI_BYTE, proc, tag, comm,
                      &requests[static_cast<std::size_t>(nreq++)]);
        };
        for (int s = begin; s < end; ++s) {
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
    n *= sizeof(std::byte);
    for (const auto& buf : b->send_pack)
        n += buf.size() * sizeof(double);
    for (const auto& buf : b->recv_pack)
        n += buf.size() * sizeof(double);
    return n;
}
} // namespace

CommBrick::CommBrick() = default;
CommBrick::~CommBrick() = default;
CommBrick::CommBrick(CommBrick&&) noexcept = default;
CommBrick& CommBrick::operator=(CommBrick&&) noexcept = default;

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
                        std::size_t payload_bytes, PackPlan& plan,
                        FaceBufs& bufs) {
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

    // Two streams occupy FaceBufs slots 0 (hi) and 1 (lo). Tags are 0 and 1
    // — the two concurrent messages, same reason the halo round uses the
    // swap index. Count Waitall finishes before payload, so both phases
    // reuse those tags (LAMMPS uses 0 because it is sequential).
    bufs.resize_swaps(std::max(bufs.send_pos.size(), std::size_t{2}));
    auto& send_hi = bufs.send_pos[0];
    auto& send_lo = bufs.send_pos[1];
    auto& recv_lo = bufs.recv_pos[0];
    auto& recv_hi = bufs.recv_pos[1];
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
                if (keep != i)
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
        bufs.send_count[0] = n_send_hi;
        bufs.send_count[1] = n_send_lo;
        bufs.exchange_round(
            comm, 0, 2,
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
            comm, 0, 2,
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
                          const GhostContract& contract, TimerRegistry* timer) {
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

    // Concurrent posts to the same neighbour (dims[d]==2) need distinct
    // MPI tags: the swap index. LAMMPS uses tag 0 because it is sequential.
    if (full_rebuild) {
        ScopedTimer scope(timer, "halo/brd_count", -1, kCommCat);
        bufs.exchange_round(
            dd.cart_comm, begin, end,
            [&](int s, auto post) {
                if (swaps[s].is_silent())
                    return;
                post(&bufs.recv_count[static_cast<std::size_t>(s)],
                     static_cast<int>(sizeof(int)), swaps[s].recvproc,
                     Halo::mirror_swap(begin, end, s));
            },
            [&](int s, auto post) {
                if (swaps[s].is_silent())
                    return;
                post(&bufs.send_count[static_cast<std::size_t>(s)],
                     static_cast<int>(sizeof(int)), swaps[s].sendproc, s);
            });
    }

    const std::size_t row_bytes = plan.bytes(1);
    for (int s = begin; s < end; ++s) {
        auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        const std::size_t nbytes =
            checked_mul(static_cast<std::size_t>(
                            bufs.recv_count[static_cast<std::size_t>(s)]),
                        row_bytes, "halo receive buffer");
        if (rbuf.size() != nbytes)
            rbuf.resize(nbytes);
    }

    // Recvs first so packing overlaps a standing receive (same shape as
    // `forward_comm`). Cascade round r+1 still waits on r's unpack.
    const int width = std::max(0, end - begin);
    const std::size_t need = 2 * static_cast<std::size_t>(width);
    if (bufs.requests.size() < need)
        bufs.requests.resize(need);
    int nreq = 0;
    for (int s = begin; s < end; ++s) {
        if (swaps[s].is_silent()
            || bufs.recv_count[static_cast<std::size_t>(s)] <= 0)
            continue;
        auto& rbuf = bufs.recv_pos[static_cast<std::size_t>(s)];
        MPI_Irecv(rbuf.data(),
                  checked_mpi_byte_count(rbuf.size(), "halo forward receive"),
                  MPI_BYTE, swaps[s].recvproc, Halo::mirror_swap(begin, end, s),
                  dd.cart_comm,
                  &bufs.requests[static_cast<std::size_t>(nreq++)]);
    }
    {
        ScopedTimer scope(timer, "halo/brd_pack", -1, kCommCat);
        for (int s = begin; s < end; ++s) {
            auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
            if (swaps[s].is_silent())
                sbuf.clear();
            else
                pack_forward_for_swap(frame, swaps[s], plan, sbuf);
        }
    }
    {
        ScopedTimer scope(timer, "halo/brd_wait", -1, kCommCat);
        for (int s = begin; s < end; ++s) {
            if (swaps[s].is_silent()
                || bufs.send_count[static_cast<std::size_t>(s)] <= 0)
                continue;
            auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
            MPI_Isend(sbuf.data(),
                      checked_mpi_byte_count(sbuf.size(), "halo forward send"),
                      MPI_BYTE, swaps[s].sendproc, s, dd.cart_comm,
                      &bufs.requests[static_cast<std::size_t>(nreq++)]);
        }
        if (nreq > 0)
            MPI_Waitall(nreq, bufs.requests.data(), MPI_STATUSES_IGNORE);
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
            plan = make_wire_plan(frame, plan.kind, contract);
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
                            PackPlan& plan, const GhostContract& contract,
                            TimerRegistry* timer) {
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
                          slot_in_out, plan, contract, timer);

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

// Zero every schema-declared reverse accumulator on ghost rows.
void zero_ghost_forces(Frame& frame, const md::mpi::DomainDecomposition& dd,
                       PackPlan& reverse_plan, const GhostContract& contract) {
    const int lo = dd.n_local;
    const int hi = dd.n_local + dd.n_ghost;
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
    require_collective_plan(bufs_->border_plan, dd->cart_comm, "halo/border");
    require_collective_plan(bufs_->comm_plan, dd->cart_comm, "halo/comm");
    require_collective_plan(bufs_->reverse_plan, dd->cart_comm, "halo/reverse");
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
    setup_done_ = true;
}

void CommBrick::borders(const cpu::LaunchContext& ctx) {
    remember_(ctx);
    if (ctx.runtime.mpi_size <= 1 || ghost_cutoff_ <= 0.0)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;
    // `exchange()` and `reverse_comm()` also allocate `bufs_` on demand, so
    // its existence says nothing about whether the wire schemas were ever
    // cached. Key the one-time setup on the thing setup() actually produces.
    if (!setup_done_)
        setup(ctx);
    if (!setup_done_)
        return;

    Frame& frame = ctx.frame;
    FaceBufs& bufs = *bufs_;

    PackPlan& forward_plan = bufs.border_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Border)
        forward_plan = make_wire_plan(frame, PackKind::Border, contract_);

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);
    int frame_cap = frame.capacity();

    if (forward_plan.fingerprint != verified_border_fingerprint_)
        throw std::runtime_error(
            "halo/border: the Frame wire schema changed after setup "
            "anchored it; schema mutation mid-run is not supported.");
    if (md::AtomSortState* sort = ctx.runtime.md.atom_sort) {
        sort->step = ctx.step;
        if (sort->due()) {
            const double bs = sort->resolved_binsize(ghost_cutoff_);
            if (md::sort_owned_atoms(frame, dd->n_local, bs))
                md::notify_sorted(dd, frame, &swaps_, /*clear_ghosts=*/true);
            sort->mark_sorted();
        }
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
                    my_rank, slot, forward_plan, contract_, ctx.runtime.timer);
    dd->n_ghost = slot - dd->n_local;
    frame.set_nghost(dd->n_ghost);
    ++dd->halo_generation;
    zero_ghost_forces(frame, *dd, bufs.reverse_plan, contract_);
    int max_atoms = 1;
    for (std::size_t s = 0; s < swaps_.size(); ++s)
        max_atoms =
            std::max({max_atoms, static_cast<int>(swaps_[s].sendlist.size()),
                      swaps_[s].nrecv});
    bufs.reserve_packer(packer_nsize_, max_atoms);
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

    PackPlan& forward_plan = bufs.comm_plan;
    if (!forward_plan.valid_for(frame) || !forward_plan.writable
        || forward_plan.kind != PackKind::Comm)
        forward_plan = make_wire_plan(frame, PackKind::Comm, contract_);

    const int nswap = static_cast<int>(swaps_.size());
    bufs.resize_swaps(static_cast<std::size_t>(nswap));
    const std::size_t row_bytes = forward_plan.bytes(1);

    // LAMMPS `CommBrick::forward_comm` (`src/comm_brick.cpp:485`) shape:
    // standing receive, pack, blocking `MPI_Send`, wait — no send-side
    // `Isend`/`Waitall` (that Waitall was the measured Comm wall at 256
    // ranks).
    //
    // Posted per ROUND, not per swap. LAMMPS can run one swap at a time on
    // tag 0 because there `sendproc != recvproc`: a swap ships one way and
    // hears back from the other, so both ranks' swap `i` are each other's
    // partner. ATV's table instead gives both faces of a round the SAME
    // partner (`Swap::sendproc == recvproc`, `halo.cpp` setup_uniform), so
    // this rank's swap `s` is answered by the partner's
    // `Halo::mirror_swap(s)`. Draining a round one swap at a time would
    // pair the wrong faces — and deadlock outright at `dims[d] == 2`, where
    // both faces of the round address the same rank. The round is therefore
    // the unit: post every receive in it, then send into partners that are
    // already listening.
    for (int begin = 0; begin < nswap;) {
        const int end = swaps_.round_end(begin);
        const int width = end - begin;
        if (bufs.requests.size() < static_cast<std::size_t>(width))
            bufs.requests.resize(static_cast<std::size_t>(width));
        int nreq = 0;
        for (int s = begin; s < end; ++s) {
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
                      MPI_BYTE, swap.recvproc, Halo::mirror_swap(begin, end, s),
                      dd->cart_comm,
                      &bufs.requests[static_cast<std::size_t>(nreq++)]);
        }
        {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.sendlist.empty())
                    continue;
                auto& sbuf = bufs.send_pos[static_cast<std::size_t>(s)];
                pack_forward_for_swap(frame, swap, forward_plan, sbuf);
                MPI_Send(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo refresh send"),
                    MPI_BYTE, swap.sendproc, s, dd->cart_comm);
            }
        }
        if (nreq > 0) {
            ATV_PROFILE_SCOPE(ctx, "halo/fwd_wait", -1,
                              static_cast<int>(md::StepTiming::Comm));
            MPI_Waitall(nreq, bufs.requests.data(), MPI_STATUSES_IGNORE);
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
    zero_ghost_forces(frame, *dd, bufs.reverse_plan, contract_);
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

    // Mirror of `forward_comm`: standing receives for the whole round, then
    // blocking `MPI_Send` into partners that are already listening. Rounds
    // run backwards — a shallower round re-ships ghost forces the deeper
    // round just folded in — and the round, not the swap, is the unit for
    // the same mirrored-partner reason (see `forward_comm`).
    for (int end = nswap; end > 0;) {
        const int begin = swaps_.round_begin(end);
        const int width = end - begin;
        if (bufs.requests.size() < static_cast<std::size_t>(width))
            bufs.requests.resize(static_cast<std::size_t>(width));
        int nreq = 0;
        for (int s = begin; s < end; ++s) {
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
                      MPI_BYTE, swap.sendproc, Halo::mirror_swap(begin, end, s),
                      dd->cart_comm,
                      &bufs.requests[static_cast<std::size_t>(nreq++)]);
        }
        {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_pack", -1,
                              static_cast<int>(md::StepTiming::Comm));
            for (int s = begin; s < end; ++s) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.nrecv == 0)
                    continue;
                auto& sbuf = bufs.send_force[static_cast<std::size_t>(s)];
                reverse_plan.pack_rows(swap.firstrecv, swap.nrecv, sbuf);
                MPI_Send(
                    sbuf.data(),
                    checked_mpi_byte_count(sbuf.size(), "halo reverse send"),
                    MPI_BYTE, swap.recvproc, s, dd->cart_comm);
            }
        }
        if (nreq > 0) {
            ATV_PROFILE_SCOPE(ctx, "halo/rev_wait", -1,
                              static_cast<int>(md::StepTiming::Comm));
            MPI_Waitall(nreq, bufs.requests.data(), MPI_STATUSES_IGNORE);
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

    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
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
        for (int i = 0; i < dd->n_local; ++i) {
            if (sp_x[i] < dd->sublo[0] || sp_x[i] >= dd->subhi[0]
                || sp_y[i] < dd->sublo[1] || sp_y[i] >= dd->subhi[1]
                || sp_z[i] < dd->sublo[2] || sp_z[i] >= dd->subhi[2]) {
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
    dd->n_ghost = 0;

    bufs_->resize_swaps(std::max(swaps_.size(), std::size_t{2}));
    exchange_along_dim(frame, dd, 0, payload_bytes, plan, *bufs_);
    exchange_along_dim(frame, dd, 1, payload_bytes, plan, *bufs_);
    exchange_along_dim(frame, dd, 2, payload_bytes, plan, *bufs_);

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
    // LAMMPS `CommBrick::forward_comm(Fix*)` (src/comm_brick.cpp:1078)
    // is sequential per swap, tag 0. Concurrent lo/hi of one round uses
    // the swap index as tag (same dims[d]==2 reason as the position halo).
    if (mpi_size_ <= 1)
        return;
    if (!dd_ || dd_->cart_comm == MPI_COMM_NULL || swaps_.size() == 0)
        return;
    const int nsize = packer.comm_forward();
    if (nsize <= 0)
        return;
    packer_nsize_ = std::max(packer_nsize_, nsize);
    if (!bufs_)
        bufs_ = std::make_unique<detail::FaceBufs>();
    FaceBufs& bufs = *bufs_;
    const int nswap = static_cast<int>(swaps_.size());
    bufs.resize_swaps(static_cast<std::size_t>(nswap));
    int max_atoms = 1;
    for (int s = 0; s < nswap; ++s)
        max_atoms =
            std::max({max_atoms, static_cast<int>(swaps_[s].sendlist.size()),
                      swaps_[s].nrecv});
    bufs.reserve_packer(packer_nsize_, max_atoms);

    std::vector<int> packed(static_cast<std::size_t>(nswap), 0);
    for (int begin = 0; begin < nswap;) {
        const int end = swaps_.round_end(begin);
        for (int s = begin; s < end; ++s) {
            const Swap& swap = swaps_[s];
            if (swap.is_silent())
                continue;
            const int n_send = static_cast<int>(swap.sendlist.size());
            const int pbc_flag =
                (swap.pbc[0] != 0 || swap.pbc[1] != 0 || swap.pbc[2] != 0) ? 1
                                                                           : 0;
            auto& sbuf = bufs.send_pack[static_cast<std::size_t>(s)];
            auto& rbuf = bufs.recv_pack[static_cast<std::size_t>(s)];
            sbuf.resize(static_cast<std::size_t>(std::max(n_send * nsize, 1)));
            packed[static_cast<std::size_t>(s)] = packer.pack_forward_comm(
                n_send, swap.sendlist.data(), sbuf.data(), pbc_flag, swap.pbc);
            rbuf.resize(
                static_cast<std::size_t>(std::max(swap.nrecv * nsize, 1)));
        }
        bufs.exchange_round(
            dd_->cart_comm, begin, end,
            [&](int s, auto post) {
                const Swap& swap = swaps_[s];
                if (swap.is_silent() || swap.nrecv <= 0)
                    return;
                auto& rbuf = bufs.recv_pack[static_cast<std::size_t>(s)];
                post(rbuf.data(),
                     checked_mpi_byte_count(packer_bytes(swap.nrecv, nsize),
                                            "packer recv"),
                     swap.recvproc, Halo::mirror_swap(begin, end, s));
            },
            [&](int s, auto post) {
                const Swap& swap = swaps_[s];
                const int n = packed[static_cast<std::size_t>(s)];
                if (swap.is_silent() || n <= 0)
                    return;
                auto& sbuf = bufs.send_pack[static_cast<std::size_t>(s)];
                post(sbuf.data(),
                     checked_mpi_byte_count(
                         checked_mul(static_cast<std::size_t>(n),
                                     sizeof(double), "packer send"),
                         "packer send"),
                     swap.sendproc, s);
            });
        for (int s = begin; s < end; ++s) {
            const Swap& swap = swaps_[s];
            if (swap.is_silent() || swap.nrecv <= 0)
                continue;
            packer.unpack_forward_comm(
                swap.nrecv, swap.firstrecv,
                bufs.recv_pack[static_cast<std::size_t>(s)].data());
        }
        begin = end;
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
