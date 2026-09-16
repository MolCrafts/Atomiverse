/// @file cpu/communication/halo.h
/// @brief `Swap` / `Halo` — the LAMMPS CommBrick swap table of the CPU halo.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/comm_brick.cpp` sendlist (checkout `970d4ba`, 2026-08-27).
/// Structural reimplementation of the swap-table derivation (`maxneed`,
/// `sendlist`, `firstrecv`, `sendself`) — not a verbatim dump. Public API
/// is ATV-native (`comm::Halo`).
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// A `Halo` is the ordered table of face exchanges a rank performs to
/// grow its ghost shell: derived **once per full halo rebuild** from the
/// `DomainDecomposition` + `Box` + halo cutoff, then replayed by the halo
/// components for the rest of the rebuild interval. It replaces the
/// per-step, per-face geometry re-derivation of the old cascade halo.
///
/// `Halo` is **purely local**: it performs no MPI and does not depend on
/// `ATV_HAS_MPI`. Count handshakes, `Isend`/`Irecv`, packing and unpacking
/// stay in `CommBrick` — `Halo` is to the halo what `CellList` /
/// `PairList` are to `NeighborList`.
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039 — the original swap-list forward/reverse
///     comm (`maxneed`, `sendlist`, `firstrecv`, `sendself`).
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — modern LAMMPS borders/forward/reverse.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace ATV {
namespace cpu {

class Box;
class Frame;

namespace md {
namespace mpi {
struct DomainDecomposition;
} // namespace mpi
} // namespace md

namespace comm {

/// @brief One face exchange: the rows this rank ships out of one side of its
/// subdomain, and where the ghosts coming back land.
///
/// Swaps are ordered axis-major, then round, then face: axis 0's rounds
/// `1..maxneed(0)` each contributing a `(lo, hi)` pair, then axis 1, then
/// axis 2. At one hop per axis that degenerates to
/// `(0,lo), (0,hi), (1,lo), (1,hi), (2,lo), (2,hi)` — bit-for-bit the
/// `DomainDecomposition::face_neighbors` order `{-x, +x, -y, +y, -z, +z}`.
///
/// The geometry members (`dim` … `slabhi`) are derived once by
/// `Halo::setup`; the index members (`sendlist`, `nrecv`, `firstrecv`)
/// are refilled per rebuild by `Halo::rebuild_sendlist` /
/// `Halo::commit_recv` and dropped by `Halo::invalidate`.
struct Swap {
    /// @brief Axis this swap crosses: 0 = x, 1 = y, 2 = z.
    int dim = 0;

    /// @brief Which face of `dim`: -1 = the low face, +1 = the high face.
    int sign = -1;

    /// @brief Rank the rows in `sendlist` are shipped to.
    ///
    /// `face_neighbors[2 * dim + (sign > 0)]` — the neighbour across the
    /// face this swap points at. A face with no partner (`kNoPartner` /
    /// `MPI_PROC_NULL`) is skipped; under the all-periodic brick that is
    /// only an undecomposed axis (`sendself`) or a defensive unused slot.
    int sendproc = -1;

    /// @brief Rank whose ghosts land in `[firstrecv, firstrecv + nrecv)`.
    ///
    /// The **opposite** face's neighbour,
    /// `face_neighbors[2 * dim + (sign < 0)]` — LAMMPS `CommBrick::setup`'s
    /// `sendproc = procneigh[dim][0], recvproc = procneigh[dim][1]` pairing
    /// (send west, hear back from east). That is what makes swap `s`
    /// self-matching: the table is rank-uniform and every rank walks it in
    /// the same order, so the message this rank sends on swap `s` is the one
    /// its partner unpacks on **its** swap `s`. Exactly one message per pair
    /// per swap, so every post keys on tag 0 and the **swap**, not the
    /// round, is the transport unit — including at `dims[dim] == 2`, where
    /// both faces address the same rank.
    ///
    /// The mirrored convention (`sendproc == recvproc`: this rank's band out
    /// and the partner's band across the same plane back) moves the
    /// identical atom set, but pairs swap `s` with the partner's *opposite*
    /// swap, which forces a per-swap tag and forbids draining a round one
    /// swap at a time.
    int recvproc = -1;

    /// @brief The Cartesian partner on this face **is** this rank
    /// (`dims[dim] == 1`), so this swap is a **skip, not a LAMMPS memcpy**.
    ///
    /// LAMMPS copies the band locally to create self-image ghosts. This
    /// engine must not: `CellList` marks an axis extended only when
    /// `dims[d] > 1` (`src/cpu/neighbor/cell_list.cpp:55`) and keeps
    /// applying the minimum-image convention on every non-extended axis, so
    /// a self-image ghost would be counted once as a ghost row *and* once as
    /// its own min-image — the silent double count pinned by
    /// `tests/mpi/test_halo_dual_image_ghosts.cpp`. A `sendself` swap
    /// therefore sends nothing, receives nothing, and keeps `nrecv == 0`;
    /// periodicity along that axis is carried by the pair kernel's
    /// min-image. Deliberate deviation from LAMMPS, reaffirmed by spec
    /// `comm-topology-02`: an undecomposed axis derives **zero** hops
    /// (`Halo::max_hops(d) == 0`) and keeps exactly one placeholder
    /// `(lo, hi)` pair in the table.
    bool sendself = false;

    /// @brief Integer periodic image flags, one per axis, in `{-1, 0, +1}`.
    ///
    /// Applied at pack time as `pbc[d] * box.length(d)` [Angstrom] using the
    /// **current** box, so an NPT / deform rescale cannot leave a stale
    /// metric displacement behind. Only a rank on the global periodic edge
    /// of `dim` shifts: `+1` on the low face at `coords[dim] == 0`, `-1` on
    /// the high face at `coords[dim] == dims[dim] - 1`. A `sendself` swap is
    /// all zero.
    int pbc[3] = {0, 0, 0};

    /// @brief Which hop of its axis this swap carries, 1-based.
    ///
    /// A dim with `Halo::maxneed(dim) == k` expands to `k` `(lo, hi)`
    /// pairs, `round = 1 … k`; the whole chain goes through the **same** face
    /// neighbours (`sendproc`/`recvproc` are round-independent), each round
    /// forwarding one rank further what the previous round received.
    /// Rounds are the unit the scan window advances on (`Halo::round_end`).
    int round = 1;

    /// @brief Low edge of the eligibility band along `dim` [Angstrom].
    ///
    /// **Round 1** bounds the rank's own boundary band:
    /// `sublo[dim] + halo_cutoff` bounds the high side of a low-face swap
    /// and `subhi[dim] - halo_cutoff` the low side of a high-face swap; the
    /// outward side carries a `±1e30` sentinel, far outside any physical
    /// box. The band is **inclusive on both edges** — a row qualifies for
    /// `slablo <= q <= slabhi` — matching LAMMPS `CommBrick::borders`: an
    /// extra ghost on the edge is free, a missing one is a lost pair, and a
    /// coordinate that crossed a periodic boundary can land exactly on the
    /// edge by round-off.
    ///
    /// **Rounds 2 and deeper** carry the LAMMPS `ineed >= 2` band — a
    /// low-face swap takes `(mid, sublo[dim] + halo_cutoff)` and a
    /// high-face swap `(subhi[dim] - halo_cutoff, mid)`, where `mid` is
    /// this rank's own mid-plane. *Which* rows are offered is the caller's
    /// scan window — everything the previous round of this axis delivered,
    /// both faces at once, never the owners round 1 already shipped. The
    /// **mid-plane edge is what separates the two faces inside that shared
    /// window**: ghosts a face received sit past the opposite edge of the
    /// subdomain, so they fail their own face's band and pass the other's.
    /// The far edge clips **depth**: a forwarded row only matters to the
    /// partner while it is inside the partner's shell, whose far edge is
    /// `sublo[dim] + halo_cutoff` for the low face — the chain is still
    /// complete because the next hop clips at a strictly lower bound
    /// again. Leaving it open forwards the whole slab instead of the part
    /// still in reach, and the surplus compounds once per hop.
    double slablo = 0.0;

    /// @brief High edge of the eligibility band along `dim` [Angstrom].
    /// See `slablo`.
    double slabhi = 0.0;

    /// @brief Frame row indices shipped by this swap, ascending.
    /// Owners in the first axis stage, owners + earlier-axis ghosts later —
    /// the scan bound is the caller's (see `Halo::rebuild_sendlist`).
    std::vector<int> sendlist;

    /// @brief Ghost rows received by this swap in the current rebuild.
    int nrecv = 0;

    /// @brief Frame row index of this swap's first received ghost.
    /// `firstrecv = n_local + sum_{j < iswap} nrecv[j]` — the prefix sum
    /// maintained by `Halo::commit_recv`.
    int firstrecv = 0;

    /// @brief `sendproc` value meaning "this face has no partner".
    ///
    /// `Halo` is MPI-free by contract, so it cannot name `MPI_PROC_NULL`.
    /// The transport that does (`comm_brick.cpp`) static-asserts the two
    /// are equal, so this stays a compile-time-checked mirror rather than
    /// an assumption.
    static constexpr int kNoPartner = -1;

    /// @brief True when this swap moves no data at all.
    ///
    /// Either the face has no partner (`kNoPartner`) or the partner
    /// is this rank itself (`sendself`, an undecomposed axis). The latter
    /// is a **skip, not a LAMMPS local memcpy** — see `sendself` for why a
    /// self-image ghost would double-count against the pair kernel's
    /// minimum image. Testing the send side alone is complete: on an
    /// undecomposed axis both face neighbours are this rank, and under the
    /// all-periodic brick a decomposed axis has a partner on both faces.
    /// @return True if the transport side must ignore this swap.
    bool is_silent() const noexcept {
        return sendself || sendproc == kNoPartner;
    }

    /// @brief Periodic image displacement this swap applies at pack time.
    ///
    /// `pbc` is an integer image flag; the metric displacement is only
    /// formed here, against the **current** box, so an NPT / deform
    /// rescale cannot ship a stale shift.
    /// @param box Current simulation box.
    /// @param out Displacement per axis [Angstrom].
    void pbc_shift(const Box& box, double out[3]) const;
};

/// @brief The ordered swap table of one rank's halo.
///
/// Three primitives, in the order a full rebuild uses them:
///   1. `setup` — derive the per-axis hop counts (`derive_hops`) and expand
///      them into the swap geometry from the current decomposition and box
///      (called at every full rebuild, so both the round count and the slab
///      bands follow an NPT / deform box rescale automatically),
///   2. `rebuild_sendlist` — refill one swap's send list from the frame,
///   3. `commit_recv` — record what came back and advance the ghost-slot
///      prefix sum.
///
/// `invalidate()` drops (2) and (3) while keeping (1), which is the
/// post-sort / post-migration failure mode: the geometry is still valid but
/// every stored row index is not.
///
/// Construction, rebuild cadence and error handling mirror
/// `NeighborList` + `CellList`: a default-constructed list is empty and
/// `setup` is what makes it usable; bad input throws `std::runtime_error`
/// with a how-to rather than degrading silently.
class Halo {
  public:
    /// @brief Hops one rank needs in each send direction, per axis.
    ///
    /// `minus[d]` counts the hops toward *decreasing* `coords[d]` (the chain
    /// of partners `coords[d] - 1, coords[d] - 2, …`), `plus[d]` toward
    /// increasing coords. Both are 0 on an undecomposed axis (`dims[d] <= 1`)
    /// — that axis exchanges nothing.
    ///
    /// The two directions differ only under a non-uniform partition
    /// (`DomainDecomposition::cuts`), where the partner slab widths along the
    /// two chains differ; a uniform brick is symmetric.
    struct HopNeed {
        /// @brief Hops toward decreasing `coords[d]`, one entry per axis.
        int minus[3] = {0, 0, 0};
        /// @brief Hops toward increasing `coords[d]`, one entry per axis.
        int plus[3] = {0, 0, 0};
    };

    /// @brief Derive one rank's per-direction hop counts. Pure, no MPI.
    ///
    /// LAMMPS `Comm::setup` semantics, per axis `d` with `dims[d] > 1`:
    ///
    ///   * **uniform** partition (`dd.cuts[d]` empty), slab width
    ///     `w_d = L_d / dims_d`: the closed form
    ///     `hops = ceil(shell_width * dims_d / L_d)`, identical in both
    ///     directions. A shell exactly as wide as one slab stays a *single*
    ///     hop.
    ///   * **non-uniform** partition (`dd.cuts[d]` non-empty, RCB /
    ///     `Rebalance`): walk the send direction rank by rank with periodic
    ///     wrap, accumulating the **partner** slab widths, and stop at the
    ///     first partner whose cumulative width covers `shell_width`. It is
    ///     the partner chain, not this rank's own width, that decides whether
    ///     the shell can be filled.
    ///
    /// An axis with `dims[d] <= 1` yields 0 in both directions and is
    /// **exempt from every width check**: it produces no ghosts at all and
    /// its periodicity is the pair kernel's minimum image, so an EAM-style
    /// `2 * cutoff` shell on a thin undecomposed axis stays legal.
    ///
    /// Every decomposed axis is periodic. There is no non-periodic hop
    /// truncation: a shell that would wrap onto this rank's own image is
    /// refused (minimum-image bound). See `DomainDecomposition`.
    ///
    /// @param dd This rank's decomposition: `dims`, `coords`, and
    ///        the optional fractional `cuts`.
    /// @param box Current simulation box; `length(d)` [Angstrom] is the axis
    ///        length the slab widths are measured against.
    /// @param shell_width Ghost shell width [Angstrom],
    ///        `max(pair cutoff, bonded extension)`.
    /// @return Hops needed in each direction of each axis.
    /// @throws std::runtime_error if `shell_width` is negative or NaN, if an
    ///         axis length is non-positive, if `dd.cuts[d]` is present but
    ///         not `dims[d] + 1` long, or if a **periodic** decomposed axis
    ///         needs more hops than `dims[d] - 1` — the minimum-image bound
    ///         `shell_width <= L_d - w_self`, beyond which an atom would
    ///         interact with its own periodic image.
    static HopNeed derive_hops(const md::mpi::DomainDecomposition& dd,
                               const Box& box, double shell_width);

    /// @brief `derive_hops` with a coordinate override — the sweep entry
    /// used by `derive_uniform_rounds`, so scanning another rank's
    /// position on an axis never copies the decomposition (its tag→local
    /// map makes that copy O(atoms)).
    /// @param coords Per-axis coordinates to derive for.
    static HopNeed derive_hops(const md::mpi::DomainDecomposition& dd,
                               const Box& box, double shell_width,
                               const int (&coords)[3]);

    /// @brief Rank-uniform per-axis round counts. Pure, no MPI.
    ///
    /// This is the round vector that used to be an `MPI_Allreduce(MAX)` in
    /// `setup_swaps`. For each axis `d` it copies `dd`, walks
    /// `coords[d] = 0 … dims[d)`, calls `derive_hops`, and keeps
    /// `max(minus[d], plus[d])`. Complexity `O(Σ dims[d])`, independent of
    /// the rank count. Hop math is not rewritten: negative/NaN shells,
    /// cut-table errors, and the minimum-image bound all come from
    /// `derive_hops`.
    ///
    /// Every input is rank-consistent (`dims` / `cuts` / `box` /
    /// `shell_width`), so every rank produces the same vector. A
    /// minimum-image refusal throws inside the local scan — every rank sees
    /// every coordinate — which is strictly better than a generic "another
    /// rank refused" Allreduce flag.
    ///
    /// @param dd Any rank's decomposition: `dims` and the
    ///        optional fractional `cuts`. `coords` is overwritten per scan
    ///        point; `sublo`/`subhi` [Angstrom] appear only in a refusal
    ///        message.
    /// @param box Current simulation box; `length(d)` [Angstrom] is the axis
    ///        length the slab widths are measured against.
    /// @param shell_width Ghost shell width [Angstrom],
    ///        `max(pair cutoff, bonded extension)`.
    /// @return Hop count per axis — the rank-uniform round vector a
    ///         transport caller passes to the five-argument `setup`.
    /// @throws std::runtime_error under the same conditions as `derive_hops`.
    static std::array<int, 3>
    derive_uniform_rounds(const md::mpi::DomainDecomposition& dd,
                          const Box& box, double shell_width);

    /// @brief Derive the swap table for this rank from its decomposition.
    ///
    /// Calls `derive_hops` and expands the result: axis `d` contributes
    /// `maxneed(d)` `(lo, hi)` swap pairs, in round order `1 … maxneed(d)`,
    /// so `size() == 2 * sum_d maxneed(d)`. Every round of an axis uses that
    /// axis' two face neighbours — the chain is a **forwarding** chain, each
    /// round passing what the previous one received one rank further, never
    /// a direct message to a rank two hops away.
    ///
    /// `maxneed(d) = max(max_hops(d), 1)` keeps one placeholder pair on an
    /// axis that needs no hop at all (undecomposed, or a zero-width shell);
    /// those swaps are `sendself` and move nothing.
    ///
    /// This runs at the head of **every full rebuild**, so it is also the
    /// re-derivation hook: an NPT / deform box shrink narrows the slabs and
    /// raises the round count in place instead of throwing or silently
    /// truncating the ghost shell. It is idempotent for identical arguments.
    ///
    /// The derivation is **per rank and serial** — `Halo` performs no
    /// MPI. What is stored here is this rank's own maximum over the two send
    /// directions; a transport caller that needs the round count to be
    /// rank-uniform obtains it from `derive_uniform_rounds` (the MPI-free
    /// local sweep that replaced the spec's `MPI_Allreduce(MAX)`) and then
    /// calls the five-argument overload.
    ///
    /// Rebuilds the table in place, preserving the send lists' heap
    /// capacity. On a throw the previous table is left untouched.
    ///
    /// @param dd This rank's decomposition: `dims`, `coords`,
    ///        `cuts`, `sublo/subhi` [Angstrom] and `face_neighbors`.
    /// @param box Current simulation box; edge lengths [Angstrom] drive the
    ///        slab widths and are reported in the refusal message. Image
    ///        displacements are *not* baked in here — `Swap::pbc` stays an
    ///        integer flag.
    /// @param halo_cutoff Ghost shell width [Angstrom],
    ///        `max(cutoff, bonded_extension)`; readable back as
    ///        `shell_width()`.
    /// @param my_rank This rank's id in the Cartesian communicator; a face
    ///        neighbour equal to it marks the swap `sendself`.
    /// @throws std::runtime_error if `halo_cutoff` is negative or NaN, if a
    ///         decomposed axis has a non-positive slab width, or if the
    ///         minimum-image bound of `derive_hops` is exceeded.
    void setup(const md::mpi::DomainDecomposition& dd, const Box& box,
               double halo_cutoff, int my_rank);

    /// @brief Derive the swap table against an externally reconciled round
    /// count.
    ///
    /// Same expansion as the four-argument `setup`, except that axis `d`
    /// contributes `max(locally derived hops, rounds[d])` `(lo, hi)` pairs
    /// (clamped to `dims[d] - 1`, beyond which a further round reaches no new
    /// rank, and to 0 on an undecomposed axis).
    ///
    /// This is the entry point of a **transport** caller. Rounds are matched
    /// `Sendrecv` pairs: a rank that stops one round earlier than its partner
    /// leaves that partner posting a receive nobody answers. The locally
    /// derived count is not automatically rank-uniform — a non-uniform (RCB)
    /// partition has different partner-slab chains per rank — so the
    /// caller obtains a uniform vector from `derive_uniform_rounds` and
    /// hands it back here. The extra rounds are not wasted: they carry
    /// exactly what the deepest rank on that axis still needs.
    ///
    /// `Halo` performs no MPI; the sweep is a pure local function.
    ///
    /// @param dd This rank's decomposition, as for the four-argument form.
    /// @param box Current simulation box; edge lengths [Angstrom].
    /// @param halo_cutoff Ghost shell width [Angstrom].
    /// @param my_rank This rank's id in the Cartesian communicator.
    /// @param rounds Reconciled hop count per axis, one entry per axis;
    ///        negative entries are read as 0.
    /// @throws std::runtime_error under the same conditions as the
    ///         four-argument form — the minimum-image bound is re-checked
    ///         here, so an imposed round count cannot buy past it.
    void setup(const md::mpi::DomainDecomposition& dd, const Box& box,
               double halo_cutoff, int my_rank, const int rounds[3]);

    /// @brief Derive the table with a **rank-uniform** round count.
    ///
    /// The entry point a transport caller wants. `derive_uniform_rounds`
    /// sweeps every coordinate of each axis, so every rank gets the same
    /// vector by construction; the four-argument `setup` derives only
    /// *this* rank's hops, which is not rank-uniform and would leave a
    /// partner posting a receive nobody answers. Pure and local — no MPI.
    /// @param dd This rank's decomposition (current `sublo/subhi` [Å]).
    /// @param box Current simulation box.
    /// @param halo_cutoff Ghost shell width [Angstrom].
    /// @param my_rank This rank's id in the Cartesian communicator.
    /// @throws std::runtime_error on the minimum-image refusal.
    void setup_uniform(const md::mpi::DomainDecomposition& dd, const Box& box,
                       double halo_cutoff, int my_rank);

    /// @brief Refill one swap's send list from the current positions.
    ///
    /// Scans rows `[0, sources_end)` of the frame's `x`/`y`/`z` column for
    /// the swap's `dim` and keeps every row inside the band
    /// `slablo <= q <= slabhi`. The list is cleared first (refill, not
    /// append); its heap capacity is preserved.
    ///
    /// Cascading is the caller's to express through `sources_end`: axis 0
    /// scans owners, axis 1 scans owners + axis-0 ghosts, and so on. The
    /// swap table itself knows nothing about staging.
    ///
    /// @param iswap Swap index in `[0, size())`.
    /// @param frame Frame holding the positions [Angstrom] to scan.
    /// @param sources_end Exclusive upper row bound of the scan.
    /// @throws std::runtime_error if `iswap` or `sources_end` is out of
    ///         range.
    void rebuild_sendlist(int iswap, const Frame& frame, int sources_end);

    /// @brief Refill one swap's send list from a bounded row window.
    ///
    /// The same band predicate over rows `[first, last)` instead of
    /// `[0, last)`. The lower bound is what a **forwarding round** needs:
    /// round `r > 1` of an axis offers exactly the ghost rows round `r - 1`
    /// delivered — one contiguous slot range covering both of that round's
    /// faces, LAMMPS `CommBrick::borders`' `[nfirst, nlast)` — and must not
    /// re-ship the owners round 1 already sent, which would land the same
    /// atom twice at the same periodic image on the partner. The window
    /// picks *which* rows are offered; the `round > 1` band then both clips
    /// how deep they may be and sorts the shared window between the two
    /// faces at the mid-plane (see `Swap::slablo`).
    ///
    /// @param iswap Swap index in `[0, size())`.
    /// @param frame Frame holding the positions [Angstrom] to scan.
    /// @param first Inclusive lower row bound of the scan.
    /// @param last Exclusive upper row bound of the scan.
    /// @throws std::runtime_error if `iswap` is out of range, if the window is
    ///         inverted (`first > last`), or if it leaves the frame's
    ///         allocated rows.
    void rebuild_sendlist(int iswap, const Frame& frame, int first, int last);

    /// @brief Record one swap's received ghost count and its first slot.
    ///
    /// Stores `nrecv` and `firstrecv = next_slot`, making the ghost-slot
    /// prefix sum an invariant of this type instead of a count hand-rolled
    /// by each consumer. Drive it in swap order starting from `n_local`.
    ///
    /// @param iswap Swap index in `[0, size())`.
    /// @param nrecv Ghost rows received by this swap (`>= 0`).
    /// @param next_slot Frame row index this swap's ghosts start at.
    /// @return The next free ghost slot, `next_slot + nrecv`.
    /// @throws std::runtime_error if any argument is out of range.
    int commit_recv(int iswap, int nrecv, int next_slot);

    /// @brief Total ghost rows the whole table received.
    /// @return `sum_i nrecv[i]`; must equal `dd.n_ghost` after a rebuild.
    int total_recv() const;

    /// @brief Hops the ghost shell needs along one axis.
    ///
    /// The reduced (`max`) of the two send directions of `derive_hops`, as
    /// derived by the last `setup`: 0 on an undecomposed axis, 0 before the
    /// first `setup`.
    ///
    /// @param d Axis index in `[0, 3)`.
    /// @return Hop count on axis `d`.
    /// @throws std::runtime_error if `d` is out of range.
    int max_hops(int d) const;

    /// @brief Swap rounds the hop derivation asks for on one axis.
    /// @param d Axis index in `[0, 3)`.
    /// @return `2 * max_hops(d)` — one `(lo, hi)` pair per hop, 0 on an
    ///         undecomposed axis.
    /// @throws std::runtime_error if `d` is out of range.
    int nswap(int d) const;

    /// @brief Swap **pairs** the table actually holds for one axis.
    ///
    /// `max(max_hops(d), 1)`: the table keeps one `sendself` placeholder pair
    /// on an axis that needs no hop, so `size() == 2 * sum_d maxneed(d)` and
    /// every axis owns a contiguous run of swaps. Use `nswap(d)` for the
    /// number of rounds that actually transmit.
    ///
    /// @param d Axis index in `[0, 3)`.
    /// @return Swap pairs on axis `d`; 0 before the first `setup`.
    /// @throws std::runtime_error if `d` is out of range.
    int maxneed(int d) const;

    /// @brief Ghost shell width the current table was derived against.
    /// @return The `halo_cutoff` [Angstrom] passed to the last `setup`;
    ///         0 before the first one.
    double shell_width() const noexcept { return shell_width_; }

    /// @brief One past the last swap of the axis stage starting at `begin`.
    ///
    /// The swaps of one axis are contiguous in the table (`setup` expands
    /// axis-major, then round, then face), and a stage is exactly one such
    /// run — the unit whose round 1 shares a scan bound (owners plus the
    /// ghosts every earlier axis delivered).
    /// @param begin First swap of the stage.
    /// @return One past the stage's last swap.
    int stage_end(int begin) const;

    /// @brief One past the last swap of the forwarding round starting at
    /// `begin`: the `(lo, hi)` pair of one hop of one axis.
    ///
    /// The round is the **forwarding** unit, not the transport unit: its two
    /// swaps share one scan window (everything the previous round of this
    /// axis delivered, LAMMPS's `nfirst`/`nlast`), and consecutive rounds
    /// are strictly serialized because round `r` ships what round `r - 1`
    /// received. Transport is per swap — see `Swap::recvproc`.
    /// @param begin First swap of the round.
    /// @return One past the round's last swap.
    int round_end(int begin) const;

    /// @brief Drop every stored row index, keep the swap geometry.
    ///
    /// The failure mode after an atom sort or a migration: `dim`, `sign`,
    /// `sendproc`/`recvproc`, `sendself`, `pbc` and the slab bands still
    /// describe this rank correctly, but every `sendlist` entry, `nrecv`
    /// and `firstrecv` refers to rows that have moved. Clears those and
    /// leaves `size()` unchanged; send-list heap capacity is preserved.
    void invalidate();

    /// @brief Number of swaps, `2 * sum_d maxneed_d`.
    /// @return Swap count; 0 before the first `setup`.
    std::size_t size() const noexcept { return swaps_.size(); }

    /// @brief Access one swap. Caller iterates `[0, size())`; unchecked,
    /// like `std::vector::operator[]`.
    /// @param iswap Swap index in `[0, size())`.
    /// @return The swap's geometry and current indices.
    const Swap& operator[](int iswap) const {
        return swaps_[static_cast<std::size_t>(iswap)];
    }

  private:
    /// @brief Expand a per-axis round count into the swap table.
    ///
    /// The shared tail of both `setup` entry points: validates this rank's
    /// slab widths, records the round counts and rewrites `swaps_` in axis /
    /// round / face order, preserving send-list heap capacity.
    ///
    /// @param dd This rank's decomposition (`sublo/subhi` [Angstrom],
    ///        `coords`, `dims`, `face_neighbors`).
    /// @param halo_cutoff Ghost shell width [Angstrom]; the round-1 band edge.
    /// @param my_rank This rank's id in the Cartesian communicator.
    /// @param rounds Hops to expand per axis (already reconciled and clamped).
    void expand(const md::mpi::DomainDecomposition& dd, double halo_cutoff,
                int my_rank, const int rounds[3]);

    std::vector<Swap> swaps_;
    /// Reduced hops per axis (`max` of the two send directions); 0 = none.
    int hops_[3] = {0, 0, 0};
    /// Swap pairs per axis, `max(hops_, 1)` after `setup`, 0 before it.
    int maxneed_[3] = {0, 0, 0};
    /// Ghost shell width [Angstrom] the table was derived against.
    double shell_width_ = 0.0;
};

} // namespace comm
} // namespace cpu
} // namespace ATV
