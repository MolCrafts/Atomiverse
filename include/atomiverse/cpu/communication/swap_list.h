/// @file cpu/communication/swap_list.h
/// @brief `Swap` / `SwapList` — the LAMMPS CommBrick swap table of the CPU halo.
///
/// A `SwapList` is the ordered table of face exchanges a rank performs to
/// grow its ghost shell: derived **once per full halo rebuild** from the
/// `DomainDecomposition` + `Box` + halo cutoff, then replayed by the halo
/// components for the rest of the rebuild interval. It replaces the
/// per-step, per-face geometry re-derivation of the old cascade halo.
///
/// `SwapList` is **purely local**: it performs no MPI and does not depend on
/// `ATV_HAS_MPI`. Count handshakes, `Isend`/`Irecv`, packing and unpacking
/// stay in `HaloSend` / `HaloRecv` / `HaloRefresh` — `SwapList` is to the
/// halo what `CellList` / `PairList` are to `NeighborList`.
///
/// References:
///   * S. Plimpton, J. Comput. Phys. **117**, 1 (1995),
///     DOI 10.1006/jcph.1995.1039 — the original swap-list forward/reverse
///     comm (`maxneed`, `sendlist`, `firstrecv`, `sendself`).
///   * A. P. Thompson et al., Comput. Phys. Commun. **271**, 108171 (2022),
///     DOI 10.1016/j.cpc.2021.108171 — modern LAMMPS borders/forward/reverse.
#pragma once

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
/// Swaps are ordered `(0,lo), (0,hi), (1,lo), (1,hi), (2,lo), (2,hi)` inside
/// `SwapList`, bit-for-bit the `DomainDecomposition::face_neighbors` face
/// order `{-x, +x, -y, +y, -z, +z}`.
///
/// The geometry members (`dim` … `slabhi`) are derived once by
/// `SwapList::setup`; the index members (`sendlist`, `nrecv`, `firstrecv`)
/// are refilled per rebuild by `SwapList::rebuild_sendlist` /
/// `SwapList::commit_recv` and dropped by `SwapList::invalidate`.
struct Swap {
    /// @brief Axis this swap crosses: 0 = x, 1 = y, 2 = z.
    int dim = 0;

    /// @brief Which face of `dim`: -1 = the low face, +1 = the high face.
    int sign = -1;

    /// @brief Rank the rows in `sendlist` are shipped to.
    ///
    /// This is `face_neighbors[2 * dim + (sign > 0)]`: the halo exchanges
    /// symmetrically across a face, so `sendproc == recvproc` — this rank's
    /// boundary band goes out, the partner's band across the same plane
    /// comes back. (LAMMPS instead pairs a send on one face with a receive
    /// from the opposite face; the two conventions move the identical atom
    /// set, because the partner's opposite face points back here.) A face
    /// with no partner (non-periodic axis) keeps whatever `face_neighbors`
    /// holds — `MPI_PROC_NULL` — and the transport side skips it.
    int sendproc = -1;

    /// @brief Rank whose ghosts land in `[firstrecv, firstrecv + nrecv)`.
    /// Equal to `sendproc`; see there.
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
    /// (`SwapList::max_hops(d) == 0`) and keeps exactly one placeholder
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
    /// A dim with `SwapList::maxneed(dim) == k` expands to `k` `(lo, hi)`
    /// pairs, `round = 1 … k`; the whole chain goes through the **same** face
    /// neighbours (`sendproc`/`recvproc` are round-independent), each round
    /// forwarding one rank further what the previous round received.
    int round = 1;

    /// @brief Low edge of the eligibility band along `dim` [Angstrom].
    ///
    /// **Round 1** bounds the rank's own boundary band:
    /// `sublo[dim] + halo_cutoff` bounds the high side of a low-face swap
    /// and `subhi[dim] - halo_cutoff` the low side of a high-face swap; the
    /// outward side carries a `±1e30` sentinel, far outside any physical
    /// box. The band is **strict on both edges**: a row qualifies for
    /// `slablo < q < slabhi`. That reproduces the outgoing predicate
    /// (`q - sublo < cutoff` / `subhi - q < cutoff`) and is deliberately
    /// *not* LAMMPS' `<=`, which would drift the ghost set by the atoms
    /// sitting exactly on the band edge.
    ///
    /// **Rounds 2 and deeper** carry the *open* band `(-1e30, +1e30)`, the
    /// LAMMPS `ineed >= 2` convention: a forwarding round ships everything
    /// the caller offers it, and *which* rows those are — the ghosts the
    /// previous round received, not the owners already shipped by round 1 —
    /// is the caller's scan window (`rebuild_sendlist`'s `sources_end`, plus
    /// whatever lower bound the forward pass keeps). The geometry does not
    /// re-filter them: after one hop the rows sit in the *partner's* band,
    /// which this rank's own `sublo/subhi` no longer describes.
    double slablo = 0.0;

    /// @brief High edge of the eligibility band along `dim` [Angstrom].
    /// See `slablo`.
    double slabhi = 0.0;

    /// @brief Frame row indices shipped by this swap, ascending.
    /// Owners in the first axis stage, owners + earlier-axis ghosts later —
    /// the scan bound is the caller's (see `SwapList::rebuild_sendlist`).
    std::vector<int> sendlist;

    /// @brief Ghost rows received by this swap in the current rebuild.
    int nrecv = 0;

    /// @brief Frame row index of this swap's first received ghost.
    /// `firstrecv = n_local + sum_{j < iswap} nrecv[j]` — the prefix sum
    /// maintained by `SwapList::commit_recv`.
    int firstrecv = 0;
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
class SwapList {
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
    /// On a **non-periodic** axis (`dd.periods[d] == 0`) the chain simply
    /// ends at the global boundary: the count is truncated to the partners
    /// that exist in that direction, never refused.
    ///
    /// @param dd This rank's decomposition: `dims`, `coords`, `periods` and
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
    /// The derivation is **per rank and serial** — `SwapList` performs no
    /// MPI. What is stored here is this rank's own maximum over the two send
    /// directions; a transport caller that needs the round count to be
    /// rank-uniform (the `MPI_Allreduce(MAX)` of the spec, required whenever
    /// the partition is non-uniform or the axis non-periodic) reduces it and
    /// calls the five-argument overload instead.
    ///
    /// Rebuilds the table in place, preserving the send lists' heap
    /// capacity. On a throw the previous table is left untouched.
    ///
    /// @param dd This rank's decomposition: `dims`, `coords`, `periods`,
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
    /// partition has different partner-slab chains per rank, and a
    /// non-periodic axis truncates its chain at the global boundary — so the
    /// caller reduces it (`MPI_Allreduce(MAX)`, the spec's reconciliation)
    /// and hands the result back here. The extra rounds are not wasted: they
    /// carry exactly what the deepest rank on that axis still needs.
    ///
    /// `SwapList` performs no MPI; the reduction stays with the caller
    /// (`HaloSend`).
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

    /// @brief Refill one swap's send list from the current positions.
    ///
    /// Scans rows `[0, sources_end)` of the frame's `x`/`y`/`z` column for
    /// the swap's `dim` and keeps every row strictly inside the band
    /// `slablo < q < slabhi`. The list is cleared first (refill, not
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
    /// round `r > 1` of a face ships exactly the ghosts round `r - 1`
    /// delivered on the *opposite* face of the same axis — one contiguous
    /// slot range, `[firstrecv, firstrecv + nrecv)` of that swap — and must
    /// not re-ship the owners round 1 already sent, which would land the same
    /// atom twice at the same periodic image on the partner. Those rows carry
    /// the open band of a `round > 1` swap, so the window alone selects them.
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
