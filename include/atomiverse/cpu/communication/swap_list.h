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
    /// min-image. Deliberate deviation from LAMMPS, up for review when
    /// multi-hop swaps land (spec `comm-topology-02`).
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

    /// @brief Low edge of the eligibility band along `dim` [Angstrom].
    ///
    /// `sublo[dim] + halo_cutoff` bounds the high side of a low-face swap
    /// and `subhi[dim] - halo_cutoff` the low side of a high-face swap; the
    /// outward side carries a `±1e30` sentinel, far outside any physical
    /// box. The band is **strict on both edges**: a row qualifies for
    /// `slablo < q < slabhi`. That reproduces the outgoing predicate
    /// (`q - sublo < cutoff` / `subhi - q < cutoff`) and is deliberately
    /// *not* LAMMPS' `<=`, which would drift the ghost set by the atoms
    /// sitting exactly on the band edge.
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
///   1. `setup` — derive `maxneed` and the swap geometry from the current
///      decomposition and box (called at every full rebuild, so slab bands
///      follow an NPT / deform box rescale automatically),
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
    /// @brief Derive the swap table for this rank from its decomposition.
    ///
    /// Per axis, the number of hops the ghost shell needs is
    /// `need_d = ceil(r_halo / sub_d)` where `sub_d = subhi[d] - sublo[d]`
    /// is this rank's own slab width [Angstrom]. For the uniform brick
    /// decomposition `sub_d == L_d / dims_d`, so this is exactly the
    /// `ceil(r_halo * dims_d / L_d)` of the spec; using the rank's own slab
    /// keeps the refusal exact under a non-uniform partition
    /// (`DomainDecomposition::cuts`), where the two differ.
    ///
    /// **Undecomposed axes are exempt.** An axis with `dims[d] <= 1` is
    /// forced to `need_d = 1` and both its swaps to `sendself` without any
    /// width check, mirroring the `if (dims[d] <= 1) continue` short circuit
    /// of the outgoing guard: such an axis produces no ghosts at all, and a
    /// halo shell wider than the whole (thin) axis is legal there because
    /// periodicity is the pair kernel's min-image. Refusing it would break
    /// runs that are correct today, e.g. an EAM-style `2 * cutoff` shell on
    /// a thin undecomposed axis.
    ///
    /// A **decomposed** axis is constrained to `need_d == 1`: the halo
    /// exchanges with immediate neighbours only, so a deeper shell could
    /// never be filled. `r_halo == sub_d` stays a single hop (the `ceil`
    /// convention), matching the outgoing guard, which refused only
    /// `r_halo > sub_d`. Multi-hop chains are spec `comm-topology-02`.
    ///
    /// Rebuilds the table in place, preserving the send lists' heap
    /// capacity. On a throw the previous table is left untouched.
    ///
    /// @param dd This rank's decomposition: `dims`, `coords`, `sublo/subhi`
    ///        [Angstrom] and `face_neighbors`.
    /// @param box Current simulation box; edge lengths [Angstrom] are
    ///        reported in the refusal message. Image displacements are
    ///        *not* baked in here — `Swap::pbc` stays an integer flag.
    /// @param halo_cutoff Ghost shell width [Angstrom],
    ///        `max(cutoff, bonded_extension)`.
    /// @param my_rank This rank's id in the Cartesian communicator; a face
    ///        neighbour equal to it marks the swap `sendself`.
    /// @throws std::runtime_error if `halo_cutoff` is negative or NaN, if a
    ///         decomposed axis has a non-positive slab width, or if a
    ///         decomposed axis needs more than one hop.
    void setup(const md::mpi::DomainDecomposition& dd, const Box& box,
               double halo_cutoff, int my_rank);

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
    /// @param d Axis index in `[0, 3)`.
    /// @return `need_d`, always 1 at this stage (see `setup`).
    /// @throws std::runtime_error if `d` is out of range.
    int maxneed(int d) const;

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
    std::vector<Swap> swaps_;
    int maxneed_[3] = {0, 0, 0};
};

} // namespace comm
} // namespace cpu
} // namespace ATV
