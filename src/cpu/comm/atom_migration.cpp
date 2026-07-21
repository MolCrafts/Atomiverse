/// @file src/cpu/comm/atom_migration.cpp
/// @brief `Migrate` -- MPI_Alltoallv owner migration implementation.
///
/// For each owner whose post-integration position lies outside this rank's
/// spatial slab, the atom (and all its per-atom fields) is shipped to the
/// rank whose slab contains the new position. Implemented as a single
/// `MPI_Alltoallv` over byte buffers so every world rank participates and
/// either packs/unpacks zero bytes or some multiple of the communicator-agreed
/// per-atom payload size (`atom_payload_bytes`).
///
/// Bonded-tuple migration is **not** handled here; that is tracked under
/// See msu-pe-validation.md (MSU-PE multi-rank validation).

#include "atomiverse/cpu/comm/atom_migration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "atomiverse/cpu/comm/atom_pack.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/comm/schema_mpi.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace comm {

#ifdef ATV_HAS_MPI

namespace {

// Per-atom payload layout: worst-case fixed-size dump from
// `comm::pack_exchange`. Carries all SoA fields plus per-atom topology
// (num_bond + bond_atom + bond_type + angles + dihedrals + nspecial +
// special). Atoms with fewer bonds than the max simply leave the
// trailing slots zero-padded — receiver reads `num_bond` first and
// only consumes that many partner entries.
//
// The slot size follows the RUNTIME topology strides (PE: 39 doubles vs the
// old compile-time worst case of 84), reduced MAX across the communicator so
// every rank counts buffer atoms with the same divisor.
std::size_t atom_payload_bytes(const cpu::Frame& frame, const PackPlan& plan,
                               MPI_Comm communicator) {
    require_collective_schema(plan, communicator, "atom_migration");
    const std::uint64_t local = max_exchange_bytes(frame, plan);
    std::uint64_t global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_MAX, communicator);
    if (global > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "atom_migration: one native row exceeds MPI datatype count");
    return static_cast<std::size_t>(global);
}

inline int rank_along_dim(double q, double L, int dims_d) {
    if (dims_d <= 1)
        return 0;
    const int r = static_cast<int>(q * static_cast<double>(dims_d) / L);
    return std::max(0, std::min(dims_d - 1, r));
}

// Locate the slab index `k` (in [0, dims_d)) whose [cuts[k], cuts[k+1])
// contains the box-length fraction `frac` (already wrapped into [0, 1)).
// `cuts` has size `dims_d + 1` with `cuts.front() == 0` and `cuts.back() == 1`.
inline int slab_along_dim_from_cuts(double frac,
                                    const std::vector<double>& cuts,
                                    int dims_d) {
    if (dims_d <= 1)
        return 0;
    // upper_bound gives the first cuts[i] strictly greater than frac;
    // its predecessor is the slab index. Clamp against FP rounding at the
    // top edge.
    auto it = std::upper_bound(cuts.begin(), cuts.end(), frac);
    int k = static_cast<int>(std::distance(cuts.begin(), it)) - 1;
    if (k < 0)
        k = 0;
    if (k > dims_d - 1)
        k = dims_d - 1;
    return k;
}

// Compute the rank that owns position (x, y, z) under the current DD.
//
// When `dd.cuts[d]` is empty along axis d, the uniform `coord = floor(q*dims/L)`
// formula applies. When non-empty (set by `Rebalance`), the destination is
// derived from the box-length fraction via binary search through the cut
// planes -- which guarantees the same rank assignment that
// `md::mpi::DomainDecomposition::compute_sublo_subhi` would produce on the destination
// side.
inline int dest_rank_for(const md::mpi::DomainDecomposition& dd, double x,
                         double y, double z, const cpu::Box& box) {
    auto axis_coord = [&](int d, double q) {
        const double L = box.length(d);
        const double wrapped = box.wrap_axis(q, d);
        if (static_cast<int>(dd.cuts[d].size()) == dd.dims[d] + 1) {
            const double frac = (L > 0.0) ? (wrapped / L) : 0.0;
            return slab_along_dim_from_cuts(frac, dd.cuts[d], dd.dims[d]);
        }
        return rank_along_dim(wrapped, L, dd.dims[d]);
    };
    const int cx = axis_coord(0, x);
    const int cy = axis_coord(1, y);
    const int cz = axis_coord(2, z);
    return dd.rank_of(cx, cy, cz);
}

// Pack one atom (full SoA + per-atom topology) into `buf` at `offset`.
// Delegates to `comm::pack_exchange` for the payload layout; pads with
// zeros up to the communicator-agreed `payload_bytes` so every atom
// occupies the same byte count on the wire (atom_migration's
// `MPI_Alltoallv` needs uniform per-atom strides).
std::size_t pack_atom(std::vector<std::byte>& buf, std::size_t offset,
                      const Frame& frame, const PackPlan& plan, int i,
                      std::size_t payload_bytes) {
    std::byte* dst = buf.data() + offset;
    // Zero-pad the slot first so the trailing topology entries don't
    // carry stale data.
    std::memset(dst, 0, payload_bytes);
    (void)comm::pack_exchange(frame, plan, i, dst, payload_bytes);
    return offset + payload_bytes;
}

std::size_t unpack_atom(const std::vector<std::byte>& buf, std::size_t offset,
                        Frame& frame, PackPlan& plan, int slot,
                        std::size_t payload_bytes) {
    (void)comm::unpack_exchange(frame, plan, slot, buf.data() + offset,
                                payload_bytes);
    return offset + payload_bytes;
}

// Append owned atom `i`'s fixed-stride payload to the back of `buf`.
inline void pack_to(std::vector<std::byte>& buf, const Frame& frame,
                    const PackPlan& plan, int i, std::size_t payload_bytes) {
    const std::size_t off = buf.size();
    buf.resize(off + payload_bytes);
    std::byte* dst = buf.data() + off;
    std::memset(dst, 0, payload_bytes);
    (void)comm::pack_exchange(frame, plan, i, dst, payload_bytes);
}

// Staged nearest-neighbor owner exchange along one Cartesian axis (LAMMPS
// comm_brick::exchange style). Owned atoms whose `dim` coordinate falls outside
// [sublo, subhi) are shipped to the lo/hi FACE NEIGHBOR; the loop forwards
// multi-subdomain movers (and the rare PBC wrap-around atom) one hop per round
// until no rank has any in-flight atom. Uses only nearest-neighbor Sendrecv +
// a single-int Allreduce per round — O(1) point-to-point + O(log P) reduce
// fabric state, versus MPI_Alltoallv's O(world) per-peer state that made AV's
// cross-node RSS blow up (3.16x LAMMPS at 1024 ranks). Positions are already
// wrapped into [0, L) by the integrator's wrap_orthorhombic, so a raw
// coord-vs-slab compare matches the old dest_rank_for binning exactly.
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

    std::vector<std::byte> send_lo, send_hi, recv_lo, recv_hi;
    while (true) {
        // Re-fetch each round: set_capacity below may have reallocated the SoA,
        // invalidating any span resolved in a previous round.
        const StorageReal* __restrict q = (dim == 0)   ? frame.span(kAtom_x)
                                          : (dim == 1) ? frame.span(kAtom_y)
                                                       : frame.span(kAtom_z);
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
        int local_out = n_send_lo + n_send_hi;
        int global_out = 0;
        MPI_Allreduce(&local_out, &global_out, 1, MPI_INT, MPI_SUM, comm);
        if (global_out == 0)
            return; // every rank's atoms are within their slab in this dim

        // Two independent streams (distinct tags so they never collide even
        // when dims[dim]==2 and lo_nbr==hi_nbr): +dim atoms go to hi_nbr and
        // arrive from lo_nbr; −dim atoms go to lo_nbr and arrive from hi_nbr.
        int n_recv_lo = 0, n_recv_hi = 0; // +dim from lo, −dim from hi
        MPI_Status st;
        MPI_Sendrecv(&n_send_hi, 1, MPI_INT, hi_nbr, 0, &n_recv_lo, 1, MPI_INT,
                     lo_nbr, 0, comm, &st);
        MPI_Sendrecv(&n_send_lo, 1, MPI_INT, lo_nbr, 1, &n_recv_hi, 1, MPI_INT,
                     hi_nbr, 1, comm, &st);

        recv_lo.assign(static_cast<std::size_t>(n_recv_lo) * payload_bytes,
                       std::byte{0});
        recv_hi.assign(static_cast<std::size_t>(n_recv_hi) * payload_bytes,
                       std::byte{0});
        MPI_Sendrecv(send_hi.data(),
                     checked_mpi_byte_count(send_hi.size(), "migrate/send_hi"),
                     MPI_BYTE, hi_nbr, 2, recv_lo.data(),
                     checked_mpi_byte_count(recv_lo.size(), "migrate/recv_lo"),
                     MPI_BYTE, lo_nbr, 2, comm, &st);
        MPI_Sendrecv(send_lo.data(),
                     checked_mpi_byte_count(send_lo.size(), "migrate/send_lo"),
                     MPI_BYTE, lo_nbr, 3, recv_hi.data(),
                     checked_mpi_byte_count(recv_hi.size(), "migrate/recv_hi"),
                     MPI_BYTE, hi_nbr, 3, comm, &st);

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
        int slot = dd->n_local;
        std::size_t o = 0;
        for (int k = 0; k < n_recv_lo; ++k, o += payload_bytes)
            (void)comm::unpack_exchange(frame, plan, slot++, recv_lo.data() + o,
                                        payload_bytes);
        o = 0;
        for (int k = 0; k < n_recv_hi; ++k, o += payload_bytes)
            (void)comm::unpack_exchange(frame, plan, slot++, recv_hi.data() + o,
                                        payload_bytes);
        dd->n_local = slot;
        frame.set_nlocal(slot);
        // Loop: freshly received atoms are re-classified and may forward
        // another hop (multi-subdomain drift / PBC wrap-around).
    }
}

} // namespace

void Migrate::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_size <= 1)
        return;
    // Skip migration during a line-search trial re-evaluation: the optimizer
    // is replaying the force phase at a trial geometry x_prev + alpha*p, and
    // its x_prev_/direction_ buffers are keyed to the *current* owned-row
    // order. Migrating mid-search would reshuffle those rows and corrupt the
    // search. Halo exchange still refreshes ghosts, so forces stay correct for
    // the bounded trial step; real migration resumes when the driver's force
    // phase runs at the committed geometry.
    if (ctx.runtime.opt.line_search_trial)
        return;
    auto* dd = ctx.runtime.md.dd;
    if (!dd || dd->cart_comm == MPI_COMM_NULL)
        return;

    Frame& frame = ctx.frame;

    int my_rank = 0;
    MPI_Comm_rank(dd->cart_comm, &my_rank);

    const bool first_launch = last_observed_generation_ == ~std::uint64_t{0};
    const bool generation_changed =
        !first_launch && dd->neighbor_generation != last_observed_generation_;
    if ((ctx.step % every_) != 0 && !generation_changed) {
        return;
    }

    // Quick global early-out: if no rank has any atom outside its 3D slab,
    // skip the exchange entirely (cheap 1-int LOR Allreduce). dest_rank_for
    // wraps defensively, matching the per-dim [sublo,subhi) classification.
    int any_local = 0;
    {
        const StorageReal* __restrict sp_x = frame.span(kAtom_x);
        const StorageReal* __restrict sp_y = frame.span(kAtom_y);
        const StorageReal* __restrict sp_z = frame.span(kAtom_z);
        for (int i = 0; i < dd->n_local; ++i) {
            if (dest_rank_for(*dd, sp_x[i], sp_y[i], sp_z[i], frame.simbox)
                != my_rank) {
                any_local = 1;
                break;
            }
        }
    }
    int any_global = 0;
    MPI_Allreduce(&any_local, &any_global, 1, MPI_INT, MPI_LOR, dd->cart_comm);
    if (any_global == 0) {
        last_observed_generation_ = dd->neighbor_generation;
        return;
    }

    // Old ghost rows are not owners and never enter migration. Retire their
    // population before any capacity or owner-count mutation.
    frame.set_nghost(0);
    dd->n_ghost = 0;

    PackPlan plan = make_exchange_plan(frame);
    const std::size_t payload_bytes =
        atom_payload_bytes(frame, plan, dd->cart_comm);

    // Staged nearest-neighbor exchange x → y → z. After all three axes every
    // owned atom lies within [sublo,subhi) on every axis, i.e. on the rank
    // dest_rank_for would assign it — identical end state to the old
    // MPI_Alltoallv, but with O(1)/O(log P) fabric state instead of O(world).
    exchange_along_dim(frame, dd, 0, payload_bytes, plan);
    exchange_along_dim(frame, dd, 1, payload_bytes, plan);
    exchange_along_dim(frame, dd, 2, payload_bytes, plan);

    dd->n_ghost = 0;
    ++dd->neighbor_generation;

    // Rebuild tag → local index map. Bonded indices that key off `tag_to_local`
    // re-resolve through this map on the next neighbor / bonded build.
    // Pure pairwise systems never query it — skip the rebuild entirely.
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

    // Also clear `ghost_tags`. The previous step's ghost-tag log is now
    // stale, and worse, leaving it populated lets `HaloSend`'s
    // full-rebuild path `erase` those tags from the FRESH `tag_to_local`
    // we just rebuilt. Any atom that was a ghost on this rank
    // pre-migration AND just migrated IN as a now-owned atom would have
    // its owner entry silently removed, causing the bonded
    // `select_owned_tuples` filter to drop every bond owned by that atom
    // on every rank (no rank sees it as `owned`, slot < n_local). For
    // MSU PE that was ~15 bonds dropped per migration step, ~5 eV PE
    // loss per halo cycle, accumulating to a catastrophic ε̇ runaway by
    // step ~1000.
    dd->ghost_tags.clear();

    last_observed_generation_ = dd->neighbor_generation;
}

void scatter_atoms_from_root(cpu::Frame& frame) {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited)
        return;
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world <= 1)
        return;

    // Wire agreement precedes every payload, including the cell broadcast.
    PackPlan plan = make_exchange_plan(frame);
    require_collective_schema(plan, MPI_COMM_WORLD, "scatter_atoms_from_root");
    const std::uint64_t expected_fingerprint = plan.fingerprint;

    // 1. Broadcast the nine physical cell scalars explicitly. Box owns a
    // cache and is never a raw-memory wire object.
    std::array<double, 9> cell{};
    if (rank == 0) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[static_cast<std::size_t>(3 * i + j)] =
                    frame.simbox.cell()(i, j);
    }
    MPI_Bcast(cell.data(), static_cast<int>(cell.size()), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    if (rank != 0) {
        Mat3 matrix{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                matrix(i, j) = cell[static_cast<std::size_t>(3 * i + j)];
        frame.simbox.set_cell(matrix);
    }

    // 2. Topology identical to the DD the backend builds → ownership matches
    //    partition_atoms' uniform-Cartesian rule (cuts empty here).
    md::mpi::DomainDecomposition dd;
    dd.init_topology(MPI_COMM_WORLD, frame.simbox.length(0),
                     frame.simbox.length(1), frame.simbox.length(2));
    const int world_size = dd.dims[0] * dd.dims[1] * dd.dims[2];

    // 3. Bucket every locally-held atom by owner (rank 0: all; others: none).
    const int n = frame.nlocal();
    std::vector<std::vector<int>> by_dest(static_cast<std::size_t>(world_size));
    {
        const StorageReal* __restrict sp_x = frame.span(kAtom_x);
        const StorageReal* __restrict sp_y = frame.span(kAtom_y);
        const StorageReal* __restrict sp_z = frame.span(kAtom_z);
        for (int i = 0; i < n; ++i) {
            const int dest =
                dest_rank_for(dd, sp_x[i], sp_y[i], sp_z[i], frame.simbox);
            by_dest[static_cast<std::size_t>(dest)].push_back(i);
        }
    }

    // 4. Pack per destination (fixed stride) + Alltoallv. Counts/displacements
    //    are in ATOMS (via a per-atom MPI datatype), NOT bytes: rank 0 holds
    //    the whole system, so a byte-count displacement (`int`) overflows 2 GB
    //    at large N (4M atoms x payload > 2 GB → negative displ → segfault at
    //    256+ ranks). Atom counts stay < 2^31 for any realistic system.
    const std::size_t payload_bytes =
        atom_payload_bytes(frame, plan, MPI_COMM_WORLD);
    MPI_Datatype atom_t;
    MPI_Type_contiguous(
        checked_mpi_byte_count(payload_bytes, "scatter_atoms_from_root/row"),
        MPI_BYTE, &atom_t);
    MPI_Type_commit(&atom_t);
    std::vector<int> sc(world_size, 0), sd(world_size, 0), rc(world_size, 0),
        rd(world_size, 0);
    std::size_t tot_send = 0; // in atoms
    for (int r = 0; r < world_size; ++r) {
        sd[r] = static_cast<int>(tot_send);
        sc[r] = static_cast<int>(by_dest[r].size());
        tot_send += by_dest[r].size();
    }
    if (tot_send != 0
        && payload_bytes > std::numeric_limits<std::size_t>::max() / tot_send)
        throw std::overflow_error(
            "scatter_atoms_from_root: send size overflow");
    std::vector<std::byte> sbuf(tot_send * payload_bytes);
    std::size_t off = 0;
    for (int r = 0; r < world_size; ++r)
        for (int idx : by_dest[static_cast<std::size_t>(r)])
            off = pack_atom(sbuf, off, frame, plan, idx, payload_bytes);

    MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::size_t tot_recv = 0; // in atoms
    for (int r = 0; r < world_size; ++r) {
        rd[r] = static_cast<int>(tot_recv);
        tot_recv += static_cast<std::size_t>(rc[r]);
    }
    if (tot_recv != 0
        && payload_bytes > std::numeric_limits<std::size_t>::max() / tot_recv)
        throw std::overflow_error(
            "scatter_atoms_from_root: receive size overflow");
    std::vector<std::byte> rbuf(tot_recv * payload_bytes);
    MPI_Alltoallv(sbuf.data(), sc.data(), sd.data(), atom_t, rbuf.data(),
                  rc.data(), rd.data(), atom_t, MPI_COMM_WORLD);
    MPI_Type_free(&atom_t);

    // 5. Land received atoms in [0, n_recv); size to owned + ghost headroom.
    //    `tot_recv` is already in atoms (the Alltoallv used a per-atom type).
    if (tot_recv > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "scatter_atoms_from_root: owner count overflow");
    const int n_recv = static_cast<int>(tot_recv);
    // Ghost headroom: 25% (typical halo influx is 20-35% of owners; the
    // chunked Atom::grow covers thin-subdomain outliers). The previous
    // `max(n_recv, 16384)` headroom doubled every per-atom array.
    const std::int64_t requested =
        static_cast<std::int64_t>(n_recv) + std::max(n_recv / 4, 16384);
    if (requested > std::numeric_limits<int>::max())
        throw std::overflow_error("scatter_atoms_from_root: capacity overflow");
    frame.set_nghost(0);
    frame.set_nlocal(0);
    frame.set_capacity(static_cast<int>(requested));
    plan = make_exchange_plan(frame);
    require_schema_fingerprint(plan, expected_fingerprint, plan.version);
    std::size_t r_off = 0;
    for (int k = 0; k < n_recv; ++k)
        r_off = unpack_atom(rbuf, r_off, frame, plan, k, payload_bytes);
    frame.set_nlocal(n_recv);
    frame.set_nghost(0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n_recv);
}

#else // !ATV_HAS_MPI

void Migrate::launch(const cpu::LaunchContext& /*ctx*/) {}
void scatter_atoms_from_root(cpu::Frame& /*frame*/) {}

#endif // ATV_HAS_MPI

} // namespace comm
} // namespace cpu
} // namespace ATV
