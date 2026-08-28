/// @file src/cpu/communication/domain.cpp
/// @brief `DomainDecomposition` implementation: single-rank `bind`, plus
/// multi-rank `init_topology`, `compute_sublo_subhi`, and `partition_atoms`.
///
/// Algorithm and communication structure transplanted from LAMMPS
/// `src/domain.cpp` / `src/comm.cpp` / `src/procmap.cpp` (checkout `78a7ddf`,
/// 2026-08-25). Public API is ATV-native.
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).

#include "atomiverse/cpu/communication/domain.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_set>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

void DomainDecomposition::bind(const Frame& frame) {
    const int n = frame.nlocal();
    n_local = n;
    n_ghost = 0;
    capacity = n;

    tag_to_local.clear();
    // The tag→local map exists solely for bonded-tuple resolution; pure
    // pairwise systems never query it. Skipping the fill (and freeing any
    // prior buckets) saves ~50 B/atom of hash-node heap.
    if (frame.has_bonded()) {
        tag_to_local.reserve(static_cast<std::size_t>(n));
        const TagInt* __restrict sp_tag = frame.span(kAtom_tag);
        for (int i = 0; i < n; ++i) {
            tag_to_local.emplace(sp_tag[i], i);
        }
    } else {
        std::unordered_map<TagInt, int>{}.swap(tag_to_local);
    }
}

void DomainDecomposition::compute_sublo_subhi(const ATV::cpu::Box& box) {
    for (int d = 0; d < 3; ++d) {
        const double L = box.length(d);
        if (static_cast<int>(cuts[d].size()) == dims[d] + 1) {
            // Non-uniform cut planes (set by `Rebalance`). cuts[d] is in
            // box-length fractions, so the partition survives NPT/Deform box
            // rescaling without recomputation.
            sublo[d] = cuts[d][static_cast<std::size_t>(coords[d])] * L;
            subhi[d] = cuts[d][static_cast<std::size_t>(coords[d] + 1)] * L;
        } else {
            const double slab = L / static_cast<double>(dims[d]);
            sublo[d] = slab * static_cast<double>(coords[d]);
            subhi[d] = sublo[d] + slab;
        }
    }
}

#ifdef ATV_HAS_MPI

void DomainDecomposition::release_comm(bool /*mpi_is_finalizing*/) {
    // We do not own a Cartesian sub-comm (we use MPI_COMM_WORLD directly --
    // see init_topology). Just clear the handle.
    cart_comm = MPI_COMM_NULL;
}

void DomainDecomposition::choose_dims(int world_size, double lx, double ly,
                                      double lz, int dims_out[3]) {
    // Env override first — shared by every caller (DD and builder-side
    // pre-filtering must agree on the grid or atoms get dropped at the
    // initial partition).
    if (const char* env = std::getenv("ATV_FORCE_DIMS")) {
        int dx, dy, dz;
        if (std::sscanf(env, "%d,%d,%d", &dx, &dy, &dz) == 3
            && dx * dy * dz == world_size) {
            dims_out[0] = dx;
            dims_out[1] = dy;
            dims_out[2] = dz;
            return;
        }
    }
    if (!(lx > 0.0) || !(ly > 0.0) || !(lz > 0.0))
        lx = ly = lz = 1.0; // box unknown: near-cubic factorization

    // Enumerate every factor triple (dx, dy, dz) of P and minimize the
    // per-rank half-surface wx*wy + wy*wz + wz*wx (wd = Ld/dd): the halo
    // ghost volume each rank ships per step is proportional to it. Strict
    // `<` with DESCENDING iteration keeps the choice deterministic AND
    // makes equal-cost ties (e.g. any cubic box) resolve to the same
    // descending-dims orientation `MPI_Dims_create` used to produce —
    // cubic-box decompositions are bit-compatible with the old behavior.
    double best = std::numeric_limits<double>::infinity();
    for (int dx = world_size; dx >= 1; --dx) {
        if (world_size % dx)
            continue;
        const int pyz = world_size / dx;
        for (int dy = pyz; dy >= 1; --dy) {
            if (pyz % dy)
                continue;
            const int dz = pyz / dy;
            const double wx = lx / dx, wy = ly / dy, wz = lz / dz;
            const double cost = wx * wy + wy * wz + wz * wx;
            if (cost < best) {
                best = cost;
                dims_out[0] = dx;
                dims_out[1] = dy;
                dims_out[2] = dz;
            }
        }
    }
}

void DomainDecomposition::init_topology(MPI_Comm world, double lx, double ly,
                                        double lz) {
    int world_size = 0;
    int my_rank = 0;
    MPI_Comm_size(world, &world_size);
    MPI_Comm_rank(world, &my_rank);

    if (world_size <= 1) {
        // Single rank: keep defaults (dims={1,1,1}, no cart_comm).
        return;
    }

    // Do NOT use MPI_Cart_create -- on OpenMPI 5.0.x (macOS) the resulting
    // sub-communicator triggers a segfault inside MPI_Finalize when any
    // user MPI_Comm_free has been issued or when the same process repeatedly
    // creates Cartesian comms (e.g. across GoogleTest cases). Compute the
    // brick decomposition directly on top of the world communicator using
    // simple rank arithmetic; this is functionally equivalent and avoids
    // the OpenMPI bug entirely.
    cart_comm = world;

    // Box-aware processor grid (LAMMPS ProcMap::best_factors): more cuts
    // along longer axes, minimizing per-rank halo surface. `choose_dims`
    // honors the `ATV_FORCE_DIMS=Nx,Ny,Nz` diagnostic override and falls
    // back to a near-cubic factorization when the box is not yet known
    // (lx/ly/lz == 0, e.g. Backend::init before a frame is bound — the
    // grid is re-derived box-aware in Backend::set_frame before the
    // initial partition).
    choose_dims(world_size, lx, ly, lz, dims);

    // Rank ↔ grid-coords mapping: see rank_of/coords_of (x-slowest
    // row-major brick).
    coords_of(my_rank, coords);

    auto rank_at = [&](int cx, int cy, int cz) {
        // Periodic wrap on all axes.
        cx = ((cx % dims[0]) + dims[0]) % dims[0];
        cy = ((cy % dims[1]) + dims[1]) % dims[1];
        cz = ((cz % dims[2]) + dims[2]) % dims[2];
        return rank_of(cx, cy, cz);
    };

    // Face order: -x, +x, -y, +y, -z, +z.
    face_neighbors[0] = rank_at(coords[0] - 1, coords[1], coords[2]);
    face_neighbors[1] = rank_at(coords[0] + 1, coords[1], coords[2]);
    face_neighbors[2] = rank_at(coords[0], coords[1] - 1, coords[2]);
    face_neighbors[3] = rank_at(coords[0], coords[1] + 1, coords[2]);
    face_neighbors[4] = rank_at(coords[0], coords[1], coords[2] - 1);
    face_neighbors[5] = rank_at(coords[0], coords[1], coords[2] + 1);
}

namespace {

// Orthorhombic PBC wrap into [0, L) is provided by `Box::wrap_axis`
// (`cpu/box.h`) — the single canonical implementation. partition_atoms()
// below calls `frame.box.wrap_axis(coord, axis)` directly.

// Rank index (along one dim) that owns wrapped position `q` with `dims_d`
// equal-width slabs over `[0, L)`. Clamps to [0, dims_d - 1] against FP
// rounding at the upper edge.
inline int rank_along_dim(double q, double L, int dims_d) {
    if (dims_d <= 1)
        return 0;
    const int r = static_cast<int>(q * static_cast<double>(dims_d) / L);
    return std::max(0, std::min(dims_d - 1, r));
}

// Slab index whose [cuts[k], cuts[k+1]) contains the box-length fraction
// `frac` (already wrapped into [0, 1)). `cuts` has dims_d + 1 entries with
// front() == 0 and back() == 1. Clamped against FP rounding at both edges.
inline int slab_along_dim_from_cuts(double frac,
                                    const std::vector<double>& cuts,
                                    int dims_d) {
    if (dims_d <= 1)
        return 0;
    auto it = std::upper_bound(cuts.begin(), cuts.end(), frac);
    int k = static_cast<int>(std::distance(cuts.begin(), it)) - 1;
    if (k < 0)
        k = 0;
    if (k > dims_d - 1)
        k = dims_d - 1;
    return k;
}

} // namespace

int DomainDecomposition::owner_of(double x, double y, double z,
                                  const ATV::cpu::Box& box) const {
    const double q[3] = {x, y, z};
    int c[3];
    for (int d = 0; d < 3; ++d) {
        const double L = box.length(d);
        const double wrapped = box.wrap_axis(q[d], d);
        if (static_cast<int>(cuts[static_cast<std::size_t>(d)].size())
            == dims[d] + 1) {
            const double frac = (L > 0.0) ? (wrapped / L) : 0.0;
            c[d] = slab_along_dim_from_cuts(
                frac, cuts[static_cast<std::size_t>(d)], dims[d]);
        } else {
            c[d] = rank_along_dim(wrapped, L, dims[d]);
        }
    }
    return rank_of(c[0], c[1], c[2]);
}

void DomainDecomposition::partition_atoms(ATV::cpu::Frame& frame) {
    const int world_size = dims[0] * dims[1] * dims[2];
    if (world_size <= 1 || cart_comm == MPI_COMM_NULL) {
        // Single-rank fallback: identical to `bind`.
        bind(frame);
        return;
    }

    compute_sublo_subhi(frame.box);

    const double Lx = frame.box.length(0);
    const double Ly = frame.box.length(1);
    const double Lz = frame.box.length(2);

    // First-run vs subsequent-run scoping. On the first call (n_local == 0)
    // we scan the entire user-supplied atom array because we don't yet know
    // which atoms belong to this rank. On subsequent calls -- e.g. when a
    // user invokes `Driver::run()` twice in a row -- the array's tail rows
    // hold stale ghost positions and old owner copies left over by halo and
    // by the previous compaction. Re-scanning those would let the same
    // physical atom be picked again as an extra owner (its stale FCC
    // position is unchanged, its post-step compacted copy is at a different
    // slot), and the resulting two near-identical "owner" rows blow up LJ.
    // On a re-partition the previously-owned rows [0, n_local) are the
    // authoritative owner set (Migrate already re-homed any cross-slab
    // drifters at pre_force); we only need to re-scan that range, not the
    // stale ghost/old-owner rows beyond it.
    const bool first_partition = (n_local == 0);
    const int n_to_iterate = first_partition ? frame.nlocal() : n_local;
    std::vector<int> owned_indices;
    owned_indices.reserve(static_cast<std::size_t>(n_to_iterate));

    // First-partition deduplication. When the caller passed an
    // already-partitioned frame (e.g. polyethylene_tension's per-stage
    // driver rebuild: stage 1 partitioned the frame into [0..n_local) +
    // stale tail; stage 2's fresh DD enters here with n_local==0 so the
    // tail rows are also scanned), the same tag can appear at both the
    // compacted slot and the stale slot. Without dedup both copies are
    // claimed, LJ then finds two atoms at identical positions, and the
    // run blows up at stage 2 step 0. Tag-keyed `first-wins` keeps the
    // live (compacted) copy and drops the stale one — physics is
    // identical to a clean global-frame first-partition.
    std::unordered_set<std::int64_t> seen_tags;
    if (first_partition)
        seen_tags.reserve(static_cast<std::size_t>(n_to_iterate));

    // Hoisted ONCE at the launch boundary: no grow/set_capacity happens
    // between here and the compaction pass below (the first reallocation in
    // this function is the post-compaction shrink further down), so these
    // stay valid across both the eligibility scan and the compaction.
    TagInt* __restrict sp_tag = frame.span(kAtom_tag);
    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);

    for (int i = 0; i < n_to_iterate; ++i) {
        if (first_partition) {
            const std::int64_t t = sp_tag[i];
            if (!seen_tags.insert(t).second)
                continue; // duplicate tag from stale tail — keep first
        }
        const double xw = frame.box.wrap_axis(sp_x[i], 0);
        const double yw = frame.box.wrap_axis(sp_y[i], 1);
        const double zw = frame.box.wrap_axis(sp_z[i], 2);
        const int cx = rank_along_dim(xw, Lx, dims[0]);
        const int cy = rank_along_dim(yw, Ly, dims[1]);
        const int cz = rank_along_dim(zw, Lz, dims[2]);
        if (cx == coords[0] && cy == coords[1] && cz == coords[2]) {
            owned_indices.push_back(i);
        }
    }

    n_local = static_cast<int>(owned_indices.size());
    n_ghost = 0;
    capacity = frame.capacity();

    // One structural row operation covers every migrating column: canonical
    // state, topology and runtime columns. Increasing destination order keeps
    // every source row live until it is copied.
    for (int ni = 0; ni < n_local; ++ni) {
        const int oi = owned_indices[static_cast<std::size_t>(ni)];
        if (oi != ni)
            frame.copy_row(ni, oi);
    }
    frame.set_nlocal(n_local);
    frame.set_nghost(0);

    // Release the global-N backing store. Each rank loaded the whole file
    // (load_lammps_data is called per-rank) but owns only `n_local`; the
    // arrays were left at global-N capacity "as slack for ghosts", which made
    // per-rank memory scale with the TOTAL system size (measured: 237 GB vs
    // LAMMPS 20 GB at 256 ranks). Shrink to owned + a generous ghost/migration
    // headroom. Halo/migration grow via `set_capacity` on overflow, so this
    // bound only needs to cover the common (roughly cubic subdomain) case;
    // ghost shells run ~n_local for a compact subdomain.
    {
        // 25% ghost headroom (typical halo influx is 20-35% of owners;
        // halo_exchange grows on demand for thin-subdomain outliers). The
        // previous max(n_local, 16384) headroom doubled every per-atom
        // array for the whole run.
        const int headroom = std::max(n_local / 4, 16384);
        const int new_cap = n_local + headroom;
        if (new_cap < frame.capacity()) {
            frame.set_capacity(new_cap);
        }
        capacity = frame.capacity();
    }

    tag_to_local.clear();
    // Bonded-resolution map only — pure pairwise systems skip it (see bind()).
    if (frame.has_bonded()) {
        tag_to_local.reserve(static_cast<std::size_t>(n_local));
        // Re-resolved: `set_capacity` just above may have reallocated the
        // SoA, invalidating the `sp_tag` resolved earlier in this function.
        const TagInt* __restrict sp_tag2 = frame.span(kAtom_tag);
        for (int i = 0; i < n_local; ++i) {
            tag_to_local.emplace(sp_tag2[i], i);
        }
    } else {
        std::unordered_map<TagInt, int>{}.swap(tag_to_local);
    }

    // No topology filter at partition. Each owned row keeps its full bonded topology
    // / angles / dihedrals so that:
    //   * bonded force kernels can iterate all tuples and filter by their
    //     owner-tag (LAMMPS-style ownership rule) at force-eval time;
    //   * LJ::set_special_bonds() can build a complete tag-indexed
    //     exclusion map without missing bonds whose owner lives on another
    //     rank -- otherwise an owner-ghost LJ pair that is bonded but
    //     ownership-routed away would not be excluded and would blow up.
    //
    // Trade-off: each rank scans all O(N) tuples per force call instead
    // of just its O(N/ranks) owned subset. For < 1M-atom systems this is
    // a ~1 ms tax per step. A future optimisation can split: keep full
    // topology for the LJ exclusion build, filter for the force pass.
    ++neighbor_generation;
}

#endif // ATV_HAS_MPI

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
