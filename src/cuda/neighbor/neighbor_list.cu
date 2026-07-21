/// @file md/neighbor_list.cu
/// @brief NeighborList Component — build/launch/destroy lifecycle with
///        Verlet-skin rebuild gating.
///
/// build():   allocates PairList + internal buffers + pinned host scratch
///            + cudaEvents; pre-sizes CUB sort/scan temps so launch() never
///            cudaMalloc's via DeviceVector::resize.
/// launch():  consults the lagged-D2H `rebuild_flag` from the previous step
///            and either skips the full rebuild pipeline (Verlet skin still
///            valid) or runs Morton sort → radix sort → scatter+aabb → grid
///            → coarse-count → fill → compact → csr_to_flat → smart prune
///            → export and snapshots ref_x/y/z. Always runs
///            launch_atom_displacement_max to write the next step's flag.
/// destroy(): frees PairList + pinned scratch + events.
///
/// Skin-gate semantics:
///   - First launch: rebuild unconditionally (no prior flag).
///   - Subsequent launches: cudaEventSynchronize on the previous step's
///     rebuild_flag event, read pinned host int, rebuild iff set.
///   - Lag is 1 step: a violation at step N triggers rebuild at step N+1.
///     skin width (default 0.3 · cutoff) absorbs the extra step of motion;
///     LAMMPS / GROMACS use the same 1-step-lag pattern.

#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "nlist_impl.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

using namespace ATV::cuda::md::nlist;

static constexpr int kMaxStencil = 512;

namespace ATV {
namespace cuda {

using cuda::prec_t;
using cuda::real_t;
using cuda::work_t;

// Right-size the per-cell bin capacity to the actual cell occupancy. binsize is
// 0.5*(cutoff+skin) (setup.cu), so cells are fine: ~1.5 atoms/cell for an LJ
// liquid, ~5 for the PE melt. A fixed cap of 32 therefore left ~95% of every
// cell's slots empty — and atom_bin_pos is a float4 per slot, the single
// largest neighbor-list allocation (~410 B/atom for LJ). Start near the
// expected peak occupancy (mean + headroom); the per-build overflow probe grows
// it on the rare dense-cell overflow, and the grown cap then sticks for the run
// (the resize gate keys on cell-count change only, so it never re-shrinks and
// thrashes). Clamped to [8, 256].
static inline int bin_cap_for(int n, int ncells) {
    if (ncells <= 0 || n <= 0)
        return 16;
    const int mean = (n + ncells - 1) / ncells; // ceil(n / ncells)
    int cap = mean * 2 + 8;                     // headroom over the mean
    if (cap < 8)
        cap = 8;
    if (cap > 256)
        cap = 256;
    return cap;
}

// Tight regrow target after a bin overflow: the observed max occupancy `bo`
// plus 50% headroom. Replaces the prior `bo*1.3 + 16`, whose +16 floor jumped a
// 12-slot cap back to ~30 (re-introducing the waste) on a single overflow.
static inline int bin_cap_grow(int bo) {
    return static_cast<int>(bo * 1.5) + 2;
}

// ============================================================================
// State
// ============================================================================

struct NeighborList::State {
    NListBuffers<real_t> buf;

    DeviceVector<real_t> d_cell{9};
    DeviceVector<real_t> d_inv_cell{9};
    DeviceVector<int> d_dims{3};
    DeviceVector<int> d_ncells{1};
    DeviceVector<int> stencil_dx{kMaxStencil};
    DeviceVector<int> stencil_dy{kMaxStencil};
    DeviceVector<int> stencil_dz{kMaxStencil};
    DeviceVector<int> d_n_stencil{1};
    DeviceVector<int> d_pair_count{1};
    DeviceVector<int> d_rebuild_flag{1};

    int num_atoms = 0;
    int num_clusters = 0;
    int max_atoms = 0; // build-time atom capacity (buffers sized for this)
    int pair_capacity = 0;
    int atom_pair_cap = 0; // flat (i,j,...) export capacity; allocated lazily
    int dims_x = 0, dims_y = 0, dims_z = 0;
    bool periodic = true;

    // Cached scalars (computed once in build())
    work_t cut_skin_sq = work_t(0);
    work_t cutoff_sq = work_t(0);
    work_t half_skin_sq =
        work_t(0); // (skin/2)² — Verlet-skin trigger threshold

    // Skin-gate state
    bool first_launch = true;
    int rebuild_count = 0; // host-side counter for tests / introspection
    int last_flat_pair_count =
        0; // lagged readback; sized to pair_capacity on first call

    // Pinned host scratch + events for lagged D2H of rebuild_flag /
    // flat_pair_counter. cudaMallocHost'd in build(), freed in destroy().
    int* h_rebuild_flag = nullptr;
    int* h_flat_pair_count = nullptr;
    // Multi-GPU concurrent reneighbor (rebuild_issue/finalize_rebuild): separate
    // pinned overflow probes for the bin and Verlet passes (both in flight when
    // the rebuild is issued without an intervening sync) + the ncells the
    // in-flight rebuild used (for the rare resize in finalize_rebuild).
    DeviceVector<int> d_bin_overflow{1};
    int* h_bin_overflow = nullptr;
    int* h_vl_overflow = nullptr;
    int rebuild_ncells = 0;
    cudaEvent_t rebuild_flag_event = nullptr;
    cudaEvent_t flat_pair_count_event = nullptr;
    bool flat_pair_count_event_valid = false;
};

// ============================================================================
// build() — one-time resource preparation
// ============================================================================

void NeighborList::build(const BuildContext& ctx) {
    state_ = std::make_unique<State>();
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    s.num_atoms = ctx.frame.n_atoms();
    int n = s.num_atoms;
    int nc = (n + kClusterSize - 1) / kClusterSize;
    s.num_clusters = nc;
    s.max_atoms = n; // all buffers below are sized for this atom count
    s.periodic = true;

    work_t r_search = cutoff_ + skin_;

    // Cache squared cutoffs so launch() doesn't recompute every step.
    s.cut_skin_sq = r_search * r_search;
    s.cutoff_sq = work_t(cutoff_) * work_t(cutoff_);
    work_t half_skin = work_t(skin_) * work_t(0.5);
    s.half_skin_sq = half_skin * half_skin;

    // GPU setup: compute cell/inv_cell/dims/stencil from device box.
    // Re-run in launch() only when the rebuild gate decides to rebuild,
    // since the stencil is invariant for NVE/NVT and changes at most as
    // often as the rebuild rate for NPT.
    launch_setup_cell_device(frame.box, s.d_cell.data(), s.d_inv_cell.data(),
                             s.d_dims.data(), s.stencil_dx.data(),
                             s.stencil_dy.data(), s.stencil_dz.data(),
                             s.d_n_stencil.data(), s.d_ncells.data(), r_search,
                             s.periodic, kMaxStencil, stream);

    // Read dims from device (build-time, allowed)
    cudaStreamSynchronize(stream);
    cudaMemcpy(&s.dims_x, s.d_dims.data(), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s.dims_y, s.d_dims.data() + 1, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&s.dims_z, s.d_dims.data() + 2, sizeof(int),
               cudaMemcpyDeviceToHost);
    int ncells = s.dims_x * s.dims_y * s.dims_z;

    // Box volume — read once here, reused for both the cluster-pair capacity
    // (just below) and the atom-pair capacity further down.
    double box_vol = 0.0;
    {
        prec_t bd[3] = {prec_t(0), prec_t(0), prec_t(0)};
        cudaMemcpy(&bd[0], &frame.box[0], sizeof(prec_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&bd[1], &frame.box[4], sizeof(prec_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&bd[2], &frame.box[8], sizeof(prec_t),
                   cudaMemcpyDeviceToHost);
        box_vol = double(bd[0]) * double(bd[1]) * double(bd[2]);
    }

    // ── Cluster-pair capacity (density-based, NOT the dense upper bound) ──
    // The CSR cluster-pair arrays (col_idx / interaction_mask / flat_ci /
    // flat_cj / flat_mask) are PACKED: they store the actual number of
    // interacting cluster pairs, indexed sequentially via row_ptr — not a
    // dense nc*nc triangle. Sizing them to nc*(nc+1)/2 wasted ~21 GB at 312k
    // atoms (the sole cause of the backend's ~95 GB/million-atom footprint and
    // its OOM ceiling at ~558k). Size from physical density instead: a cluster
    // sees ~density*sphere_vol(r_search) atoms => /kClusterSize neighbor
    // clusters. The 6x factor absorbs the cluster's spatial extent (a cluster's
    // atoms reach a shell ~one cluster-diameter beyond r_search) plus
    // inhomogeneity. This is only an INITIAL guess: launch() measures the exact
    // cluster-pair count each rebuild and SELF-HEALS (grows the CSR buffers and
    // continues) if the guess undershoots, so a wrong estimate can never
    // corrupt the CSR — it just triggers a one-time grow. Clamped to int range.
    // NOTE: the all-pairs figure nc*(nc+1)/2 is NOT a valid upper bound and is
    // deliberately NOT used as a clamp. Under PBC a cluster can interact with
    // periodic IMAGES of another cluster (and of itself), so for boxes that are
    // not >> cutoff the true cluster-pair count exceeds nc*(nc+1)/2 (e.g. a
    // frozen-lattice fixture with nc=8 produces 64 pairs > 36). Capacity is:
    //   max( nc * 256                                  (floor; cheap + ample for
    //                                                   small / sparse systems),
    //        nc * 12 * (density * sphere(r_search)/kClusterSize)  (large melts) )
    // The 12x absorbs the cubic-cell-stencil over-count vs a sphere plus the
    // cluster's spatial extent (~7-8x measured; 1.5x margin). launch()'s
    // self-heal grows the CSR on any residual undershoot, so this guess only
    // affects how often the one-time grow fires, never correctness.
    const double kPi = 3.141592653589793;
    long long cluster_cap_ll = static_cast<long long>(nc) * 64;
    if (box_vol > 0.0 && n > 0) {
        double density = double(n) / box_vol;
        double rs = double(r_search);
        double sphere = (4.0 / 3.0) * kPi * rs * rs * rs;
        double neigh_clusters = density * sphere / double(kClusterSize);
        // Now that launch() sizes the CSR from an EXACT AABB-filtered count
        // (and self-heals on undershoot), this is only an initial allocation
        // hint — sized to the realistic neighbor-cluster count, not a loose
        // over-bound. 8x absorbs the cluster's spatial extent (a cluster's
        // atoms reach ~one cluster-diameter past r_search) + the full
        // (both-direction) list; any residual undershoot just triggers a
        // one-time grow. Keeps the initial footprint ~density-proportional
        // instead of the old 24x loose-bound allocation (gigabytes at ~1M).
        double per_cluster = std::max(64.0, 8.0 * neigh_clusters);
        cluster_cap_ll = std::max(
            cluster_cap_ll, static_cast<long long>(double(nc) * per_cluster));
    }
    cluster_cap_ll =
        std::min(cluster_cap_ll,
                 static_cast<long long>(std::numeric_limits<int>::max()));
    s.pair_capacity = static_cast<int>(cluster_cap_ll);

    // Always-needed per-atom buffers (Verlet-skin reference positions).
    s.buf.resize_atoms(n);

    // Per-atom Verlet column capacity: start at kMaxVerletNeigh (covers an LJ
    // liquid's ~101 measured max). A density estimate is intentionally NOT used —
    // for a multi-GPU subdomain n is owned+ghost but box_vol is the GLOBAL cell,
    // so n/box_vol underestimates the local density and would undersize the cap.
    // Instead launch() self-heals: if any atom's neighbor count exceeds the
    // current capacity it grows the buffer and rebuilds, so the denser PE-melt
    // shell (~400) is handled automatically with no hand-tuned constant.
    s.buf.resize_verlet(n, kMaxVerletNeigh);

    // Cluster-CSR + grid + flat (i,j) pair machinery serves ONLY the opt-in
    // flat-pair export; the MD force path uses the per-atom Verlet list built
    // in launch(). It is ~115 B/atom (Morton + cluster SoA + AABB + CSR
    // col_idx/mask + flat arrays + CUB temps), so for any flat=false run (all
    // multi-GPU MD — multi_gpu.cu forces it false — and plain single-GPU MD)
    // it is pure dead weight. Allocate it only when flat pairs are requested;
    // otherwise keep size-1 placeholders and pair_capacity = 1 so launch()'s
    // lazy path grows everything the first time a requester turns the flag on
    // (a requester's build() may run after this build()). No in-tree
    // component sets the flag today — the RDF observer streams positions into
    // the molrs accumulator instead (arch-refactor-07).
    if (ctx.runtime.md.flat_pairs_requested) {
        s.buf.resize_cluster_csr(n, nc);
        s.buf.resize_displacement(nc);
        s.buf.resize_grid(ncells);
        int pc = s.pair_capacity > 0 ? s.pair_capacity : 1;
        s.buf.col_idx.resize(pc);
        s.buf.interaction_mask.resize(pc);
        s.buf.resize_flat_pairs(pc);

        // Pre-size CUB sort + scan temp storage so launch() never resizes
        // (and never cudaMalloc's via DeviceVector::resize).
        CubTempBytes cub_bytes = query_cub_temp_bytes(n, nc);
        if (cub_bytes.sort_bytes > 0)
            s.buf.sort_temp.resize(cub_bytes.sort_bytes);
        // launch() runs `launch_exclusive_sum` over `pair_counts[nc]` via
        // both `prefix_temp` (post-fill) and the previous `scan_temp` for
        // grid CSR (over `ncells`). Size to the larger of the two.
        std::size_t scan_max =
            std::max(cub_bytes.scan_bytes_clusters,
                     cub_bytes.scan_bytes_atoms); // atoms covers ncells too
        if (scan_max > 0) {
            s.buf.scan_temp.resize(scan_max);
            s.buf.prefix_temp.resize(scan_max);
        }
    } else {
        s.buf.col_idx.resize(1);
        s.buf.interaction_mask.resize(1);
        s.buf.resize_flat_pairs(1);
        s.pair_capacity = 1; // lazy grow in launch() sizes up if flat turns on
    }

    // Allocate the owned PairList.
    //
    // Capacity sized from physical density × neighborhood volume rather
    // than a flat per-atom guess. For dense molecular systems (e.g. UA
    // polyethylene at ρ ≈ 0.0195 atoms/Å³, cutoff 10.5 Å + skin 2 Å) a
    // pair can have ~160 neighbors within the search shell; a single
    // (i,j) entry is emitted per pair from the cluster export, so the
    // tight upper bound is n_pairs ≈ n · N_neighbors / 2. We multiply
    // by a 2× safety factor for inhomogeneous regions (the typical hot
    // spot during stage-1 NVELimit relaxation of an "as-built" PE
    // configuration is local overlap that briefly inflates the neighbor
    // count). Falls back to `max(n·50, 64k)` when the density estimate
    // is degenerate (atomless or zero-volume frame).
    //
    // The previous flat `n · 50` cap silently TRUNCATED overflow pairs:
    // a 10000-atom PE system with ~160 neighbors per atom hit ~800k
    // pairs while capacity was 500k → ~37% of pairs missing per step.
    // The dropped pairs had no LJ contribution, so atoms drifted toward
    // each other under bond/angle forces alone until LJ "rediscovered"
    // them at near-zero r → catastrophic E_pot spikes (10¹⁴ eV
    // observed) and irrecoverable energy injection that no Langevin
    // damp could compensate.
    // box_vol was already read above (reused here for the atom-pair capacity).
    double cut_search = static_cast<double>(cutoff_ + skin_);
    int atom_pair_cap = 0;
    if (box_vol > 0.0 && n > 0) {
        double density = double(n) / box_vol;
        double sphere_vol = (4.0 / 3.0) * 3.141592653589793 * cut_search
                            * cut_search * cut_search;
        double n_neighbors_est = density * sphere_vol;
        // Cluster-based export emits each pair TWICE (Ci→Cj and Cj→Ci);
        // empirical entry count is ~2.4× density·sphere_vol per atom
        // (cluster padding rounds neighbor lists up to the next
        // 8-atom group, so the effective sphere is slightly larger
        // than the geometric r_search). Use 3× as a comfortable cap.
        double per_atom = std::max(100.0, 3.0 * n_neighbors_est);
        atom_pair_cap = static_cast<int>(std::min(
            double(n) * per_atom, // each atom appears in `per_atom` entries
            double(std::numeric_limits<int>::max())));
    }
    atom_pair_cap = std::max(atom_pair_cap, std::max(1, n * 50));
    s.atom_pair_cap = atom_pair_cap;
    // The flat (i,j,dx,dy,dz,r²) atom-pair arrays exist ONLY for the opt-in
    // flat export. The MD force path reads the per-atom Verlet list, so for
    // it (and any flat=false run) these big arrays — n·max(100,…) entries ×
    // 24 B, e.g. ~2.4 GB per million atoms — are dead allocation. Allocate
    // them lazily on the first export instead (build() can't rely on
    // flat_pairs_requested being set yet, since a requester's build() may run
    // after NeighborList's). n_pairs/capacity are tiny and always live.
    if (ctx.runtime.md.flat_pairs_requested) {
        cudaMalloc(&pairs_.i, atom_pair_cap * sizeof(int));
        cudaMalloc(&pairs_.j, atom_pair_cap * sizeof(int));
        cudaMalloc(&pairs_.dx, atom_pair_cap * sizeof(real_t));
        cudaMalloc(&pairs_.dy, atom_pair_cap * sizeof(real_t));
        cudaMalloc(&pairs_.dz, atom_pair_cap * sizeof(real_t));
        cudaMalloc(&pairs_.dist_sq, atom_pair_cap * sizeof(real_t));
    }
    cudaMalloc(&pairs_.n_pairs, sizeof(int));
    cudaMemset(pairs_.n_pairs, 0, sizeof(int));
    cudaMalloc(&pairs_.capacity, sizeof(int));
    int cap_val = static_cast<int>(atom_pair_cap);
    cudaMemcpy(pairs_.capacity, &cap_val, sizeof(int), cudaMemcpyHostToDevice);

    // Pinned host scratch + events for the skin-gate's lagged D2H of
    // rebuild_flag (and lagged flat_pair_counter for right-sized
    // smart_rolling_prune / export grids).
    cudaMallocHost(&s.h_rebuild_flag, sizeof(int));
    *s.h_rebuild_flag = 0;
    cudaMallocHost(&s.h_flat_pair_count, sizeof(int));
    *s.h_flat_pair_count = 0;
    cudaMallocHost(&s.h_bin_overflow, sizeof(int));
    *s.h_bin_overflow = 0;
    cudaMallocHost(&s.h_vl_overflow, sizeof(int));
    *s.h_vl_overflow = 0;
    cudaEventCreateWithFlags(&s.rebuild_flag_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&s.flat_pair_count_event, cudaEventDisableTiming);

    // Publish the cluster CSR pointers. Buffer addresses are stable across
    // the run (DeviceVector capacity is fixed at build), but the CONTENTS are
    // only populated by the flat_pairs_requested rebuild block in launch() —
    // with the flag off these point at never-filled size-1 placeholders.
    // TODO(roykid, 2026-07-10): CoulLong's cluster kernels read cl_* yet
    // nothing in-tree sets flat_pairs_requested since the RDF observer
    // migrated to the molrs accumulator (arch-refactor-07) — the Coulomb
    // path needs its own cluster-CSR request (GPU-node verification owed).
    pairs_.cl_row_ptr = s.buf.row_ptr.data();
    pairs_.cl_col_idx = s.buf.col_idx.data();
    pairs_.cl_orig_index = s.buf.original_index.data();
    // flat_mask doubles as the interaction mask (1:1 with col_idx, since
    // csr_to_flat preserves CSR ordering); the address is stable, so publish
    // it here.
    pairs_.cl_mask = s.buf.flat_mask.data();
    pairs_.num_clusters = nc;

    // Per-atom Verlet list (the force path). Addresses are stable across the
    // run; contents rebuilt in place each skin-gated rebuild. vl_stride is the
    // build-time atom count (column stride); vl_max the column capacity.
    pairs_.vl_nbr = s.buf.verlet_nbr.data();
    pairs_.vl_cnt = s.buf.verlet_cnt.data();
    pairs_.vl_stride = s.buf.verlet_stride;
    pairs_.vl_max = s.buf.verlet_max;

    // Publish the pair list to the execution context for downstream pair
    // styles (LJ via the per-atom Verlet list, Coulomb via the cluster CSR),
    // mirroring cpu::md::NeighborList.
    ctx.runtime.md.pair_list = &pairs_;
}

// ============================================================================
// set_active_count() — reuse build()-time buffers for a different atom count
// ============================================================================
//
// Multi-GPU domain decomposition: a subdomain's owned+ghost count changes every
// reneighbor interval, but the per-subdomain buffers were sized once for the
// frame's full capacity at build(). Rather than destroy()+build() (reallocating
// GB-scale buffers each reneighbor — the dominant multi-GPU overhead), the
// driver calls this to reset the active atom/cluster count (<= the build-time
// capacity) and force a full CSR rebuild on the next launch(). No allocation.
// The single-GPU path never calls this, so its behaviour is unchanged.
void NeighborList::set_active_count(int n_active) {
    auto& s = *state_;
    if (n_active < 0)
        n_active = 0;
    if (n_active > s.max_atoms)
        throw std::runtime_error(
            "NeighborList::set_active_count: " + std::to_string(n_active)
            + " exceeds build capacity " + std::to_string(s.max_atoms));
    s.num_atoms = n_active;
    s.num_clusters = (n_active + kClusterSize - 1) / kClusterSize;
    s.first_launch = true; // the atom SET changed — skin-gate cannot be trusted
}

// ============================================================================
// launch() — per-step GPU work (pure GPU, zero D2H/sync/malloc)
// ============================================================================

void NeighborList::launch(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;

    // Re-publish the pair list (address is stable across the run). num_clusters
    // can change between intervals via set_active_count(), so republish it too
    // (the force kernel sizes its grid from pairs_.num_clusters).
    ctx.runtime.md.pair_list = &pairs_;

    int n = s.num_atoms;
    int nc = s.num_clusters;
    pairs_.num_clusters = nc;
    work_t r_search = cutoff_ + skin_;

    // ── Skin-gate decision ──────────────────────────────────────────
    //
    // On the first launch we always rebuild (no prior flag exists). On
    // subsequent launches the previous step's `launch_atom_displacement_max`
    // wrote the device flag; we issued an async D2H + event then. Sync
    // on that event (host wait — by the time the next step launches,
    // the GPU has typically completed the small int copy) and read the
    // pinned host int.
    //
    // 1-step lag: if a displacement at step N exceeded half_skin, the
    // flag arrives at step N+1 and we rebuild then. The skin width
    // absorbs the extra step of motion (standard LAMMPS/GROMACS
    // pattern). cuda::stream_synchronize is intentionally not used —
    // cudaEventSynchronize blocks only the host thread, not the GPU
    // stream, and the event has typically already fired by the next
    // launch() call.
    // Future: a fully event-driven pattern (`cudaStreamWaitEvent` on
    // next launch so the host thread never waits) is plausible but is
    // explicitly out of scope for the cleanup chain — link 03's
    // launch-purity contract is satisfied by the lagged-readback +
    // cudaEventSynchronize pattern here.
    bool rebuild_needed;
    if (s.first_launch) {
        rebuild_needed = true;
        s.first_launch = false;
    } else if (rebuild_on_request_) {
        // Driver-owned cadence (multi-GPU): rebuild only when set_active_count
        // re-arms first_launch; never via the per-step skin auto-gate (which
        // false-fires on cross-GPU ghost rows). No displacement readback needed.
        rebuild_needed = false;
    } else {
        cudaEventSynchronize(s.rebuild_flag_event);
        rebuild_needed = (*s.h_rebuild_flag != 0);
    }

    if (rebuild_needed) {
        s.rebuild_count++;

        // Re-run cell/inv_cell/dims/stencil setup — gated on rebuild so
        // NVE/NVT (fixed box) skips entirely after the build() call and
        // NPT re-runs only when the box has drifted enough to invalidate
        // the cached neighbor list (which is what triggered the rebuild).
        launch_setup_cell_device(
            frame.box, s.d_cell.data(), s.d_inv_cell.data(), s.d_dims.data(),
            s.stencil_dx.data(), s.stencil_dy.data(), s.stencil_dz.data(),
            s.d_n_stencil.data(), s.d_ncells.data(), r_search, s.periodic,
            kMaxStencil, stream);

        // ── LAMMPS-style direct cell-bin FULL per-atom neighbor list ──────
        // One pass: bin atoms into cells, then one thread per atom scans the
        // stencil cells and appends every neighbor within (cutoff+skin) into
        // the per-atom Verlet list. Replaces the cluster-CSR machinery
        // (Morton sort → scatter+AABB → grid CSR → count/fill → csr_to_flat →
        // build_verlet_list), which spent ~65% of GPU time building a
        // redundant second neighbor structure. Both bins and the Verlet list
        // auto-grow on overflow (LAMMPS resize-and-rebuild).
        int ncells = s.dims_x * s.dims_y * s.dims_z;
        if (s.buf.bin_ncells != ncells)
            s.buf.resize_bins(ncells, bin_cap_for(n, ncells));
        for (int attempt = 0; attempt < 6; ++attempt) {
            cudaMemsetAsync(s.buf.atom_bincount.data(), 0, ncells * sizeof(int),
                            stream);
            cudaMemsetAsync(s.buf.verlet_overflow.data(), 0, sizeof(int),
                            stream); // reused as bin-overflow probe
            launch_bin_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z, n,
                             ctx.runtime.host_box, s.dims_x, s.dims_y, s.dims_z,
                             s.periodic, s.buf.atom_bincount.data(),
                             s.buf.atom_bin_pos.data(), s.buf.bin_cap,
                             s.buf.verlet_overflow.data(), stream);
            int bin_max = 0;
            cudaMemcpyAsync(&bin_max, s.buf.verlet_overflow.data(), sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (bin_max <= s.buf.bin_cap)
                break;
            s.buf.resize_bins(ncells, bin_cap_grow(bin_max));
        }
        // LAMMPS-style special-bonds baking (find_special at build time): when
        // full exclusion is active (s_excl == 0, published by PairLJCut::build)
        // and the per-atom special SoA is on-device, drop 1-2/1-3/1-4 partners
        // straight from the neighbor list — the force kernel then needs zero
        // per-pair exclusion work AND we avoid rebuilding any host-side
        // exclusion CSR (the dominant multi-GPU per-reneighbor cost). Scaled
        // exclusions (s_excl != 0) fall back to the force-kernel scan.
        const bool bake_excl =
            (frame.atoms.special != nullptr && ctx.runtime.md.s_excl == 0.0);
        const std::int64_t* bake_tag = bake_excl ? frame.atoms.tag : nullptr;
        const int* bake_nsp = bake_excl ? frame.atoms.nspecial : nullptr;
        const std::int64_t* bake_sp = bake_excl ? frame.atoms.special : nullptr;
        ctx.runtime.md.exclusions_baked = bake_excl;
        for (int attempt = 0; attempt < 6; ++attempt) {
            cudaMemsetAsync(s.buf.verlet_overflow.data(), 0, sizeof(int),
                            stream);
            launch_build_verlet_direct<work_t>(
                frame.atoms.x, frame.atoms.y, frame.atoms.z, n,
                ctx.runtime.host_box, s.dims_x, s.dims_y, s.dims_z, s.periodic,
                s.stencil_dx.data(), s.stencil_dy.data(), s.stencil_dz.data(),
                s.d_n_stencil.data(), s.buf.atom_bincount.data(),
                s.buf.atom_bin_pos.data(), s.buf.bin_cap, s.cut_skin_sq,
                bake_tag, bake_nsp, bake_sp, frame.atoms.maxspecial,
                s.buf.verlet_nbr.data(), s.buf.verlet_cnt.data(),
                s.buf.verlet_stride, s.buf.verlet_max,
                s.buf.verlet_overflow.data(), stream);
            int v_max = 0;
            cudaMemcpyAsync(&v_max, s.buf.verlet_overflow.data(), sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (v_max <= s.buf.verlet_max)
                break;
            s.buf.resize_verlet(s.buf.verlet_stride,
                                static_cast<int>(v_max * 1.3) + 32);
            pairs_.vl_nbr = s.buf.verlet_nbr.data();
            pairs_.vl_cnt = s.buf.verlet_cnt.data();
            pairs_.vl_max = s.buf.verlet_max;
        }

        // Snapshot current per-atom positions as the new reference for the
        // next step's Verlet-skin displacement check.
        launch_snapshot_ref_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z,
                                  s.buf.ref_x.data(), s.buf.ref_y.data(),
                                  s.buf.ref_z.data(), n, stream);

        // Cluster CSR + flat (i,j) pair list — ONLY for the opt-in flat-pair
        // export. The MD force path uses the per-atom Verlet list built
        // above; this whole block is dead for it. No in-tree component
        // requests the flat list today (the RDF observer streams positions
        // into the molrs accumulator instead — arch-refactor-07).
        if (ctx.runtime.md.flat_pairs_requested) {
            // Lazily allocate the cluster-CSR pipeline if build() skipped it
            // (flat pairs were requested only after that build — a
            // requester's build() can run later). col_idx/mask/flat then grow below via the
            // total_cluster_pairs path (pair_capacity is 1); the CUB temp
            // DeviceVectors self-size in their launch_* wrappers.
            if (!s.buf.cluster_ready(n)) {
                s.buf.resize_cluster_csr(n, nc);
                s.buf.resize_displacement(nc);
                s.buf.resize_grid(ncells);
                // Re-publish the cluster pointers build() left null (it ran with
                // flat off); col_idx/mask are re-published again on the grow.
                pairs_.cl_row_ptr = s.buf.row_ptr.data();
                pairs_.cl_col_idx = s.buf.col_idx.data();
                pairs_.cl_orig_index = s.buf.original_index.data();
                pairs_.cl_mask = s.buf.flat_mask.data();
            }
            launch_compute_morton<work_t>(frame.atoms.x, frame.atoms.y,
                                          frame.atoms.z, n, s.d_inv_cell.data(),
                                          s.periodic, s.buf.morton_codes.data(),
                                          s.buf.atom_indices.data(), stream);
            launch_radix_sort(
                s.buf.morton_codes.data(), s.buf.morton_sorted.data(),
                s.buf.atom_indices.data(), s.buf.indices_sorted.data(), n,
                s.buf.sort_temp, stream);
            launch_scatter_and_aabb<work_t>(
                s.buf.indices_sorted.data(), frame.atoms.x, frame.atoms.y,
                frame.atoms.z, n, nc, s.buf.cluster_x.data(),
                s.buf.cluster_y.data(), s.buf.cluster_z.data(),
                s.buf.original_index.data(), s.buf.cluster_count.data(),
                s.buf.aabb_min_x.data(), s.buf.aabb_min_y.data(),
                s.buf.aabb_min_z.data(), s.buf.aabb_max_x.data(),
                s.buf.aabb_max_y.data(), s.buf.aabb_max_z.data(), stream);
            launch_assign_clusters_to_cells<work_t>(
                s.buf.aabb_min_x.data(), s.buf.aabb_min_y.data(),
                s.buf.aabb_min_z.data(), s.buf.aabb_max_x.data(),
                s.buf.aabb_max_y.data(), s.buf.aabb_max_z.data(), nc,
                s.d_inv_cell.data(), s.dims_x, s.dims_y, s.dims_z, s.periodic,
                s.buf.cluster_cell_idx.data(), stream);
            launch_build_grid_csr(
                s.buf.cluster_cell_idx.data(), nc, ncells,
                s.buf.cell_start.data(), s.buf.cell_clusters.data(),
                s.buf.histogram.data(), s.buf.scan_temp, stream);
            launch_count_cluster_pairs<work_t>(
                s.buf.cell_start.data(), s.buf.cell_clusters.data(),
                s.buf.cluster_cell_idx.data(), s.dims_x, s.dims_y, s.dims_z,
                s.stencil_dx.data(), s.stencil_dy.data(), s.stencil_dz.data(),
                s.d_n_stencil.data(), s.buf.aabb_min_x.data(),
                s.buf.aabb_min_y.data(), s.buf.aabb_min_z.data(),
                s.buf.aabb_max_x.data(), s.buf.aabb_max_y.data(),
                s.buf.aabb_max_z.data(), nc, s.periodic, s.cut_skin_sq,
                ctx.runtime.host_box, s.buf.pair_counts.data(), stream);
            launch_exclusive_sum(s.buf.pair_counts.data(),
                                 s.buf.row_ptr_upper.data(), nc,
                                 s.buf.prefix_temp, stream);
            launch_write_row_ptr_sentinel(s.buf.row_ptr_upper.data(),
                                          s.buf.pair_counts.data(), nc, stream);
            int total_cluster_pairs = 0;
            cudaMemcpy(&total_cluster_pairs, s.buf.row_ptr_upper.data() + nc,
                       sizeof(int), cudaMemcpyDeviceToHost);
            if (total_cluster_pairs > s.pair_capacity) {
                long long want =
                    static_cast<long long>(total_cluster_pairs) * 13 / 10
                    + 4096;
                int new_cap = static_cast<int>(std::min(
                    want,
                    static_cast<long long>(std::numeric_limits<int>::max())));
                s.pair_capacity = new_cap;
                s.buf.col_idx.resize(new_cap);
                s.buf.interaction_mask.resize(new_cap);
                s.buf.resize_flat_pairs(new_cap);
                pairs_.cl_col_idx = s.buf.col_idx.data();
                pairs_.cl_mask = s.buf.flat_mask.data();
            }
            launch_fill_cluster_pairs<work_t>(
                s.buf.cell_start.data(), s.buf.cell_clusters.data(),
                s.buf.cluster_cell_idx.data(), s.dims_x, s.dims_y, s.dims_z,
                s.stencil_dx.data(), s.stencil_dy.data(), s.stencil_dz.data(),
                s.d_n_stencil.data(), s.buf.aabb_min_x.data(),
                s.buf.aabb_min_y.data(), s.buf.aabb_min_z.data(),
                s.buf.aabb_max_x.data(), s.buf.aabb_max_y.data(),
                s.buf.aabb_max_z.data(), nc, s.periodic, s.cut_skin_sq,
                ctx.runtime.host_box, s.buf.row_ptr_upper.data(),
                s.buf.col_idx.data(), s.buf.pair_counts.data(), stream);
            launch_exclusive_sum(s.buf.pair_counts.data(), s.buf.row_ptr.data(),
                                 nc, s.buf.prefix_temp, stream);
            launch_write_row_ptr_sentinel(s.buf.row_ptr.data(),
                                          s.buf.pair_counts.data(), nc, stream);
            launch_compact_pairs(
                s.buf.col_idx.data(), s.buf.row_ptr_upper.data(),
                s.buf.row_ptr.data(), s.buf.pair_counts.data(), nc, stream);
            int mask_size = std::max(s.pair_capacity, 1);
            cudaMemsetAsync(s.buf.interaction_mask.data(), 0xFF,
                            mask_size * sizeof(uint64_t), stream);
            cudaMemcpyAsync(s.buf.flat_pair_counter.data(),
                            s.buf.row_ptr.data() + nc, sizeof(int),
                            cudaMemcpyDeviceToDevice, stream);
            launch_csr_to_flat(s.buf.row_ptr.data(), s.buf.col_idx.data(),
                               s.buf.interaction_mask.data(), nc,
                               s.buf.flat_ci.data(), s.buf.flat_cj.data(),
                               s.buf.flat_mask.data(), stream);
        }
    }

    // ── Per-step prune + flat-pair export: OPT-IN ───────────────────────
    // The MD force kernel (PairLJCut) consumes the per-atom Verlet list and
    // computes distances from CURRENT positions, so the flat (i,j,dx,dy,dz,r²)
    // PairList is dead work for force evaluation — skipped by default (this
    // prune + export were ~60% of the per-step time). A flat-list consumer
    // would set ctx.runtime.md.flat_pairs_requested in build() to re-enable
    // them; none exists in-tree today.
    if (!ctx.runtime.md.flat_pairs_requested) {
        cudaMemsetAsync(pairs_.n_pairs, 0, sizeof(int), stream);
    } else {
        // Lazy one-time allocation of the flat export arrays (see build()): a
        // flat consumer requested them, possibly after build() ran.
        if (pairs_.i == nullptr) {
            int cap = std::max(1, s.atom_pair_cap);
            cudaMalloc(&pairs_.i, cap * sizeof(int));
            cudaMalloc(&pairs_.j, cap * sizeof(int));
            cudaMalloc(&pairs_.dx, cap * sizeof(real_t));
            cudaMalloc(&pairs_.dy, cap * sizeof(real_t));
            cudaMalloc(&pairs_.dz, cap * sizeof(real_t));
            cudaMalloc(&pairs_.dist_sq, cap * sizeof(real_t));
        }
        int prune_grid_pairs = (s.last_flat_pair_count > 0)
                                   ? s.last_flat_pair_count
                                   : s.pair_capacity;
        launch_smart_rolling_prune<work_t>(
            s.buf.cluster_x.data(), s.buf.cluster_y.data(),
            s.buf.cluster_z.data(), s.buf.flat_ci.data(), s.buf.flat_cj.data(),
            s.buf.flat_pair_counter.data(), prune_grid_pairs,
            s.buf.dirty_flags.data(), s.cutoff_sq, ctx.runtime.host_box,
            s.buf.flat_mask.data(), /*prune_step*/ 0, /*prune_period*/ 1,
            stream);

        int cap = s.pair_capacity;
        if (cap > 0) {
            cudaMemsetAsync(s.d_pair_count.data(), 0, sizeof(int), stream);
            int export_grid_pairs =
                (s.last_flat_pair_count > 0) ? s.last_flat_pair_count : cap;
            launch_export_atom_pairs(
                s.buf.flat_ci.data(), s.buf.flat_cj.data(),
                s.buf.flat_mask.data(), s.buf.flat_pair_counter.data(),
                export_grid_pairs, s.buf.original_index.data(), frame.atoms.x,
                frame.atoms.y, frame.atoms.z, s.periodic, ctx.runtime.host_box,
                pairs_.i, pairs_.j, pairs_.dx, pairs_.dy, pairs_.dz,
                pairs_.dist_sq, s.d_pair_count.data(), pairs_.capacity, stream);
            cudaMemcpyAsync(pairs_.n_pairs, s.d_pair_count.data(), sizeof(int),
                            cudaMemcpyDeviceToDevice, stream);

            // Lagged flat_pair_counter readback for next step's grid sizing.
            cudaMemcpyAsync(s.h_flat_pair_count, s.buf.flat_pair_counter.data(),
                            sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(s.flat_pair_count_event, stream);
            if (s.flat_pair_count_event_valid) {
                cudaEventSynchronize(s.flat_pair_count_event);
                s.last_flat_pair_count = *s.h_flat_pair_count;
            }
            s.flat_pair_count_event_valid = true;
        } else {
            cudaMemsetAsync(pairs_.n_pairs, 0, sizeof(int), stream);
        }
    }

    // ── Compute next step's rebuild flag ────────────────────────────
    // Clear device flag, run per-atom displacement vs ref_x/y/z, issue
    // async D2H of the result + event. The next launch() call will
    // synchronize on this event before reading the pinned host int.
    //
    // Driver-owned cadence (rebuild_on_request_, multi-GPU): the per-step skin
    // auto-gate is never consulted (the driver reneighbors on a fixed interval),
    // so computing the flag is pure waste — a memset + an all-atom displacement
    // kernel + a D2H + an event record EVERY step. At ~3M atoms/GPU this
    // displacement kernel dominated the per-step neighbor-launch host time
    // (~30 ms/step over the non-rebuild steps). Skip it entirely in that mode.
    if (rebuild_on_request_)
        return;
    cudaMemsetAsync(s.d_rebuild_flag.data(), 0, sizeof(int), stream);
    launch_atom_displacement_max<work_t>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, s.buf.ref_x.data(),
        s.buf.ref_y.data(), s.buf.ref_z.data(), n, s.half_skin_sq,
        s.d_rebuild_flag.data(), stream);
    cudaMemcpyAsync(s.h_rebuild_flag, s.d_rebuild_flag.data(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(s.rebuild_flag_event, stream);
}

// ============================================================================
// rebuild_issue() / finalize_rebuild() — multi-GPU concurrent reneighbor
// ============================================================================
//
// Split of the launch() rebuild into an ISSUE pass (no host sync) and a
// FINALIZE pass (after the driver's single cross-device barrier). The MD force
// path never requests the flat (i,j) export, so this is the Verlet-list build
// only — none of launch()'s `flat_pairs_requested` CSR machinery.

void NeighborList::rebuild_issue(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    ctx.runtime.md.pair_list = &pairs_;
    int n = s.num_atoms;
    pairs_.num_clusters = s.num_clusters;
    work_t r_search = cutoff_ + skin_;
    s.rebuild_count++;

    // Cell/stencil setup (box is fixed for NVE/NVT, so the host dims cached in
    // build() stay valid — no D2H read of dims needed here).
    launch_setup_cell_device(frame.box, s.d_cell.data(), s.d_inv_cell.data(),
                             s.d_dims.data(), s.stencil_dx.data(),
                             s.stencil_dy.data(), s.stencil_dz.data(),
                             s.d_n_stencil.data(), s.d_ncells.data(), r_search,
                             s.periodic, kMaxStencil, stream);
    int ncells = s.dims_x * s.dims_y * s.dims_z;
    s.rebuild_ncells = ncells;
    if (s.buf.bin_ncells != ncells)
        s.buf.resize_bins(ncells, bin_cap_for(n, ncells));

    // Bin (single pass, no sync) → bin overflow probe to its OWN pinned int.
    cudaMemsetAsync(s.buf.atom_bincount.data(), 0, ncells * sizeof(int),
                    stream);
    cudaMemsetAsync(s.d_bin_overflow.data(), 0, sizeof(int), stream);
    launch_bin_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z, n,
                     ctx.runtime.host_box, s.dims_x, s.dims_y, s.dims_z,
                     s.periodic, s.buf.atom_bincount.data(),
                     s.buf.atom_bin_pos.data(), s.buf.bin_cap,
                     s.d_bin_overflow.data(), stream);
    cudaMemcpyAsync(s.h_bin_overflow, s.d_bin_overflow.data(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    // Verlet build (single pass, no sync) → Verlet overflow probe to its pinned
    // int. Same special-bonds baking as launch().
    const bool bake_excl =
        (frame.atoms.special != nullptr && ctx.runtime.md.s_excl == 0.0);
    const std::int64_t* bake_tag = bake_excl ? frame.atoms.tag : nullptr;
    const int* bake_nsp = bake_excl ? frame.atoms.nspecial : nullptr;
    const std::int64_t* bake_sp = bake_excl ? frame.atoms.special : nullptr;
    ctx.runtime.md.exclusions_baked = bake_excl;
    cudaMemsetAsync(s.buf.verlet_overflow.data(), 0, sizeof(int), stream);
    launch_build_verlet_direct<work_t>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, n, ctx.runtime.host_box,
        s.dims_x, s.dims_y, s.dims_z, s.periodic, s.stencil_dx.data(),
        s.stencil_dy.data(), s.stencil_dz.data(), s.d_n_stencil.data(),
        s.buf.atom_bincount.data(), s.buf.atom_bin_pos.data(), s.buf.bin_cap,
        s.cut_skin_sq, bake_tag, bake_nsp, bake_sp, frame.atoms.maxspecial,
        s.buf.verlet_nbr.data(), s.buf.verlet_cnt.data(), s.buf.verlet_stride,
        s.buf.verlet_max, s.buf.verlet_overflow.data(), stream);
    cudaMemcpyAsync(s.h_vl_overflow, s.buf.verlet_overflow.data(), sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    // Republish the Verlet pointers (stable unless a prior interval resized) and
    // clear the flat-pair count — the force path consumes only the Verlet list.
    pairs_.vl_nbr = s.buf.verlet_nbr.data();
    pairs_.vl_cnt = s.buf.verlet_cnt.data();
    pairs_.vl_stride = s.buf.verlet_stride;
    pairs_.vl_max = s.buf.verlet_max;
    cudaMemsetAsync(pairs_.n_pairs, 0, sizeof(int), stream);

    // Rebuild done for the common (no-overflow) case → the next launch()
    // republishes only. finalize_rebuild re-arms first_launch iff it overflowed.
    s.first_launch = false;
}

bool NeighborList::finalize_rebuild(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    const int bo =
        *s.h_bin_overflow; // settled by the driver's cross-device barrier
    const int vo = *s.h_vl_overflow;
    bool grew = false;
    if (bo > s.buf.bin_cap) {
        s.buf.resize_bins(s.rebuild_ncells, bin_cap_grow(bo));
        grew = true;
    }
    if (vo > s.buf.verlet_max) {
        s.buf.resize_verlet(s.buf.verlet_stride,
                            static_cast<int>(vo * 1.3) + 32);
        pairs_.vl_nbr = s.buf.verlet_nbr.data();
        pairs_.vl_cnt = s.buf.verlet_cnt.data();
        pairs_.vl_max = s.buf.verlet_max;
        grew = true;
    }
    if (grew) {
        // Overflow (rare: first interval / density change). The Verlet list from
        // rebuild_issue may be truncated — re-arm a synchronous rebuild so the
        // following launch() rebuilds THIS device correctly (and only this one).
        s.first_launch = true;
        return true;
    }
    // Snapshot the skin reference (kept for parity even though the driver-owned
    // cadence never consults the skin gate).
    launch_snapshot_ref_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z,
                              s.buf.ref_x.data(), s.buf.ref_y.data(),
                              s.buf.ref_z.data(), s.num_atoms, stream);
    return false;
}

// ============================================================================
// destroy() — release build-time resources
// ============================================================================

void NeighborList::destroy(const DestroyContext& ctx) {
    // Free the owned PairList (allocated in build)
    cudaFree(pairs_.i);
    cudaFree(pairs_.j);
    cudaFree(pairs_.dx);
    cudaFree(pairs_.dy);
    cudaFree(pairs_.dz);
    cudaFree(pairs_.dist_sq);
    cudaFree(pairs_.n_pairs);
    cudaFree(pairs_.capacity);
    pairs_ = {};
    ctx.runtime.md.pair_list = nullptr;

    // Free pinned host scratch + events (skin gate plumbing).
    if (state_) {
        auto& s = *state_;
        if (s.rebuild_flag_event) {
            cudaEventDestroy(s.rebuild_flag_event);
            s.rebuild_flag_event = nullptr;
        }
        if (s.flat_pair_count_event) {
            cudaEventDestroy(s.flat_pair_count_event);
            s.flat_pair_count_event = nullptr;
        }
        if (s.h_rebuild_flag) {
            cudaFreeHost(s.h_rebuild_flag);
            s.h_rebuild_flag = nullptr;
        }
        if (s.h_flat_pair_count) {
            cudaFreeHost(s.h_flat_pair_count);
            s.h_flat_pair_count = nullptr;
        }
        if (s.h_bin_overflow) {
            cudaFreeHost(s.h_bin_overflow);
            s.h_bin_overflow = nullptr;
        }
        if (s.h_vl_overflow) {
            cudaFreeHost(s.h_vl_overflow);
            s.h_vl_overflow = nullptr;
        }
    }

    // Free internal state (DeviceVectors auto-free in State destructor)
    state_.reset();
}

// ============================================================================
// Constructor / move / introspection
// ============================================================================

NeighborList::NeighborList(double cutoff, double skin)
    : cutoff_(cutoff), skin_(skin < 0.0 ? 0.3 * cutoff : skin) {}

NeighborList::~NeighborList() = default;
NeighborList::NeighborList(NeighborList&&) noexcept = default;
NeighborList& NeighborList::operator=(NeighborList&&) noexcept = default;

int NeighborList::rebuild_count() const noexcept {
    return state_ ? state_->rebuild_count : 0;
}

} // namespace cuda
} // namespace ATV
