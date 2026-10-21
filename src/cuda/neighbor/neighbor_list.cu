/// @file cuda/neighbor/neighbor_list.cu
/// @brief NeighborList Component — build/launch/destroy lifecycle with
///        Verlet-skin rebuild gating.
///
/// build():   allocates PairList + internal buffers + pinned host scratch
///            + cudaEvents; pre-sizes CUB sort/scan temps so launch() never
///            cudaMalloc's via DeviceVector::resize.
/// launch(): checks current periodic displacements before force evaluation;
///            rebuilds Hilbert clusters and exact CSR when skin is exhausted.
///            Mapped per-block flags avoid a copy-engine transfer and clear.
/// destroy(): frees PairList, pinned scratch and events.

#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "nlist_impl.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

using namespace ATV::cuda::md::nlist;

namespace ATV {
namespace cuda {

namespace {
// Process-wide publication IDs survive provider replacement and allocator
// address reuse. Zero denotes a list whose masks have not been built.
std::atomic<std::uint64_t> next_rebuild_generation{1};
} // namespace

// ============================================================================
// State
// ============================================================================

struct NeighborList::State {
    NListBuffers<real_t> buf;

    DeviceVector<real_t> d_inv_cell{9};
    DeviceVector<int> d_pair_count{1};

    int num_atoms = 0;
    int num_clusters = 0;
    int max_atoms = 0; // build-time atom capacity (buffers sized for this)
    int pair_capacity = 0;
    int atom_pair_cap = 0; // flat atom-pair export capacity, fixed in build
    bool flat_export = false;
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

    // One mapped flag per displacement block; GPU overwrites every live
    // entry before the host reads it after the event. Allocated only in build.
    int* h_rebuild_flag = nullptr;
    int* mapped_rebuild_flags = nullptr;
    Box reference_box{};
    int* h_flat_pair_count = nullptr;
    int* h_cluster_pair_count = nullptr;
    cudaEvent_t rebuild_flag_event = nullptr;
    cudaEvent_t flat_pair_count_event = nullptr;
    bool flat_pair_count_event_valid = false;
    ~State() {
        if (h_cluster_pair_count)
            cudaFreeHost(h_cluster_pair_count);
    }
};

// ============================================================================
// build() — one-time resource preparation
// ============================================================================

void NeighborList::build(const BuildContext& ctx) {
    state_ = std::make_unique<State>();
    auto& s = *state_;
    ATV_CUDA_CHECK(cudaMallocHost(&s.h_cluster_pair_count, sizeof(int)));
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

    // Hilbert keys need the inverse cell; the hierarchy needs no cell grid,
    // atom-sized stencil, or dimensions readback.
    launch_update_inverse_cell(frame.box, s.d_inv_cell.data(), stream);
    const double box_vol = double(ctx.runtime.host_box.volume());

    // The full directed CSR has at most nc*nc entries. The default reserve
    // combines a density estimate with the measured initial hierarchy count
    // below; an explicit constructor reserve keeps its exact override meaning.
    const double kPi = 3.141592653589793;
    long long cluster_cap_ll = static_cast<long long>(nc) * 128;
    if (box_vol > 0.0 && n > 0) {
        double density = double(n) / box_vol;
        double rs = double(r_search);
        double sphere = (4.0 / 3.0) * kPi * rs * rs * rs;
        double neigh_clusters = density * sphere / double(kClusterSize);
        double per_cluster = std::max(128.0, 8.0 * neigh_clusters);
        cluster_cap_ll = std::max(
            cluster_cap_ll, static_cast<long long>(double(nc) * per_cluster));
    }
    cluster_cap_ll =
        std::min(cluster_cap_ll,
                 static_cast<long long>(std::numeric_limits<int>::max()));
    const long long dense_limit = static_cast<long long>(nc) * nc;
    cluster_cap_ll = std::min(cluster_cap_ll, dense_limit);
    if (cluster_pair_capacity_ > 0)
        cluster_cap_ll = std::min(
            dense_limit, static_cast<long long>(cluster_pair_capacity_));
    s.pair_capacity = static_cast<int>(cluster_cap_ll);
    s.flat_export = ctx.runtime.md.flat_pairs_requested;

    // Always-needed per-atom buffers (Verlet-skin reference positions).
    s.buf.resize_atoms(n);

    // OpenMM cluster pipeline is the CUDA neighbor source of truth.
    s.buf.resize_cluster_csr(n, nc);
    s.buf.resize_displacement(nc);
    CubTempBytes cub_bytes = query_cub_temp_bytes(n, nc);
    if (cub_bytes.sort_bytes > 0)
        s.buf.sort_temp.resize(cub_bytes.sort_bytes);
    {
        std::size_t scan_max =
            std::max(cub_bytes.scan_bytes_clusters, cub_bytes.scan_bytes_atoms);
        if (scan_max > 0) {
            s.buf.scan_temp.resize(scan_max);
            s.buf.prefix_temp.resize(scan_max);
        }
    }
    // MultiGpu may temporarily set n_atoms() to its reservation size while
    // building scratch. Only n_all() owned+ghost rows contain coordinates;
    // never sort the uninitialized reservation tail when measuring occupancy.
    const int initial_atoms = frame.n_all();
    if (initial_atoms < 0 || initial_atoms > n)
        throw std::invalid_argument(
            "NeighborList: active owned+ghost count exceeds build reservation");
    const int initial_clusters =
        (initial_atoms + kClusterSize - 1) / kClusterSize;

    // Measure the initial hierarchy after all count/sort scratch is allocated.
    // Wide Hilbert clusters can overlap more groups than a homogeneous-density
    // estimate predicts. Reserve twice this exact count during build, while
    // keeping every subsequent rebuild allocation-free and overflow-checked.
    if (cluster_pair_capacity_ == 0 && initial_clusters > 0) {
        launch_compute_spatial_keys<work_t>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, initial_atoms,
            s.d_inv_cell.data(), s.periodic, s.buf.spatial_keys.data(),
            s.buf.atom_indices.data(), stream);
        launch_radix_sort(
            s.buf.spatial_keys.data(), s.buf.spatial_sorted.data(),
            s.buf.atom_indices.data(), s.buf.indices_sorted.data(),
            initial_atoms, s.buf.sort_temp, stream);
        launch_scatter_and_aabb<work_t>(
            s.buf.indices_sorted.data(), frame.atoms.x, frame.atoms.y,
            frame.atoms.z, initial_atoms, initial_clusters,
            s.buf.cluster_x.data(), s.buf.cluster_y.data(),
            s.buf.cluster_z.data(), s.buf.original_index.data(),
            s.buf.cluster_count.data(), s.buf.aabb_min_x.data(),
            s.buf.aabb_min_y.data(), s.buf.aabb_min_z.data(),
            s.buf.aabb_max_x.data(), s.buf.aabb_max_y.data(),
            s.buf.aabb_max_z.data(), stream);
        ClusterBounds bounds{s.buf.aabb_min_x.data(), s.buf.aabb_min_y.data(),
                             s.buf.aabb_min_z.data(), s.buf.aabb_max_x.data(),
                             s.buf.aabb_max_y.data(), s.buf.aabb_max_z.data()};
        launch_large_cluster_bounds(bounds, initial_clusters,
                                    s.buf.large_bounds.data(), stream);
        launch_count_cluster_pairs<work_t>(bounds, s.buf.large_bounds.data(),
                                           initial_clusters, s.periodic,
                                           s.cut_skin_sq, ctx.runtime.host_box,
                                           s.buf.pair_counts.data(), stream);
        launch_exclusive_sum(s.buf.pair_counts.data(), s.buf.row_ptr.data(),
                             initial_clusters, s.buf.prefix_temp, stream);
        launch_write_row_ptr_sentinel(s.buf.row_ptr.data(),
                                      s.buf.pair_counts.data(),
                                      initial_clusters, stream);
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            s.h_cluster_pair_count, s.buf.row_ptr.data() + initial_clusters,
            sizeof(int), cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        const int initial_count = *s.h_cluster_pair_count;
        if (initial_count < 0)
            throw std::overflow_error(
                "NeighborList: initial cluster CSR exceeds int indexing");
        const long long measured_reserve =
            static_cast<long long>(initial_count) * 2;
        s.pair_capacity = static_cast<int>(std::min(
            std::min(dense_limit,
                     static_cast<long long>(std::numeric_limits<int>::max())),
            std::max(cluster_cap_ll, measured_reserve)));
    }
    s.buf.resize_pairs(std::max(s.pair_capacity, 1));

    // Flat (i,j) export buffers only when a consumer requested them.
    if (ctx.runtime.md.flat_pairs_requested) {
        int pc = s.pair_capacity > 0 ? s.pair_capacity : 1;
        s.buf.resize_flat_pairs(pc);
    }

    // Optional atom-level export uses a fixed density-based reserve. Directed
    // cluster export may include the same atom pair twice, so reserve three
    // times the physical neighbor-shell estimate, with a per-atom floor.
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
    // Flat export is opt-in at build time. Late requests cannot allocate
    // method scratch during launch().
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

    const int flag_blocks = (s.max_atoms + 255) / 256;
    ATV_CUDA_CHECK(cudaHostAlloc(&s.h_rebuild_flag,
                                 std::max(1, flag_blocks) * sizeof(int),
                                 cudaHostAllocMapped));
    ATV_CUDA_CHECK(
        cudaHostGetDevicePointer(&s.mapped_rebuild_flags, s.h_rebuild_flag, 0));
    ATV_CUDA_CHECK(cudaMallocHost(&s.h_flat_pair_count, sizeof(int)));
    *s.h_flat_pair_count = 0;
    ATV_CUDA_CHECK(cudaEventCreateWithFlags(&s.rebuild_flag_event,
                                            cudaEventDisableTiming));
    ATV_CUDA_CHECK(cudaEventCreateWithFlags(&s.flat_pair_count_event,
                                            cudaEventDisableTiming));

    // Publish OpenMM cluster CSR (force source of truth). Contents filled on
    // first rebuild in launch(); addresses stable for the run.
    pairs_.cl_row_ptr = s.buf.row_ptr.data();
    pairs_.cl_col_idx = s.buf.col_idx.data();
    pairs_.cl_orig_index = s.buf.original_index.data();
    pairs_.cl_mask = s.buf.interaction_mask.data();
    pairs_.cl_x = s.buf.cluster_x.data();
    pairs_.cl_y = s.buf.cluster_y.data();
    pairs_.cl_z = s.buf.cluster_z.data();
    pairs_.num_clusters = nc;
    pairs_.cluster_pair_capacity = s.pair_capacity;
    pairs_.rebuild_generation = 0;

    // Publish the pair list to the execution context for downstream pair
    // mirroring cpu::md::NeighborList.
    ctx.runtime.md.pair_list = &pairs_;
    ctx.runtime.md.exclusions_baked = false;
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
// launch() — skin gate, GPU rebuild, optional flat export
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
    if (n > s.max_atoms)
        throw std::runtime_error(
            "NeighborList: active atoms exceed build capacity");
    if (ctx.runtime.md.flat_pairs_requested && !s.flat_export)
        throw std::runtime_error(
            "NeighborList: flat_pairs_requested must be set before build()");

    // The current positions must satisfy the skin bound before this force
    // pass. A previous-step flag leaves an unbounded one-step gap. Periodic
    // displacement avoids rebuilding merely because an atom crossed a face.
    bool rebuild_needed = s.first_launch;
    s.first_launch = false;
    for (int d = 0; d < 9; ++d)
        rebuild_needed |=
            s.reference_box.cell[d] != ctx.runtime.host_box.cell[d];
    if (!rebuild_needed && !rebuild_on_request_ && n > 0) {
        launch_atom_displacement_max<work_t>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, s.buf.ref_x.data(),
            s.buf.ref_y.data(), s.buf.ref_z.data(), n, s.half_skin_sq,
            ctx.runtime.host_box, s.mapped_rebuild_flags, stream);
        ATV_CUDA_CHECK(cudaEventRecord(s.rebuild_flag_event, stream));
        ATV_CUDA_CHECK(cudaEventSynchronize(s.rebuild_flag_event));
        const int blocks = (n + 255) / 256;
        for (int block = 0; block < blocks; ++block)
            rebuild_needed |= s.h_rebuild_flag[block] != 0;
    }

    if (rebuild_needed) {
        s.rebuild_count++;
        s.reference_box = ctx.runtime.host_box;

        launch_update_inverse_cell(frame.box, s.d_inv_cell.data(), stream);
        pairs_.cl_row_ptr = s.buf.row_ptr.data();
        pairs_.cl_col_idx = s.buf.col_idx.data();
        pairs_.cl_orig_index = s.buf.original_index.data();
        pairs_.cl_mask = s.buf.interaction_mask.data();
        pairs_.cl_x = s.buf.cluster_x.data();
        pairs_.cl_y = s.buf.cluster_y.data();
        pairs_.cl_z = s.buf.cluster_z.data();
        pairs_.num_clusters = nc;

        launch_compute_spatial_keys<work_t>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, n, s.d_inv_cell.data(),
            s.periodic, s.buf.spatial_keys.data(), s.buf.atom_indices.data(),
            stream);
        launch_radix_sort(
            s.buf.spatial_keys.data(), s.buf.spatial_sorted.data(),
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
        ClusterBounds bounds{s.buf.aabb_min_x.data(), s.buf.aabb_min_y.data(),
                             s.buf.aabb_min_z.data(), s.buf.aabb_max_x.data(),
                             s.buf.aabb_max_y.data(), s.buf.aabb_max_z.data()};
        launch_large_cluster_bounds(bounds, nc, s.buf.large_bounds.data(),
                                    stream);
        launch_count_cluster_pairs<work_t>(
            bounds, s.buf.large_bounds.data(), nc, s.periodic, s.cut_skin_sq,
            ctx.runtime.host_box, s.buf.pair_counts.data(), stream);
        launch_exclusive_sum(s.buf.pair_counts.data(), s.buf.row_ptr.data(), nc,
                             s.buf.prefix_temp, stream);
        launch_write_row_ptr_sentinel(s.buf.row_ptr.data(),
                                      s.buf.pair_counts.data(), nc, stream);
        ATV_CUDA_CHECK(cudaMemcpyAsync(s.h_cluster_pair_count,
                                       s.buf.row_ptr.data() + nc, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        const int total_cluster_pairs = *s.h_cluster_pair_count;
        pairs_.num_cluster_pairs = total_cluster_pairs;
        if (total_cluster_pairs > s.pair_capacity) {
            throw std::runtime_error(
                "NeighborList: cluster pair capacity "
                + std::to_string(s.pair_capacity)
                + " is too small; construct NeighborList(cutoff, skin, "
                + std::to_string(total_cluster_pairs)
                + ") or a larger reserve");
        }
        launch_fill_cluster_pairs<work_t>(
            bounds, s.buf.large_bounds.data(), nc, s.periodic, s.cut_skin_sq,
            ctx.runtime.host_box, s.buf.row_ptr.data(), s.buf.col_idx.data(),
            stream);
        launch_compute_csr_masks<work_t>(
            s.buf.cluster_x.data(), s.buf.cluster_y.data(),
            s.buf.cluster_z.data(), s.buf.row_ptr.data(), s.buf.col_idx.data(),
            nc, total_cluster_pairs, s.cut_skin_sq, ctx.runtime.host_box,
            s.buf.interaction_mask.data(), stream);
        ctx.runtime.md.exclusions_baked = launch_bake_csr_exclusions(
            s.buf.row_ptr.data(), s.buf.col_idx.data(),
            s.buf.original_index.data(), nc, total_cluster_pairs,
            ctx.runtime.md.excl_offsets, ctx.runtime.md.excl_partners,
            ctx.runtime.md.s_excl, s.buf.interaction_mask.data(), stream);
        pairs_.rebuild_generation =
            next_rebuild_generation.fetch_add(1, std::memory_order_relaxed);
        launch_snapshot_ref_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z,
                                  s.buf.ref_x.data(), s.buf.ref_y.data(),
                                  s.buf.ref_z.data(), n, stream);
    }

    // ── Per-step prune + flat-pair export: OPT-IN only ──────────────────
    // Default MD force uses cl_* CSR + masks; never csr_to_flat on this path.
    if (!ctx.runtime.md.flat_pairs_requested) {
        // Without a flat export allocation this counter is initialized once
        // in build and has no writer. A previously enabled flat consumer may
        // be disabled, in which case its old exported count must be cleared.
        if (s.flat_export)
            ATV_CUDA_CHECK(
                cudaMemsetAsync(pairs_.n_pairs, 0, sizeof(int), stream));
    } else {
        // Expand CSR → flat only when a flat consumer exists.
        launch_csr_to_flat(s.buf.row_ptr.data(), s.buf.col_idx.data(),
                           s.buf.interaction_mask.data(), nc,
                           s.buf.flat_ci.data(), s.buf.flat_cj.data(),
                           s.buf.flat_mask.data(), stream);
        // Publish pair count for prune (compact CSR length ≈ total pairs).
        int total_pairs_h = 0;
        cudaMemcpy(&total_pairs_h, s.buf.row_ptr.data() + nc, sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(s.buf.flat_pair_counter.data(), &total_pairs_h,
                        sizeof(int), cudaMemcpyHostToDevice, stream);

        int prune_grid_pairs = (s.last_flat_pair_count > 0)
                                   ? s.last_flat_pair_count
                                   : total_pairs_h;
        if (prune_grid_pairs > 0) {
            launch_smart_rolling_prune<work_t>(
                s.buf.cluster_x.data(), s.buf.cluster_y.data(),
                s.buf.cluster_z.data(), s.buf.flat_ci.data(),
                s.buf.flat_cj.data(), s.buf.flat_pair_counter.data(),
                prune_grid_pairs, s.buf.dirty_flags.data(), s.cutoff_sq,
                ctx.runtime.host_box, s.buf.flat_mask.data(),
                /*prune_step*/ 0, /*prune_period*/ 1, stream);

            cudaMemsetAsync(s.d_pair_count.data(), 0, sizeof(int), stream);
            launch_export_atom_pairs(
                s.buf.flat_ci.data(), s.buf.flat_cj.data(),
                s.buf.flat_mask.data(), s.buf.flat_pair_counter.data(),
                prune_grid_pairs, s.buf.original_index.data(), frame.atoms.x,
                frame.atoms.y, frame.atoms.z, s.periodic, ctx.runtime.host_box,
                pairs_.i, pairs_.j, pairs_.dx, pairs_.dy, pairs_.dz,
                pairs_.dist_sq, s.d_pair_count.data(), pairs_.capacity, stream);
            cudaMemcpyAsync(pairs_.n_pairs, s.d_pair_count.data(), sizeof(int),
                            cudaMemcpyDeviceToDevice, stream);

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
}

// ============================================================================
// rebuild_issue() / finalize_rebuild() — multi-GPU concurrent reneighbor
// ============================================================================
//
// OpenMM cluster rebuild (same contract as single-GPU launch()). Issue runs
// the full Hilbert→CSR→mask pipeline; capacity overflow is detected via the
// total pair count D2H before fill. Overflow throws with the required reserve;
// finalize snapshots the rebuilt positions without allocation.

void NeighborList::rebuild_issue(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    ctx.runtime.md.pair_list = &pairs_;
    int n = s.num_atoms;
    int nc = s.num_clusters;
    pairs_.num_clusters = nc;
    s.rebuild_count++;
    s.reference_box = ctx.runtime.host_box;

    launch_update_inverse_cell(frame.box, s.d_inv_cell.data(), stream);
    pairs_.cl_row_ptr = s.buf.row_ptr.data();
    pairs_.cl_col_idx = s.buf.col_idx.data();
    pairs_.cl_orig_index = s.buf.original_index.data();
    pairs_.cl_mask = s.buf.interaction_mask.data();
    pairs_.cl_x = s.buf.cluster_x.data();
    pairs_.cl_y = s.buf.cluster_y.data();
    pairs_.cl_z = s.buf.cluster_z.data();

    launch_compute_spatial_keys<work_t>(frame.atoms.x, frame.atoms.y,
                                        frame.atoms.z, n, s.d_inv_cell.data(),
                                        s.periodic, s.buf.spatial_keys.data(),
                                        s.buf.atom_indices.data(), stream);
    launch_radix_sort(s.buf.spatial_keys.data(), s.buf.spatial_sorted.data(),
                      s.buf.atom_indices.data(), s.buf.indices_sorted.data(), n,
                      s.buf.sort_temp, stream);
    launch_scatter_and_aabb<work_t>(
        s.buf.indices_sorted.data(), frame.atoms.x, frame.atoms.y,
        frame.atoms.z, n, nc, s.buf.cluster_x.data(), s.buf.cluster_y.data(),
        s.buf.cluster_z.data(), s.buf.original_index.data(),
        s.buf.cluster_count.data(), s.buf.aabb_min_x.data(),
        s.buf.aabb_min_y.data(), s.buf.aabb_min_z.data(),
        s.buf.aabb_max_x.data(), s.buf.aabb_max_y.data(),
        s.buf.aabb_max_z.data(), stream);
    ClusterBounds bounds{s.buf.aabb_min_x.data(), s.buf.aabb_min_y.data(),
                         s.buf.aabb_min_z.data(), s.buf.aabb_max_x.data(),
                         s.buf.aabb_max_y.data(), s.buf.aabb_max_z.data()};
    launch_large_cluster_bounds(bounds, nc, s.buf.large_bounds.data(), stream);
    launch_count_cluster_pairs<work_t>(
        bounds, s.buf.large_bounds.data(), nc, s.periodic, s.cut_skin_sq,
        ctx.runtime.host_box, s.buf.pair_counts.data(), stream);
    launch_exclusive_sum(s.buf.pair_counts.data(), s.buf.row_ptr.data(), nc,
                         s.buf.prefix_temp, stream);
    launch_write_row_ptr_sentinel(s.buf.row_ptr.data(),
                                  s.buf.pair_counts.data(), nc, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(s.h_cluster_pair_count,
                                   s.buf.row_ptr.data() + nc, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int total_cluster_pairs = *s.h_cluster_pair_count;
    pairs_.num_cluster_pairs = total_cluster_pairs;
    if (total_cluster_pairs > s.pair_capacity) {
        throw std::runtime_error(
            "NeighborList: cluster pair capacity "
            + std::to_string(s.pair_capacity)
            + " is too small; construct NeighborList(cutoff, skin, "
            + std::to_string(total_cluster_pairs) + ") or a larger reserve");
    }
    launch_fill_cluster_pairs<work_t>(
        bounds, s.buf.large_bounds.data(), nc, s.periodic, s.cut_skin_sq,
        ctx.runtime.host_box, s.buf.row_ptr.data(), s.buf.col_idx.data(),
        stream);
    launch_compute_csr_masks<work_t>(
        s.buf.cluster_x.data(), s.buf.cluster_y.data(), s.buf.cluster_z.data(),
        s.buf.row_ptr.data(), s.buf.col_idx.data(), nc, total_cluster_pairs,
        s.cut_skin_sq, ctx.runtime.host_box, s.buf.interaction_mask.data(),
        stream);
    ctx.runtime.md.exclusions_baked = launch_bake_csr_exclusions(
        s.buf.row_ptr.data(), s.buf.col_idx.data(), s.buf.original_index.data(),
        nc, total_cluster_pairs, ctx.runtime.md.excl_offsets,
        ctx.runtime.md.excl_partners, ctx.runtime.md.s_excl,
        s.buf.interaction_mask.data(), stream);

    pairs_.rebuild_generation =
        next_rebuild_generation.fetch_add(1, std::memory_order_relaxed);

    pairs_.cl_mask = s.buf.interaction_mask.data();
    cudaMemsetAsync(pairs_.n_pairs, 0, sizeof(int), stream);
    s.first_launch = false;
}

bool NeighborList::finalize_rebuild(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
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
    ctx.runtime.md.exclusions_baked = false;

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
    }

    // Free internal state (DeviceVectors auto-free in State destructor)
    state_.reset();
}

// ============================================================================
// Constructor / move / introspection
// ============================================================================

NeighborList::NeighborList(double cutoff, double skin,
                           int cluster_pair_capacity)
    : cutoff_(cutoff), skin_(skin < 0.0 ? 0.3 * cutoff : skin),
      cluster_pair_capacity_(cluster_pair_capacity) {
    if (!std::isfinite(cutoff_) || cutoff_ <= 0 || !std::isfinite(skin_)
        || skin_ < 0 || cluster_pair_capacity_ < 0)
        throw std::invalid_argument(
            "NeighborList: positive finite cutoff, finite nonnegative skin and "
            "nonnegative cluster pair capacity required");
}

NeighborList::~NeighborList() = default;
NeighborList::NeighborList(NeighborList&&) noexcept = default;
NeighborList& NeighborList::operator=(NeighborList&&) noexcept = default;

int NeighborList::rebuild_count() const noexcept {
    return state_ ? state_->rebuild_count : 0;
}

} // namespace cuda
} // namespace ATV
