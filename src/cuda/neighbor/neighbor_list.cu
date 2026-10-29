/// @file cuda/neighbor/neighbor_list.cu
/// @brief NeighborList Component — build/launch/destroy lifecycle with a
///        device-decided Verlet-skin rebuild.
///
/// build():   allocates PairList + internal buffers + device control words;
///            pre-sizes CUB sort/scan temps so no rebuild allocates.
/// launch(): one graph launch per step. A skin-gate kernel decides on the
///            device whether the current positions need a rebuild and sets
///            a conditional node whose body is the full Hilbert → CSR →
///            mask pipeline. No host readback (iron law 10).
/// destroy(): frees PairList, graph and buffers.

#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "nlist_impl.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/graph_ring.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include <cuda_runtime.h>

using namespace ATV::cuda::md::nlist;

namespace ATV {
namespace cuda {

namespace {
// Reference cell for the next skin gate and the narrowed inverse cell for the
// Hilbert keys, in one launch at the head of every rebuild.
__global__ void begin_rebuild_k(Box box, Box* reference, const geom_t* cell,
                                force_t* inverse) {
    *reference = box.current();
    geom_t inv[9];
    invert_3x3(cell, inv);
    for (int i = 0; i < 9; ++i)
        inverse[i] = static_cast<force_t>(inv[i]);
}

// Everything a captured graph bakes into its kernel arguments. A change in
// any field re-captures before the next launch.
struct GraphKey {
    int num_atoms = -1;
    int num_clusters = -1;
    const pos_t* x = nullptr;
    const pos_t* y = nullptr;
    const pos_t* z = nullptr;
    const geom_t* cell = nullptr;
    ForceFault* fault = nullptr;
    const Box* resident = nullptr;
    bool ortho = false;
    const int* excl_offsets = nullptr;
    const int* excl_partners = nullptr;
    double s_excl = 1.0;
    std::uint64_t hooks = 0;
    bool operator==(const GraphKey&) const = default;
};
} // namespace

// ============================================================================
// State
// ============================================================================

struct NeighborList::State {
    NListBuffers<force_t> buf;

    DeviceVector<force_t> d_inv_cell{9}; // narrowed from the FP64 cell
    DeviceVector<Control> control{1};
    DeviceVector<Box> reference_box{1};

    int num_atoms = 0;
    int num_clusters = 0;
    int max_atoms = 0; // build-time atom capacity (buffers sized for this)
    int pair_capacity = 0;
    bool periodic = true;

    // Cached force-path scalars (computed once in build())
    force_t cut_skin_sq = force_t(0);
    force_t cutoff_sq = force_t(0);
    force_t half_skin_sq =
        force_t(0); // (skin/2)² — Verlet-skin trigger threshold

    bool first_launch = true;
    std::uint64_t epoch = 0; // bumps when the active set changes

    // Step graph: skin gate → IF(rebuild). Captured on a private stream.
    cudaStream_t capture_stream = nullptr;
    GraphRing step_graph;
    GraphKey key;
    bool exclusions_baked = false;

    // Build-time measurement and the multi-GPU host-cadence rebuild only.
    int* h_cluster_pair_count = nullptr;

    void release_graph() {
        step_graph.reset();
        key = {};
    }
    ~State() {
        release_graph();
        if (capture_stream)
            cudaStreamDestroy(capture_stream);
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
    ATV_CUDA_CHECK(
        cudaStreamCreateWithFlags(&s.capture_stream, cudaStreamNonBlocking));
    ATV_CUDA_CHECK(cudaMemset(s.control.data(), 0, sizeof(Control)));
    ATV_CUDA_CHECK(cudaMemset(s.reference_box.data(), 0, sizeof(Box)));
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;
    s.num_atoms = ctx.frame.n_atoms();
    int n = s.num_atoms;
    int nc = (n + kClusterSize - 1) / kClusterSize;
    s.num_clusters = nc;
    s.max_atoms = n; // all buffers below are sized for this atom count
    s.periodic = true;

    force_t r_search = force_t(cutoff_ + skin_);

    // Cache squared cutoffs so launch() doesn't recompute every step.
    s.cut_skin_sq = r_search * r_search;
    s.cutoff_sq = force_t(cutoff_) * force_t(cutoff_);
    force_t half_skin = force_t(skin_) * force_t(0.5);
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

    // Always-needed per-atom buffers (Verlet-skin reference positions).
    s.buf.resize_atoms(n);

    // OpenMM cluster pipeline is the CUDA neighbor source of truth.
    s.buf.resize_cluster_csr(n, nc);
    CubTempBytes cub_bytes = query_cub_temp_bytes(n, nc);
    if (cub_bytes.sort_bytes > 0)
        s.buf.sort_temp.resize(cub_bytes.sort_bytes);
    {
        std::size_t scan_max =
            std::max(cub_bytes.scan_bytes_clusters, cub_bytes.scan_bytes_atoms);
        if (scan_max > 0)
            s.buf.prefix_temp.resize(scan_max);
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
    // Build-time synchronization is allowed; the run loop never reads back.
    if (cluster_pair_capacity_ == 0 && initial_clusters > 0) {
        launch_compute_spatial_keys<force_t>(
            frame.atoms.x, frame.atoms.y, frame.atoms.z, initial_atoms,
            s.d_inv_cell.data(), s.periodic, s.buf.spatial_keys.data(),
            s.buf.atom_indices.data(), stream);
        launch_radix_sort(
            s.buf.spatial_keys.data(), s.buf.spatial_sorted.data(),
            s.buf.atom_indices.data(), s.buf.indices_sorted.data(),
            initial_atoms, s.buf.sort_temp, stream);
        launch_scatter_and_aabb<force_t>(
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
        launch_count_cluster_pairs<force_t>(bounds, s.buf.large_bounds.data(),
                                            initial_clusters, s.periodic,
                                            s.cut_skin_sq, ctx.runtime.host_box,
                                            s.buf.pair_counts.data(), stream);
        launch_exclusive_sum(s.buf.pair_counts.data(), s.buf.row_ptr.data(),
                             initial_clusters, s.buf.prefix_temp, stream);
        launch_close_csr(s.buf.row_ptr.data(), s.buf.pair_counts.data(),
                         initial_clusters, std::numeric_limits<int>::max(),
                         &s.control.data()->overflow, nullptr, stream);
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
    pairs_.rebuilt = &s.control.data()->rebuilt;
    pairs_.generation = &s.control.data()->generation;
    pairs_.hooks = &hooks_;
    ++s.epoch;

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
void NeighborList::invalidate_inputs(InputChange changes,
                                     const BuildContext& ctx) {
    if (has_input_change(changes, InputChange::Geometry)
        || has_input_change(changes, InputChange::Cell))
        set_active_count(ctx.frame.n_atoms());
}

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
    ++s.epoch;
}

// ============================================================================
// Rebuild body — enqueued into the conditional graph body (and, for the
// multi-GPU host cadence, eagerly). Same pipeline either way.
// ============================================================================

namespace {
struct RebuildArgs {
    NListBuffers<force_t>* buf;
    force_t* inv_cell;
    Control* control;
    Box* reference_box;
    int n;
    int nc;
    int capacity;
    bool periodic;
    force_t cut_skin_sq;
};

void enqueue_hierarchy(const RebuildArgs& a, const LaunchContext& ctx,
                       cudaStream_t stream) {
    auto& b = *a.buf;
    auto& frame = ctx.frame;
    begin_rebuild_k<<<1, 1, 0, stream>>>(ctx.runtime.host_box, a.reference_box,
                                         frame.box, a.inv_cell);
    ATV_CUDA_CHECK(cudaGetLastError());
    launch_compute_spatial_keys<force_t>(
        frame.atoms.x, frame.atoms.y, frame.atoms.z, a.n, a.inv_cell,
        a.periodic, b.spatial_keys.data(), b.atom_indices.data(), stream);
    launch_radix_sort(b.spatial_keys.data(), b.spatial_sorted.data(),
                      b.atom_indices.data(), b.indices_sorted.data(), a.n,
                      b.sort_temp, stream);
    launch_scatter_and_aabb<force_t>(
        b.indices_sorted.data(), frame.atoms.x, frame.atoms.y, frame.atoms.z,
        a.n, a.nc, b.cluster_x.data(), b.cluster_y.data(), b.cluster_z.data(),
        b.original_index.data(), b.cluster_count.data(), b.aabb_min_x.data(),
        b.aabb_min_y.data(), b.aabb_min_z.data(), b.aabb_max_x.data(),
        b.aabb_max_y.data(), b.aabb_max_z.data(), stream);
    ClusterBounds bounds{b.aabb_min_x.data(), b.aabb_min_y.data(),
                         b.aabb_min_z.data(), b.aabb_max_x.data(),
                         b.aabb_max_y.data(), b.aabb_max_z.data()};
    launch_large_cluster_bounds(bounds, a.nc, b.large_bounds.data(), stream);
    launch_count_cluster_pairs<force_t>(
        bounds, b.large_bounds.data(), a.nc, a.periodic, a.cut_skin_sq,
        ctx.runtime.host_box, b.pair_counts.data(), stream);
    launch_exclusive_sum(b.pair_counts.data(), b.row_ptr.data(), a.nc,
                         b.prefix_temp, stream);
    launch_close_csr(b.row_ptr.data(), b.pair_counts.data(), a.nc, a.capacity,
                     &a.control->overflow, frame.force_fault, stream);
}

bool enqueue_csr(const RebuildArgs& a, const LaunchContext& ctx,
                 cudaStream_t stream) {
    auto& b = *a.buf;
    auto& frame = ctx.frame;
    ClusterBounds bounds{b.aabb_min_x.data(), b.aabb_min_y.data(),
                         b.aabb_min_z.data(), b.aabb_max_x.data(),
                         b.aabb_max_y.data(), b.aabb_max_z.data()};
    launch_fill_cluster_pairs<force_t>(
        bounds, b.large_bounds.data(), a.nc, a.periodic, a.cut_skin_sq,
        ctx.runtime.host_box, b.row_ptr.data(), b.col_idx.data(),
        &a.control->overflow, stream);
    launch_compute_csr_masks<force_t>(
        b.cluster_x.data(), b.cluster_y.data(), b.cluster_z.data(),
        b.row_ptr.data(), b.col_idx.data(), a.nc, a.capacity, a.cut_skin_sq,
        ctx.runtime.host_box, b.interaction_mask.data(), stream);
    const bool baked = launch_bake_csr_exclusions(
        b.row_ptr.data(), b.col_idx.data(), b.original_index.data(), a.nc,
        ctx.runtime.md.excl_offsets, ctx.runtime.md.excl_partners,
        ctx.runtime.md.s_excl, b.interaction_mask.data(), stream);
    launch_snapshot_ref_atoms(frame.atoms.x, frame.atoms.y, frame.atoms.z,
                              b.ref_x.data(), b.ref_y.data(), b.ref_z.data(),
                              a.n, stream);
    return baked;
}

void run_hooks(const RebuildHooks& hooks, cudaStream_t stream) {
    for (const auto& hook : hooks.list)
        hook.enqueue(stream);
}

void check_graph(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("NeighborList: ") + what + ": "
                                 + cudaGetErrorString(e));
}
} // namespace

// ============================================================================
// launch() — one graph launch: device skin gate + conditional rebuild
// ============================================================================

void NeighborList::launch(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    auto& frame = ctx.frame;

    // Re-publish the pair list (address is stable across the run). num_clusters
    // can change between intervals via set_active_count(), so republish it too
    // (the force kernel sizes its grid from pairs_.num_clusters).
    ctx.runtime.md.pair_list = &pairs_;
    const int n = s.num_atoms;
    const int nc = s.num_clusters;
    pairs_.num_clusters = nc;
    if (n > s.max_atoms)
        throw std::runtime_error(
            "NeighborList: active atoms exceed build capacity");

    // Multi-GPU owns the cadence: rebuilds are issued explicitly; a launch
    // only republishes (a set_active_count without an issued rebuild still
    // rebuilds once, on that loop's host path).
    if (rebuild_on_request_) {
        if (s.first_launch) {
            rebuild_issue(ctx);
            finalize_rebuild(ctx);
        }
        return;
    }

    const RebuildArgs args{&s.buf,
                           s.d_inv_cell.data(),
                           s.control.data(),
                           s.reference_box.data(),
                           n,
                           nc,
                           s.pair_capacity,
                           s.periodic,
                           s.cut_skin_sq};
    // Inside a step segment the gate and the conditional rebuild join the
    // step graph: one launch for the whole step instead of a graph of our
    // own (each separately launched conditional graph costs ~16 us of idle
    // device time on GH200). The forced first rebuild never gets here
    // (graph_capturable() is false until it ran).
    if (graph_capture::active(stream)) {
        graph_capture::splice_if(
            stream, s.capture_stream,
            [&](cudaGraphConditionalHandle handle) {
                launch_skin_gate<force_t>(
                    frame.atoms.x, frame.atoms.y, frame.atoms.z,
                    s.buf.ref_x.data(), s.buf.ref_y.data(), s.buf.ref_z.data(),
                    n, s.half_skin_sq, ctx.runtime.host_box,
                    s.reference_box.data(), s.control.data(), handle, stream);
            },
            [&](cudaStream_t body) {
                enqueue_hierarchy(args, ctx, body);
                s.exclusions_baked = enqueue_csr(args, ctx, body);
                run_hooks(hooks_, body);
            });
        ctx.runtime.md.exclusions_baked = s.exclusions_baked;
        return;
    }

    const GraphKey key{n,
                       nc,
                       frame.atoms.x,
                       frame.atoms.y,
                       frame.atoms.z,
                       frame.box,
                       frame.force_fault,
                       ctx.runtime.host_box.resident,
                       ctx.runtime.host_box.ortho,
                       ctx.runtime.md.excl_offsets,
                       ctx.runtime.md.excl_partners,
                       ctx.runtime.md.s_excl,
                       hooks_.version};
    if (!s.step_graph.ready() || !(s.key == key)) {
        s.release_graph();
        const auto cs = s.capture_stream;
        constexpr auto mode = cudaStreamCaptureModeThreadLocal;
        // One capture per ring slot: a graph with a conditional node can be
        // instantiated only once.
        auto make = [&]() -> cudaGraph_t {
            cudaGraph_t graph = nullptr;
            check_graph(cudaGraphCreate(&graph, 0), "graph create");
            try {
                cudaGraphConditionalHandle handle;
                check_graph(cudaGraphConditionalHandleCreate(
                                &handle, graph, 0, cudaGraphCondAssignDefault),
                            "conditional handle");
                // Root: the skin gate, which sets the handle for this launch.
                check_graph(cudaStreamBeginCaptureToGraph(cs, graph, nullptr,
                                                          nullptr, 0, mode),
                            "capture gate");
                try {
                    launch_skin_gate<force_t>(
                        frame.atoms.x, frame.atoms.y, frame.atoms.z,
                        s.buf.ref_x.data(), s.buf.ref_y.data(),
                        s.buf.ref_z.data(), n, s.half_skin_sq,
                        ctx.runtime.host_box, s.reference_box.data(),
                        s.control.data(), handle, cs);
                } catch (...) {
                    cudaGraph_t discard = nullptr;
                    cudaStreamEndCapture(cs, &discard);
                    throw;
                }
                cudaGraph_t captured = nullptr;
                check_graph(cudaStreamEndCapture(cs, &captured), "end gate");
                std::size_t roots = 1;
                cudaGraphNode_t gate = nullptr;
                check_graph(cudaGraphGetRootNodes(graph, &gate, &roots),
                            "gate node");
                if (roots != 1)
                    throw std::logic_error(
                        "NeighborList: skin gate must be one kernel node");

                // IF(rebuild): the whole hierarchy → CSR → mask pipeline.
                cudaGraphNodeParams params{};
                params.type = cudaGraphNodeTypeConditional;
                params.conditional.handle = handle;
                params.conditional.type = cudaGraphCondTypeIf;
                params.conditional.size = 1;
                cudaGraphNode_t branch = nullptr;
                check_graph(cudaGraphAddNode(&branch, graph, &gate, nullptr, 1,
                                             &params),
                            "conditional node");
                cudaGraph_t body = params.conditional.phGraph_out[0];
                check_graph(cudaStreamBeginCaptureToGraph(cs, body, nullptr,
                                                          nullptr, 0, mode),
                            "capture rebuild");
                try {
                    enqueue_hierarchy(args, ctx, cs);
                    s.exclusions_baked = enqueue_csr(args, ctx, cs);
                    run_hooks(hooks_, cs);
                } catch (...) {
                    cudaGraph_t discard = nullptr;
                    cudaStreamEndCapture(cs, &discard);
                    throw;
                }
                check_graph(cudaStreamEndCapture(cs, &captured), "end rebuild");
            } catch (...) {
                cudaGraphDestroy(graph);
                throw;
            }
            return graph;
        };
        try {
            s.step_graph.build(make);
        } catch (...) {
            s.release_graph();
            throw;
        }
        s.key = key;
    }

    // The first launch after build or an input change always rebuilds.
    if (s.first_launch)
        ATV_CUDA_CHECK(
            cudaMemsetAsync(&s.control.data()->force, 1, sizeof(int), stream));
    s.first_launch = false;
    ATV_CUDA_CHECK(s.step_graph.launch(stream));
    ctx.runtime.md.exclusions_baked = s.exclusions_baked;
}

// ============================================================================
// rebuild_issue() / finalize_rebuild() — multi-GPU concurrent reneighbor
// ============================================================================
//
// OpenMM cluster rebuild (same pipeline as the single-GPU graph body) on the
// multi-GPU driver's host cadence. That loop still reads the total back before
// fill and throws on overflow; it is retired with MultiGpu (md-unification).

void NeighborList::rebuild_issue(const LaunchContext& ctx) {
    auto& s = *state_;
    auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    ctx.runtime.md.pair_list = &pairs_;
    const int n = s.num_atoms;
    const int nc = s.num_clusters;
    pairs_.num_clusters = nc;
    const RebuildArgs args{&s.buf,
                           s.d_inv_cell.data(),
                           s.control.data(),
                           s.reference_box.data(),
                           n,
                           nc,
                           s.pair_capacity,
                           s.periodic,
                           s.cut_skin_sq};
    enqueue_hierarchy(args, ctx, stream);
    ATV_CUDA_CHECK(cudaMemcpyAsync(s.h_cluster_pair_count,
                                   s.buf.row_ptr.data() + nc, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int total_cluster_pairs = *s.h_cluster_pair_count;
    if (total_cluster_pairs > s.pair_capacity) {
        throw std::runtime_error(
            "NeighborList: cluster pair capacity "
            + std::to_string(s.pair_capacity)
            + " is too small; construct NeighborList(cutoff, skin, "
            + std::to_string(total_cluster_pairs) + ") or a larger reserve");
    }
    ctx.runtime.md.exclusions_baked = enqueue_csr(args, ctx, stream);
    run_hooks(hooks_, stream);
    launch_publish_generation(s.control.data(), stream);
    s.first_launch = false;
}

bool NeighborList::finalize_rebuild(const LaunchContext&) {
    // The reference snapshot is part of the issued rebuild.
    return false;
}

// ============================================================================
// destroy() — release build-time resources
// ============================================================================

void NeighborList::destroy(const DestroyContext& ctx) {
    pairs_ = {};
    ctx.runtime.md.pair_list = nullptr;
    ctx.runtime.md.exclusions_baked = false;
    // Graph, capture stream, pinned scratch and DeviceVectors are released by
    // the State destructor.
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

bool NeighborList::graph_capturable() const {
    return state_ && !rebuild_on_request_ && !state_->first_launch;
}

std::uint64_t NeighborList::graph_epoch() const {
    return state_ ? (state_->epoch << 32) ^ hooks_.version : 0;
}

int NeighborList::rebuild_count() const {
    if (!state_)
        return 0;
    int count = 0;
    ATV_CUDA_CHECK(cudaMemcpy(&count, &state_->control.data()->rebuilds,
                              sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

int NeighborList::candidate_tiles() const {
    if (!state_)
        return 0;
    int count = 0;
    ATV_CUDA_CHECK(cudaMemcpy(&count,
                              state_->buf.row_ptr.data() + state_->num_clusters,
                              sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

} // namespace cuda
} // namespace ATV
