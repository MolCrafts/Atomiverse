/// @file cuda/neighbor/neighbor_list.cuh
/// @brief NeighborList -- cluster-based pair-building component with full lifecycle.
///
/// Demonstrates the build/launch/destroy lifecycle:
/// - build(): allocates owned PairList and all internal GPU buffers.
/// - launch(): skin-gated hierarchical tile search; no allocation.
///   Rebuild decisions and capacity checks use scalar device readbacks.
/// - destroy(): frees PairList and all internal buffers.
///
/// Cross-driver component: returns empty default_bindings(), must be
/// bound explicitly via `sim.on(phase).add(NeighborList{...})`.
/// CUDA-free header.
#pragma once
#include "atomiverse/core/role.h"

#include <cstdint>
#include <memory>
#include <string>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/neighbor_protocol.cuh"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief OpenMM cluster tile width (atoms per cluster). Must match
///        `md::nlist::kClusterSize` in the neighbor implementation.
inline constexpr int kOpenmmClusterSize = 32;

/// @brief uint64 words of interaction mask per CSR cluster-pair (32×32 bits).
inline constexpr int kOpenmmMaskWords = 16;

/// @brief Read the 32-bit j-interaction flags for i_slot of CSR pair p.
__host__ __device__ inline std::uint32_t
openmm_mask_jbits(const std::uint64_t* mask, int pair_idx, int i_slot) {
    const auto* rows = reinterpret_cast<const std::uint32_t*>(
        mask + static_cast<std::size_t>(pair_idx) * kOpenmmMaskWords);
    return rows[i_slot];
}

/// @brief Neighbor pair list -- all arrays on device.
///
/// Computed method-specific scratch, **not** part of `cuda::Frame` (see the
/// CLAUDE.md Frame purity rule). Owned by `NeighborList` -- allocated in
/// `build()`, populated in `launch()`, freed in `destroy()` -- and published
/// to consumers as a non-owning `const PairList*` on `cuda::Context`,
/// mirroring `cpu::md::NeighborList` / `cpu::Context::pair_list`.
struct PairList {
    int* i = nullptr;     ///< First atom index [device].
    int* j = nullptr;     ///< Second atom index [device].
    real_t* dx = nullptr; ///< x-displacement (j - i) [Angstrom, device].
    real_t* dy = nullptr; ///< y-displacement (j - i) [Angstrom, device].
    real_t* dz = nullptr; ///< z-displacement (j - i) [Angstrom, device].
    real_t* dist_sq =
        nullptr; ///< Squared pair distance dx²+dy²+dz² [Å², device].
    ///<
    ///< Per cuda-md-review-cleanup-06-launch-config-streams, the
    ///< neighbor-list export stores r² rather than r — pair styles
    ///< need inv_r² for the LJ force calculation and can skip the
    ///< sqrt entirely. A consumer that needs r takes the sqrt at
    ///< read time.
    int* n_pairs = nullptr;  ///< Current pair count [device, 1 element].
    int* capacity = nullptr; ///< Pre-allocated slot count [device, 1 element].

    // ── OpenMM cluster neighbor CSR (CUDA MD force source of truth) ──────
    // FULL neighbor: every neighboring cluster in both directions. Force
    // kernels (tile / cluster) consume these only — not vl_*. Addresses are
    // stable across the run; rebuilt in place on each skin-gated rebuild.
    // orig_index maps (cluster * kClusterSize + slot) → atom index; pad slots
    // hold a negative sentinel. Null until the first rebuild publishes them.
    const int* cl_row_ptr = nullptr;    ///< [num_clusters+1] CSR row offsets.
    const int* cl_col_idx = nullptr;    ///< neighbor cluster indices (full).
    const int* cl_orig_index = nullptr; ///< cluster slot -> global atom index.
    /// Per-CSR-entry multi-word interaction mask: `kOpenmmMaskWords` uint64s
    /// per entry (32×32 bitflags). See `openmm_mask_jbits`. Null until rebuild.
    const uint64_t* cl_mask = nullptr;
    /// Optional Hilbert-sorted cluster SoA positions (Å), length
    /// `num_clusters * kOpenmmClusterSize`. When non-null, force kernels use
    /// these directly and skip a per-step gather from `frame.atoms.x/y/z`.
    const real_t* cl_x = nullptr;
    const real_t* cl_y = nullptr;
    const real_t* cl_z = nullptr;
    int num_clusters = 0;          ///< cluster count (host scalar).
    int num_cluster_pairs = 0;     ///< Directed CSR entries (host scalar).
    int cluster_pair_capacity = 0; ///< Build-time directed CSR reserve.
    std::uint64_t rebuild_generation = 0; ///< Unique baked-mask publication.
};

/// @brief Cluster-based neighbor list builder.
///
/// Builds atom-level pair lists using a cluster-based algorithm:
/// Hilbert sort, hierarchical AABB rejection, exact directed cluster CSR,
/// and geometric interaction masks; atom-level export is opt-in at build time.
///
/// @see PairList for the output data structure
class NeighborList : public Component, public NeighborProvider {
  public:
    /// @brief Construct with cutoff and optional skin width.
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin   Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
    /// @param cluster_pair_capacity Reserved directed cluster-pair entries;
    ///        zero measures the initial list and adds density-based headroom.
    ///        Overflow throws before fill.
    explicit NeighborList(double cutoff, double skin = -1.0,
                          int cluster_pair_capacity = 0);

    ~NeighborList() override;
    NeighborList(NeighborList&&) noexcept;
    NeighborList& operator=(NeighborList&&) noexcept;

    /// @brief Component name.
    /// @return "neighbor_list".
    std::string name() const override { return "neighbor_list"; }
    Role role() const override { return role_bit(RoleFlag::FrameImmutable); }

    /// @brief Identifies this component as the neighbor-list supplier
    ///        for `cuda::md::MD::install_infra_`'s typed query.
    const NeighborProvider* as_neighbor_provider() const override {
        return this;
    }

    /// @brief Allocate the owned PairList + all internal GPU buffers, and
    ///        publish the pair list to `ctx.runtime.md.pair_list`.
    ///
    /// Cluster capacity uses the larger of twice the exact initial count
    /// and max(128, 8*rho*sphere(search)/32) entries per cluster, bounded by
    /// the full nc*nc list. An explicit constructor reserve overrides this
    /// measurement. Flat consumers must request export
    /// before this call. Capacity overflow throws before writing the CSR.
    ///
    /// @param ctx Build context with Runtime and DeviceFrame.
    void build(const BuildContext& ctx) override;

    /// @brief Rebuild pair list from current positions, gated by the
    ///        Verlet-skin rebuild check.
    ///
    /// Rebuild: Hilbert sort, scatter+AABB, 32-cluster group bounds, exact
    /// count, prefix scan, CSR fill, geometric masks and optional exclusion
    /// baking, then snapshot reference positions. No buffers grow in launch.
    /// Optional flat export follows; every automatic-gate step checks current
    /// motion before deciding whether this force evaluation needs a rebuild.
    ///
    /// First launch always rebuilds. Later launches check current minimum-
    /// image motion against skin/2 before force evaluation. Any box change
    /// also rebuilds. Mapped block flags are read only after their CUDA event.
    ///
    /// Re-publishes the owned pair list to `ctx.runtime.md.pair_list`.
    ///
    /// @param ctx Launch context with Runtime and DeviceFrame.
    void launch(const LaunchContext& ctx) override;

    /// @brief Reset the active atom count to reuse build()-time buffers for a
    ///        different (smaller-or-equal) atom count, forcing a full rebuild
    ///        on the next launch() — no reallocation.
    ///
    /// For single-process multi-GPU domain decomposition: a subdomain's
    /// owned+ghost count changes each reneighbor interval, but its buffers are
    /// sized once at build() for the frame's full capacity. Calling this each
    /// reneighbor (with the current owned+ghost count) avoids the
    /// destroy()+build() realloc that otherwise dominates the multi-GPU step.
    /// `n_active` must be <= the atom count passed to build(). The single-GPU
    /// driver never calls this; its behaviour is unchanged.
    /// @param n_active Current active atom count (<= build-time capacity).
    void set_active_count(int n_active);

    /// @brief Rebuild the list ONLY on explicit request (first launch after a
    ///        set_active_count), never via the per-step Verlet-skin auto-gate.
    ///
    /// The multi-GPU driver reneighbors on a fixed cadence (every N steps) and
    /// owns the rebuild trigger, so the skin auto-gate is redundant — and it
    /// false-fires on ghost rows (refreshed cross-GPU each step, ref mismatch),
    /// causing many spurious rebuilds. Single-GPU leaves this false and keeps
    /// the displacement-driven gate.
    void set_rebuild_on_request(bool on) { rebuild_on_request_ = on; }

    /// @brief Submit a full cluster rebuild for driver-owned cadence.
    /// Uses the same fixed-capacity hierarchy and masks as launch(). The
    /// exact count is read back before CSR fill; overflow throws with the
    /// required constructor reserve. Requires a prior set_active_count().
    /// Subsequent launch() calls republish without rebuilding.
    void rebuild_issue(const LaunchContext& ctx);

    /// @brief Snapshot reference positions after an issued rebuild.
    /// @return False; fixed-capacity rebuilds never resize or require retry.
    bool finalize_rebuild(const LaunchContext& ctx);

    /// @brief Free the owned PairList + all internal buffers.
    /// @param ctx Destroy context with Runtime and DeviceFrame.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Access the owned neighbor pair list.
    /// @return Device-resident pair list (valid after `build()`).
    const PairList& pairs() const { return pairs_; }

    /// @brief Get interaction cutoff.
    /// @return Interaction cutoff [Angstrom].
    double cutoff() const { return cutoff_; }

    /// @brief Get Verlet skin width.
    /// @return Skin width [Angstrom].
    double skin() const { return skin_; }

    /// @brief Number of full neighbor-list rebuilds since `build()`.
    ///
    /// Test introspection only — used by `test_neighborlist_skin_gating`
    /// to assert that a quiescent system rebuilds at most O(1) times
    /// over many steps. Production code should not depend on this
    /// counter (the rebuild cadence is a perf-tuning detail, not a
    /// public contract).
    /// @return Rebuild count over the lifetime of this component.
    int rebuild_count() const noexcept;

  private:
    double cutoff_;                 ///< Interaction cutoff [Angstrom].
    double skin_;                   ///< Verlet skin width [Angstrom].
    int cluster_pair_capacity_ = 0; ///< Explicit CSR capacity; zero = estimate.
    bool rebuild_on_request_ = false; ///< Skip the skin auto-gate (multi-GPU).
    PairList pairs_; ///< Owned device pair list (allocated in build()).
    struct State;    ///< Opaque GPU state (pimpl, defined in .cu).
    std::unique_ptr<State> state_; ///< Internal GPU buffers.
};

} // namespace cuda
} // namespace ATV
