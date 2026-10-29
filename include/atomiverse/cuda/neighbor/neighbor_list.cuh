/// @file cuda/neighbor/neighbor_list.cuh
/// @brief NeighborList -- cluster-based pair-building component with full lifecycle.
///
/// Demonstrates the build/launch/destroy lifecycle:
/// - build(): allocates owned PairList and all internal GPU buffers.
/// - launch(): one graph launch: a device skin gate and a conditional
///   hierarchical tile rebuild. No allocation and no host readback; the
///   rebuild decision and capacity verdict stay on the device.
/// - destroy(): frees PairList and all internal buffers.
///
/// Cross-driver component: returns empty default_bindings(), must be
/// bound explicitly via `sim.on(phase).add(NeighborList{...})`.
/// CUDA-free header.
#pragma once
#include "atomiverse/core/role.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
/// docs/developer/project-contract.md Frame purity rule). Owned by `NeighborList` -- allocated in
/// `build()`, populated in `launch()`, freed in `destroy()` -- and published
/// to consumers as a non-owning `const PairList*` on `cuda::Context`,
/// mirroring `cpu::md::NeighborList` / `cpu::Context::pair_list`.
///
/// @brief Rebuild-time work a consumer attaches to its neighbor list.
///
/// Each hook enqueues on the given stream (a `cudaStream_t`) inside the
/// rebuild, after the CSR masks are final, so a consumer that keeps derived
/// per-rebuild state (packed tiles) runs exactly when the list changes,
/// without a separate device decision or launch. Owners remove their hook
/// before they are destroyed.
struct RebuildHooks {
    struct Hook {
        const void* owner;
        std::function<void(void* stream)> enqueue;
    };
    std::vector<Hook> list;
    std::uint64_t version = 0; ///< Bumped on every change (recapture).
    void add(const void* owner, std::function<void(void*)> enqueue) {
        list.push_back({owner, std::move(enqueue)});
        ++version;
    }
    void remove(const void* owner) {
        std::erase_if(list, [&](const Hook& h) { return h.owner == owner; });
        ++version;
    }
};

/// The cluster coordinate copies keep the stored coordinate type `pos_t`.
struct PairList {
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
    const pos_t* cl_x = nullptr;
    const pos_t* cl_y = nullptr;
    const pos_t* cl_z = nullptr;
    int num_clusters = 0;          ///< cluster count (host scalar).
    int cluster_pair_capacity = 0; ///< Build-time directed CSR reserve.
    /// Device word: nonzero when the current step rebuilt the CSR. The live
    /// directed entry count is `cl_row_ptr[num_clusters]` on the device.
    const int* rebuilt = nullptr;
    /// Device word: unique id of the published masks; zero before the first
    /// rebuild. Survives provider replacement and address reuse.
    const unsigned long long* generation = nullptr;
    /// Rebuild-time consumer work, run inside every rebuild (owned by the
    /// NeighborList).
    RebuildHooks* hooks = nullptr;
};

/// @brief Cluster-based neighbor list builder.
///
/// Builds the cluster pair list: Hilbert sort, hierarchical AABB rejection,
/// exact directed cluster CSR and geometric interaction masks.
///
/// @see PairList for the output data structure
class NeighborList : public Component, public NeighborProvider {
  public:
    /// @brief Construct with cutoff and optional skin width.
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @param skin   Verlet skin width [Angstrom]. Negative = auto (0.3 * cutoff).
    /// @param cluster_pair_capacity Reserved directed cluster-pair entries;
    ///        zero measures the initial list and adds density-based headroom.
    ///        A rebuild that outgrows it raises `ForceFault::kCapacity`.
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
    /// measurement.
    ///
    /// @param ctx Build context with Runtime and DeviceFrame.
    void build(const BuildContext& ctx) override;

    /// @brief Rebuild pair list from current positions, gated by the
    ///        Verlet-skin rebuild check — decided on the device.
    ///
    /// One graph launch per step: the skin gate checks current minimum-image
    /// motion against skin/2 (and any cell change) and selects a conditional
    /// rebuild body: Hilbert sort, scatter+AABB, 32-cluster group bounds,
    /// exact count, prefix scan, capacity verdict, CSR fill, geometric masks
    /// and optional exclusion baking, then snapshot reference positions. The
    /// first launch forces a rebuild. An outgrown reserve empties the CSR and
    /// raises `ForceFault::kCapacity`, freezing the run at that step. The
    /// graph is captured on the first launch after any input change.
    ///
    /// Re-publishes the owned pair list to `ctx.runtime.md.pair_list`.
    ///
    /// @param ctx Launch context with Runtime and DeviceFrame.
    void launch(const LaunchContext& ctx) override;
    void initialize_forces(const LaunchContext& ctx) override { launch(ctx); }

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
    void invalidate_inputs(InputChange changes,
                           const BuildContext& ctx) override;

    /// @brief Rebuild the list ONLY on explicit request (first launch after a
    ///        set_active_count), never via the per-step Verlet-skin auto-gate.
    ///
    /// The multi-GPU driver reneighbors on a fixed cadence (every N steps) and
    /// owns the rebuild trigger, so the skin auto-gate is redundant — and it
    /// false-fires on ghost rows (refreshed cross-GPU each step, ref mismatch),
    /// causing many spurious rebuilds. Single-GPU leaves this false and keeps
    /// the displacement-driven gate.
    void set_rebuild_on_request(bool on) { rebuild_on_request_ = on; }

    /// @brief Submit a full cluster rebuild for driver-owned cadence
    ///        (multi-GPU only; that loop still reads the count back and
    ///        throws on overflow before fill). Requires a prior
    ///        set_active_count().
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
    /// Reads the device counter; call outside a run (it synchronizes).
    /// @return Rebuild count over the lifetime of this component.
    int rebuild_count() const;

    /// @brief Candidate cluster pairs (directed CSR entries) of the latest
    ///        rebuild, before exact masks (diagnostics). Reads the device
    ///        CSR; call outside a run (it synchronizes).
    /// @return Entry count [dimensionless].
    int candidate_tiles() const;

    /// Captured into the step graph after the forced first rebuild.
    bool graph_capturable() const override;
    std::uint64_t graph_epoch() const override;

  private:
    double cutoff_;                 ///< Interaction cutoff [Angstrom].
    double skin_;                   ///< Verlet skin width [Angstrom].
    int cluster_pair_capacity_ = 0; ///< Explicit CSR capacity; zero = estimate.
    bool rebuild_on_request_ = false; ///< Skip the skin auto-gate (multi-GPU).
    PairList pairs_;     ///< Owned device pair list (allocated in build()).
    RebuildHooks hooks_; ///< Consumer rebuild work (see `PairList::hooks`).
    struct State;        ///< Opaque GPU state (pimpl, defined in .cu).
    std::unique_ptr<State> state_; ///< Internal GPU buffers.
};

} // namespace cuda
} // namespace ATV
