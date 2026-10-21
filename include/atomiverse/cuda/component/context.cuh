/// @file cuda/component/context.cuh
/// @brief cuda::Context -- CUDA execution context: flat core state (streams
/// + host-side box + the active loop's status/metrics) plus the classical
/// force-pipeline slice (arch-refactor-05-context).
///
/// Holds stream handles as void* to keep this header CUDA-free.
/// Components cast to cudaStream_t in their .cu implementations.
///
/// Homomorphic with `cpu::Context`: same-named slices carry the same
/// semantics. CUDA has no Sampling / Path driver and no analysis
/// executor (async observation goes through `AsyncBridge`), so only the
/// `loop` and `md` slices exist on this backend — homomorphism means
/// "same slice, same meaning", not "same field set".
///
/// @see cuda::Frame for the device-side data
#pragma once

#include <type_traits>

#include "atomiverse/core/iteration_status.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cuda/frame/box.cuh"

namespace ATV {
namespace cuda {

struct PairList;

/// @brief Classical force-pipeline services (publishers: `NeighborList`,
///        `PairLJCut::build`, the multi-GPU comm layer). Mirrors
///        `cpu::DynamicsServices`; the exclusion CSR and the force-row
///        split live here because they belong to the same
///        publish-consume chain as `pair_list`.
///
/// CUDA MD nonbonded path is the **OpenMM cluster/tile engine** only
/// (see CLAUDE.md + `cuda-openmm-md` specs). There is no style enum.
struct DynamicsServices {
    /// @brief Non-owning handle to the active neighbor pair list.
    ///
    /// Published by `NeighborList::build()` (and re-published on each
    /// `launch()`) and consumed by downstream force components (LJ) and
    /// observers. Kept here -- rather than on `cuda::Frame` -- so the
    /// frame stays confined to fundamental physical quantities. Null on
    /// drivers / configurations that carry no neighbor list. Mirrors
    /// `cpu::DynamicsServices::pair_list`.
    /// @see cuda::PairList
    const PairList* pair_list = nullptr;

    /// @brief Opt-in request for the flat (i,j,dx,dy,dz,r²) atom pair list.
    ///
    /// Cluster force kernels read the cluster CSR and do not need the flat
    /// list; NeighborList skips prune+export by default. Consumers that
    /// genuinely need the flat list set this true in their build();
    /// NeighborList::launch() then emits it every step. Set before the
    /// first launch (all build()s precede any launch()).
    bool flat_pairs_requested = false;

    /// @brief Sorted exclusion CSR in original/local atom indices.
    /// Pair evaluators publish their fully excluded ordinary interactions
    /// here. NeighborList removes these pairs from geometric tile masks at
    /// rebuild only when s_excl is zero; nonzero LJ scaling stays in the force
    /// kernel. Fused Ewald evaluators restore scaled special interactions in
    /// their sparse exception pass.
    const int* excl_offsets = nullptr;
    const int* excl_partners = nullptr;
    double s_excl = 1.0;
    bool exclusions_baked = false; ///< Current tile masks contain exclusions.

    /// @brief Pair-force row restriction (multi-GPU owned/ghost split).
    ///
    /// A pair style computes one force row per atom. On a multi-GPU
    /// subdomain the frame's atom range is [owned | ghost]; ghost rows
    /// must NOT be computed: each owned atom's force is already complete
    /// from its own full neighbor list, so a ghost row's pair force is (a)
    /// wasted work and (b) WRONG when `reverse_comm` later ships ghost
    /// rows home — the pair part would double onto the owner (only bonded
    /// ghost deposits may travel). It also double-counts cross-boundary
    /// pair energy (each half-pair contribution appears on both GPUs).
    ///
    /// `n_force_rows < 0` (default): compute every atom row (single-GPU /
    /// driver path — unchanged). `n_force_rows >= 0` with `force_rows ==
    /// nullptr`: compute the owned prefix [0, n_force_rows). With
    /// `force_rows != nullptr`: compute the explicit row subset
    /// `force_rows[0..n_force_rows)` — the multi-GPU interior/boundary
    /// split that overlaps interior force with the halo exchange. Neighbor
    /// POSITIONS are always read from the full owned+ghost set regardless.
    int n_force_rows = -1;
    const int* force_rows = nullptr;
};

/// @brief The active loop node's status + metrics (publisher: the running Driver).
///
/// Non-owning: a Driver owns its own `IterationStatus` and `LoopMetrics` and points
/// these here for the duration of its loop. So a nested driver's components write to
/// the NESTED loop's status, never the parent's. Mirrors `cpu::LoopServices`.
struct LoopServices {
    /// @brief Stop channel. A component may set `stop_requested`; it must NOT set
    /// `converged` — declaring success is a Criterion's job, not a component's.
    IterationStatus* status = nullptr;

    /// @brief Where components publish what they measured, for handlers to judge.
    /// `cuda::minimizer::ConvCheck` writes it after its one pinned D2H per iteration.
    LoopMetrics* metrics = nullptr;
};

/// @brief CUDA execution context passed to components via
///        build/launch/destroy contexts.
///
/// Components must launch kernels onto these streams, never the default
/// stream.
struct Context {
    void* stream = nullptr; ///< Main CUDA stream (cast to cudaStream_t in .cu).
    void* defer_stream =
        nullptr; ///< Secondary stream for deferrable operations.

    /// @brief The **active loop node's** status and metrics.
    ///
    /// Both are OWNED BY THE LOOP (`cuda::Driver` holds them as members), not by
    /// the Context — the Context only points at whichever loop is currently running.
    /// It used to be an `IterationStatus` here BY VALUE, and that is the shape of a bug
    /// that shipped twice on the CPU side: a component (there `LBFGS`, here
    /// `ConvCheck`) wrote `converged = true` onto the one shared status, so an inner
    /// loop's convergence could stop the loop ABOVE it. A pointer that points at the
    /// current loop cannot reach the loop above it.
    ///
    /// A component **measures** into `metrics` and may **request a stop** via `status`.
    /// It does not declare convergence: that is a `cuda::Criterion`'s job.
    /// @see cuda/driver/criterion.cuh
    LoopServices loop;

    /// @brief Host-resident snapshot of the simulation cell, passed by
    ///        value into integrator + bonded + pair kernels so they
    ///        avoid per-thread FP64 division and uncoalesced
    ///        widely-spaced `frame.box[0]/[4]/[8]` loads.
    ///
    /// Populated by `Backend::materialize(cpu::Frame)` at build time
    /// from the canonical `cpu::Box` in `cpu::Frame`. Mutated by
    /// integrators that change the cell (`cuda::md::NoseHooverNPT`,
    /// `cuda::md::Barostat`, `cuda::md::Deform`) after they write the
    /// new diagonal to device-side `frame.box`. All consumers read
    /// from this field at launch — no per-step D2H of box[0,4,8]
    /// (cuda-md-review-cleanup-04-box-by-value-pbc). Core state, not a
    /// nullable service — stays flat.
    cuda::Box host_box;

    /// Cell/inverse retained in integration precision, independently of force math.
    cuda::MotionBox host_motion_box;

    DynamicsServices md; ///< Classical force-pipeline slice.
};

// Slices are plain aggregates and the Context stays non-polymorphic — components
// receive it by reference in every lifecycle hook, and a vtable here would mean the
// Context could be subclassed into carrying behaviour. Mirrors the CPU asserts.
static_assert(std::is_trivially_copyable_v<DynamicsServices>);
static_assert(std::is_trivially_copyable_v<LoopServices>);
static_assert(!std::is_polymorphic_v<Context>);

} // namespace cuda
} // namespace ATV
