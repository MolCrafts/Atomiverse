#pragma once

/// @file cuda/communication/multi_gpu.cuh
/// @brief MultiGpu — single-process multi-GPU MD orchestrator. Owns the
///        DeviceManager (P2P over NVLink) and a 3D SpatialGrid, splits
///        a global cpu::Frame into one device-resident subdomain frame per GPU,
///        and drives the full Verlet-cadence step loop over NVLink: partition /
///        gather, the 3-axis cascade halo (build_halo) + per-step position
///        forward_comm, atom migration across subdomains, the per-subdomain
///        cluster neighbor-list LJ force path, and velocity-Verlet NVE — all
///        without MPI (ghost/migration traffic is cudaMemcpyPeer).
///
/// gather()/gather_forces() merge owned atoms back into a global cpu::Frame by
/// persistent tag (so they survive migration). What is NOT yet here: device-
/// side cross-GPU reduction of energy/virial (gather is host-side) and bonded
/// interactions across subdomain boundaries.

#include <functional>
#include <memory>
#include <vector>

#include "atomiverse/cuda/communication/spatial_grid.cuh"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/device_manager.cuh"
#include "atomiverse/cuda/frame/frame.cuh"
#include <array>

#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
// Force families are composed by abstract base pointer + factory (de-hardcoded
// from PairLJCut/BondHarmonic/...); concrete styles are pulled in only by the
// default factories in multi_gpu.cu, not by this header.
#include "atomiverse/cuda/longrange/kspace.cuh"
#include "atomiverse/cuda/potentials/angle.cuh"
#include "atomiverse/cuda/potentials/bond.cuh"
#include "atomiverse/cuda/potentials/dihedral.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"

namespace ATV {
namespace cpu {
class Frame;
}
namespace cuda {
namespace md {

namespace nlist {
struct AtomCellSortBuffers; // reused per-slot cell-sort scratch (impl detail)
}

class MultiGpu {
  public:
    /// @param n_gpus  Subdomains / GPUs to use (<=0 ⇒ all visible).
    explicit MultiGpu(int n_gpus = -1);
    ~MultiGpu();
    MultiGpu(const MultiGpu&) = delete;
    MultiGpu& operator=(const MultiGpu&) = delete;

    int count() const { return dm_.count(); }
    const DeviceManager& devices() const { return dm_; }
    const SpatialGrid& domain() const { return *grid_; }

    /// Split `global` by owner slab and materialize each subdomain onto its
    /// GPU. `ghost_headroom` over-allocates each subdomain's per-atom arrays so
    /// halo ghosts can be appended later without reallocation (capacity =
    /// (n_local * (1 + ghost_headroom)) rounded up). Frees any prior frames.
    void partition(const cpu::Frame& global, double ghost_headroom = 0.5);

    /// Owned atom count on subdomain `slot`.
    int n_local(int slot) const;
    /// Ghost atom count currently appended on subdomain `slot` (after a halo).
    int n_ghost(int slot) const;
    /// The device frame for `slot` (owned in [0,n_local), ghosts in [n_local,
    /// n_local+n_ghost)).
    const cuda::Frame& frame(int slot) const { return frames_[slot]; }

    /// Establish the persistent ghost set (a "reneighbor"-cadence operation).
    /// For each face, each subdomain selects its owned atoms within `cutoff` of
    /// that face boundary and sends them to the face-neighbor GPU, appended as
    /// ghosts in [n_local, n_local+n_ghost). Positions are raw global
    /// coordinates (the neighbor list min-images against the global box, so no
    /// PBC shift is needed). Ghost x/y/z/tag/type are transferred over NVLink
    /// via cudaMemcpyPeer. Self-neighbors (dims along an axis == 1) are skipped.
    /// Unlike a per-step refresh, build_halo also RECORDS, per transfer, the
    /// source owned-atom indices and the destination ghost offset (the
    /// `HaloPlan`) so subsequent steps can refresh ghost POSITIONS cheaply via
    /// `forward_comm()` without re-selecting the ghost set. Call this every K
    /// steps (when the neighbor list is rebuilt); call `forward_comm()` on the
    /// intervening steps. Communication is a 3-axis CASCADE (x, then y, then z):
    /// the y-pass forwards owned+x-ghost atoms and the z-pass forwards
    /// owned+x+y-ghosts, so edge/corner ghosts reach diagonal-neighbor GPUs.
    /// Within an axis the sendable set is snapshotted at the axis start, so
    /// ghosts received during that axis are not immediately re-forwarded.
    void build_halo(double cutoff);

    /// Refresh ghost POSITIONS only, reusing the `HaloPlan` recorded by the
    /// last `build_halo`. For each recorded transfer, gather the current owned
    /// positions of the previously-selected source atoms and cudaMemcpyPeer
    /// them into the same destination ghost slots. The ghost SET (which atoms,
    /// their tags/types, and the per-GPU counts) is unchanged — only x/y/z move.
    /// This is the per-step "forward communicate" of standard domain-decomposed
    /// MD; it is valid as long as no atom has moved more than the neighbor-list
    /// skin since `build_halo` (the reneighbor cadence guarantees that).
    void forward_comm();

    /// Fully asynchronous forward halo: identical data movement to
    /// `forward_comm()` but issued on each device's COMM stream (the
    /// DeviceManager defer stream) with pure event ordering — zero host sync.
    /// Cascade ordering holds because each device's comm ops are issued on
    /// its single comm stream in axis order, and cross-device copy→unpack
    /// edges use the per-plan `ev_src` events. On entry each comm stream
    /// waits `ev_pos_[s]` (positions current — recorded by the step loop
    /// after the first integrator half-kick); on exit each device's comm
    /// stream records `ev_ghosts_[s]` (ghost positions landed), which the
    /// force path waits on before launching ghost-dependent (boundary)
    /// kernels. External callers that need completed ghosts on return should
    /// call `forward_comm()` (== this + one device barrier).
    void forward_comm_async();

    /// Reverse halo (force communicate): the adjoint of `build_halo`/`forward_comm`.
    /// Replays the recorded `HaloPlan` transfers in REVERSE cascade order
    /// (z, then y, then x): for each transfer, gather the destination's ghost-row
    /// forces, ship them back over NVLink, and scatter-add them into the source's
    /// owned (or prior-axis ghost) rows. After this, any force a force kernel
    /// deposited onto a ghost row (e.g. a bonded partner owned on another GPU) has
    /// been accumulated onto its true owner. The LJ path computes the full force
    /// on owned atoms directly (its force kernel computes OWNED rows only, so
    /// ghost rows carry nothing but bonded deposits) and so does NOT need this;
    /// the cross-GPU bonded path does. Reverse-once invariant: call exactly once
    /// per step, after all force kernels, mirroring the single forward halo.
    void reverse_comm();

    /// Fully asynchronous reverse halo (comm streams + events, zero host
    /// sync), the adjoint of `forward_comm_async`. On entry each comm stream
    /// waits `ev_force_[s]` (ghost-row bonded deposits final — recorded by
    /// `compute_forces` after its force kernels); on exit each device's comm
    /// stream records `ev_rev_[s]` (scatter-adds into its owned rows landed),
    /// which the integrator's second half-kick waits on. External callers
    /// that need completed forces on return should call `reverse_comm()`.
    void reverse_comm_async();

    /// Rebuild the per-subdomain flat bonded tuple lists (BondList/AngleList/
    /// DihedralList) from the per-atom bonded topology (global tags), resolving
    /// each partner tag → local (owned or ghost) row via a freshly built device
    /// tag→local map over the current owned+ghost set. A reneighbor-cadence
    /// operation (call after `build_halo`); no-op if the system has no bonded
    /// topology. Unresolvable partners (halo too thin) are dropped and counted.
    /// Public so multi-GPU bonded force paths and tests can drive it explicitly.
    void rebuild_bonded_tuples();

    // ── Force-family factories (de-hardcoded composition) ─────────────────
    // The engine composes force families by abstract base; the concrete style
    // is chosen by a caller-supplied factory (one instance built PER subdomain,
    // so each GPU owns its own component). An empty factory selects the
    // built-in default (LJ pair / harmonic bonded), preserving prior behavior.
    using PairFactory = std::function<std::unique_ptr<cuda::Pair>()>;
    using KSpaceFactory = std::function<std::unique_ptr<cuda::KSpace>()>;
    using BondFactory = std::function<std::unique_ptr<cuda::Bond>()>;
    using AngleFactory = std::function<std::unique_ptr<cuda::Angle>()>;
    using DihedralFactory = std::function<std::unique_ptr<cuda::Dihedral>()>;

    /// Set up the per-subdomain cross-GPU bonded force engine. Call after
    /// `setup_force_engine` (it reuses the per-GPU Context's box/stream). One
    /// `cuda::BondHarmonic` per subdomain, built with the given per-type spring
    /// constants `bond_K` [eV/Å²] and equilibria `bond_r0` [Å]. Angle
    /// (`angle_K` [eV/rad²], `angle_theta0` [deg]) and dihedral
    /// (`dihedral_C`, per-type 5-coeff multi/harmonic) components are created
    /// only when their coefficient lists are non-empty.
    ///
    /// The optional factories override the concrete bonded styles (default:
    /// harmonic bond/angle + multi/harmonic dihedral from the coefficient
    /// lists). A non-empty `angle_K` / `dihedral_C` selects whether the angle /
    /// dihedral family is created at all (unchanged); the factories only change
    /// WHICH concrete style is built.
    void setup_bonded(std::vector<double> bond_K, std::vector<double> bond_r0,
                      std::vector<double> angle_K = {},
                      std::vector<double> angle_theta0 = {},
                      std::vector<std::array<double, 5>> dihedral_C = {},
                      BondFactory bond_factory = {},
                      AngleFactory angle_factory = {},
                      DihedralFactory dihedral_factory = {});

    /// Compute cross-GPU bonded forces: zero each subdomain's owned+ghost force
    /// rows, run BondHarmonic over the re-materialized per-GPU bond tuples
    /// (depositing on owned AND ghost partner rows), then `reverse_comm()` to
    /// accumulate the ghost-row contributions onto their true owners. Each bond
    /// is owned by exactly one GPU, so the summed energy is the total bond energy
    /// (available via `bonded_energy()`). Requires a prior `rebuild_bonded_tuples`.
    void compute_bonded_forces();

    /// Total bond potential energy from the last `compute_bonded_forces` (summed
    /// across subdomains; each bond counted once on its owner GPU).
    double bonded_energy() const { return bonded_energy_; }

    /// Enable special-bonds exclusion (1-2/1-3/1-4 fully excluded) on every
    /// subdomain's PairLJCut. The exclusion CSR is (re)built from the
    /// re-materialized topology each `rebuild_bonded_tuples`, so excluded
    /// partners resolve to local OR ghost rows. Call after `setup_bonded`.
    void enable_special_bonds();

    /// Combined per-step force: zero owned+ghost, LJ over each subdomain's
    /// neighbor list (owned, full, special-bonds-excluded) + bonded
    /// (bond/angle/dihedral, owned+ghost), then a single `reverse_comm`. Owned
    /// forces are then complete (LJ + bonded). Requires a prior `reneighbor`.
    /// @param read_energy  If true, D2H the per-subdomain energy scalar and sum
    ///        into bonded_energy() (a host sync). The integration loop passes
    ///        false to skip this per-step readback (throughput); callers that
    ///        need the energy (tests, thermo) pass true.
    void compute_forces(bool read_energy = true);

    /// Merge all subdomains' owned atoms back into one cpu::Frame (positions /
    /// velocities by tag order matching the original global frame size). Used
    /// to validate partition round-trips and, later, full multi-GPU runs.
    void gather(cpu::Frame& out) const;

    /// Compute LJ 12-6 forces on each subdomain's OWNED atoms from its
    /// owned+ghost set (brute force, global-box min-image). Run after
    /// `build_halo`: each owned atom's force is complete (ghosts supply the
    /// cross-boundary neighbors), so this needs no reverse halo. Halo
    /// correctness oracle for tests — production force is `compute_lj_nlist`.
    void compute_lj_bruteforce(double cutoff, double epsilon, double sigma);

    /// Gather owned-atom forces back to global tag/index order (like gather()
    /// but for fx/fy/fz). `out` sized to the global atom count.
    void gather_forces(cpu::Frame& out) const;

    /// Run `steps` of velocity-Verlet NVE across the GPUs (brute-force LJ):
    /// each step refreshes the halo, recomputes forces (owned, from
    /// owned+ghost), and integrates owned atoms. Acceleration is f/m (consistent
    /// units; this validates parity, not a specific unit system). Assumes atoms
    /// stay in their subdomain over the run (no migration yet).
    void run_nve(int steps, double dt, double cutoff, double epsilon,
                 double sigma);

    // ── Cluster neighbor-list force path (production performance) ──────────
    // The real per-subdomain force engine: each GPU runs the optimized
    // atom-centric cluster LJ kernel over its owned+ghost set, instead of the
    // O(n²) brute force above. The neighbor list is rebuilt only on reneighbor
    // steps (with a fresh halo over cutoff+skin); intervening steps just
    // forward_comm ghost positions and let the Verlet skin-gating reuse the CSR.

    /// Construct the per-subdomain force engine (Context + NeighborList + a
    /// `cuda::Pair`). Call once after `partition`. `skin` < 0 ⇒ auto (0.3*cutoff).
    /// The halo is built over `cutoff + skin` so ghosts cover the skin region
    /// and no cross-boundary neighbor appears mid-interval.
    ///
    /// `pair_factory` builds the pair style per subdomain; when empty, the
    /// default builds `PairLJCut(epsilon, sigma, cutoff)` with stress gated off
    /// (the prior hardcoded behavior). A custom factory ignores epsilon/sigma.
    void setup_force_engine(double cutoff, double epsilon, double sigma,
                            double skin = -1.0, PairFactory pair_factory = {});

    /// Attach a reciprocal-space `cuda::KSpace` component per subdomain (built
    /// by `kspace_factory` and `build()`-initialized). Optional; call after
    /// `setup_force_engine`. The reciprocal sum needs a cross-GPU structure-
    /// factor reduction on >1 GPU (force-infra-04) — single-GPU only today.
    void set_kspace(KSpaceFactory kspace_factory);

    /// Reneighbor step: migrate atoms that crossed subdomain boundaries to
    /// their new owner GPU, rebuild the halo (cutoff+skin), and rebuild the
    /// per-subdomain cluster neighbor list over the new owned+ghost set. Call
    /// every K steps.
    void reneighbor();

    /// Re-bucket owned atoms by their CURRENT owner subdomain: each atom that
    /// drifted out of its subdomain is gathered with full per-atom state on its
    /// old owner GPU and cudaMemcpyPeer'd to its new owner (single-process
    /// equivalent of MPI_Alltoallv). Ghosts are dropped (rebuilt by build_halo).
    /// Owned-atom global identity is carried by the persistent tag; gather()
    /// resolves by tag afterwards. No-op on a single GPU.
    void migrate();

    /// Compute LJ forces on every subdomain's owned atoms via the cluster
    /// kernel (zero forces over owned+ghost, refresh/launch the neighbor list,
    /// run PairLJCut). Assumes the halo/forward_comm for this step already ran.
    void compute_lj_nlist();

    /// Velocity-Verlet NVE using the cluster neighbor-list force path.
    /// `reneigh_every` steps between full halo+neighbor rebuilds; positions are
    /// forward-communicated on the intervening steps.
    void run_nve_nlist(int steps, double dt, double cutoff, double epsilon,
                       double sigma, double skin = -1.0,
                       int reneigh_every = 10);

    /// Velocity-Verlet NVE using the combined LJ + bonded force path
    /// (compute_forces). Requires setup_force_engine + setup_bonded (and, for a
    /// physical PE run, enable_special_bonds) to have been called first. Every
    /// `reneigh_every` steps does a full migrate + halo + topology
    /// re-materialize + neighbor rebuild; intervening steps forward-communicate
    /// ghost positions. The multi-GPU equivalent of the single-GPU PE NVE step.
    void run_nve_pe(int steps, double dt, int reneigh_every = 10);

  private:
    /// One recorded P2P ghost transfer (one src subdomain → one dst face
    /// neighbor). Stable for the life of a reneighbor interval.
    struct HaloPlan {
        int src = 0;
        int dst = 0;
        int axis = 0;
        int cnt = 0;
        int base = 0;
        int* d_src_idx = nullptr;
        real_t* d_pack_src = nullptr;
        real_t* d_pack_dst = nullptr;
        cudaEvent_t ev_src = nullptr;
        cudaEvent_t ev_dst = nullptr;
        cudaEvent_t ev_fwd_done = nullptr;
        cudaEvent_t ev_rev_done = nullptr;
    };

    void destroy_frames();
    void destroy_halo_plan();
    void destroy_force_engine();
    // Partition each subdomain's OWNED rows by ghost dependence (interior =
    // no ghost in the atom's Verlet row) into rows_[s]; reneighbor-cadence.
    void classify_force_rows();
    // Build the per-subdomain device tag→local map over owned+ghost (sorted
    // tags + parallel local-index array), used by rebuild_bonded_tuples.
    void build_tag_maps();
    void destroy_tag_maps();

    DeviceManager dm_;
    std::unique_ptr<SpatialGrid> grid_;
    std::vector<cuda::Frame> frames_; // per-slot device frame
    std::vector<int> n_local_;        // per-slot owned count
    std::vector<HaloPlan>
        halo_plan_; // recorded by build_halo, replayed by forward_comm
    int n_global_ = 0;
    double box_L_[3] = {0, 0, 0}; // global orthorhombic box edge lengths

    // Cluster force engine (one per subdomain); built by setup_force_engine.
    std::vector<cuda::Context> ctx_;
    std::vector<std::unique_ptr<cuda::NeighborList>> nlist_;
    // Force families held by ABSTRACT BASE pointer (de-hardcoded): any
    // cuda::Pair / KSpace / Bond / Angle / Dihedral subclass, built by the
    // factories passed to setup_force_engine / set_kspace / setup_bonded.
    std::vector<std::unique_ptr<cuda::Pair>> pair_;
    std::vector<std::unique_ptr<cuda::KSpace>> kspace_; // reciprocal (optional)
    std::vector<char>
        nlist_built_; // per-slot: NeighborList::build called (vector<bool> avoided)
    std::vector<int>
        nlist_cap_; // per-slot: atom count the nlist buffers were built for
    // Reused per-slot cell-sort scratch (cell-key/perm/cub-temp/staging) so the
    // per-reneighbor owned-atom cell-sort does NO cudaMalloc/Free in steady
    // state. unique_ptr to a forward-declared impl struct (defined in the .cu).
    std::vector<std::unique_ptr<nlist::AtomCellSortBuffers>> sort_buf_;
    bool engine_ready_ = false;
    double force_cutoff_ = 0, force_skin_ = 0;

    // ── Async per-step pipeline (multi-GPU comm/compute overlap) ──────────
    // Per-device events (created by setup_force_engine, DisableTiming).
    // Per-step ordering is carried ENTIRELY by streams + these events — the
    // steady step loop performs no host synchronization. Compute/integration
    // run on ctx_[s].stream; halo traffic runs on the device's comm stream
    // (DeviceManager defer stream). Both are created with cudaStreamCreate
    // (legacy-blocking), so they overlap with each other as long as no
    // per-step work lands on stream 0 — which this pipeline guarantees.
    std::vector<cudaEvent_t> ev_pos_; // ctx: vv_first done, positions current
    std::vector<cudaEvent_t> ev_ghosts_; // comm: forward unpacks landed here
    std::vector<cudaEvent_t>
        ev_force_; // ctx: force kernels done (ghost rows final)
    std::vector<cudaEvent_t> ev_rev_; // comm: reverse scatter-adds landed here
    // Owned-row partition by ghost dependence, rebuilt each reneighbor:
    // rows_[s][0..n_interior_[s]) = owned atoms whose Verlet row holds no
    // ghost index (force computable before the halo lands — overlapped with
    // the forward comm); rows_[s][n_interior_[s]..n_local) = boundary atoms
    // (launched after ev_ghosts_[s]). n_interior_[s] < 0 ⇒ no valid split
    // (fall back to one owned-prefix launch after the ghost wait).
    std::vector<int*> rows_;                // [capacity] partitioned owned rows
    std::vector<unsigned char*> row_flags_; // [capacity] interior flags
    std::vector<int*> row_nsel_;            // [1] device interior count
    std::vector<void*> row_tmp_;            // CUB DevicePartition temp storage
    std::vector<std::size_t> row_tmp_sz_;
    std::vector<int> n_interior_;

    // Cross-GPU bonded topology (multigpu-bonded). `bonded_` is set by
    // partition when the system carries bonded topology (per-atom SoA filled).
    // The tag→local maps are rebuilt each reneighbor over owned+ghost.
    bool bonded_ = false;
    std::vector<std::int64_t*>
        tagmap_tag_; // [capacity] sorted global tags, per slot
    std::vector<int*>
        tagmap_idx_; // [capacity] parallel local indices, per slot
    std::vector<int*>
        drop_ctr_; // [1] persistent dropped-tuple counter, per slot
    // GPU tag-map sort scratch (CUB DeviceRadixSort SortPairs): per slot, an
    // iota value buffer [capacity] and a persistent CUB temp-storage blob. The
    // sort runs fully on device — no D2H/host-sort/H2D round-trip per reneighbor.
    std::vector<int*> tagmap_iota_; // [capacity] 0..na-1 (sort values-in)
    std::vector<void*> tagmap_tmp_; // CUB temp storage, per slot
    std::vector<std::size_t> tagmap_tmp_sz_; // bytes of tagmap_tmp_[s]

    // Cross-GPU bonded force engine (one component per subdomain per type).
    std::vector<std::unique_ptr<cuda::Bond>> bond_;
    std::vector<std::unique_ptr<cuda::Angle>> angle_;
    std::vector<std::unique_ptr<cuda::Dihedral>> dihedral_;
    double bonded_energy_ = 0.0;
    bool special_bonds_ = false; // rebuild PairLJCut exclusions each reneighbor
    // Single-GPU (count()==1) has no migration and no ghosts, so the tag→local
    // map, the rematerialized bonded tuples, and the special-bonds exclusion CSR
    // are INVARIANT across the run. Build them once and skip the per-reneighbor
    // rebuild (which D2H-sorts every tag on the host — the dominant single-GPU
    // overhead). Reset on partition(). Multi-GPU rebuilds every reneighbor.
    bool bonded_static_built_ = false;
};

} // namespace md
} // namespace cuda
} // namespace ATV
