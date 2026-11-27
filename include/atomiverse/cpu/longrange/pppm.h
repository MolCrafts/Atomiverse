/// @file cpu/longrange/pppm.h
/// @brief Particle–mesh reciprocal electrostatics — brick PPPM (CPU).
///
/// ## Vocabulary
///
/// Coulomb interactions are split ("Ewald splitting") into a short-ranged part
/// summed atom by atom and a smooth long-ranged part evaluated on a regular
/// grid — the **mesh**. PPPM (particle–particle particle–mesh) spreads each
/// charge onto the mesh points near it, solves the smooth part in **reciprocal
/// space** (the Fourier picture of the periodic cell, where that part needs
/// only a few terms), and interpolates the result back onto the atoms as a
/// force. The **FFT** (fast Fourier transform) is what moves the mesh between
/// the two pictures; **FFTW** is the library that provides the 1-D
/// transforms, r2c/c2r being its real→complex / complex→real ones.
/// **MPI** (Message Passing Interface) runs the job as `nprocs` processes
/// called **ranks**, numbered `0 … nprocs-1`. A **brick** is the block of
/// mesh points one rank owns under the spatial domain decomposition — the
/// ranks laid out as a 3-D grid, the **Cartesian communicator** — ringed by
/// the **ghost** layers that its owned atoms' stencils reach and its
/// neighbours also write into; **reverse_comm** sums those ghosts back into
/// their owners and **forward_comm** refills them from the owners afterwards.
/// A **pencil** is a block of whole mesh lines along one axis; the
/// distributed FFT (`PencilFft`, the LAMMPS `fft_3d` port) transforms one
/// axis at a time on pencils along it and moves the data between pencil
/// layouts, and between bricks and pencils, with a **remap** (a plan-based
/// point-to-point exchange between overlapping ranks, LAMMPS `remap_3d`).
/// Forces are **ik** differentiation, the LAMMPS `pppm` default: the mesh
/// potential's Cartesian gradient is i·k·G(k)·ρ̂(k), inverse-transformed, and
/// gathered with the same B-spline θ that spread the charge.
///
/// Charge assignment is LAMMPS's centred stencil (`core/pppm_ik.h`): an
/// order-p charge at grid coordinate u covers nx + nlower … nx + nupper,
/// nx = ⌊u⌋ for even p and the nearest grid point for odd p. The brick's
/// ghost extents are LAMMPS `Grid3d::ghost_grid` for that stencil
/// (`set_stencil_atom(-nlower, nupper)`, `set_shift_atom` 0.5 for odd p,
/// `set_distance(0.5·skin)`): an owned atom may sit half the neighbor skin
/// outside its subdomain before the next reneighbouring, and the skin is
/// the one the active neighbor provider publishes (`Context::md` slice,
/// `neighbor_skin`). Like LAMMPS the extents are fixed when the mesh, the
/// processor grid or the skin changes, not on every cell change.
///
/// ## Architecture (LAMMPS-aligned)
///
/// | Step | Action |
/// |---|---|
/// | 1 | Local **density brick** (owned mesh + order ghost layers) |
/// | 2 | Spread **owned atoms only** onto brick |
/// | 3 | **mesh reverse fold**: ghost density to owners (Grid3d plan, p2p) |
/// | 4 | Pencil r2c FFT of the owned brick cells (remap into pencils inside) |
/// | 5 | Hockney–Eastwood G(k) → three ik backward transforms, each landing in its field brick |
/// | 6 | One **mesh forward fill** of the three field bricks (LAMMPS `FORWARD_IK`) |
/// | 7 | θ interpolation of the Cartesian electric field |
///
/// ### Mesh arms
///
/// Every solve runs on exactly one of three decompositions — the arm
/// `mesh_arm()` reports afterwards, sized by `fft_ranks()`:
///
/// | Arm | Mesh layout | FFT group |
/// |---|---|---|
/// | `Replicated` | full mesh on every rank | none: each rank repeats it |
/// | `BrickWorld` | brick + pencil FFT | all `nprocs` ranks |
/// | `BrickSubgroup` | brick + pencil FFT | `fft_ranks()` of `nprocs` ranks |
///
/// `Replicated` is the floor: each rank spreads its own atoms, one
/// `MPI_Allreduce` sums the whole mesh, and then *every* rank runs the same
/// serial r2c/c2r transform on that identical grid. Nothing is broadcast and no
/// rank idles; rank 0 alone banks the reciprocal energy into the Frame. It is
/// taken at one rank, without a domain decomposition, and when
/// `ATV_PPPM_ALLREDUCE=1` forces it.
///
/// The two brick arms both run steps 1–7 above and differ only in who holds
/// pencils. `BrickWorld`, the multi-rank default, gives every rank pencils,
/// as LAMMPS does. `BrickSubgroup` is taken when the constructor's
/// `fft_ranks` names fewer: cart ranks `[0, fft_ranks())` hold the pencils
/// and the others only send their brick to the transform and receive the
/// field back. On both, each rank banks its own share of the reciprocal
/// energy and virial.
///
/// Requires `ATV_BUILD_KSPACE` (FFTW3); the multi-rank arms need only MPI.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/pppm_accuracy.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/longrange/kspace.h"

#ifdef ATV_HAS_MPI
#include <memory>

#include <mpi.h>

#include "atomiverse/cpu/longrange/pencil_fft.h"
#endif

namespace ATV {
namespace cpu {

/// @brief Which mesh decomposition the reciprocal solve runs on.
///
/// The selection rules are in the arm table at the top of this file;
/// `mesh_arm()` reports the arm a solve actually took.
enum class PppmMeshArm {
    /// Full mesh on every rank: local spread, one `MPI_Allreduce` of the whole
    /// grid, then the *same* serial FFT on every rank — no broadcast, no idle
    /// rank. Rank 0 alone banks the reciprocal energy.
    Replicated,
    /// Brick + pencil FFT with pencils on every rank of the Cartesian
    /// communicator.
    BrickWorld,
    /// Brick + pencil FFT with pencils on the `fft_ranks()` lowest-numbered
    /// cart ranks only.
    BrickSubgroup
};

/// @brief Pick the mesh arm for a mesh/rank/FFT-group triple.
///
/// Pure policy — no MPI, no environment, no state, so it is unit-testable
/// without launching ranks. The `ATV_PPPM_ALLREDUCE` / `ATV_PPPM_BRICK`
/// overrides stay at the call site.
///
/// @param planes_k0 Mesh planes along x, i.e. the FFT dim-0 extent (count).
/// @param nprocs Ranks in the parent communicator (count).
/// @param fft_ranks Members of the FFT group (rank count).
/// @return `Replicated` for a single rank, `BrickWorld` when every rank is an
///         FFT group member, `BrickSubgroup` otherwise.
/// @throws std::invalid_argument if any argument is ≤ 0, or if @p fft_ranks
///         exceeds @p nprocs (an FFT group cannot outgrow its parent).
inline PppmMeshArm pppm_select_mesh_arm(int planes_k0, int nprocs,
                                        int fft_ranks) {
    if (planes_k0 <= 0 || nprocs <= 0 || fft_ranks <= 0)
        throw std::invalid_argument(
            "pppm_select_mesh_arm: planes_k0, nprocs and fft_ranks must all "
            "be > 0 (got planes_k0="
            + std::to_string(planes_k0) + ", nprocs=" + std::to_string(nprocs)
            + ", fft_ranks=" + std::to_string(fft_ranks) + ")");
    if (fft_ranks > nprocs)
        throw std::invalid_argument(
            "pppm_select_mesh_arm: fft_ranks=" + std::to_string(fft_ranks)
            + " exceeds nprocs=" + std::to_string(nprocs)
            + "; the FFT group is a subset of the parent communicator");
    if (nprocs <= 1)
        return PppmMeshArm::Replicated;
    if (fft_ranks == nprocs)
        return PppmMeshArm::BrickWorld;
    return PppmMeshArm::BrickSubgroup;
}

/// @brief Brick-decomposed particle–mesh kspace (Hockney–Eastwood ik PPPM).
///
/// Same parameters, defaults and mesh rule as `cuda::KSpacePPPM`.
class KSpacePPPM : public cpu::KSpace {
  public:
    /// @brief Construct a brick PPPM solver.
    /// @param alpha Ewald splitting parameter (Å⁻¹), > 0 — the inverse width
    ///        of the Gaussian that separates the short-ranged real-space sum
    ///        from this mesh part; a larger alpha moves work onto the mesh.
    /// @param grid_spacing Target mesh spacing (Å), > 0; the mesh counts
    ///        `K0,K1,K2` are ⌈cell length / spacing⌉ rounded up to sizes with
    ///        prime factors 2, 3 and 5 (LAMMPS `factorable`), at least
    ///        order + 1, derived at grid setup.
    /// @param order Charge-assignment order (count), in [3, 12] — how many
    ///        mesh points along each axis one charge is spread over (order 5
    ///        spreads it over 5³ points), on LAMMPS's centred stencil. It
    ///        also sets the ghost depth. The ik denominator keeps 12
    ///        coefficient slots, which is why the order stops at 12; LAMMPS
    ///        itself stops at 7.
    /// @param fft_ranks Ranks that hold FFT pencils (rank count); 0 (the
    ///        default) is every rank. Ignored on the replicated arm and when
    ///        `ATV_PPPM_BRICK=1` forces the world arm.
    /// @throws std::invalid_argument at construction on a non-positive alpha
    ///         or grid_spacing, an order outside [3, 12], or a negative
    ///         fft_ranks; and at the first multi-rank reciprocal solve when
    ///         fft_ranks exceeds the rank count.
    explicit KSpacePPPM(double alpha, double grid_spacing = 1.0, int order = 6,
                        int fft_ranks = 0);
    /// @brief Construct on an explicit mesh (LAMMPS `kspace_modify mesh`).
    /// @param alpha Ewald α (Å⁻¹), > 0.
    /// @param mesh  Grid points per axis (count), each > 0; raised to sizes
    ///        with factors 2, 3 and 5 and to at least order + 1, as LAMMPS
    ///        and the spacing rule do. The mesh does not follow the cell.
    /// @param order, fft_ranks As for the spacing constructor.
    /// @throws std::invalid_argument as the spacing constructor, and on a
    ///         non-positive mesh entry.
    KSpacePPPM(double alpha, std::array<int, 3> mesh, int order = 6,
               int fft_ranks = 0);
    ~KSpacePPPM() override;

    KSpacePPPM(const KSpacePPPM&) = delete;
    KSpacePPPM& operator=(const KSpacePPPM&) = delete;
    KSpacePPPM(KSpacePPPM&& other) noexcept;
    KSpacePPPM& operator=(KSpacePPPM&&) = delete;

    /// @brief α and mesh from an accuracy target (LAMMPS `kspace_style pppm
    ///        <accuracy>` + `kspace_modify gewald / mesh / order`).
    ///
    /// `longrange::pppm_from_accuracy` decides; the same request gives the
    /// same solver on `cuda::KSpacePPPM`. Read the chosen α back from
    /// `long_range_contract().alpha` for the real-space `CoulLong`.
    /// @throws std::invalid_argument / std::runtime_error from
    ///         `pppm_from_accuracy`.
    static KSpacePPPM from_accuracy(const longrange::PppmAccuracyRequest& req,
                                    int fft_ranks = 0) {
        const auto r = longrange::pppm_from_accuracy(req);
        return KSpacePPPM(r.alpha, {r.mesh[0], r.mesh[1], r.mesh[2]}, r.order,
                          fft_ranks);
    }

    /// @brief Mesh of the last grid setup; zero before build().
    /// @return {K0, K1, K2} (grid points).
    std::array<int, 3> mesh() const noexcept { return {K_[0], K_[1], K_[2]}; }

    std::string name() const override { return "pppm"; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, alpha_, AccReal(0), AccReal(0)};
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, 100)};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

    /// @brief FFT group size actually used by the last reciprocal solve.
    ///
    /// The same value on every rank, member or not — which is what lets a
    /// parity test assert that the arm it meant to exercise is the arm that
    /// ran. On the two brick arms it counts the ranks holding FFT pencils;
    /// on `PppmMeshArm::Replicated` there are no pencils, so it reports every
    /// rank sharing the mesh (1 without a domain decomposition) because each of
    /// them repeats the whole serial FFT.
    /// @return Member rank count (count); 0 before the first compute().
    int fft_ranks() const noexcept { return fft_ranks_resolved_; }

    /// @brief Mesh arm actually taken by the last reciprocal solve.
    ///
    /// Also identical on every rank; the selection rules are in the arm table
    /// at the top of this file.
    /// @return The selected PppmMeshArm; `Replicated` before the first
    ///         compute().
    PppmMeshArm mesh_arm() const noexcept { return mesh_arm_; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    // ── Grid-exchange plan types (LAMMPS `Grid3d` + `remap_3d` port) ──
    //
    // Message extents are inclusive cell boxes **in the executing side's
    // own indexing** (brick: unwrapped out-range coords; slab: global
    // coords); wire order is x-outer / y-mid / z-inner.
    struct GridMsg {
        int proc = -1; ///< partner cart rank
        int xlo = 0, xhi = 0;
        int ylo = 0, yhi = 0;
        int zlo = 0, zhi = 0;
        int bufloc = 0; ///< offset into the direction's buffer [doubles]
        int size = 0;   ///< extent volume [doubles]
    };
    struct GridPlan {
        std::vector<GridMsg> sends;
        std::vector<GridMsg> recvs;
        /// Self overlaps (periodic images / slab-on-same-rank), moved by
        /// local copy: pack `self_src[k]`, unpack `self_dst[k]`.
        std::vector<GridMsg> self_src;
        std::vector<GridMsg> self_dst;
        int sendtot = 0; ///< doubles across sends (excl. self)
        int recvtot = 0; ///< doubles across recvs (excl. self)
        int selftot = 0; ///< doubles across self copies
    };

    void ensure_grid(const cpu::Frame& frame, bool initial_build);
    void ensure_atom_capacity(int required, bool initial_build);
    void release_storage() noexcept;
    /// @brief Owned cells (LAMMPS `Grid3d::partition_grid`) and the
    /// ghost-extended brick (`Grid3d::ghost_grid`) of the rank at @p coords
    /// in a uniform @p dims processor grid.
    /// LAMMPS `set_distance(0.5·skin + qdist)` (qdist = 0, point charges)
    /// enters as @p dist_cells, the drift distance in grid cells per axis.
    void setup_brick_partition_(const int (&coords)[3], const int (&dims)[3],
                                const double (&dist_cells)[3]);
    /// @brief Re-partition the brick for the decomposition, re-deriving the
    /// drift distance (and dropping the grid plans) only when the mesh, the
    /// processor grid or the neighbor skin changed — the same on every rank.
    void partition_for_decomposition_(const int (&coords)[3],
                                      const int (&dims)[3], double skin,
                                      const Mat3& inv_cell);
    /// @brief Stencil start and θ of every owned atom (LAMMPS
    /// `particle_map` + `compute_rho1d` on the centred stencil).
    void fill_stencils_(const StorageReal* x, const StorageReal* y,
                        const StorageReal* z, int n_local,
                        const Mat3& inv_cell);
    /// @brief Image-map every atom's stencil into the contiguous brick.
    /// @return Empty, or the error for an atom that drifted beyond the ghost
    ///         range (LAMMPS "out of range atoms — cannot compute PPPM"); the
    ///         caller agrees on it across ranks before throwing.
    std::string map_stencils_(int n_local);
#ifdef ATV_HAS_MPI
    /// @brief Build the two mesh-ghost plans against the Allgathered owned
    /// and ghost-extended brick extents (once per mesh shape /
    /// decomposition).
    void build_grid_plans_(MPI_Comm comm, int me, int nprocs);
    /// @brief Replay one plan: Irecv all, pack + Isend all, local self
    /// copies, Waitall, unpack. `add` sums on unpack (mesh reverse) instead
    /// of overwriting. `nfield` bricks travel in one message per partner
    /// (LAMMPS `FORWARD_IK` packs the three field bricks the same way);
    /// brick `c` starts at `data + c * stride` [doubles].
    void run_grid_plan_(const GridPlan& plan, MPI_Comm comm, int tag_base,
                        double* data, std::size_t stride, bool add, int nfield);
#endif

    double alpha_;
    double spacing_;
    int order_;
    int fft_ranks_requested_ = 0; ///< 0 = derive from the mesh at first solve.
    /// Explicit mesh (`kspace_modify mesh`); zeros = derive from spacing_.
    int fixed_K_[3] = {0, 0, 0};

    // Decomposition the brick extents were last derived for (rank-uniform).
    int part_K_[3] = {0, 0, 0};
    int part_dims_[3] = {0, 0, 0};
    double part_skin_ = -1.0;
    double part_dist_cells_[3] = {0.0, 0.0, 0.0};

    int K_[3] = {0, 0, 0};
    const cpu::Box* box_identity_ = nullptr;
    std::uint64_t cell_revision_ = 0;
    bool have_grid_ = false;

    // ── Brick (owned + ghost), global indices inclusive ───────────────
    int nxlo_in_ = 0, nxhi_in_ = 0;
    int nylo_in_ = 0, nyhi_in_ = 0;
    int nzlo_in_ = 0, nzhi_in_ = 0;
    int nxlo_out_ = 0, nxhi_out_ = 0;
    int nylo_out_ = 0, nyhi_out_ = 0;
    int nzlo_out_ = 0, nzhi_out_ = 0;
    int nxl_ = 0, nyl_ = 0, nzl_ = 0; ///< brick dimensions incl. ghosts
    std::vector<double> brick_;       ///< density, size nxl*nyl*nzl
    /// The three ik field components, each one brick (LAMMPS
    /// `vdx_brick` / `vdy_brick` / `vdz_brick`), size 3*nxl*nyl*nzl.
    std::vector<double> field_brick_;

    // Serial full-mesh buffers (1-rank path / green tables).
    std::vector<double> dens_full_; ///< full K0·K1·K2 real buffer (serial FFT)
    std::vector<std::complex<double>> work_;
    std::vector<std::complex<double>>
        density_; ///< ρ̂/V kept across the 3 ik solves
    /// Folded half-spectrum tables (`longrange::pppm_half_mode`), full
    /// K0×K1×n2c: energy weight, three ik planes, six virial planes.
    std::vector<double> green_;
    std::vector<double> ik_coef_;
    std::vector<double> virial_coef_;

    void* plan_fwd_ = nullptr;
    void* plan_bwd_ = nullptr;

    // Multi-rank pencil FFT (LAMMPS `fft_3d` port), built at the first
    // multi-rank solve and rebuilt when the mesh, the bricks, the rank count
    // or the FFT group changes.
    bool mpi_fft_ready_ = false;
#ifdef ATV_HAS_MPI
    std::unique_ptr<longrange::PencilFft> fft_;
#endif
    /// ρ̂/V on this rank's pencils (`PencilFft` spectrum layout), kept
    /// across the three ik solves.
    std::vector<std::complex<double>> spectrum_;
    std::vector<double> owned_; ///< owned brick cells, no ghosts
    int mpi_fft_K_[3] = {0, 0, 0};
    int mpi_nprocs_ = 1;
    /// Resolved FFT group size; 0 until the reciprocal solve sets it.
    int fft_ranks_resolved_ = 0;
    /// Arm taken by the reciprocal solve; Replicated until it sets it.
    PppmMeshArm mesh_arm_ = PppmMeshArm::Replicated;
    // ── Plan-based mesh-ghost exchanges (LAMMPS `Grid3d` port) ──
    //
    // Two plans, built once per (mesh shape, decomposition) and replayed
    // every step as point-to-point value messages between
    // actually-overlapping ranks — no collectives, no indices on the wire:
    //
    //   * `mesh_rev_` — ghost density folded into owners (sum),
    //   * `mesh_fwd_` — ghost field refilled from owners (copy).
    //
    // The plans enumerate periodic images of the ghost-extended brick, so a
    // ghost shell deeper than a neighbour's brick (small mesh, many ranks)
    // is routed to the rank that actually owns the cells — the dim-by-dim
    // face cascade this replaces silently dropped that charge.
    GridPlan mesh_rev_;
    GridPlan mesh_fwd_;
    std::vector<double> grid_send_;
    std::vector<double> grid_recv_;
    std::vector<double> grid_self_;
#ifdef ATV_HAS_MPI
    std::vector<MPI_Request> grid_reqs_;
#endif
    bool mesh_plans_ready_ = false;

    int atom_capacity_ = 0;
    /// Per-atom first stencil point nx + nlower (3·capacity), nx from the
    /// position wrapped into the cell, so in [nlower, K + nlower].
    std::vector<int> base_;
    /// Per-atom weights (3·order·capacity): θ[(3i+d)·order + a] belongs to
    /// grid point base_[3i+d] + a and equals LAMMPS rho1d[d][nlower + a].
    std::vector<double> theta_;
    /// Per-atom stencil-min brick coordinate (3·capacity), image-mapped so
    /// the whole `order³` stencil is contiguous inside the out-range brick.
    std::vector<int> stencil_min_;

    /// @brief Flat index into brick_ for global (ix,iy,iz) in out-range.
    /// z-fastest, matching the FFTW slab layout so remap/halo rows are
    /// contiguous runs.
    int brick_index_(int ix, int iy, int iz) const noexcept {
        return ((ix - nxlo_out_) * nyl_ + (iy - nylo_out_)) * nzl_
               + (iz - nzlo_out_);
    }
};

} // namespace cpu
} // namespace ATV
