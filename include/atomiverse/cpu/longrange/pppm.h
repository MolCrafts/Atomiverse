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
/// the two pictures; **FFTW** is the library that provides it, **FFTW-MPI** its
/// distributed build, and r2c/c2r its real→complex / complex→real transforms.
/// **MPI** (Message Passing Interface) runs the job as `nprocs` processes
/// called **ranks**, numbered `0 … nprocs-1`. A **mesh plane** is one
/// `x = const` layer of the mesh — there are `K0` of them — and FFTW-MPI gives
/// each rank a **slab** of whole planes, so a rank owning no plane has no FFT
/// work at all. A **brick** is the block of mesh points one rank owns under the
/// spatial domain decomposition — the ranks laid out as a 3-D grid, the
/// **Cartesian communicator** — ringed by `order - 1` **ghost** layers that its
/// neighbours also write into; **reverse_comm** sums those ghosts back into
/// their owners and **forward_comm** refills them from the owners afterwards.
/// Brick layout and slab layout cut the mesh differently, so a **remap** (an
/// all-to-all exchange) converts between them before and after the FFT.
/// Forces are taken in the **AD** (analytic differentiation) scheme: each atom
/// differentiates the interpolation weights it spread with, rather than reading
/// a separately differentiated mesh.
///
/// ## Architecture (LAMMPS-aligned)
///
/// | Step | Action |
/// |---|---|
/// | 1 | Local **density brick** (owned mesh + order ghost layers) |
/// | 2 | Spread **owned atoms only** onto brick |
/// | 3 | **reverse_comm**: sum ghost density into neighbors' owned cells |
/// | 4 | **brick→FFT remap** (Alltoallv) into FFTW-MPI dim-0 slabs |
/// | 5 | Distributed r2c FFT → SPME H(m) → c2r |
/// | 6 | **FFT→brick remap** + **forward_comm** potential ghosts |
/// | 7 | AD force interpolation from local brick |
///
/// ### Mesh arms
///
/// Every solve runs on exactly one of three decompositions — the arm
/// `mesh_arm()` reports afterwards, sized by `fft_ranks()`:
///
/// | Arm | Mesh layout | FFT group |
/// |---|---|---|
/// | `Replicated` | full mesh on every rank | none: each rank repeats it |
/// | `BrickWorld` | brick + remap | all `nprocs` ranks |
/// | `BrickSubgroup` | brick + remap | `fft_ranks()` of `nprocs` ranks |
///
/// `Replicated` is the floor: each rank spreads its own atoms, one
/// `MPI_Allreduce` sums the whole mesh, and then *every* rank runs the same
/// serial r2c/c2r transform on that identical grid. Nothing is broadcast and no
/// rank idles; rank 0 alone banks the reciprocal energy into the Frame. It is
/// taken at one rank, without a domain decomposition, when FFTW-MPI is not
/// compiled in, and when `ATV_PPPM_ALLREDUCE=1` forces it.
///
/// The two brick arms both run steps 1–7 above and differ only in who holds the
/// FFT slabs. `BrickWorld` keeps them on the whole Cartesian communicator
/// (aliased, never split); it is taken when the FFT group is every rank
/// (`fft_ranks() == nprocs`) or when `ATV_PPPM_BRICK=1` forces it.
/// `BrickSubgroup` — the multi-rank default whenever
/// `pppm_fft_ranks(K0, nprocs) < nprocs`, i.e. whenever the mesh has fewer
/// usable planes than there are ranks — confines the slabs to a subgroup of
/// cart ranks `[0, fft_ranks())` split off that communicator. Non-members still
/// spread, halo, remap and interpolate; they skip only the FFT itself and the
/// reciprocal-energy reduction.
///
/// Requires `ATV_BUILD_KSPACE` (FFTW3). Multi-rank brick needs `ATV_HAS_FFTW_MPI`.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/longrange/kspace.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

/// @brief Match real-space α to a PPPM accuracy setting (LAMMPS-style guess).
inline double pppm_alpha_from_accuracy(double accuracy, double real_cutoff) {
    if (!(accuracy > 0.0) || !(accuracy < 1.0) || !(real_cutoff > 0.0))
        throw std::invalid_argument("pppm_alpha_from_accuracy: bad args");
    return (1.35 - 0.15 * std::log(accuracy)) / real_cutoff;
}

/// @brief Which mesh decomposition the reciprocal solve runs on.
///
/// The selection rules are in the arm table at the top of this file;
/// `mesh_arm()` reports the arm a solve actually took.
enum class PppmMeshArm {
    /// Full mesh on every rank: local spread, one `MPI_Allreduce` of the whole
    /// grid, then the *same* serial FFT on every rank — no broadcast, no idle
    /// rank. Rank 0 alone banks the reciprocal energy.
    Replicated,
    /// Brick + remap, with the FFT slabs spread over every rank of the
    /// Cartesian communicator (aliased, never split).
    BrickWorld,
    /// Brick + remap, with the FFT slabs confined to the `fft_ranks()`
    /// lowest-numbered cart ranks; the rest skip the FFT collectives.
    BrickSubgroup
};

/// @brief Largest FFT group size that divides the mesh and fits the ranks.
///
/// Pure arithmetic — no MPI, no environment, no state. The result is a
/// *divisor* of @p planes_k0 rather than a plain minimum so that every group
/// member gets the same whole number of planes: the FFTW-MPI dim-0 slab split
/// is then exact, never ragged. It also never exceeds the rank budget
/// `min(nprocs, planes_k0 / min_planes_per_rank)` (clamped to ≥ 1, since 1
/// divides every mesh). FFTW's MPI slab decomposition gives a rank work only
/// when it owns at least one plane, which is the origin of the `≤ planes_k0`
/// bound (Frigo & Johnson, Proc. IEEE 93, 216 (2005)).
///
/// @param planes_k0 Mesh planes along x, i.e. the FFT dim-0 extent (count).
/// @param nprocs Ranks in the parent communicator (count).
/// @param min_planes_per_rank Minimum planes a group member must own (count).
/// @return FFT group size (rank count), ≥ 1 and an exact divisor of
///         @p planes_k0.
/// @throws std::invalid_argument if any argument is ≤ 0 (no fallback value).
inline int pppm_fft_ranks(int planes_k0, int nprocs,
                          int min_planes_per_rank = 1) {
    if (planes_k0 <= 0 || nprocs <= 0 || min_planes_per_rank <= 0)
        throw std::invalid_argument(
            "pppm_fft_ranks: planes_k0, nprocs and min_planes_per_rank must "
            "all be > 0 (got planes_k0="
            + std::to_string(planes_k0) + ", nprocs=" + std::to_string(nprocs)
            + ", min_planes_per_rank=" + std::to_string(min_planes_per_rank)
            + ")");
    const int budget =
        std::max(1, std::min(nprocs, planes_k0 / min_planes_per_rank));
    for (int d = budget; d > 1; --d)
        if (planes_k0 % d == 0)
            return d;
    return 1;
}

/// @brief Pick the mesh arm for a mesh/rank/FFT-group triple.
///
/// Pure policy — no MPI, no environment, no state, so it is unit-testable
/// without launching ranks. Compile-time FFTW-MPI availability and the
/// `ATV_PPPM_ALLREDUCE` / `ATV_PPPM_BRICK` overrides stay at the call site.
///
/// @param planes_k0 Mesh planes along x, i.e. the FFT dim-0 extent (count).
/// @param nprocs Ranks in the parent communicator (count).
/// @param fft_ranks Members of the FFT group (rank count), typically from
///        pppm_fft_ranks().
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

/// @brief Brick-decomposed particle–mesh kspace.
class KSpacePPPM : public cpu::KSpace {
  public:
    /// @brief Construct a brick PPPM solver.
    /// @param alpha Ewald splitting parameter (Å⁻¹), > 0 — the inverse width
    ///        of the Gaussian that separates the short-ranged real-space sum
    ///        from this mesh part; a larger alpha moves work onto the mesh.
    /// @param grid_spacing Target mesh spacing (Å), > 0; the mesh counts
    ///        `K0,K1,K2` are derived from it and the cell at grid setup.
    /// @param order B-spline interpolation order (count), ≥ 3 — how many mesh
    ///        points along each axis one charge is spread over (order 5 spreads
    ///        it over 5³ points). It also fixes the ghost depth, `order - 1`.
    /// @param fft_ranks Requested FFT group size (rank count); 0 (the default)
    ///        derives it from the mesh with pppm_fft_ranks() at the first
    ///        reciprocal solve, and again whenever the mesh or the rank count
    ///        changes. Ignored on the replicated arm and when
    ///        `ATV_PPPM_BRICK=1` forces the world arm.
    /// @throws std::invalid_argument at construction on a non-positive alpha
    ///         or grid_spacing, an order < 3, or a negative fft_ranks; and at
    ///         the first multi-rank reciprocal solve when a positive fft_ranks
    ///         exceeds `min(nprocs, K0)` — an FFTW-MPI slab member needs at
    ///         least one mesh plane, so no such group can exist.
    explicit KSpacePPPM(double alpha, double grid_spacing = 1.0, int order = 5,
                        int fft_ranks = 0);
    ~KSpacePPPM() override;

    KSpacePPPM(const KSpacePPPM&) = delete;
    KSpacePPPM& operator=(const KSpacePPPM&) = delete;
    KSpacePPPM(KSpacePPPM&& other) noexcept;
    KSpacePPPM& operator=(KSpacePPPM&&) = delete;

    static KSpacePPPM from_accuracy(double accuracy, double real_cutoff,
                                    double grid_spacing = -1.0) {
        const double alpha = pppm_alpha_from_accuracy(accuracy, real_cutoff);
        const double spacing =
            grid_spacing > 0.0 ? grid_spacing : (real_cutoff / 2.5);
        return KSpacePPPM(alpha, spacing, /*order=*/5);
    }

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
    /// ran. On the two brick arms it counts the ranks holding FFTW-MPI slabs;
    /// on `PppmMeshArm::Replicated` there are no slabs, so it reports every
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
    void ensure_grid(const cpu::Frame& frame, bool initial_build);
    void ensure_atom_capacity(int required, bool initial_build);
    void release_storage() noexcept;
    void setup_brick_partition_(const int (&coords)[3], const int (&dims)[3]);

    double alpha_;
    double spacing_;
    int order_;
    int fft_ranks_requested_ = 0; ///< 0 = derive from the mesh at first solve.

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
    int nghost_ = 0;
    std::vector<double> brick_; ///< density or potential, size nxl*nyl*nzl

    // Serial full-mesh buffers (1-rank path / green tables).
    std::vector<double> dens_full_; ///< full K0·K1·K2 real buffer (serial FFT)
    std::vector<std::complex<double>> work_;
    std::vector<double> green_; ///< H(m) half-spectrum, full K0×K1×n2c
    std::vector<double> bsp_mod_[3];

    void* plan_fwd_ = nullptr;
    void* plan_bwd_ = nullptr;

    // FFTW-MPI slab
    bool mpi_fft_ready_ = false;
    long long mpi_local_n0_ = 0;
    long long mpi_local_0_start_ = 0;
    long long mpi_alloc_ = 0;
    double* mpi_data_ = nullptr;
    void* plan_mpi_fwd_ = nullptr;
    void* plan_mpi_bwd_ = nullptr;
    int mpi_fft_K_[3] = {0, 0, 0};
    int mpi_nprocs_ = 1;
    /// Resolved FFT group size; 0 until the reciprocal solve sets it.
    int fft_ranks_resolved_ = 0;
    /// Arm taken by the reciprocal solve; Replicated until it sets it.
    PppmMeshArm mesh_arm_ = PppmMeshArm::Replicated;
#ifdef ATV_HAS_MPI
    /// FFT slab communicator: an alias of `dd->cart_comm` on the world arm, a
    /// split of it on the subgroup arm, `MPI_COMM_NULL` for a non-member and
    /// before the first multi-rank solve.
    MPI_Comm fft_comm_ = MPI_COMM_NULL;
    /// True only when `fft_comm_` came from `MPI_Comm_split` (an alias is not
    /// ours to free).
    bool owns_fft_comm_ = false;
#endif
    /// Per-rank FFT m0 ownership (length nprocs+1 prefix starts).
    std::vector<int> fft_m0_starts_;
    std::vector<int> fft_m0_counts_;

    // Remap scratch
    std::vector<double> remap_send_;
    std::vector<double> remap_recv_;
    std::vector<int> sendcounts_, senddispls_, recvcounts_, recvdispls_;

    int atom_capacity_ = 0;
    std::vector<int> base_;
    std::vector<double> theta_;
    std::vector<double> dtheta_;

    /// @brief Flat index into brick_ for global (ix,iy,iz) in out-range.
    int brick_index_(int ix, int iy, int iz) const noexcept {
        return (iz - nzlo_out_) * nyl_ * nxl_ + (iy - nylo_out_) * nxl_
               + (ix - nxlo_out_);
    }
};

} // namespace cpu
} // namespace ATV
