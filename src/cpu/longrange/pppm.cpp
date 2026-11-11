/// @file src/cpu/longrange/pppm.cpp
/// @brief Brick-decomposed PPPM: local brick + plan-based mesh ghost fold /
///        fill + plan-based brick↔FFTW-MPI-slab remap, all point-to-point.
///
/// The mesh ghost exchange and the brick↔FFT remap are structural ports of
/// LAMMPS `src/grid3d.cpp` (`Grid3d::setup_comm` ghost-box overlap with
/// periodic images) and `src/KSPACE/remap.cpp` (`remap_3d_create_plan` /
/// `remap_3d`), checkout `78a7ddf`: plans built once per mesh shape /
/// decomposition / FFT group, replayed each step as value-only
/// Irecv/Isend/Waitall between overlapping ranks — no per-step collectives.
///
/// Charge assignment and the brick extents it needs are ports of LAMMPS
/// `src/KSPACE/pppm.cpp` (`set_grid_local`, `particle_map`, `make_rho`,
/// `fieldforce_ik`, `init`'s `set_distance(0.5*neighbor->skin + qdist)`) and
/// `src/grid3d.cpp` (`partition_grid`, `ghost_grid`, with
/// `MathExtra::tribbox` for the triclinic distance), checkout e8c4c278a8: the
/// centred order-p stencil nx + nlower … nx + nupper and the owned/ghost cell
/// ranges LAMMPS derives from it. The r2c/c2r ik field and the half-spectrum
/// energy/virial fold each stored mode with its conjugate partner
/// (`core/pppm_ik.h` `pppm_half_mode`) to equal LAMMPS `poisson_ik`'s full
/// complex transform and real part.
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.
#include "atomiverse/cpu/longrange/pppm.h"

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/device/fftw_threads.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fftw3.h>

#ifdef ATV_HAS_FFTW_MPI
#include <fftw3-mpi.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/driver/step_timing.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme_influence.h"
#include "atomiverse/cpu/longrange/pme_spline.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/schema_mpi.h"
#endif

namespace ATV {
namespace cpu {
namespace {

// Distributed mesh bricks currently follow uniform fractional spatial cuts.
// Check the replicated cut table locally before FFT setup or MPI collectives;
// the explicit replicated mesh path has no such ownership constraint.
void require_supported_mesh_partition(const cpu::Context& ctx) {
#if defined(ATV_HAS_MPI) && defined(ATV_HAS_FFTW_MPI)
    if (!cpu::multi_rank_dd(ctx)
        || std::getenv("ATV_PPPM_ALLREDUCE") != nullptr)
        return;
    const auto& dd = *ctx.md.dd;
    for (int d = 0; d < 3; ++d) {
        const auto& cuts = dd.cuts[d];
        if (cuts.empty())
            continue;
        if (dd.dims[d] <= 0
            || cuts.size() != static_cast<std::size_t>(dd.dims[d] + 1))
            throw std::invalid_argument(
                "KSpacePPPM: invalid spatial cut table");
        for (int i = 0; i <= dd.dims[d]; ++i)
            if (cuts[i] != static_cast<double>(i) / dd.dims[d])
                throw std::invalid_argument(
                    "KSpacePPPM: distributed mesh requires uniform spatial "
                    "cuts; "
                    "nonuniform Rebalance cuts are unsupported. "
                    "Use the explicit replicated mesh path "
                    "ATV_PPPM_ALLREDUCE=1.");
    }
#else
    (void)ctx;
#endif
}

// Fold a stencil index into [0, K). Indices lie in [nlower, K + nupper] and
// K ≥ order + 1, so one period either way suffices (no integer division in
// the spread / gather loops).
inline int wrap_k(int g, int K) noexcept {
    if (g < 0)
        g += K;
    else if (g >= K)
        g -= K;
    return g;
}

// Sub-key convention for ATV_PROFILE_SCOPE (this file is its first consumer):
// "<component>/<region>", one key per region and the SAME key on every mesh
// arm, so the three-arm matrix is read by comparing like-named rows. The five
// regions are spread / halo / remap / fft / interp; a region that runs twice
// per compute (both mesh halos, both brick↔FFT remaps) accumulates into its
// one key. `category` is the component's own MD timing bucket, so a scoped row
// rolls up beside the KSpacePPPM launch row it lives inside.
constexpr int kTimerCategory = static_cast<int>(md::StepTiming::Kspace);

} // namespace

KSpacePPPM::KSpacePPPM(double alpha, double grid_spacing, int order,
                       int fft_ranks)
    : alpha_(alpha), spacing_(grid_spacing), order_(order),
      fft_ranks_requested_(fft_ranks) {
    if (!(alpha > 0.0))
        throw std::invalid_argument("KSpacePPPM: alpha must be > 0");
    if (!(grid_spacing > 0.0))
        throw std::invalid_argument("KSpacePPPM: grid_spacing must be > 0");
    if (order < 3 || order > 12)
        throw std::invalid_argument("KSpacePPPM: order must be in [3, 12]");
    if (fft_ranks < 0)
        throw std::invalid_argument(
            "KSpacePPPM: fft_ranks must be ≥ 0 (0 = derive from the mesh)");
}

KSpacePPPM::KSpacePPPM(double alpha, std::array<int, 3> mesh, int order,
                       int fft_ranks)
    : KSpacePPPM(alpha, /*grid_spacing=*/1.0, order, fft_ranks) {
    for (int d = 0; d < 3; ++d) {
        if (mesh[d] <= 0)
            throw std::invalid_argument(
                "KSpacePPPM: every mesh entry must be > 0");
        fixed_K_[d] = mesh[d];
    }
}

KSpacePPPM::~KSpacePPPM() { release_storage(); }

KSpacePPPM::KSpacePPPM(KSpacePPPM&& other) noexcept
    : KSpace(std::move(other)), alpha_(other.alpha_), spacing_(other.spacing_),
      order_(other.order_), fft_ranks_requested_(other.fft_ranks_requested_),
      part_skin_(other.part_skin_), box_identity_(other.box_identity_),
      cell_revision_(other.cell_revision_), have_grid_(other.have_grid_),
      nxlo_in_(other.nxlo_in_), nxhi_in_(other.nxhi_in_),
      nylo_in_(other.nylo_in_), nyhi_in_(other.nyhi_in_),
      nzlo_in_(other.nzlo_in_), nzhi_in_(other.nzhi_in_),
      nxlo_out_(other.nxlo_out_), nxhi_out_(other.nxhi_out_),
      nylo_out_(other.nylo_out_), nyhi_out_(other.nyhi_out_),
      nzlo_out_(other.nzlo_out_), nzhi_out_(other.nzhi_out_), nxl_(other.nxl_),
      nyl_(other.nyl_), nzl_(other.nzl_), brick_(std::move(other.brick_)),
      // The serial FFTW plans are bound to these two buffers; a vector move
      // keeps the storage they point at.
      dens_full_(std::move(other.dens_full_)), work_(std::move(other.work_)),
      density_(std::move(other.density_)), green_(std::move(other.green_)),
      ik_coef_(std::move(other.ik_coef_)),
      virial_coef_(std::move(other.virial_coef_)), plan_fwd_(other.plan_fwd_),
      plan_bwd_(other.plan_bwd_), mpi_fft_ready_(other.mpi_fft_ready_),
      mpi_local_n0_(other.mpi_local_n0_),
      mpi_local_0_start_(other.mpi_local_0_start_),
      mpi_alloc_(other.mpi_alloc_), mpi_data_(other.mpi_data_),
      plan_mpi_fwd_(other.plan_mpi_fwd_), plan_mpi_bwd_(other.plan_mpi_bwd_),
      mpi_nprocs_(other.mpi_nprocs_),
      fft_ranks_resolved_(other.fft_ranks_resolved_),
      mesh_arm_(other.mesh_arm_),
      fft_m0_starts_(std::move(other.fft_m0_starts_)),
      fft_m0_counts_(std::move(other.fft_m0_counts_)),
      mesh_rev_(std::move(other.mesh_rev_)),
      mesh_fwd_(std::move(other.mesh_fwd_)),
      remap_fwd_(std::move(other.remap_fwd_)),
      remap_bwd_(std::move(other.remap_bwd_)),
      grid_send_(std::move(other.grid_send_)),
      grid_recv_(std::move(other.grid_recv_)),
      grid_self_(std::move(other.grid_self_)),
      mesh_plans_ready_(other.mesh_plans_ready_),
      atom_capacity_(other.atom_capacity_), base_(std::move(other.base_)),
      theta_(std::move(other.theta_)),
      stencil_min_(std::move(other.stencil_min_)) {
    other.mesh_plans_ready_ = false;
    for (int d = 0; d < 3; ++d) {
        fixed_K_[d] = other.fixed_K_[d];
        part_K_[d] = other.part_K_[d];
        part_dims_[d] = other.part_dims_[d];
        part_dist_cells_[d] = other.part_dist_cells_[d];
        K_[d] = other.K_[d];
        mpi_fft_K_[d] = other.mpi_fft_K_[d];
        other.K_[d] = 0;
        other.mpi_fft_K_[d] = 0;
    }
    other.box_identity_ = nullptr;
    other.cell_revision_ = 0;
    other.have_grid_ = false;
    other.plan_fwd_ = other.plan_bwd_ = nullptr;
    other.plan_mpi_fwd_ = other.plan_mpi_bwd_ = nullptr;
    other.mpi_data_ = nullptr;
    other.mpi_fft_ready_ = false;
    other.fft_ranks_resolved_ = 0;
    other.mesh_arm_ = PppmMeshArm::Replicated;
    other.atom_capacity_ = 0;
#ifdef ATV_HAS_MPI
    // The FFT subgroup follows the plans that were built on it; the source
    // must not free a communicator this object now uses.
    fft_comm_ = other.fft_comm_;
    owns_fft_comm_ = other.owns_fft_comm_;
    other.fft_comm_ = MPI_COMM_NULL;
    other.owns_fft_comm_ = false;
    // Sized with the plans: mesh_plans_ready_ moved true needs them.
    grid_reqs_ = std::move(other.grid_reqs_);
#endif
}

void KSpacePPPM::setup_brick_partition_(const int (&coords)[3],
                                        const int (&dims)[3],
                                        const double (&dist_cells)[3]) {
    // Owned cells: LAMMPS Grid3d::partition_grid with shift_grid = 0.5 —
    // grid point i belongs to the rank whose fractional slab holds i + 0.5.
    // Uniform spatial cuts only (require_supported_mesh_partition), so the
    // slab of coordinate c is [c/p, (c+1)/p).
    auto partition = [](int ngrid, double fraclo, double frachi, int& lo,
                        int& hi) {
        constexpr double shift_grid = 0.5;
        lo = static_cast<int>(fraclo * ngrid);
        while (lo + shift_grid < fraclo * ngrid)
            ++lo;
        hi = static_cast<int>(frachi * ngrid);
        while (hi + shift_grid >= frachi * ngrid)
            --hi;
    };
    // Ghost cells: LAMMPS Grid3d::ghost_grid with PPPM's
    // set_stencil_atom(-nlower, nupper), set_shift_atom(shift, shift) and
    // set_distance(0.5·skin + qdist): an owned atom lies within dist_cells[d]
    // grid cells of its subdomain until it is exchanged, so its grid point
    // nx is in [lo, hi] below and its stencil in [lo + nlower, hi + nupper].
    // Farther drift makes map_stencils_ throw rather than deposit wrong
    // (LAMMPS "Out of range atoms - cannot compute PPPM").
    const int nlower = longrange::pppm_stencil_lower(order_);
    const int nupper = longrange::pppm_stencil_upper(order_);
    const double shift_atom = longrange::pppm_stencil_shift(order_);
    constexpr int offset = longrange::kPppmOffset;
    int in_lo[3], in_hi[3], out_lo[3], out_hi[3];
    for (int d = 0; d < 3; ++d) {
        const double fraclo = static_cast<double>(coords[d]) / dims[d];
        const double frachi = static_cast<double>(coords[d] + 1) / dims[d];
        partition(K_[d], fraclo, frachi, in_lo[d], in_hi[d]);
        const int lo = static_cast<int>(fraclo * K_[d] - dist_cells[d]
                                        + shift_atom + offset)
                       - offset;
        const int hi = static_cast<int>(frachi * K_[d] + dist_cells[d]
                                        + shift_atom + offset)
                       - offset;
        out_lo[d] = std::min(lo + nlower, in_lo[d]);
        out_hi[d] = std::max(hi + nupper, in_hi[d]);
    }
    nxlo_in_ = in_lo[0];
    nxhi_in_ = in_hi[0];
    nylo_in_ = in_lo[1];
    nyhi_in_ = in_hi[1];
    nzlo_in_ = in_lo[2];
    nzhi_in_ = in_hi[2];
    nxlo_out_ = out_lo[0];
    nxhi_out_ = out_hi[0];
    nylo_out_ = out_lo[1];
    nyhi_out_ = out_hi[1];
    nzlo_out_ = out_lo[2];
    nzhi_out_ = out_hi[2];
    nxl_ = nxhi_out_ - nxlo_out_ + 1;
    nyl_ = nyhi_out_ - nylo_out_ + 1;
    nzl_ = nzhi_out_ - nzlo_out_ + 1;
    const std::size_t n = static_cast<std::size_t>(nxl_) * nyl_ * nzl_;
    // Resize only when the brick shape changes; compute() zeros each step.
    if (brick_.size() != n)
        brick_.assign(n, 0.0);
}

void KSpacePPPM::partition_for_decomposition_(const int (&coords)[3],
                                              const int (&dims)[3], double skin,
                                              const Mat3& inv_cell) {
    bool same = skin == part_skin_;
    for (int d = 0; d < 3; ++d)
        same = same && part_K_[d] == K_[d] && part_dims_[d] == dims[d];
    if (!same) {
        // Grid3d::ghost_grid: dist = maxdist per axis for an orthogonal box,
        // MathExtra::tribbox(h, maxdist) in lamda units otherwise — both are
        // maxdist·|row d of H⁻¹| (the inverse spacing of the lattice planes
        // normal to axis d), here in grid cells. maxdist = 0.5·skin + qdist,
        // qdist = 0 (no TIP4P site).
        const double maxdist = 0.5 * skin;
        for (int d = 0; d < 3; ++d) {
            const double row = std::sqrt(inv_cell(d, 0) * inv_cell(d, 0)
                                         + inv_cell(d, 1) * inv_cell(d, 1)
                                         + inv_cell(d, 2) * inv_cell(d, 2));
            part_dist_cells_[d] = maxdist * row * K_[d];
            part_K_[d] = K_[d];
            part_dims_[d] = dims[d];
        }
        part_skin_ = skin;
        // Every rank sees the same mesh, processor grid and skin, so every
        // rank drops its plans together: the rebuild's Allgather matches.
        mesh_plans_ready_ = false;
    }
    setup_brick_partition_(coords, dims, part_dist_cells_);
}

void KSpacePPPM::fill_stencils_(const StorageReal* x, const StorageReal* y,
                                const StorageReal* z, int n_local,
                                const Mat3& inv_cell) {
    // LAMMPS particle_map + make_rho / compute_rho1d on the position wrapped
    // into the cell (u in [0, K)); the brick path image-maps the stencil
    // back next to the owner in map_stencils_. θ is the centred cardinal
    // B-spline: θ[a] at grid point nx + nlower + a is rho1d[nlower + a].
    const int n = order_;
    const int nlower = longrange::pppm_stencil_lower(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        double dtheta[12];
        const double r[3] = {x[i], y[i], z[i]};
        for (int d = 0; d < 3; ++d) {
            const double s = inv_cell(d, 0) * r[0] + inv_cell(d, 1) * r[1]
                             + inv_cell(d, 2) * r[2];
            const int K = K_[d];
            double u = s * K;
            u -= std::floor(u / K) * K; // wrap into [0, K)
            double frac = 0.0;
            const int nx = longrange::pppm_particle_map(u, n, frac);
            longrange::bspline(frac, n, &theta_[(3 * i + d) * n], dtheta);
            base_[3 * i + d] = nx + nlower;
        }
    }
}

void KSpacePPPM::ensure_grid(const cpu::Frame& frame, bool initial_build) {
    const auto& box = frame.box;
    if (have_grid_ && box_identity_ == &box && box.revision() == cell_revision_)
        return;

    // ⌈|a_d| / spacing⌉ (or the explicit mesh) on LAMMPS factorable sizes
    // (2, 3, 5), at least order + 1 — the cuda::KSpacePPPM rule.
    int next_K[3];
    for (int d = 0; d < 3; ++d) {
        next_K[d] = longrange::pppm_fft_size(
            fixed_K_[d] > 0
                ? fixed_K_[d]
                : static_cast<int>(std::ceil(box.lengths()[d] / spacing_)));
        if (next_K[d] < order_ + 1)
            next_K[d] = longrange::pppm_fft_size(order_ + 1);
    }

    const bool shape_changed = !have_grid_ || next_K[0] != K_[0]
                               || next_K[1] != K_[1] || next_K[2] != K_[2];
    if (have_grid_ && !initial_build) {
        if (shape_changed)
            note_capacity_rebuild();
        else
            note_cell_refresh();
    }

    if (shape_changed) {
        if (plan_fwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_fwd_));
        if (plan_bwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_bwd_));
        plan_fwd_ = plan_bwd_ = nullptr;
#ifdef ATV_HAS_FFTW_MPI
        if (plan_mpi_fwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_fwd_));
        if (plan_mpi_bwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_bwd_));
        plan_mpi_fwd_ = plan_mpi_bwd_ = nullptr;
        if (mpi_data_) {
            fftw_free(mpi_data_);
            mpi_data_ = nullptr;
        }
        mpi_fft_ready_ = false;
#endif
        for (int d = 0; d < 3; ++d)
            K_[d] = next_K[d];
        const int ntot = K_[0] * K_[1] * K_[2];
        const int ncomplex = K_[0] * K_[1] * (K_[2] / 2 + 1);
        dens_full_.assign(static_cast<std::size_t>(ntot), 0.0);
        work_.assign(static_cast<std::size_t>(ncomplex),
                     std::complex<double>(0, 0));
        density_.assign(static_cast<std::size_t>(ncomplex),
                        std::complex<double>(0, 0));
        green_.assign(static_cast<std::size_t>(ncomplex), 0.0);
        ik_coef_.assign(3 * static_cast<std::size_t>(ncomplex), 0.0);
        virial_coef_.assign(6 * static_cast<std::size_t>(ncomplex), 0.0);

        // Serial plans bind to dens_full_ / work_ (reused every step).
        // FFTW_ESTIMATE, as LAMMPS: a MEASURE plan is picked by timing, so two
        // processes could run different FFT algorithms and round differently;
        // a minimization then followed a different path from identical input.
        // Plans run on the OpenMP team (fftw_threads.h).
        device::ensure_fftw_threads();
        plan_fwd_ = fftw_plan_dft_r2c_3d(
            K_[0], K_[1], K_[2], dens_full_.data(),
            reinterpret_cast<fftw_complex*>(work_.data()), FFTW_ESTIMATE);
        plan_bwd_ = fftw_plan_dft_c2r_3d(
            K_[0], K_[1], K_[2], reinterpret_cast<fftw_complex*>(work_.data()),
            dens_full_.data(), FFTW_ESTIMATE);
        if (!plan_fwd_ || !plan_bwd_) {
            release_storage();
            throw std::runtime_error("KSpacePPPM: serial r2c/c2r plan failed");
        }
    }

    // Hockney–Eastwood G(k) on the r2c half-spectrum (not the Essmann
    // Euler factor), folded with each mode's conjugate partner.
    {
        const auto& ic = box.inv_cell();
        const auto& h = box.cell();
        double inv[9], cell[9];
        for (int d = 0; d < 3; ++d)
            for (int r = 0; r < 3; ++r) {
                inv[3 * d + r] = ic(d, r);
                cell[3 * d + r] = h(d, r);
            }
        longrange::pppm_fill_half_modes(K_, cell, inv, alpha_, order_,
                                        green_.data(), ik_coef_.data(),
                                        virial_coef_.data());
    }

    // Default serial brick = full mesh (overwritten under multi-rank).
    int coords[3] = {0, 0, 0};
    int dims[3] = {1, 1, 1};
    const double no_drift[3] = {0.0, 0.0, 0.0};
    setup_brick_partition_(coords, dims, no_drift);

    box_identity_ = &box;
    cell_revision_ = box.revision();
    have_grid_ = true;
}

void KSpacePPPM::ensure_atom_capacity(int required, bool initial_build) {
    if (required <= atom_capacity_)
        return;
    const int grown = atom_capacity_ > 0 ? atom_capacity_ * 2 : 1;
    const int next = std::max(required, grown);
    base_.resize(static_cast<std::size_t>(3 * next));
    theta_.resize(static_cast<std::size_t>(3 * order_ * next));
    stencil_min_.resize(static_cast<std::size_t>(3 * next));
    atom_capacity_ = next;
    if (!initial_build)
        note_capacity_rebuild();
}

std::string KSpacePPPM::map_stencils_(int n_local) {
    const int nm1 = order_ - 1;
    const int lo[3] = {nxlo_out_, nylo_out_, nzlo_out_};
    const int hi[3] = {nxhi_out_, nyhi_out_, nzhi_out_};
    for (int i = 0; i < n_local; ++i) {
        for (int d = 0; d < 3; ++d) {
            const int K = K_[d];
            const int s = base_[3 * i + d];
            int sm = s;
            if (!(sm >= lo[d] && sm + nm1 <= hi[d])) {
                if (s + K >= lo[d] && s + K + nm1 <= hi[d])
                    sm = s + K;
                else if (s - K >= lo[d] && s - K + nm1 <= hi[d])
                    sm = s - K;
                else
                    return std::string(
                        "KSpacePPPM: atom " + std::to_string(i)
                        + " maps to mesh cells outside this rank's "
                          "ghost-extended brick on axis "
                        + std::to_string(d) + " (stencil min "
                        + std::to_string(s) + ", brick ["
                        + std::to_string(lo[d]) + ", " + std::to_string(hi[d])
                        + "], K=" + std::to_string(K)
                        + "). The atom drifted farther than the ghost "
                          "depth since the last reneighbouring — the "
                          "LAMMPS analogue is 'out of range atoms - "
                          "cannot compute PPPM'. Reduce the timestep or "
                          "reneighbour more often.");
            }
            stencil_min_[3 * i + d] = sm;
        }
    }
    return {};
}

#if defined(ATV_HAS_MPI) && defined(ATV_HAS_FFTW_MPI)

namespace {

/// @brief Inclusive box intersection; false when empty.
inline bool grid_isect(const int a[6], const int b[6], int out[6]) noexcept {
    for (int d = 0; d < 3; ++d) {
        out[2 * d] = std::max(a[2 * d], b[2 * d]);
        out[2 * d + 1] = std::min(a[2 * d + 1], b[2 * d + 1]);
        if (out[2 * d] > out[2 * d + 1])
            return false;
    }
    return true;
}

inline int grid_volume(const int b[6]) noexcept {
    return (b[1] - b[0] + 1) * (b[3] - b[2] + 1) * (b[5] - b[4] + 1);
}

} // namespace

void KSpacePPPM::build_grid_plans_(MPI_Comm comm, int me, int nprocs) {
    mesh_rev_ = {};
    mesh_fwd_ = {};
    remap_fwd_ = {};
    remap_bwd_ = {};

    // Owned and ghost-extended extents of every rank (Grid3d::ghost_grid
    // rounds per rank, so ghost depths are not uniform). 12 ints per rank,
    // gathered once per rebuild: [0, 6) owned, [6, 12) out.
    const int mine[6] = {nxlo_in_, nxhi_in_, nylo_in_,
                         nyhi_in_, nzlo_in_, nzhi_in_};
    const int myout[6] = {nxlo_out_, nxhi_out_, nylo_out_,
                          nyhi_out_, nzlo_out_, nzhi_out_};
    int mine_both[12];
    for (int k = 0; k < 6; ++k) {
        mine_both[k] = mine[k];
        mine_both[6 + k] = myout[k];
    }
    std::vector<int> extents(static_cast<std::size_t>(12 * nprocs));
    MPI_Allgather(mine_both, 12, MPI_INT, extents.data(), 12, MPI_INT, comm);
    std::vector<int> owned(static_cast<std::size_t>(6 * nprocs));
    std::vector<int> outs(static_cast<std::size_t>(6 * nprocs));
    for (int r = 0; r < nprocs; ++r)
        for (int k = 0; k < 6; ++k) {
            owned[static_cast<std::size_t>(6 * r + k)] =
                extents[static_cast<std::size_t>(12 * r + k)];
            outs[static_cast<std::size_t>(6 * r + k)] =
                extents[static_cast<std::size_t>(12 * r + 6 + k)];
        }

    const int K0 = K_[0], K1 = K_[1], K2 = K_[2];

    auto grid_push = [](std::vector<GridMsg>& list, int& tot, int proc,
                        const int b[6]) {
        GridMsg m;
        m.proc = proc;
        m.xlo = b[0];
        m.xhi = b[1];
        m.ylo = b[2];
        m.yhi = b[3];
        m.zlo = b[4];
        m.zhi = b[5];
        m.size = grid_volume(b);
        m.bufloc = tot;
        tot += m.size;
        list.push_back(m);
    };

    auto push_self = [](GridPlan& plan, const int src[6], const int dst[6]) {
        GridMsg s;
        s.proc = -1;
        s.xlo = src[0];
        s.xhi = src[1];
        s.ylo = src[2];
        s.yhi = src[3];
        s.zlo = src[4];
        s.zhi = src[5];
        s.size = grid_volume(src);
        s.bufloc = plan.selftot;
        GridMsg d = s;
        d.xlo = dst[0];
        d.xhi = dst[1];
        d.ylo = dst[2];
        d.yhi = dst[3];
        d.zlo = dst[4];
        d.zhi = dst[5];
        plan.selftot += s.size;
        plan.self_src.push_back(s);
        plan.self_dst.push_back(d);
    };

    // ── Mesh ghost plans (LAMMPS Grid3d::setup_comm ghost-box drop) ────
    //
    // For every rank pair and periodic image: the ghost side's
    // out-box + S·K intersected with the owner's owned box. Both sides
    // enumerate ranks ascending and images in the same lexical order, so
    // the per-pair message sequences match without any index traffic.
    for (int r = 0; r < nprocs; ++r) {
        const int* rown = &owned[static_cast<std::size_t>(6 * r)];
        for (int sx = -1; sx <= 1; ++sx) {
            for (int sy = -1; sy <= 1; ++sy) {
                for (int sz = -1; sz <= 1; ++sz) {
                    // Ghost side = me. Global box of my shifted out-box.
                    if (!(r == me && sx == 0 && sy == 0 && sz == 0)) {
                        const int shifted[6] = {
                            myout[0] + sx * K0, myout[1] + sx * K0,
                            myout[2] + sy * K1, myout[3] + sy * K1,
                            myout[4] + sz * K2, myout[5] + sz * K2};
                        int b[6];
                        if (grid_isect(shifted, rown, b)) {
                            const int local[6] = {
                                b[0] - sx * K0, b[1] - sx * K0, b[2] - sy * K1,
                                b[3] - sy * K1, b[4] - sz * K2, b[5] - sz * K2};
                            if (r == me) {
                                // Periodic self-fold: ghost image of my own
                                // owned cells.
                                push_self(mesh_rev_, local, b);
                                push_self(mesh_fwd_, b, local);
                            } else {
                                grid_push(mesh_rev_.sends, mesh_rev_.sendtot, r,
                                          local);
                                grid_push(mesh_fwd_.recvs, mesh_fwd_.recvtot, r,
                                          local);
                            }
                        }
                    }
                    // Owner side = me. r's shifted out-box vs my owned box.
                    if (r != me) {
                        const int* rbox =
                            &outs[static_cast<std::size_t>(6 * r)];
                        const int rout[6] = {
                            rbox[0] + sx * K0, rbox[1] + sx * K0,
                            rbox[2] + sy * K1, rbox[3] + sy * K1,
                            rbox[4] + sz * K2, rbox[5] + sz * K2};
                        int b[6];
                        if (grid_isect(rout, mine, b)) {
                            grid_push(mesh_rev_.recvs, mesh_rev_.recvtot, r, b);
                            grid_push(mesh_fwd_.sends, mesh_fwd_.sendtot, r, b);
                        }
                    }
                }
            }
        }
    }

    // ── Brick ↔ FFT-slab remap plans (LAMMPS remap_3d_create_plan) ─────
    //
    // Slabs live in unshifted global coordinates, so no images here.
    for (int r = 0; r < nprocs; ++r) {
        const int n0 = fft_m0_counts_[static_cast<std::size_t>(r)];
        if (n0 > 0) {
            const int s0 = fft_m0_starts_[static_cast<std::size_t>(r)];
            const int slab[6] = {s0, s0 + n0 - 1, 0, K1 - 1, 0, K2 - 1};
            int b[6];
            if (grid_isect(mine, slab, b)) {
                if (r == me) {
                    push_self(remap_fwd_, b, b);
                    push_self(remap_bwd_, b, b);
                } else {
                    grid_push(remap_fwd_.sends, remap_fwd_.sendtot, r, b);
                    grid_push(remap_bwd_.recvs, remap_bwd_.recvtot, r, b);
                }
            }
        }
        const int myn0 = static_cast<int>(mpi_local_n0_);
        if (myn0 > 0 && r != me) {
            const int myl0 = static_cast<int>(mpi_local_0_start_);
            const int myslab[6] = {myl0, myl0 + myn0 - 1, 0, K1 - 1, 0, K2 - 1};
            const int* rown = &owned[static_cast<std::size_t>(6 * r)];
            int b[6];
            if (grid_isect(myslab, rown, b)) {
                grid_push(remap_fwd_.recvs, remap_fwd_.recvtot, r, b);
                grid_push(remap_bwd_.sends, remap_bwd_.sendtot, r, b);
            }
        }
    }

    // One scratch sized for the widest plan; plans never run concurrently.
    const int max_send =
        std::max(std::max(mesh_rev_.sendtot, mesh_fwd_.sendtot),
                 std::max(remap_fwd_.sendtot, remap_bwd_.sendtot));
    const int max_recv =
        std::max(std::max(mesh_rev_.recvtot, mesh_fwd_.recvtot),
                 std::max(remap_fwd_.recvtot, remap_bwd_.recvtot));
    const int max_self =
        std::max(std::max(mesh_rev_.selftot, mesh_fwd_.selftot),
                 std::max(remap_fwd_.selftot, remap_bwd_.selftot));
    grid_send_.resize(static_cast<std::size_t>(max_send));
    grid_recv_.resize(static_cast<std::size_t>(max_recv));
    grid_self_.resize(static_cast<std::size_t>(max_self));
    std::size_t max_reqs = 0;
    for (const GridPlan* p : {&mesh_rev_, &mesh_fwd_, &remap_fwd_, &remap_bwd_})
        max_reqs = std::max(max_reqs, p->sends.size() + p->recvs.size());
    grid_reqs_.resize(max_reqs);
    mesh_plans_ready_ = true;
}

void KSpacePPPM::run_grid_plan_(const GridPlan& plan, MPI_Comm comm,
                                int tag_base, bool src_brick, bool dst_brick,
                                bool add) {
    const int K1 = K_[1];
    const int n2pad = 2 * (K_[2] / 2 + 1);
    const ptrdiff_t l0s = static_cast<ptrdiff_t>(mpi_local_0_start_);

    auto src_row = [&](const GridMsg& m, int gx, int gy) -> const double* {
        if (src_brick)
            return &brick_[static_cast<std::size_t>(
                brick_index_(gx, gy, m.zlo))];
        return mpi_data_
               + ((static_cast<ptrdiff_t>(gx) - l0s) * K1 + gy) * n2pad + m.zlo;
    };
    auto dst_row = [&](const GridMsg& m, int gx, int gy) -> double* {
        if (dst_brick)
            return &brick_[static_cast<std::size_t>(
                brick_index_(gx, gy, m.zlo))];
        return mpi_data_
               + ((static_cast<ptrdiff_t>(gx) - l0s) * K1 + gy) * n2pad + m.zlo;
    };
    auto pack_box = [&](const GridMsg& m, double* buf) {
        const std::size_t nz = static_cast<std::size_t>(m.zhi - m.zlo + 1);
        for (int gx = m.xlo; gx <= m.xhi; ++gx)
            for (int gy = m.ylo; gy <= m.yhi; ++gy) {
                std::memcpy(buf, src_row(m, gx, gy), nz * sizeof(double));
                buf += nz;
            }
    };
    auto unpack_box = [&](const GridMsg& m, const double* buf) {
        const int nz = m.zhi - m.zlo + 1;
        for (int gx = m.xlo; gx <= m.xhi; ++gx)
            for (int gy = m.ylo; gy <= m.yhi; ++gy) {
                double* row = dst_row(m, gx, gy);
                if (add) {
                    for (int c = 0; c < nz; ++c)
                        row[c] += buf[c];
                } else {
                    std::memcpy(row, buf,
                                static_cast<std::size_t>(nz) * sizeof(double));
                }
                buf += nz;
            }
    };

    int nreq = 0;
    for (const GridMsg& m : plan.recvs)
        MPI_Irecv(grid_recv_.data() + m.bufloc, m.size, MPI_DOUBLE, m.proc,
                  tag_base, comm,
                  &grid_reqs_[static_cast<std::size_t>(nreq++)]);
    for (const GridMsg& m : plan.sends) {
        pack_box(m, grid_send_.data() + m.bufloc);
        MPI_Isend(grid_send_.data() + m.bufloc, m.size, MPI_DOUBLE, m.proc,
                  tag_base, comm,
                  &grid_reqs_[static_cast<std::size_t>(nreq++)]);
    }
    for (std::size_t k = 0; k < plan.self_src.size(); ++k) {
        const GridMsg& s = plan.self_src[k];
        pack_box(s, grid_self_.data() + s.bufloc);
        unpack_box(plan.self_dst[k], grid_self_.data() + s.bufloc);
    }
    if (nreq > 0)
        MPI_Waitall(nreq, grid_reqs_.data(), MPI_STATUSES_IGNORE);
    for (const GridMsg& m : plan.recvs)
        unpack_box(m, grid_recv_.data() + m.bufloc);
}

#endif // ATV_HAS_MPI && ATV_HAS_FFTW_MPI

void KSpacePPPM::build(const BuildContext& ctx) {
    require_supported_mesh_partition(ctx.runtime);
    note_build();
    ensure_grid(ctx.frame, true);
    ensure_atom_capacity(ctx.frame.capacity(), true);
    // The multi-rank brick is partitioned by compute(), with the neighbor
    // skin the Runtime publishes by then.
}

void KSpacePPPM::compute(const LaunchContext& ctx) {
    require_supported_mesh_partition(ctx.runtime);
    auto& f = ctx.frame;
    ensure_grid(f, false);

    // Serial: Frame only — no DomainDecomposition. Multi-rank: DD bricks.
    const int n_local = owned_count(ctx);
#ifdef ATV_HAS_MPI
    const bool mpi = multi_rank_dd(ctx.runtime);
    const auto* dd = mpi ? ctx.runtime.md.dd : nullptr;
    const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;
    if (mpi)
        partition_for_decomposition_(dd->coords, dd->dims,
                                     ctx.runtime.md.neighbor_skin,
                                     f.box.inv_cell());
#else
    const bool first_rank = true;
#endif
    ensure_atom_capacity(n_local, false);

    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    const StorageReal* __restrict q = f.span(kAtom_charge);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    const auto& ic = f.box.inv_cell();
    const double vol = f.box.volume();
    const int K0 = K_[0], K1 = K_[1], K2 = K_[2];
    const int n = order_;

    fill_stencils_(x, y, z, n_local, ic);

    double e_recip = 0;
    double stress[6]{};
    // ρ̂ here is the unnormalized FFT of q·θ, with no N/V. LAMMPS stores
    // (N/V) times that density, so its (G/N) ρ̂_lammps is our (G/V) ρ̂ and
    // the energy prefactor is coulomb/(2V), not the SPME coulomb/(2πV).
    const double energy_scale = ctx.runtime.units.qqrd2e() / (2.0 * vol);
    const double coulomb = ctx.runtime.units.qqrd2e();
    const int n2c = K2 / 2 + 1;
    const std::size_t nhalf = static_cast<std::size_t>(K0)
                              * static_cast<std::size_t>(K1)
                              * static_cast<std::size_t>(n2c);
    // Folded virial planes (pppm_half_mode): ug = herm·|ρ̂|² times the mode's
    // coefficient, the same (k_e/2V) normalization as the energy.
    auto tally_mode = [&](std::size_t gidx, double ug) {
        if (!ctx.stress_requested)
            return;
        for (int c = 0; c < 6; ++c)
            stress[c] += ug * virial_coef_[c * nhalf + gidx];
    };

    // ── Mesh arm selection (three arms, decided once per compute) ──────
    // Ranks sharing the mesh: 1 without a domain decomposition.
    int mesh_nprocs = 1;
#ifdef ATV_HAS_MPI
    if (mpi && dd->cart_comm != MPI_COMM_NULL)
        MPI_Comm_size(dd->cart_comm, &mesh_nprocs);
#endif
    // Replicated is the floor: local spread → one Allreduce of the whole
    // grid → every rank runs the *same* serial FFT + influence → local
    // interpolation. Its "FFT group" is therefore every mesh rank (at one
    // rank that is pppm_fft_ranks(K0, 1) == 1). It is what is left when
    // FFTW-MPI is absent, when there is no DD, or when ATV_PPPM_ALLREDUCE
    // forces it.
    mesh_arm_ = PppmMeshArm::Replicated;
    fft_ranks_resolved_ = mesh_nprocs;
    bool use_brick_mpi = false;
#if defined(ATV_HAS_MPI) && defined(ATV_HAS_FFTW_MPI)
    // The brick family is the multi-rank default. The old "mesh ≤ 64³ &&
    // ranks ≥ 16 → replicated" rule is deleted: it was falsified by the
    // three-arm GAFF PEO timing matrix (2026-08-26, 24048-atom MIL0PEGM18,
    // 32³ mesh; harness: benchmarks/compare/gaff-peo-lammps), where at np=64 the brick
    // arm spends 4.78 s in Kspace against 5.43 s replicated — the interval
    // the old rule sent to replicated. The same matrix shows np=256 with
    // the two arms tied (brick 9.85 s vs replicated 9.87 s) because 32
    // planes cannot feed 256 ranks; that tie is exactly what the subgroup
    // arm breaks, so the group size is the plane-bounded, divisor-exact
    // pppm_fft_ranks(K0, nprocs) rather than the whole communicator.
    // ATV_PPPM_ALLREDUCE=1 forces replicated, ATV_PPPM_BRICK=1 forces the
    // world-comm brick arm (both debug-only; no third key is introduced).
    if (mpi && mesh_nprocs > 1 && dd->cart_comm != MPI_COMM_NULL
        && std::getenv("ATV_PPPM_ALLREDUCE") == nullptr) {
        use_brick_mpi = true;
        if (std::getenv("ATV_PPPM_BRICK") != nullptr) {
            fft_ranks_resolved_ = mesh_nprocs;
        } else if (fft_ranks_requested_ > 0) {
            const int bound = std::min(mesh_nprocs, K0);
            if (fft_ranks_requested_ > bound)
                throw std::invalid_argument(
                    "KSpacePPPM: fft_ranks="
                    + std::to_string(fft_ranks_requested_)
                    + " exceeds min(nprocs=" + std::to_string(mesh_nprocs)
                    + ", mesh planes K0=" + std::to_string(K_[0])
                    + ") = " + std::to_string(bound)
                    + "; an FFTW-MPI slab member needs at least one plane. "
                      "Construct KSpacePPPM with fft_ranks ≤ "
                    + std::to_string(bound)
                    + ", or with fft_ranks=0 to derive it from the mesh.");
            fft_ranks_resolved_ = fft_ranks_requested_;
        } else {
            fft_ranks_resolved_ = pppm_fft_ranks(K0, mesh_nprocs);
        }
        mesh_arm_ = pppm_select_mesh_arm(K0, mesh_nprocs, fft_ranks_resolved_);
    }
#endif

    if (!use_brick_mpi) {
        // ── dens_full path (serial, or multi-rank via Allreduce) ───────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/spread", ctx.stage, kTimerCategory);
            std::fill(dens_full_.begin(), dens_full_.end(), 0.0);
            // LAMMPS make_rho on the periodic full mesh: θ[a] lands on grid
            // point base + a (base = nx + nlower), folded into [0, K).
            // Threads own disjoint slabs of x planes and each walks every
            // atom, so every grid point still sums its atoms in index order
            // (bitwise the serial result) with no private grids.
            double* __restrict dens = dens_full_.data();
#pragma omp parallel
            {
                int tid = 0, team = 1;
#ifdef _OPENMP
                tid = omp_get_thread_num();
                team = omp_get_num_threads();
#endif
                const int lo =
                    static_cast<int>(static_cast<long long>(K0) * tid / team);
                const int hi = static_cast<int>(static_cast<long long>(K0)
                                                * (tid + 1) / team);
                for (int i = 0; i < n_local; ++i) {
                    const double qi = q[i];
                    if (qi == 0.0)
                        continue;
                    const double* thx = &theta_[(3 * i + 0) * n];
                    const double* thy = &theta_[(3 * i + 1) * n];
                    const double* thz = &theta_[(3 * i + 2) * n];
                    const int b0 = base_[3 * i + 0];
                    const int b1 = base_[3 * i + 1];
                    const int b2 = base_[3 * i + 2];
                    for (int a = 0; a < n; ++a) {
                        const int g0 = wrap_k(b0 + a, K0);
                        if (g0 < lo || g0 >= hi)
                            continue;
                        const double wa = qi * thx[a];
                        for (int b = 0; b < n; ++b) {
                            const int g1 = wrap_k(b1 + b, K1);
                            const double wab = wa * thy[b];
                            const int plane = K2 * (g1 + K1 * g0);
                            for (int c = 0; c < n; ++c)
                                dens[plane + wrap_k(b2 + c, K2)] +=
                                    wab * thz[c];
                        }
                    }
                }
            } // omp parallel
        }
#ifdef ATV_HAS_MPI
        if (mpi) {
            // Same "pppm/halo" bucket as the brick arms' mesh halo: both are
            // the cost of agreeing the spread density across ranks, which is
            // the quantity the three-arm comparison is about.
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            MPI_Allreduce(MPI_IN_PLACE, dens_full_.data(),
                          static_cast<int>(dens_full_.size()), MPI_DOUBLE,
                          MPI_SUM, dd->cart_comm);
        }
#endif
        // Every rank runs the same forward FFT and G(k) on the Allreduced
        // grid. e_recip is identical on every rank; the caller adds it on
        // first_rank only. Forces are three ik inverse FFTs, gathered with θ.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
            fftw_execute(static_cast<fftw_plan>(plan_fwd_));
            const int n2 = n2c;
            const int m2_nyquist = (K2 % 2 == 0) ? (K2 / 2) : -1;
            const double inv_vol = 1.0 / vol;
            // ρ̂/V; the ik field of axis a is i·ik_a·ρ̂/V (pppm_half_mode).
            std::complex<double>* __restrict density = density_.data();
            const bool do_virial = ctx.stress_requested;
            const double* __restrict vcoef = virial_coef_.data();
            // Energy and virial: a fixed partition of the (m0, m1) rows into
            // ChunkedRange pieces, each summing into its own partial, folded
            // in chunk order — the same bits at any thread count and in any
            // run (an OpenMP `reduction` combines in thread-arrival order).
            constexpr long kGreenChunks = 64;
            const device::ChunkedRange rows(static_cast<long>(K0) * K1,
                                            kGreenChunks);
            double part[kGreenChunks][7] = {};
            const long n_rows_chunks = rows.chunks();
#pragma omp parallel for schedule(static)
            for (long c = 0; c < n_rows_chunks; ++c) {
                double e = 0, s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
                for (long r = rows.begin(c); r < rows.end(c); ++r) {
                    const int m0 = static_cast<int>(r / K1);
                    const int m1 = static_cast<int>(r % K1);
                    for (int m2 = 0; m2 < n2; ++m2) {
                        const std::size_t u =
                            static_cast<std::size_t>(m2 + n2 * (m1 + K1 * m0));
                        const double G = green_[u];
                        const auto w = work_[u];
                        density[u] = w * inv_vol;
                        if (G == 0.0)
                            continue;
                        const double q2 =
                            w.real() * w.real() + w.imag() * w.imag();
                        const double herm =
                            (m2 == 0 || m2 == m2_nyquist) ? 1.0 : 2.0;
                        e += herm * G * q2;
                        if (do_virial) {
                            const double ug = herm * q2;
                            s0 += ug * vcoef[0 * nhalf + u];
                            s1 += ug * vcoef[1 * nhalf + u];
                            s2 += ug * vcoef[2 * nhalf + u];
                            s3 += ug * vcoef[3 * nhalf + u];
                            s4 += ug * vcoef[4 * nhalf + u];
                            s5 += ug * vcoef[5 * nhalf + u];
                        }
                    }
                }
                double* pc = part[c];
                pc[0] = e;
                pc[1] = s0;
                pc[2] = s1;
                pc[3] = s2;
                pc[4] = s3;
                pc[5] = s4;
                pc[6] = s5;
            }
            for (long c = 0; c < n_rows_chunks; ++c) {
                e_recip += part[c][0];
                for (int d = 0; d < 6; ++d)
                    stress[d] += part[c][1 + d];
            }
            e_recip *= energy_scale;
            if (ctx.forces_requested) {
                const double* dens = dens_full_.data();
                AccReal* fout[3] = {fx, fy, fz};
                for (int axis = 0; axis < 3; ++axis) {
                    const double* ik = &ik_coef_[axis * nhalf];
                    std::complex<double>* __restrict wk = work_.data();
                    const std::ptrdiff_t nh =
                        static_cast<std::ptrdiff_t>(nhalf);
#pragma omp parallel for schedule(static)
                    for (std::ptrdiff_t u = 0; u < nh; ++u)
                        wk[u] = {-ik[u] * density[u].imag(),
                                 ik[u] * density[u].real()};
                    fftw_execute(static_cast<fftw_plan>(plan_bwd_));
                    AccReal* __restrict fa = fout[axis];
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < n_local; ++i) {
                        const double qi = q[i];
                        if (qi == 0.0)
                            continue;
                        const double* thx = &theta_[(3 * i + 0) * n];
                        const double* thy = &theta_[(3 * i + 1) * n];
                        const double* thz = &theta_[(3 * i + 2) * n];
                        const int b0 = base_[3 * i + 0];
                        const int b1 = base_[3 * i + 1];
                        const int b2 = base_[3 * i + 2];
                        double acc = 0;
                        for (int a = 0; a < n; ++a) {
                            const int g0 = wrap_k(b0 + a, K0);
                            const double tx = thx[a];
                            for (int b = 0; b < n; ++b) {
                                const int g1 = wrap_k(b1 + b, K1);
                                const int plane = K2 * (g1 + K1 * g0);
                                const double txy = tx * thy[b];
                                for (int c = 0; c < n; ++c) {
                                    const int g2 = wrap_k(b2 + c, K2);
                                    acc += dens[plane + g2] * txy * thz[c];
                                }
                            }
                        }
                        fa[i] += static_cast<AccReal>(-coulomb * qi * acc);
                    }
                }
            }
        }
    } else
#if defined(ATV_HAS_MPI) && defined(ATV_HAS_FFTW_MPI)
        if (mpi && dd->cart_comm != MPI_COMM_NULL) {
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/spread", ctx.stage, kTimerCategory);
            std::fill(brick_.begin(), brick_.end(), 0.0);
            // One image choice per atom, then a branch-free stencil deposit
            // into the contiguous ghost-extended brick (LAMMPS
            // `PPPM::make_rho` shape); the mesh reverse fold below routes
            // ghost cells to their owners, wherever they live.
            // LAMMPS `particle_map` aborts every rank (`error->one`) on an
            // out-of-range atom. Agree on the verdict collectively instead,
            // so every rank throws and none is left waiting in the next
            // collective.
            {
                const std::string drift = map_stencils_(n_local);
                int bad = drift.empty() ? 0 : 1, any_bad = 0;
                MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX,
                              dd->cart_comm);
                if (any_bad)
                    throw std::runtime_error(
                        bad ? drift
                            : std::string(
                                  "KSpacePPPM: another rank found an atom "
                                  "outside its ghost-extended brick (LAMMPS "
                                  "'out of range atoms - cannot compute "
                                  "PPPM')"));
            }
            // Disjoint x-plane slabs per thread, as in the full-mesh arm.
#pragma omp parallel
            {
                int tid = 0, team = 1;
#ifdef _OPENMP
                tid = omp_get_thread_num();
                team = omp_get_num_threads();
#endif
                const int lo =
                    static_cast<int>(static_cast<long long>(nxl_) * tid / team);
                const int hi = static_cast<int>(static_cast<long long>(nxl_)
                                                * (tid + 1) / team);
                for (int i = 0; i < n_local; ++i) {
                    const double qi = q[i];
                    if (qi == 0.0)
                        continue;
                    const double* thx = &theta_[(3 * i + 0) * n];
                    const double* thy = &theta_[(3 * i + 1) * n];
                    const double* thz = &theta_[(3 * i + 2) * n];
                    const int bx = stencil_min_[3 * i + 0] - nxlo_out_;
                    const int by = stencil_min_[3 * i + 1] - nylo_out_;
                    const int bz = stencil_min_[3 * i + 2] - nzlo_out_;
                    for (int a = 0; a < n; ++a) {
                        if (bx + a < lo || bx + a >= hi)
                            continue;
                        const double wa = qi * thx[a];
                        const int xoff = (bx + a) * nyl_;
                        for (int b = 0; b < n; ++b) {
                            const double wab = wa * thy[b];
                            double* __restrict row =
                                &brick_[static_cast<std::size_t>(
                                    (xoff + by + b) * nzl_ + bz)];
                            for (int c = 0; c < n; ++c)
                                row[c] += wab * thz[c];
                        }
                    }
                }
            } // omp parallel
        }

        // Mesh-halo and remap exchanges are plan-based point-to-point
        // (LAMMPS Grid3d / remap_3d ports); the plans are rebuilt with
        // the FFT tables below.
        MPI_Comm comm = dd->cart_comm;
        int nprocs = 0;
        MPI_Comm_size(comm, &nprocs);

        // ── Setup FFTW-MPI + m0 ownership table ───────────────────────
        static bool mpi_fft_inited = false;
        if (!mpi_fft_inited) {
            fftw_mpi_init();
            mpi_fft_inited = true;
        }
        if (!mpi_fft_ready_ || mpi_fft_K_[0] != K0 || mpi_fft_K_[1] != K1
            || mpi_fft_K_[2] != K2 || mpi_nprocs_ != nprocs) {
            if (plan_mpi_fwd_)
                fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_fwd_));
            if (plan_mpi_bwd_)
                fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_bwd_));
            if (mpi_data_)
                fftw_free(mpi_data_);
            plan_mpi_fwd_ = plan_mpi_bwd_ = nullptr;
            mpi_data_ = nullptr;

            // Old subgroup first: a shape change re-derives the group, and
            // the plans destroyed just above were the only users of it.
            if (owns_fft_comm_ && fft_comm_ != MPI_COMM_NULL) {
                int fft_comm_finalized = 0;
                MPI_Finalized(&fft_comm_finalized);
                if (!fft_comm_finalized)
                    MPI_Comm_free(&fft_comm_);
            }
            fft_comm_ = MPI_COMM_NULL;
            owns_fft_comm_ = false;

            // Divergent split parameters make MPI_Comm_split *hang*, not
            // fail. This Allreduce over the parent communicator — every
            // rank, member or not — turns that into a collective error
            // naming the operation.
            auto mix = [](std::uint64_t h, long long v) {
                return h
                       ^ (static_cast<std::uint64_t>(v) + 0x9e3779b97f4a7c15ULL
                          + (h << 6) + (h >> 2));
            };
            std::uint64_t digest = mix(0, K0);
            digest = mix(digest, K1);
            digest = mix(digest, K2);
            digest = mix(digest, fft_ranks_resolved_);
            digest = mix(digest, static_cast<long long>(mesh_arm_));
            comm::require_collective_fingerprint(digest, comm,
                                                 "KSpacePPPM::fft_comm");

            int cart_rank = 0;
            MPI_Comm_rank(comm, &cart_rank);
            if (fft_ranks_resolved_ == nprocs) {
                // World arm: alias, never split. Saves the collective and
                // keeps the tree's only derived communicator to the case
                // that actually needs one.
                fft_comm_ = comm;
                owns_fft_comm_ = false;
            } else {
                // Members are cart ranks [0, fft_ranks_resolved_); the key is
                // the cart rank, so subgroup rank i is cart rank i and the
                // slab order matches the ownership table below. Cart rank 0
                // is a member by construction — the energy accumulation on
                // first_rank depends on it, and on cart_comm being a plain
                // world alias (domain.cpp init_topology): a reordered
                // MPI_Cart_create comm could make first_rank a non-member
                // and silently drop e_recip.
                MPI_Comm_split(
                    comm, cart_rank < fft_ranks_resolved_ ? 0 : MPI_UNDEFINED,
                    cart_rank, &fft_comm_);
                owns_fft_comm_ = fft_comm_ != MPI_COMM_NULL;
            }

            // FFTW-MPI r2c: local_size uses the *complex* last dim (n2/2+1),
            // not the real n2. Plan creation still takes full n2. See FFTW
            // "Multi-dimensional MPI DFTs of Real Data".
            ptrdiff_t local_n0 = 0, local_0_start = 0;
            if (fft_comm_ != MPI_COMM_NULL) {
                ptrdiff_t alloc = fftw_mpi_local_size_3d(
                    K0, K1, n2c, fft_comm_, &local_n0, &local_0_start);
                mpi_local_n0_ = static_cast<long long>(local_n0);
                mpi_local_0_start_ = static_cast<long long>(local_0_start);
                mpi_alloc_ = static_cast<long long>(alloc);
                // In-place r2c: alloc is in complex elements → 2*alloc doubles.
                mpi_data_ = fftw_alloc_real(static_cast<size_t>(2 * alloc));
                // Non-transposed I/O so brick↔FFT remap stays on n0 slabs.
                plan_mpi_fwd_ = fftw_mpi_plan_dft_r2c_3d(
                    K0, K1, K2, mpi_data_,
                    reinterpret_cast<fftw_complex*>(mpi_data_), fft_comm_,
                    FFTW_ESTIMATE);
                plan_mpi_bwd_ = fftw_mpi_plan_dft_c2r_3d(
                    K0, K1, K2, reinterpret_cast<fftw_complex*>(mpi_data_),
                    mpi_data_, fft_comm_, FFTW_ESTIMATE);
                if (!plan_mpi_fwd_ || !plan_mpi_bwd_ || !mpi_data_)
                    throw std::runtime_error(
                        "KSpacePPPM: FFTW-MPI slab plan failed");
            } else {
                // Non-member: no plan, no buffer, no planes. It still runs
                // spread / mesh halo / both remaps / interpolation.
                mpi_local_n0_ = 0;
                mpi_local_0_start_ = 0;
                mpi_alloc_ = 0;
            }

            // Gather every rank's FFT m0 ownership for remap. Indexed by
            // *parent* rank (length nprocs) so both Alltoallv remaps stay
            // parent-comm collectives; non-members contribute count 0.
            std::vector<int> all_n0(static_cast<std::size_t>(nprocs));
            std::vector<int> all_s0(static_cast<std::size_t>(nprocs));
            int my_n0 = static_cast<int>(local_n0);
            int my_s0 = static_cast<int>(local_0_start);
            MPI_Allgather(&my_n0, 1, MPI_INT, all_n0.data(), 1, MPI_INT, comm);
            MPI_Allgather(&my_s0, 1, MPI_INT, all_s0.data(), 1, MPI_INT, comm);
            fft_m0_counts_ = all_n0;
            fft_m0_starts_ = all_s0;
            mpi_fft_K_[0] = K0;
            mpi_fft_K_[1] = K1;
            mpi_fft_K_[2] = K2;
            mpi_nprocs_ = nprocs;
            mesh_plans_ready_ = false;
            mpi_fft_ready_ = true;
        }

        // Grid plans follow the FFT tables; a shape/group rebuild above
        // dropped them.
        if (!mesh_plans_ready_) {
            int me_rank = 0;
            MPI_Comm_rank(comm, &me_rank);
            build_grid_plans_(comm, me_rank, nprocs);
        }

        // A non-member owns no plane (ln0 == 0), so every slab loop below
        // is empty for it; it still spreads, folds, remaps and interpolates.
        const bool fft_member = fft_comm_ != MPI_COMM_NULL;
        const ptrdiff_t ln0 = static_cast<ptrdiff_t>(mpi_local_n0_);
        const ptrdiff_t l0s = static_cast<ptrdiff_t>(mpi_local_0_start_);

        // ── mesh reverse fold: ghost density into owners ───────────────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            run_grid_plan_(mesh_rev_, comm, /*tag_base=*/310,
                           /*src_brick=*/true, /*dst_brick=*/true,
                           /*add=*/true);
        }

        // ── brick → FFT remap: plan-based p2p, values only ─────────────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/remap", ctx.stage, kTimerCategory);
            run_grid_plan_(remap_fwd_, comm, /*tag_base=*/330,
                           /*src_brick=*/true, /*dst_brick=*/false,
                           /*add=*/false);
        }

        // ── forward FFT + ik Green's function (FFT group only) ────────
        // ρ̂/V is kept aside. Each Cartesian force is one c2r of
        // i·ik_a·ρ̂/V (pppm_half_mode), then the same brick remap and a θ
        // gather. Energy and the
        // virial are reduced once, before that axis loop.
        std::vector<std::complex<double>> potential;
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
            if (fft_member)
                fftw_execute(static_cast<fftw_plan>(plan_mpi_fwd_));

            const int m2_nyquist = (K2 % 2 == 0) ? (K2 / 2) : -1;
            const double inv_vol = 1.0 / vol;
            double e_local = 0;
            if (fft_member) {
                auto* cdata = reinterpret_cast<fftw_complex*>(mpi_data_);
                potential.assign(static_cast<std::size_t>(ln0)
                                     * static_cast<std::size_t>(K1)
                                     * static_cast<std::size_t>(n2c),
                                 {});
                for (ptrdiff_t a = 0; a < ln0; ++a) {
                    const int m0 = static_cast<int>(l0s + a);
                    for (int m1 = 0; m1 < K1; ++m1) {
                        for (int m2 = 0; m2 < n2c; ++m2) {
                            const ptrdiff_t cidx =
                                m2
                                + static_cast<ptrdiff_t>(n2c) * (m1 + K1 * a);
                            const int gidx = m2 + n2c * (m1 + K1 * m0);
                            const double G =
                                green_[static_cast<std::size_t>(gidx)];
                            const double re = cdata[cidx][0];
                            const double im = cdata[cidx][1];
                            potential[static_cast<std::size_t>(cidx)] = {
                                re * inv_vol, im * inv_vol};
                            if (G == 0.0)
                                continue;
                            const double q2 = re * re + im * im;
                            const double herm =
                                (m2 == 0 || m2 == m2_nyquist) ? 1.0 : 2.0;
                            e_local += herm * G * q2;
                            tally_mode(static_cast<std::size_t>(gidx),
                                       herm * q2);
                        }
                    }
                }
                // Reciprocal energy lives on the FFT group. A non-member
                // keeps e_recip = 0: only first_rank (cart rank 0, always a
                // member) banks it into the Frame below.
                MPI_Allreduce(&e_local, &e_recip, 1, MPI_DOUBLE, MPI_SUM,
                              fft_comm_);
                e_recip *= energy_scale;
                if (ctx.stress_requested)
                    MPI_Allreduce(MPI_IN_PLACE, stress, 6, MPI_DOUBLE, MPI_SUM,
                                  fft_comm_);
            }
        }

        if (ctx.forces_requested) {
            AccReal* fout[3] = {fx, fy, fz};
            for (int axis = 0; axis < 3; ++axis) {
                {
                    ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage,
                                      kTimerCategory);
                    if (fft_member) {
                        auto* cdata =
                            reinterpret_cast<fftw_complex*>(mpi_data_);
                        for (ptrdiff_t a = 0; a < ln0; ++a) {
                            const int m0 = static_cast<int>(l0s + a);
                            for (int m1 = 0; m1 < K1; ++m1) {
                                for (int m2 = 0; m2 < n2c; ++m2) {
                                    const ptrdiff_t cidx =
                                        m2
                                        + static_cast<ptrdiff_t>(n2c)
                                              * (m1 + K1 * a);
                                    const std::size_t gidx =
                                        static_cast<std::size_t>(
                                            m2 + n2c * (m1 + K1 * m0));
                                    const double ik =
                                        ik_coef_[axis * nhalf + gidx];
                                    const auto rho =
                                        potential[static_cast<std::size_t>(
                                            cidx)];
                                    cdata[cidx][0] = -ik * rho.imag();
                                    cdata[cidx][1] = ik * rho.real();
                                }
                            }
                        }
                        fftw_execute(static_cast<fftw_plan>(plan_mpi_bwd_));
                    }
                }
                // Members finish the backward FFT before this collective.
                // Non-members still remap and interpolate. Waitall inside
                // the plan reclaims the tag, so the next axis can reuse it.
                {
                    ATV_PROFILE_SCOPE(ctx, "pppm/remap", ctx.stage,
                                      kTimerCategory);
                    run_grid_plan_(remap_bwd_, comm, /*tag_base=*/340,
                                   /*src_brick=*/false, /*dst_brick=*/true,
                                   /*add=*/false);
                }
                {
                    ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage,
                                      kTimerCategory);
                    run_grid_plan_(mesh_fwd_, comm, /*tag_base=*/320,
                                   /*src_brick=*/true, /*dst_brick=*/true,
                                   /*add=*/false);
                }
                {
                    ATV_PROFILE_SCOPE(ctx, "pppm/interp", ctx.stage,
                                      kTimerCategory);
                    AccReal* __restrict fa = fout[axis];
#pragma omp parallel for schedule(static)
                    for (int i = 0; i < n_local; ++i) {
                        const double qi = q[i];
                        if (qi == 0.0)
                            continue;
                        const double* thx = &theta_[(3 * i + 0) * n];
                        const double* thy = &theta_[(3 * i + 1) * n];
                        const double* thz = &theta_[(3 * i + 2) * n];
                        const int bx = stencil_min_[3 * i + 0] - nxlo_out_;
                        const int by = stencil_min_[3 * i + 1] - nylo_out_;
                        const int bz = stencil_min_[3 * i + 2] - nzlo_out_;
                        double acc = 0;
                        for (int a = 0; a < n; ++a) {
                            const int xoff = (bx + a) * nyl_;
                            const double tx = thx[a];
                            for (int b = 0; b < n; ++b) {
                                const double* __restrict row =
                                    &brick_[static_cast<std::size_t>(
                                        (xoff + by + b) * nzl_ + bz)];
                                const double txy = tx * thy[b];
                                for (int c = 0; c < n; ++c)
                                    acc += row[c] * txy * thz[c];
                            }
                        }
                        fa[i] += static_cast<AccReal>(-coulomb * qi * acc);
                    }
                }
            }
        }
    } else
#endif // MPI + FFTW_MPI
    {
#if defined(ATV_HAS_MPI) && !defined(ATV_HAS_FFTW_MPI)
        if (mpi)
            throw std::runtime_error(
                "KSpacePPPM multi-rank brick path requires FFTW-MPI "
                "(libfftw3_mpi). Rebuild with FFTW MPI or run single-rank.");
#endif
        (void)0; // serial handled above
    }

    double qsq_local = 0, q_local = 0;
    for (int i = 0; i < n_local; ++i) {
        qsq_local += q[i] * q[i];
        q_local += q[i];
    }
    const double e_self =
        longrange::self_energy_prefactor(alpha_, ctx.runtime.units.qqrd2e())
        * qsq_local;
    f.meta.get(kEnergy) += e_self;
    f.meta.get(kEnergyPair) += e_self;

    double q_total = q_local;
#ifdef ATV_HAS_MPI
    if (mpi)
        MPI_Allreduce(MPI_IN_PLACE, &q_total, 1, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
#endif
    if (first_rank) {
        const double e_net = longrange::background_energy(
            q_total, vol, alpha_, ctx.runtime.units.qqrd2e());
        f.meta.get(kEnergy) += e_recip + e_net;
        f.meta.get(kEnergyPair) += e_recip + e_net;
        if (ctx.stress_requested) {
            for (int d = 0; d < 6; ++d)
                f.meta.get(kStress)[d] +=
                    energy_scale * stress[d] + (d < 3 ? e_net : 0.0);
        }
    }
}

void KSpacePPPM::release_storage() noexcept {
    if (plan_fwd_)
        fftw_destroy_plan(static_cast<fftw_plan>(plan_fwd_));
    if (plan_bwd_)
        fftw_destroy_plan(static_cast<fftw_plan>(plan_bwd_));
    plan_fwd_ = plan_bwd_ = nullptr;
#ifdef ATV_HAS_FFTW_MPI
    // FFTW-MPI plans hold a duplicated communicator; destroying one after
    // MPI_Finalize hits MPI_Comm_free on a dead runtime (MPICH aborts and
    // eats any buffered stdout). A stack-allocated driver destroyed after
    // MPI_Finalize is legal user code, so leak the plans in that case.
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        if (plan_mpi_fwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_fwd_));
        if (plan_mpi_bwd_)
            fftw_destroy_plan(static_cast<fftw_plan>(plan_mpi_bwd_));
    }
    plan_mpi_fwd_ = plan_mpi_bwd_ = nullptr;
    if (mpi_data_) {
        fftw_free(mpi_data_);
        mpi_data_ = nullptr;
    }
    mpi_fft_ready_ = false;
#endif
#ifdef ATV_HAS_MPI
    // Same guard: the world arm only *aliases* cart_comm, so freeing is
    // conditional on having split one ourselves.
    if (owns_fft_comm_ && fft_comm_ != MPI_COMM_NULL) {
        int fft_comm_finalized = 0;
        MPI_Finalized(&fft_comm_finalized);
        if (!fft_comm_finalized)
            MPI_Comm_free(&fft_comm_);
    }
    fft_comm_ = MPI_COMM_NULL;
    owns_fft_comm_ = false;
#endif
    std::vector<double>().swap(brick_);
    std::vector<double>().swap(dens_full_);
    std::vector<std::complex<double>>().swap(work_);
    std::vector<double>().swap(green_);
    std::vector<double>().swap(ik_coef_);
    std::vector<double>().swap(virial_coef_);
    std::vector<int>().swap(base_);
    std::vector<double>().swap(theta_);
    std::vector<int>().swap(stencil_min_);
    mesh_rev_ = {};
    mesh_fwd_ = {};
    remap_fwd_ = {};
    remap_bwd_ = {};
    std::vector<double>().swap(grid_send_);
    std::vector<double>().swap(grid_recv_);
    std::vector<double>().swap(grid_self_);
    mesh_plans_ready_ = false;
    K_[0] = K_[1] = K_[2] = 0;
    atom_capacity_ = 0;
    box_identity_ = nullptr;
    cell_revision_ = 0;
    have_grid_ = false;
    for (int d = 0; d < 3; ++d) {
        part_K_[d] = 0;
        part_dims_[d] = 0;
        part_dist_cells_[d] = 0.0;
    }
    part_skin_ = -1.0;
}

void KSpacePPPM::destroy(const DestroyContext&) { release_storage(); }

std::size_t KSpacePPPM::memory_bytes() const {
    std::size_t b =
        (brick_.capacity() + dens_full_.capacity() + green_.capacity()
         + ik_coef_.capacity() + virial_coef_.capacity() + theta_.capacity()
         + grid_send_.capacity() + grid_recv_.capacity()
         + grid_self_.capacity())
            * sizeof(double)
        + work_.capacity() * sizeof(std::complex<double>)
        + (base_.capacity() + stencil_min_.capacity()) * sizeof(int);
    if (mpi_data_)
        b += static_cast<std::size_t>(2 * mpi_alloc_) * sizeof(double);
    return b;
}

} // namespace cpu
} // namespace ATV
