/// @file src/cpu/longrange/pppm.cpp
/// @brief Brick-decomposed PPPM: local brick + plan-based mesh ghost fold /
///        fill + pencil FFT (`PencilFft`), all point-to-point.
///
/// The mesh ghost exchange is a structural port of LAMMPS `src/grid3d.cpp`
/// (`Grid3d::setup_comm` ghost-box overlap with periodic images), checkout
/// `78a7ddf`: plans built once per mesh shape / decomposition, replayed each
/// step as value-only Irecv/Isend/Waitall between overlapping ranks — no
/// per-step collectives. The brick↔pencil remap and the distributed
/// transform live in `grid_remap.cpp` / `pencil_fft.cpp` (LAMMPS
/// `remap.cpp` / `fft3d.cpp`, checkout e8c4c278a8).
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
#ifdef ATV_HAS_MPI
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
      field_brick_(std::move(other.field_brick_)),
      // The serial FFTW plans are bound to these two buffers; a vector move
      // keeps the storage they point at.
      dens_full_(std::move(other.dens_full_)), work_(std::move(other.work_)),
      density_(std::move(other.density_)), green_(std::move(other.green_)),
      ik_coef_(std::move(other.ik_coef_)),
      virial_coef_(std::move(other.virial_coef_)), plan_fwd_(other.plan_fwd_),
      plan_bwd_(other.plan_bwd_), mpi_fft_ready_(other.mpi_fft_ready_),
#ifdef ATV_HAS_MPI
      fft_(std::move(other.fft_)),
#endif
      spectrum_(std::move(other.spectrum_)), owned_(std::move(other.owned_)),
      mpi_nprocs_(other.mpi_nprocs_),
      fft_ranks_resolved_(other.fft_ranks_resolved_),
      mesh_arm_(other.mesh_arm_), mesh_rev_(std::move(other.mesh_rev_)),
      mesh_fwd_(std::move(other.mesh_fwd_)),
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
    other.mpi_fft_ready_ = false;
    other.fft_ranks_resolved_ = 0;
    other.mesh_arm_ = PppmMeshArm::Replicated;
    other.atom_capacity_ = 0;
#ifdef ATV_HAS_MPI
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
    if (brick_.size() != n) {
        brick_.assign(n, 0.0);
        field_brick_.assign(3 * n, 0.0);
    }
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
#ifdef ATV_HAS_MPI
        fft_.reset();
#endif
        mpi_fft_ready_ = false;
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

#ifdef ATV_HAS_MPI

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

    // One scratch sized for the wider plan; plans never run concurrently.
    // mesh_fwd_ carries the three field bricks.
    const std::size_t max_send = static_cast<std::size_t>(
        std::max(mesh_rev_.sendtot, mesh_fwd_.sendtot));
    const std::size_t max_recv = static_cast<std::size_t>(
        std::max(mesh_rev_.recvtot, mesh_fwd_.recvtot));
    const std::size_t max_self = static_cast<std::size_t>(
        std::max(mesh_rev_.selftot, mesh_fwd_.selftot));
    grid_send_.resize(3 * max_send);
    grid_recv_.resize(3 * max_recv);
    grid_self_.resize(3 * max_self);
    grid_reqs_.resize(
        std::max(mesh_rev_.sends.size() + mesh_rev_.recvs.size(),
                 mesh_fwd_.sends.size() + mesh_fwd_.recvs.size()));
    mesh_plans_ready_ = true;
}

void KSpacePPPM::run_grid_plan_(const GridPlan& plan, MPI_Comm comm,
                                int tag_base, double* data, std::size_t stride,
                                bool add, int nfield) {
    auto pack_box = [&](const GridMsg& m, double* buf) {
        const std::size_t nz = static_cast<std::size_t>(m.zhi - m.zlo + 1);
        for (int c = 0; c < nfield; ++c) {
            const double* comp = data + static_cast<std::size_t>(c) * stride;
            for (int gx = m.xlo; gx <= m.xhi; ++gx)
                for (int gy = m.ylo; gy <= m.yhi; ++gy) {
                    std::memcpy(buf, comp + brick_index_(gx, gy, m.zlo),
                                nz * sizeof(double));
                    buf += nz;
                }
        }
    };
    auto unpack_box = [&](const GridMsg& m, const double* buf) {
        const int nz = m.zhi - m.zlo + 1;
        for (int c = 0; c < nfield; ++c) {
            double* comp = data + static_cast<std::size_t>(c) * stride;
            for (int gx = m.xlo; gx <= m.xhi; ++gx)
                for (int gy = m.ylo; gy <= m.yhi; ++gy) {
                    double* row = comp + brick_index_(gx, gy, m.zlo);
                    if (add) {
                        for (int k = 0; k < nz; ++k)
                            row[k] += buf[k];
                    } else {
                        std::memcpy(row, buf,
                                    static_cast<std::size_t>(nz)
                                        * sizeof(double));
                    }
                    buf += nz;
                }
        }
    };

    const std::size_t nf = static_cast<std::size_t>(nfield);
    int nreq = 0;
    for (const GridMsg& m : plan.recvs)
        MPI_Irecv(grid_recv_.data() + nf * m.bufloc, nfield * m.size,
                  MPI_DOUBLE, m.proc, tag_base, comm,
                  &grid_reqs_[static_cast<std::size_t>(nreq++)]);
    for (const GridMsg& m : plan.sends) {
        pack_box(m, grid_send_.data() + nf * m.bufloc);
        MPI_Isend(grid_send_.data() + nf * m.bufloc, nfield * m.size,
                  MPI_DOUBLE, m.proc, tag_base, comm,
                  &grid_reqs_[static_cast<std::size_t>(nreq++)]);
    }
    for (std::size_t k = 0; k < plan.self_src.size(); ++k) {
        const GridMsg& sm = plan.self_src[k];
        pack_box(sm, grid_self_.data() + nf * sm.bufloc);
        unpack_box(plan.self_dst[k], grid_self_.data() + nf * sm.bufloc);
    }
    if (nreq > 0)
        MPI_Waitall(nreq, grid_reqs_.data(), MPI_STATUSES_IGNORE);
    for (const GridMsg& m : plan.recvs)
        unpack_box(m, grid_recv_.data() + nf * m.bufloc);
}

#endif // ATV_HAS_MPI

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
    bool recip_is_partial = false; // e_recip / stress are this rank's share
    double q_total = 0;            // net charge of the whole system
    bool q_total_reduced = false;  // already summed over ranks
    double stress[6]{};
    // ρ̂ here is the unnormalized FFT of q·θ, with no N/V. LAMMPS stores
    // (N/V) times that density, so its (G/N) ρ̂_lammps is our (G/V) ρ̂ and
    // the energy prefactor is coulomb/(2V), not the SPME coulomb/(2πV).
    const double energy_scale =
        ctx.runtime.units.coulomb_constant() / (2.0 * vol);
    const double coulomb = ctx.runtime.units.coulomb_constant();
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
    // interpolation. Its "FFT group" is therefore every mesh rank. It is
    // what is left when there is no DD, or when ATV_PPPM_ALLREDUCE forces
    // it.
    mesh_arm_ = PppmMeshArm::Replicated;
    fft_ranks_resolved_ = mesh_nprocs;
    bool use_brick_mpi = false;
#ifdef ATV_HAS_MPI
    // The brick family is the multi-rank default, with pencils on every
    // rank as in LAMMPS: on the 20,048-atom GAFF PEO/LiTFSI melt (64³ mesh,
    // EPYC 9965, 2026-11-27, 192 ranks) every rank ran 38.9 ns/day against
    // 36.5 for 96 pencil ranks and 33.7 for 48.
    // ATV_PPPM_ALLREDUCE=1 forces replicated, ATV_PPPM_BRICK=1 forces the
    // world brick arm (both debug-only).
    if (mpi && mesh_nprocs > 1 && dd->cart_comm != MPI_COMM_NULL
        && std::getenv("ATV_PPPM_ALLREDUCE") == nullptr) {
        use_brick_mpi = true;
        if (std::getenv("ATV_PPPM_BRICK") != nullptr) {
            fft_ranks_resolved_ = mesh_nprocs;
        } else if (fft_ranks_requested_ > 0) {
            if (fft_ranks_requested_ > mesh_nprocs)
                throw std::invalid_argument(
                    "KSpacePPPM: fft_ranks="
                    + std::to_string(fft_ranks_requested_)
                    + " exceeds nprocs=" + std::to_string(mesh_nprocs)
                    + ". Construct KSpacePPPM with fft_ranks ≤ "
                    + std::to_string(mesh_nprocs)
                    + ", or with fft_ranks=0 to use every rank.");
            fft_ranks_resolved_ = fft_ranks_requested_;
        } else {
            // Every rank holds pencils, as in LAMMPS.
            fft_ranks_resolved_ = mesh_nprocs;
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
#ifdef ATV_HAS_MPI
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
            // collective. The same reduction carries the net charge the
            // background term needs, so the solve has one collective.
            {
                const std::string drift = map_stencils_(n_local);
                const int bad = drift.empty() ? 0 : 1;
                double agree[2] = {static_cast<double>(bad), 0.0};
                for (int i = 0; i < n_local; ++i)
                    agree[1] += q[i];
                {
                    ATV_PROFILE_SCOPE(ctx, "pppm/agree", ctx.stage,
                                      kTimerCategory);
                    MPI_Allreduce(MPI_IN_PLACE, agree, 2, MPI_DOUBLE, MPI_SUM,
                                  dd->cart_comm);
                }
                const bool any_bad = agree[0] > 0.0;
                q_total = agree[1];
                q_total_reduced = true;
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

        // Mesh ghosts move on point-to-point plans (LAMMPS Grid3d port) and
        // the transform is the pencil FFT (LAMMPS fft_3d port). Both follow
        // the mesh, the bricks, the rank count and the FFT group.
        MPI_Comm comm = dd->cart_comm;
        int nprocs = 0;
        MPI_Comm_size(comm, &nprocs);
        const int nxo = nxhi_in_ - nxlo_in_ + 1;
        const int nyo = nyhi_in_ - nylo_in_ + 1;
        const int nzo = nzhi_in_ - nzlo_in_ + 1;
        if (!mpi_fft_ready_ || !mesh_plans_ready_ || mpi_fft_K_[0] != K0
            || mpi_fft_K_[1] != K1 || mpi_fft_K_[2] != K2
            || mpi_nprocs_ != nprocs) {
            // Ranks that disagree on the mesh or the group would build
            // plans that never match and hang in the first exchange. This
            // Allreduce turns that into a collective error naming the
            // operation.
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
                                                 "KSpacePPPM::pencil_fft");

            fft_.reset();
            fft_ = std::make_unique<longrange::PencilFft>(
                comm, std::array<int, 3>{K0, K1, K2},
                std::array<int, 6>{nxlo_in_, nxhi_in_, nylo_in_, nyhi_in_,
                                   nzlo_in_, nzhi_in_},
                fft_ranks_resolved_);
            spectrum_.assign(fft_->spectrum_size(), {});
            owned_.assign(static_cast<std::size_t>(nxo)
                              * static_cast<std::size_t>(nyo)
                              * static_cast<std::size_t>(nzo),
                          0.0);
            mpi_fft_K_[0] = K0;
            mpi_fft_K_[1] = K1;
            mpi_fft_K_[2] = K2;
            mpi_nprocs_ = nprocs;
            mesh_plans_ready_ = false;
            mpi_fft_ready_ = true;
        }
        if (!mesh_plans_ready_) {
            int me_rank = 0;
            MPI_Comm_rank(comm, &me_rank);
            build_grid_plans_(comm, me_rank, nprocs);
        }

        // ── mesh reverse fold: ghost density into owners ───────────────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            run_grid_plan_(mesh_rev_, comm, /*tag_base=*/310, brick_.data(),
                           /*stride=*/0, /*add=*/true, /*nfield=*/1);
        }

        // This rank's modes (PencilFft spectrum layout): every m0, a block
        // of (m1, m2).
        const auto modes = fft_->spectrum_box();
        const int m1lo = modes[0], m1hi = modes[1];
        const int m2lo = modes[2], m2hi = modes[3];
        const int nm2 = m2hi - m2lo + 1;
        auto mode_index = [&](int m0, int m1, int m2) {
            return static_cast<std::size_t>(
                (static_cast<std::ptrdiff_t>(m1 - m1lo) * nm2 + (m2 - m2lo))
                    * K0
                + m0);
        };

        // ── forward FFT + energy / virial ─────────────────────────────
        // ρ̂/V is kept aside. Each Cartesian field is one backward
        // transform of i·ik_a·ρ̂/V (pppm_half_mode). Energy and virial are
        // this rank's share of the mode sum; the Frame totals are reduced
        // over ranks like every other energy term.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
            // LAMMPS brick2fft: the owned cells, without ghosts.
            for (int gx = nxlo_in_; gx <= nxhi_in_; ++gx)
                for (int gy = nylo_in_; gy <= nyhi_in_; ++gy)
                    std::memcpy(
                        &owned_[(static_cast<std::size_t>(gx - nxlo_in_) * nyo
                                 + (gy - nylo_in_))
                                * nzo],
                        &brick_[static_cast<std::size_t>(
                            brick_index_(gx, gy, nzlo_in_))],
                        static_cast<std::size_t>(nzo) * sizeof(double));
            fft_->forward(owned_.data());

            const int m2_nyquist = (K2 % 2 == 0) ? (K2 / 2) : -1;
            const double inv_vol = 1.0 / vol;
            const std::complex<double>* spec = fft_->spectrum();
            double e_local = 0;
            for (int m1 = m1lo; m1 <= m1hi; ++m1) {
                for (int m2 = m2lo; m2 <= m2hi; ++m2) {
                    const double herm =
                        (m2 == 0 || m2 == m2_nyquist) ? 1.0 : 2.0;
                    for (int m0 = 0; m0 < K0; ++m0) {
                        const std::size_t cidx = mode_index(m0, m1, m2);
                        const std::size_t gidx =
                            static_cast<std::size_t>(m2 + n2c * (m1 + K1 * m0));
                        const double G = green_[gidx];
                        const double re = spec[cidx].real();
                        const double im = spec[cidx].imag();
                        spectrum_[cidx] = {re * inv_vol, im * inv_vol};
                        if (G == 0.0)
                            continue;
                        const double q2 = re * re + im * im;
                        e_local += herm * G * q2;
                        tally_mode(gidx, herm * q2);
                    }
                }
            }
            e_recip = e_local * energy_scale;
            recip_is_partial = true;
        }

        // LAMMPS `poisson_ik` + `fieldforce_ik` shape: three backward
        // transforms into the three field bricks, then one ghost exchange
        // (`FORWARD_IK`) and one interpolation pass for all three.
        if (ctx.forces_requested) {
            const std::size_t nbrick = brick_.size();
            {
                ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
                std::complex<double>* work = fft_->spectrum();
                for (int axis = 0; axis < 3; ++axis) {
                    const double* ik_axis = &ik_coef_[axis * nhalf];
                    for (int m1 = m1lo; m1 <= m1hi; ++m1)
                        for (int m2 = m2lo; m2 <= m2hi; ++m2)
                            for (int m0 = 0; m0 < K0; ++m0) {
                                const std::size_t cidx = mode_index(m0, m1, m2);
                                const double ik =
                                    ik_axis[m2 + n2c * (m1 + K1 * m0)];
                                const auto rho = spectrum_[cidx];
                                work[cidx] = {-ik * rho.imag(),
                                              ik * rho.real()};
                            }
                    fft_->backward(owned_.data());
                    double* comp = field_brick_.data() + axis * nbrick;
                    for (int gx = nxlo_in_; gx <= nxhi_in_; ++gx)
                        for (int gy = nylo_in_; gy <= nyhi_in_; ++gy)
                            std::memcpy(
                                comp + brick_index_(gx, gy, nzlo_in_),
                                &owned_[(static_cast<std::size_t>(gx - nxlo_in_)
                                             * nyo
                                         + (gy - nylo_in_))
                                        * nzo],
                                static_cast<std::size_t>(nzo) * sizeof(double));
                }
            }
            {
                ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
                run_grid_plan_(mesh_fwd_, comm, /*tag_base=*/320,
                               field_brick_.data(), nbrick, /*add=*/false,
                               /*nfield=*/3);
            }
            {
                ATV_PROFILE_SCOPE(ctx, "pppm/interp", ctx.stage,
                                  kTimerCategory);
                const double* __restrict ex = field_brick_.data();
                const double* __restrict ey = ex + nbrick;
                const double* __restrict ez = ey + nbrick;
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
                    double ax = 0, ay = 0, az = 0;
                    for (int a = 0; a < n; ++a) {
                        const int xoff = (bx + a) * nyl_;
                        const double tx = thx[a];
                        for (int b = 0; b < n; ++b) {
                            const std::size_t row = static_cast<std::size_t>(
                                (xoff + by + b) * nzl_ + bz);
                            const double txy = tx * thy[b];
                            for (int c = 0; c < n; ++c) {
                                const double w = txy * thz[c];
                                ax += ex[row + c] * w;
                                ay += ey[row + c] * w;
                                az += ez[row + c] * w;
                            }
                        }
                    }
                    const double scale = -coulomb * qi;
                    fx[i] += static_cast<AccReal>(scale * ax);
                    fy[i] += static_cast<AccReal>(scale * ay);
                    fz[i] += static_cast<AccReal>(scale * az);
                }
            }
        }
    } else
#endif // ATV_HAS_MPI
    {
        (void)0; // serial handled above
    }

    double qsq_local = 0, q_local = 0;
    for (int i = 0; i < n_local; ++i) {
        qsq_local += q[i] * q[i];
        q_local += q[i];
    }
    const double e_self = longrange::self_energy_prefactor(
                              alpha_, ctx.runtime.units.coulomb_constant())
                          * qsq_local;
    f.meta.get(kEnergy) += e_self;
    f.meta.get(kEnergyPair) += e_self;

    if (!q_total_reduced)
        q_total = q_local;
#ifdef ATV_HAS_MPI
    if (mpi && !q_total_reduced)
        MPI_Allreduce(MPI_IN_PLACE, &q_total, 1, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
#endif
    // The pencil arm holds this rank's share of the mode sum; the serial
    // and replicated arms hold the whole sum on every rank, banked once.
    if (first_rank || recip_is_partial) {
        f.meta.get(kEnergy) += e_recip;
        f.meta.get(kEnergyPair) += e_recip;
        if (ctx.stress_requested)
            for (int d = 0; d < 6; ++d)
                f.meta.get(kStress)[d] += energy_scale * stress[d];
    }
    if (first_rank) {
        const double e_net = longrange::background_energy(
            q_total, vol, alpha_, ctx.runtime.units.coulomb_constant());
        f.meta.get(kEnergy) += e_net;
        f.meta.get(kEnergyPair) += e_net;
        if (ctx.stress_requested)
            for (int d = 0; d < 3; ++d)
                f.meta.get(kStress)[d] += e_net;
    }
}

void KSpacePPPM::release_storage() noexcept {
    if (plan_fwd_)
        fftw_destroy_plan(static_cast<fftw_plan>(plan_fwd_));
    if (plan_bwd_)
        fftw_destroy_plan(static_cast<fftw_plan>(plan_bwd_));
    plan_fwd_ = plan_bwd_ = nullptr;
#ifdef ATV_HAS_MPI
    fft_.reset();
#endif
    mpi_fft_ready_ = false;
    std::vector<double>().swap(owned_);
    std::vector<double>().swap(brick_);
    std::vector<double>().swap(field_brick_);
    std::vector<double>().swap(dens_full_);
    std::vector<std::complex<double>>().swap(work_);
    std::vector<std::complex<double>>().swap(spectrum_);
    std::vector<double>().swap(green_);
    std::vector<double>().swap(ik_coef_);
    std::vector<double>().swap(virial_coef_);
    std::vector<int>().swap(base_);
    std::vector<double>().swap(theta_);
    std::vector<int>().swap(stencil_min_);
    mesh_rev_ = {};
    mesh_fwd_ = {};
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
        (brick_.capacity() + field_brick_.capacity() + dens_full_.capacity()
         + green_.capacity() + ik_coef_.capacity() + virial_coef_.capacity()
         + theta_.capacity() + grid_send_.capacity() + grid_recv_.capacity()
         + grid_self_.capacity())
            * sizeof(double)
        + (work_.capacity() + spectrum_.capacity())
              * sizeof(std::complex<double>)
        + (base_.capacity() + stencil_min_.capacity()) * sizeof(int);
    b += owned_.capacity() * sizeof(double);
#ifdef ATV_HAS_MPI
    if (fft_)
        b += fft_->memory_bytes();
#endif
    return b;
}

} // namespace cpu
} // namespace ATV
