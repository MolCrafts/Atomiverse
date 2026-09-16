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
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.
#include "atomiverse/cpu/longrange/pppm.h"

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

inline int wrap_k(int g, int K) noexcept {
    g %= K;
    if (g < 0)
        g += K;
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
    if (order < 3)
        throw std::invalid_argument("KSpacePPPM: order must be ≥ 3");
    if (fft_ranks < 0)
        throw std::invalid_argument(
            "KSpacePPPM: fft_ranks must be ≥ 0 (0 = derive from the mesh)");
}

KSpacePPPM::~KSpacePPPM() { release_storage(); }

KSpacePPPM::KSpacePPPM(KSpacePPPM&& other) noexcept
    : KSpace(std::move(other)), alpha_(other.alpha_), spacing_(other.spacing_),
      order_(other.order_), fft_ranks_requested_(other.fft_ranks_requested_),
      box_identity_(other.box_identity_), cell_revision_(other.cell_revision_),
      have_grid_(other.have_grid_), nxlo_in_(other.nxlo_in_),
      nxhi_in_(other.nxhi_in_), nylo_in_(other.nylo_in_),
      nyhi_in_(other.nyhi_in_), nzlo_in_(other.nzlo_in_),
      nzhi_in_(other.nzhi_in_), nxlo_out_(other.nxlo_out_),
      nxhi_out_(other.nxhi_out_), nylo_out_(other.nylo_out_),
      nyhi_out_(other.nyhi_out_), nzlo_out_(other.nzlo_out_),
      nzhi_out_(other.nzhi_out_), nxl_(other.nxl_), nyl_(other.nyl_),
      nzl_(other.nzl_), nghost_lo_(other.nghost_lo_),
      nghost_hi_(other.nghost_hi_), brick_(std::move(other.brick_)),
      work_(std::move(other.work_)), green_(std::move(other.green_)),
      plan_fwd_(other.plan_fwd_), plan_bwd_(other.plan_bwd_),
      mpi_fft_ready_(other.mpi_fft_ready_), mpi_local_n0_(other.mpi_local_n0_),
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
      theta_(std::move(other.theta_)), dtheta_(std::move(other.dtheta_)),
      stencil_min_(std::move(other.stencil_min_)) {
    other.mesh_plans_ready_ = false;
    for (int d = 0; d < 3; ++d) {
        K_[d] = other.K_[d];
        bsp_mod_[d] = std::move(other.bsp_mod_[d]);
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
#endif
}

void KSpacePPPM::setup_brick_partition_(const int (&coords)[3],
                                        const int (&dims)[3]) {
    // LAMMPS-style uniform partition of the mesh among the proc grid.
    auto split = [](int n, int me, int p, int& lo, int& hi) {
        lo = me * n / p;
        hi = (me + 1) * n / p - 1;
    };
    split(K_[0], coords[0], dims[0], nxlo_in_, nxhi_in_);
    split(K_[1], coords[1], dims[1], nylo_in_, nyhi_in_);
    split(K_[2], coords[2], dims[2], nzlo_in_, nzhi_in_);
    // The one-sided B-spline stencil deposits on [base-(order-1), base]
    // with base = floor(u), so an owned atom needs order-1 ghost layers
    // BELOW its subdomain and none above. On top of the stencil reach:
    //   * one cell of drift margin per side, for owned atoms that move
    //     past their subdomain edge between reneighbourings (LAMMPS
    //     Grid3d sizes its ghosts with a 0.5*skin margin likewise) — a
    //     cell covers any skin up to ~2 cell widths, beyond which
    //     map_stencils_ throws its out-of-range error rather than
    //     depositing silently wrong;
    //   * one MORE cell on the high side for the partition mismatch:
    //     owned cells tile as `i*K/p` (integer division) while atom
    //     ownership follows the spatial cut `i*L/p`, so the topmost owned
    //     atoms can map one cell past the brick's last owned cell
    //     (exactly `floor(subhi*K/L) <= in_hi + 1`).
    // Asymmetric depths still cut the mesh-halo volume by ~2x against a
    // symmetric order-deep shell.
    nghost_lo_ = order_;
    nghost_hi_ = 2;
    nxlo_out_ = nxlo_in_ - nghost_lo_;
    nxhi_out_ = nxhi_in_ + nghost_hi_;
    nylo_out_ = nylo_in_ - nghost_lo_;
    nyhi_out_ = nyhi_in_ + nghost_hi_;
    nzlo_out_ = nzlo_in_ - nghost_lo_;
    nzhi_out_ = nzhi_in_ + nghost_hi_;
    nxl_ = nxhi_out_ - nxlo_out_ + 1;
    nyl_ = nyhi_out_ - nylo_out_ + 1;
    nzl_ = nzhi_out_ - nzlo_out_ + 1;
    const std::size_t n = static_cast<std::size_t>(nxl_) * nyl_ * nzl_;
    // Resize only when the brick shape changes; compute() zeros each step.
    if (brick_.size() != n)
        brick_.assign(n, 0.0);
}

void KSpacePPPM::ensure_grid(const cpu::Frame& frame, bool initial_build) {
    const auto& box = frame.box;
    if (have_grid_ && box_identity_ == &box && box.revision() == cell_revision_)
        return;

    int next_K[3];
    for (int d = 0; d < 3; ++d) {
        next_K[d] = longrange::good_fft_size(
            static_cast<int>(std::ceil(box.length(d) / spacing_)));
        if (next_K[d] < order_ + 1)
            next_K[d] = longrange::good_fft_size(order_ + 1);
        if (d == 2 && (next_K[d] & 1))
            next_K[d] = longrange::good_fft_size(next_K[d] + 1);
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
        for (int d = 0; d < 3; ++d) {
            K_[d] = next_K[d];
            bsp_mod_[d] = longrange::spline_moduli(K_[d], order_);
        }
        const int ntot = K_[0] * K_[1] * K_[2];
        const int ncomplex = K_[0] * K_[1] * (K_[2] / 2 + 1);
        dens_full_.assign(static_cast<std::size_t>(ntot), 0.0);
        work_.assign(static_cast<std::size_t>(ncomplex),
                     std::complex<double>(0, 0));
        green_.assign(static_cast<std::size_t>(ncomplex), 0.0);

        // Serial plans bind to dens_full_ / work_ (reused every step).
        plan_fwd_ = fftw_plan_dft_r2c_3d(
            K_[0], K_[1], K_[2], dens_full_.data(),
            reinterpret_cast<fftw_complex*>(work_.data()), FFTW_MEASURE);
        plan_bwd_ = fftw_plan_dft_c2r_3d(
            K_[0], K_[1], K_[2], reinterpret_cast<fftw_complex*>(work_.data()),
            dens_full_.data(), FFTW_MEASURE);
        if (!plan_fwd_ || !plan_bwd_) {
            release_storage();
            throw std::runtime_error("KSpacePPPM: serial r2c/c2r plan failed");
        }
    }

    // SPME H(m) on half-spectrum (full m0,m1, m2=0..K2/2).
    {
        const auto& ic = box.inv_cell();
        const int K0 = K_[0], K1 = K_[1], K2 = K_[2];
        const int n2 = K2 / 2 + 1;
        const double inv_pi2_a2 =
            ATV::units::pi * ATV::units::pi / (alpha_ * alpha_);
        const double* bm0 = bsp_mod_[0].data();
        const double* bm1 = bsp_mod_[1].data();
        const double* bm2 = bsp_mod_[2].data();
        for (int m0 = 0; m0 < K0; ++m0) {
            const int h0 = (m0 <= K0 / 2) ? m0 : m0 - K0;
            const double b0 = bm0[m0];
            for (int m1 = 0; m1 < K1; ++m1) {
                const int h1 = (m1 <= K1 / 2) ? m1 : m1 - K1;
                const double b01 = b0 * bm1[m1];
                for (int m2 = 0; m2 < n2; ++m2) {
                    const int idx = m2 + n2 * (m1 + K1 * m0);
                    if (m0 == 0 && m1 == 0 && m2 == 0) {
                        green_[static_cast<std::size_t>(idx)] = 0.0;
                        continue;
                    }
                    const int h2 = m2;
                    double mx = 0, my = 0, mz = 0;
                    const int hh[3] = {h0, h1, h2};
                    for (int d = 0; d < 3; ++d) {
                        mx += hh[d] * ic(0, d);
                        my += hh[d] * ic(1, d);
                        mz += hh[d] * ic(2, d);
                    }
                    const double msq = mx * mx + my * my + mz * mz;
                    if (msq < 1e-30) {
                        green_[static_cast<std::size_t>(idx)] = 0.0;
                        continue;
                    }
                    green_[static_cast<std::size_t>(idx)] =
                        std::exp(-inv_pi2_a2 * msq) / msq * b01 * bm2[m2];
                }
            }
        }
    }

    // Default serial brick = full mesh (overwritten under multi-rank).
    int coords[3] = {0, 0, 0};
    int dims[3] = {1, 1, 1};
    setup_brick_partition_(coords, dims);

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
    dtheta_.resize(static_cast<std::size_t>(3 * order_ * next));
    stencil_min_.resize(static_cast<std::size_t>(3 * next));
    atom_capacity_ = next;
    if (!initial_build)
        note_capacity_rebuild();
}

void KSpacePPPM::map_stencils_(int n_local) {
    const int nm1 = order_ - 1;
    const int lo[3] = {nxlo_out_, nylo_out_, nzlo_out_};
    const int hi[3] = {nxhi_out_, nyhi_out_, nzhi_out_};
    for (int i = 0; i < n_local; ++i) {
        for (int d = 0; d < 3; ++d) {
            const int K = K_[d];
            const int s = base_[3 * i + d] - nm1;
            int sm = s;
            if (!(sm >= lo[d] && sm + nm1 <= hi[d])) {
                if (s + K >= lo[d] && s + K + nm1 <= hi[d])
                    sm = s + K;
                else if (s - K >= lo[d] && s - K + nm1 <= hi[d])
                    sm = s - K;
                else
                    throw std::runtime_error(
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

    // Owned-brick extents of every rank; out extents derive from the
    // uniform ghost depth. 6 ints per rank, gathered once per rebuild.
    std::vector<int> owned(static_cast<std::size_t>(6 * nprocs));
    const int mine[6] = {nxlo_in_, nxhi_in_, nylo_in_,
                         nyhi_in_, nzlo_in_, nzhi_in_};
    MPI_Allgather(const_cast<int*>(mine), 6, MPI_INT, owned.data(), 6, MPI_INT,
                  comm);

    const int K0 = K_[0], K1 = K_[1], K2 = K_[2];
    const int myout[6] = {nxlo_out_, nxhi_out_, nylo_out_,
                          nyhi_out_, nzlo_out_, nzhi_out_};

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
                        const int rout[6] = {rown[0] - nghost_lo_ + sx * K0,
                                             rown[1] + nghost_hi_ + sx * K0,
                                             rown[2] - nghost_lo_ + sy * K1,
                                             rown[3] + nghost_hi_ + sy * K1,
                                             rown[4] - nghost_lo_ + sz * K2,
                                             rown[5] + nghost_hi_ + sz * K2};
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
    note_build();
    ensure_grid(ctx.frame, true);
    ensure_atom_capacity(ctx.frame.capacity(), true);
#ifdef ATV_HAS_MPI
    // Re-partition brick from DD if multi-rank.
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1) {
        auto* dd = ctx.runtime.md.dd;
        setup_brick_partition_(dd->coords, dd->dims);
    }
#else
    (void)ctx;
#endif
}

void KSpacePPPM::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    ensure_grid(f, false);

    // Serial: Frame only — no DomainDecomposition. Multi-rank: DD bricks.
    const int n_local = owned_count(ctx);
#ifdef ATV_HAS_MPI
    const bool mpi = multi_rank_dd(ctx.runtime);
    const auto* dd = mpi ? ctx.runtime.md.dd : nullptr;
    const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;
    if (mpi)
        setup_brick_partition_(dd->coords, dd->dims);
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
    const int nm1 = n - 1;

    longrange::pme_fill_atom_splines(x, y, z, n_local, ic, K_, order_,
                                     base_.data(), theta_.data(),
                                     dtheta_.data());

    double e_recip = 0;

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
    // three-arm matrix in tools/bench_gaff_lammps/README.md (2026-08-26
    // snapshot, 24048-atom MIL0PEGM18, 32³ mesh), where at np=64 the brick
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
            longrange::pme_spread_charges_real(q, n_local, order_, K_,
                                               base_.data(), theta_.data(),
                                               dens_full_.data());
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
        // Every rank runs the (small) FFT + influence identically on the
        // Allreduced grid — deterministic, so no Bcast is needed and no
        // rank idles. e_recip is identical on every rank; the caller adds
        // it on first_rank only.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
            fftw_execute(static_cast<fftw_plan>(plan_fwd_));
            const int n2 = K2 / 2 + 1;
            const int m2_nyquist = (K2 % 2 == 0) ? (K2 / 2) : -1;
            for (int m0 = 0; m0 < K0; ++m0)
                for (int m1 = 0; m1 < K1; ++m1)
                    for (int m2 = 0; m2 < n2; ++m2) {
                        const int idx = m2 + n2 * (m1 + K1 * m0);
                        const double H = green_[static_cast<std::size_t>(idx)];
                        auto& w = work_[static_cast<std::size_t>(idx)];
                        if (H == 0.0) {
                            w = 0;
                            continue;
                        }
                        const double q2 =
                            w.real() * w.real() + w.imag() * w.imag();
                        const double herm =
                            (m2 == 0 || m2 == m2_nyquist) ? 1.0 : 2.0;
                        e_recip += herm * H * q2;
                        w *= H;
                    }
            e_recip *= ATV::units::coulomb / (2.0 * ATV::units::pi * vol);
            fftw_execute(static_cast<fftw_plan>(plan_bwd_));
        }

        // Force from dens_full_ (same as pre-brick serial path).
        ATV_PROFILE_SCOPE(ctx, "pppm/interp", ctx.stage, kTimerCategory);
        const double fscale = ATV::units::coulomb / (ATV::units::pi * vol);
        const double ic00 = ic(0, 0), ic01 = ic(0, 1), ic02 = ic(0, 2);
        const double ic10 = ic(1, 0), ic11 = ic(1, 1), ic12 = ic(1, 2);
        const double ic20 = ic(2, 0), ic21 = ic(2, 1), ic22 = ic(2, 2);
        const double K0d = static_cast<double>(K0);
        const double K1d = static_cast<double>(K1);
        const double K2d = static_cast<double>(K2);
        const double* dens = dens_full_.data();
        for (int i = 0; i < n_local; ++i) {
            const double qi = q[i];
            if (qi == 0.0)
                continue;
            const double* thx = &theta_[(3 * i + 0) * n];
            const double* thy = &theta_[(3 * i + 1) * n];
            const double* thz = &theta_[(3 * i + 2) * n];
            const double* dthx = &dtheta_[(3 * i + 0) * n];
            const double* dthy = &dtheta_[(3 * i + 1) * n];
            const double* dthz = &dtheta_[(3 * i + 2) * n];
            const int b0 = base_[3 * i + 0] - nm1;
            const int b1 = base_[3 * i + 1] - nm1;
            const int b2 = base_[3 * i + 2] - nm1;
            double g0s = 0, g1s = 0, g2s = 0;
            for (int a = 0; a < n; ++a) {
                const int g0 = wrap_k(b0 + a, K0);
                const double tx = thx[a];
                const double dxa = dthx[a];
                for (int b = 0; b < n; ++b) {
                    const int g1 = wrap_k(b1 + b, K1);
                    const int plane = K2 * (g1 + K1 * g0);
                    double s0 = 0, s1 = 0;
                    for (int c = 0; c < n; ++c) {
                        const int g2 = wrap_k(b2 + c, K2);
                        const double cval = dens[plane + g2];
                        s0 += cval * thz[c];
                        s1 += cval * dthz[c];
                    }
                    g0s += dxa * thy[b] * s0;
                    g1s += tx * dthy[b] * s0;
                    g2s += tx * thy[b] * s1;
                }
            }
            const double fxi =
                g0s * K0d * ic00 + g1s * K1d * ic01 + g2s * K2d * ic02;
            const double fyi =
                g0s * K0d * ic10 + g1s * K1d * ic11 + g2s * K2d * ic12;
            const double fzi =
                g0s * K0d * ic20 + g1s * K1d * ic21 + g2s * K2d * ic22;
            const double s = -fscale * qi;
            fx[i] += s * fxi;
            fy[i] += s * fyi;
            fz[i] += s * fzi;
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
            map_stencils_(n_local);
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
        const int n2c = K2 / 2 + 1;
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
            comm::require_collective_fingerprint(/*version=*/1, digest, comm,
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

        // ── forward FFT + influence + backward FFT (FFT group only) ───
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/fft", ctx.stage, kTimerCategory);
            if (fft_member)
                fftw_execute(static_cast<fftw_plan>(plan_mpi_fwd_));

            auto* cdata = reinterpret_cast<fftw_complex*>(mpi_data_);
            const int m2_nyquist = (K2 % 2 == 0) ? (K2 / 2) : -1;
            double e_local = 0;
            for (ptrdiff_t a = 0; a < ln0; ++a) {
                const int m0 = static_cast<int>(l0s + a);
                for (int m1 = 0; m1 < K1; ++m1) {
                    for (int m2 = 0; m2 < n2c; ++m2) {
                        const ptrdiff_t cidx =
                            m2 + static_cast<ptrdiff_t>(n2c) * (m1 + K1 * a);
                        const int gidx = m2 + n2c * (m1 + K1 * m0);
                        const double H = green_[static_cast<std::size_t>(gidx)];
                        double& re = cdata[cidx][0];
                        double& im = cdata[cidx][1];
                        if (H == 0.0) {
                            re = im = 0;
                            continue;
                        }
                        const double q2 = re * re + im * im;
                        const double herm =
                            (m2 == 0 || m2 == m2_nyquist) ? 1.0 : 2.0;
                        e_local += herm * H * q2;
                        re *= H;
                        im *= H;
                    }
                }
            }
            // Reciprocal energy lives entirely on the FFT group, so the sum
            // stays inside it — one fewer world collective. A non-member keeps
            // e_recip = 0 and never reads it: only first_rank (cart rank 0,
            // always a member) accumulates it into the Frame below.
            if (fft_member) {
                MPI_Allreduce(&e_local, &e_recip, 1, MPI_DOUBLE, MPI_SUM,
                              fft_comm_);
                e_recip *= ATV::units::coulomb / (2.0 * ATV::units::pi * vol);

                fftw_execute(static_cast<fftw_plan>(plan_mpi_bwd_));
            }
        }

        // ── FFT → brick remap: slab potential back to brick owners ─────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/remap", ctx.stage, kTimerCategory);
            run_grid_plan_(remap_bwd_, comm, /*tag_base=*/340,
                           /*src_brick=*/false, /*dst_brick=*/true,
                           /*add=*/false);
        }

        // ── mesh forward fill: ghost potential from owners ─────────────
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            run_grid_plan_(mesh_fwd_, comm, /*tag_base=*/320,
                           /*src_brick=*/true, /*dst_brick=*/true,
                           /*add=*/false);
        }

        // AD force interpolation from the local brick — stencil rows are
        // contiguous (z-fastest) and in-range by map_stencils_; the mesh
        // forward fill above populated every ghost cell they touch.
        ATV_PROFILE_SCOPE(ctx, "pppm/interp", ctx.stage, kTimerCategory);
        const double fscale = ATV::units::coulomb / (ATV::units::pi * vol);
        const double ic00 = ic(0, 0), ic01 = ic(0, 1), ic02 = ic(0, 2);
        const double ic10 = ic(1, 0), ic11 = ic(1, 1), ic12 = ic(1, 2);
        const double ic20 = ic(2, 0), ic21 = ic(2, 1), ic22 = ic(2, 2);
        const double K0d = static_cast<double>(K0);
        const double K1d = static_cast<double>(K1);
        const double K2d = static_cast<double>(K2);
        for (int i = 0; i < n_local; ++i) {
            const double qi = q[i];
            if (qi == 0.0)
                continue;
            const double* thx = &theta_[(3 * i + 0) * n];
            const double* thy = &theta_[(3 * i + 1) * n];
            const double* thz = &theta_[(3 * i + 2) * n];
            const double* dthx = &dtheta_[(3 * i + 0) * n];
            const double* dthy = &dtheta_[(3 * i + 1) * n];
            const double* dthz = &dtheta_[(3 * i + 2) * n];
            const int bx = stencil_min_[3 * i + 0] - nxlo_out_;
            const int by = stencil_min_[3 * i + 1] - nylo_out_;
            const int bz = stencil_min_[3 * i + 2] - nzlo_out_;
            double g0s = 0, g1s = 0, g2s = 0;
            for (int a = 0; a < n; ++a) {
                const int xoff = (bx + a) * nyl_;
                const double tx = thx[a];
                const double dxa = dthx[a];
                for (int b = 0; b < n; ++b) {
                    const double* __restrict row =
                        &brick_[static_cast<std::size_t>((xoff + by + b) * nzl_
                                                         + bz)];
                    double s0 = 0, s1 = 0;
                    for (int c = 0; c < n; ++c) {
                        s0 += row[c] * thz[c];
                        s1 += row[c] * dthz[c];
                    }
                    g0s += dxa * thy[b] * s0;
                    g1s += tx * dthy[b] * s0;
                    g2s += tx * thy[b] * s1;
                }
            }
            const double fxi =
                g0s * K0d * ic00 + g1s * K1d * ic01 + g2s * K2d * ic02;
            const double fyi =
                g0s * K0d * ic10 + g1s * K1d * ic11 + g2s * K2d * ic12;
            const double fzi =
                g0s * K0d * ic20 + g1s * K1d * ic21 + g2s * K2d * ic22;
            const double s = -fscale * qi;
            fx[i] += s * fxi;
            fy[i] += s * fyi;
            fz[i] += s * fzi;
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
        -ATV::units::coulomb * alpha_ / std::sqrt(ATV::units::pi) * qsq_local;
    f.meta.get(kEnergy) += e_self;
    f.meta.get(kEnergyPair) += e_self;

    double q_total = q_local;
#ifdef ATV_HAS_MPI
    if (mpi)
        MPI_Allreduce(MPI_IN_PLACE, &q_total, 1, MPI_DOUBLE, MPI_SUM,
                      dd->cart_comm);
#endif
    if (first_rank) {
        const double e_net = -ATV::units::coulomb * ATV::units::pi
                             / (2.0 * vol * alpha_ * alpha_) * q_total
                             * q_total;
        f.meta.get(kEnergy) += e_recip + e_net;
        f.meta.get(kEnergyPair) += e_recip + e_net;
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
    for (int d = 0; d < 3; ++d)
        std::vector<double>().swap(bsp_mod_[d]);
    std::vector<int>().swap(base_);
    std::vector<double>().swap(theta_);
    std::vector<double>().swap(dtheta_);
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
}

void KSpacePPPM::destroy(const DestroyContext&) { release_storage(); }

std::size_t KSpacePPPM::memory_bytes() const {
    std::size_t b = brick_.capacity() * sizeof(double)
                    + work_.capacity() * sizeof(std::complex<double>)
                    + green_.capacity() * sizeof(double)
                    + base_.capacity() * sizeof(int)
                    + (theta_.capacity() + dtheta_.capacity()) * sizeof(double);
    for (int d = 0; d < 3; ++d)
        b += bsp_mod_[d].capacity() * sizeof(double);
    if (mpi_data_)
        b += static_cast<std::size_t>(2 * mpi_alloc_) * sizeof(double);
    return b;
}

} // namespace cpu
} // namespace ATV
