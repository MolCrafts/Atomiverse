/// @file src/cpu/longrange/pppm.cpp
/// @brief Brick-decomposed PPPM: local brick + reverse/forward mesh comm +
///        Alltoallv remap ↔ FFTW-MPI slabs (no full-grid density Allreduce).
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
      nzl_(other.nzl_), nghost_(other.nghost_), brick_(std::move(other.brick_)),
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
      atom_capacity_(other.atom_capacity_), base_(std::move(other.base_)),
      theta_(std::move(other.theta_)), dtheta_(std::move(other.dtheta_)) {
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
    nghost_ = order_ - 1;
    nxlo_out_ = nxlo_in_ - nghost_;
    nxhi_out_ = nxhi_in_ + nghost_;
    nylo_out_ = nylo_in_ - nghost_;
    nyhi_out_ = nyhi_in_ + nghost_;
    nzlo_out_ = nzlo_in_ - nghost_;
    nzhi_out_ = nzhi_in_ + nghost_;
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
    atom_capacity_ = next;
    if (!initial_build)
        note_capacity_rebuild();
}

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
                for (int b = 0; b < n; ++b) {
                    const int g1 = wrap_k(b1 + b, K1);
                    const int plane = K2 * (g1 + K1 * g0);
                    for (int c = 0; c < n; ++c) {
                        const int g2 = wrap_k(b2 + c, K2);
                        const double cval = dens[plane + g2];
                        g0s += cval * dthx[a] * thy[b] * thz[c];
                        g1s += cval * thx[a] * dthy[b] * thz[c];
                        g2s += cval * thx[a] * thy[b] * dthz[c];
                    }
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
            // Map wrapped g∈[0,K) to exactly one brick index (owned preferred,
            // else a single ±K ghost image). Depositing every ±K image *and*
            // reverse_comm double-counts PBC faces (wrap already put charge in
            // owned; ghost image + reverse adds it again).
            auto map1 = [](int g, int K, int in_lo, int in_hi, int out_lo,
                           int out_hi) -> int {
                if (g >= in_lo && g <= in_hi)
                    return g;
                if (g - K >= in_lo && g - K <= in_hi)
                    return g - K;
                if (g + K >= in_lo && g + K <= in_hi)
                    return g + K;
                if (g >= out_lo && g <= out_hi)
                    return g;
                if (g - K >= out_lo && g - K <= out_hi)
                    return g - K;
                if (g + K >= out_lo && g + K <= out_hi)
                    return g + K;
                return 0x7fffffff; // not in this brick
            };
            for (int i = 0; i < n_local; ++i) {
                const double qi = q[i];
                if (qi == 0.0)
                    continue;
                const double* thx = &theta_[(3 * i + 0) * n];
                const double* thy = &theta_[(3 * i + 1) * n];
                const double* thz = &theta_[(3 * i + 2) * n];
                const int b0 = base_[3 * i + 0] - nm1;
                const int b1 = base_[3 * i + 1] - nm1;
                const int b2 = base_[3 * i + 2] - nm1;
                for (int a = 0; a < n; ++a) {
                    const int g0 = wrap_k(b0 + a, K0);
                    const int ix =
                        map1(g0, K0, nxlo_in_, nxhi_in_, nxlo_out_, nxhi_out_);
                    if (ix == 0x7fffffff)
                        continue;
                    const double wa = qi * thx[a];
                    for (int b = 0; b < n; ++b) {
                        const int g1 = wrap_k(b1 + b, K1);
                        const int iy = map1(g1, K1, nylo_in_, nyhi_in_,
                                            nylo_out_, nyhi_out_);
                        if (iy == 0x7fffffff)
                            continue;
                        const double wab = wa * thy[b];
                        for (int c = 0; c < n; ++c) {
                            const int g2 = wrap_k(b2 + c, K2);
                            const int iz = map1(g2, K2, nzlo_in_, nzhi_in_,
                                                nzlo_out_, nzhi_out_);
                            if (iz == 0x7fffffff)
                                continue;
                            brick_[static_cast<std::size_t>(
                                brick_index_(ix, iy, iz))] += wab * thz[c];
                        }
                    }
                }
            }
        }

        // ── reverse_comm: sum ghost density into neighbors' owned cells ─
        // Dim-by-dim with matching Sendrecv (send→lo recv←hi, and dual).
        // Self-neighbor (1-proc dim) is valid and implements mesh PBC wrap.
        MPI_Comm comm = dd->cart_comm;
        int nprocs = 0;
        MPI_Comm_size(comm, &nprocs);

        // Shared by both mesh halos (reverse_comm here, forward_comm below).
        auto pack_slab = [&](int ix0, int ix1, int iy0, int iy1, int iz0,
                             int iz1, std::vector<double>& buf) -> int {
            if (ix1 < ix0 || iy1 < iy0 || iz1 < iz0)
                return 0;
            const int nsend =
                (ix1 - ix0 + 1) * (iy1 - iy0 + 1) * (iz1 - iz0 + 1);
            buf.resize(static_cast<std::size_t>(nsend));
            int p = 0;
            for (int iz = iz0; iz <= iz1; ++iz)
                for (int iy = iy0; iy <= iy1; ++iy)
                    for (int ix = ix0; ix <= ix1; ++ix)
                        buf[static_cast<std::size_t>(p++)] =
                            brick_[static_cast<std::size_t>(
                                brick_index_(ix, iy, iz))];
            return nsend;
        };
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            auto add_slab = [&](int jx0, int jy0, int jz0, int nx, int ny,
                                int nz, const std::vector<double>& buf) {
                int p = 0;
                for (int iz = 0; iz < nz; ++iz)
                    for (int iy = 0; iy < ny; ++iy)
                        for (int ix = 0; ix < nx; ++ix) {
                            const int gx = jx0 + ix;
                            const int gy = jy0 + iy;
                            const int gz = jz0 + iz;
                            if (gx >= nxlo_out_ && gx <= nxhi_out_
                                && gy >= nylo_out_ && gy <= nyhi_out_
                                && gz >= nzlo_out_ && gz <= nzhi_out_)
                                brick_[static_cast<std::size_t>(
                                    brick_index_(gx, gy, gz))] +=
                                    buf[static_cast<std::size_t>(p)];
                            ++p;
                        }
            };

            for (int dim = 0; dim < 3; ++dim) {
                const int lo_nbr =
                    dd->face_neighbors[static_cast<std::size_t>(2 * dim)];
                const int hi_nbr =
                    dd->face_neighbors[static_cast<std::size_t>(2 * dim + 1)];

                // Ghost planes on active dim; perp: owned if already reversed,
                // else full out-range (edges/corners propagate).
                int glo0[3], glo1[3], ghi0[3], ghi1[3];
                int dst_lo[3], dst_hi[3]; // where to ADD received slabs
                const int x0 = (dim > 0) ? nxlo_in_ : nxlo_out_;
                const int x1 = (dim > 0) ? nxhi_in_ : nxhi_out_;
                const int y0 = (dim > 1) ? nylo_in_ : nylo_out_;
                const int y1 = (dim > 1) ? nyhi_in_ : nyhi_out_;
                const int z0 = nzlo_out_, z1 = nzhi_out_;

                if (dim == 0) {
                    glo0[0] = nxlo_out_;
                    glo1[0] = nxlo_in_ - 1;
                    ghi0[0] = nxhi_in_ + 1;
                    ghi1[0] = nxhi_out_;
                    glo0[1] = ghi0[1] = nylo_out_;
                    glo1[1] = ghi1[1] = nyhi_out_;
                    glo0[2] = ghi0[2] = nzlo_out_;
                    glo1[2] = ghi1[2] = nzhi_out_;
                    dst_lo[0] = nxlo_in_;
                    dst_hi[0] = nxhi_in_ - nghost_ + 1;
                    dst_lo[1] = dst_hi[1] = nylo_out_;
                    dst_lo[2] = dst_hi[2] = nzlo_out_;
                } else if (dim == 1) {
                    glo0[0] = ghi0[0] = x0;
                    glo1[0] = ghi1[0] = x1;
                    glo0[1] = nylo_out_;
                    glo1[1] = nylo_in_ - 1;
                    ghi0[1] = nyhi_in_ + 1;
                    ghi1[1] = nyhi_out_;
                    glo0[2] = ghi0[2] = z0;
                    glo1[2] = ghi1[2] = z1;
                    dst_lo[0] = dst_hi[0] = x0;
                    dst_lo[1] = nylo_in_;
                    dst_hi[1] = nyhi_in_ - nghost_ + 1;
                    dst_lo[2] = dst_hi[2] = z0;
                } else {
                    glo0[0] = ghi0[0] = x0;
                    glo1[0] = ghi1[0] = x1;
                    glo0[1] = ghi0[1] = y0;
                    glo1[1] = ghi1[1] = y1;
                    glo0[2] = nzlo_out_;
                    glo1[2] = nzlo_in_ - 1;
                    ghi0[2] = nzhi_in_ + 1;
                    ghi1[2] = nzhi_out_;
                    dst_lo[0] = dst_hi[0] = x0;
                    dst_lo[1] = dst_hi[1] = y0;
                    dst_lo[2] = nzlo_in_;
                    dst_hi[2] = nzhi_in_ - nghost_ + 1;
                }
                if (glo1[dim] < glo0[dim] || ghi1[dim] < ghi0[dim])
                    continue;

                std::vector<double> s_lo, s_hi, r_from_hi, r_from_lo;
                const int n_lo = pack_slab(glo0[0], glo1[0], glo0[1], glo1[1],
                                           glo0[2], glo1[2], s_lo);
                const int n_hi = pack_slab(ghi0[0], ghi1[0], ghi0[1], ghi1[1],
                                           ghi0[2], ghi1[2], s_hi);
                if (n_lo == 0 || n_hi == 0)
                    continue;
                if (n_lo != n_hi)
                    throw std::runtime_error(
                        "KSpacePPPM reverse_comm: ghost slab size mismatch");
                r_from_hi.resize(static_cast<std::size_t>(n_lo));
                r_from_lo.resize(static_cast<std::size_t>(n_hi));
                const int tag = 100 + dim;
                // lo-ghost → lo_nbr; hi_nbr's lo-ghost → our hi owned edge
                MPI_Sendrecv(s_lo.data(), n_lo, MPI_DOUBLE, lo_nbr, tag,
                             r_from_hi.data(), n_lo, MPI_DOUBLE, hi_nbr, tag,
                             comm, MPI_STATUS_IGNORE);
                // hi-ghost → hi_nbr; lo_nbr's hi-ghost → our lo owned edge
                MPI_Sendrecv(s_hi.data(), n_hi, MPI_DOUBLE, hi_nbr, tag + 3,
                             r_from_lo.data(), n_hi, MPI_DOUBLE, lo_nbr,
                             tag + 3, comm, MPI_STATUS_IGNORE);

                const int nx_g = glo1[0] - glo0[0] + 1;
                const int ny_g = glo1[1] - glo0[1] + 1;
                const int nz_g = glo1[2] - glo0[2] + 1;
                const int nx_h = ghi1[0] - ghi0[0] + 1;
                const int ny_h = ghi1[1] - ghi0[1] + 1;
                const int nz_h = ghi1[2] - ghi0[2] + 1;
                add_slab(dst_hi[0], dst_hi[1], dst_hi[2], nx_g, ny_g, nz_g,
                         r_from_hi);
                add_slab(dst_lo[0], dst_lo[1], dst_lo[2], nx_h, ny_h, nz_h,
                         r_from_lo);
            }
        }

        // ── Setup FFTW-MPI + m0 ownership table ───────────────────────
        static bool mpi_fft_inited = false;
        if (!mpi_fft_inited) {
            fftw_mpi_init();
            mpi_fft_inited = true;
        }
        const int n2c = K2 / 2 + 1;
        const int n2pad = 2 * n2c;
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
            mpi_fft_ready_ = true;
        }

        // Rank that owns FFT slab containing global m0. No fallback: under
        // the subgroup arm rank nprocs-1 is typically a non-member, so a
        // "nearest rank" guess would silently drop or double-count planes.
        auto fft_owner = [&](int m0) -> int {
            for (int r = 0; r < nprocs; ++r)
                if (m0 >= fft_m0_starts_[static_cast<std::size_t>(r)]
                    && m0 < fft_m0_starts_[static_cast<std::size_t>(r)]
                                + fft_m0_counts_[static_cast<std::size_t>(r)])
                    return r;
            throw std::runtime_error(
                "KSpacePPPM: mesh plane m0=" + std::to_string(m0)
                + " of K0=" + std::to_string(K_[0])
                + " is owned by no rank of the FFT group (fft_ranks="
                + std::to_string(fft_ranks_resolved_) + " of "
                + std::to_string(nprocs)
                + " ranks). Reproduce with the same mesh and rank count; the "
                  "FFTW-MPI dim-0 slabs must tile [0, K0).");
        };

        // A non-member owns no plane: no buffer, ln0 == 0, so every slab loop
        // below is empty for it. Held here (outside the timed regions) because
        // both remaps and the FFT region read them.
        const bool fft_member = fft_comm_ != MPI_COMM_NULL;
        const ptrdiff_t ln0 = static_cast<ptrdiff_t>(mpi_local_n0_);
        const ptrdiff_t l0s = static_cast<ptrdiff_t>(mpi_local_0_start_);
        int stot = 0;
        int rtot = 0;
        std::vector<int> cursor;

        // ── brick → FFT remap (owned cells only, Alltoallv) ───────────
        // Pack (ix,iy,iz,val) as 4 doubles per owned brick point.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/remap", ctx.stage, kTimerCategory);
            sendcounts_.assign(static_cast<std::size_t>(nprocs), 0);
            for (int ix = nxlo_in_; ix <= nxhi_in_; ++ix)
                for (int iy = nylo_in_; iy <= nyhi_in_; ++iy)
                    for (int iz = nzlo_in_; iz <= nzhi_in_; ++iz)
                        sendcounts_[static_cast<std::size_t>(fft_owner(ix))] +=
                            4;
            senddispls_.assign(static_cast<std::size_t>(nprocs), 0);
            stot = 0;
            for (int r = 0; r < nprocs; ++r) {
                senddispls_[static_cast<std::size_t>(r)] = stot;
                stot += sendcounts_[static_cast<std::size_t>(r)];
            }
            remap_send_.assign(static_cast<std::size_t>(stot), 0.0);
            cursor = senddispls_;
            for (int ix = nxlo_in_; ix <= nxhi_in_; ++ix) {
                const int r = fft_owner(ix);
                for (int iy = nylo_in_; iy <= nyhi_in_; ++iy)
                    for (int iz = nzlo_in_; iz <= nzhi_in_; ++iz) {
                        int& p = cursor[static_cast<std::size_t>(r)];
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(ix);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(iy);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(iz);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            brick_[static_cast<std::size_t>(
                                brick_index_(ix, iy, iz))];
                    }
            }
            recvcounts_.assign(static_cast<std::size_t>(nprocs), 0);
            MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1,
                         MPI_INT, comm);
            recvdispls_.assign(static_cast<std::size_t>(nprocs), 0);
            rtot = 0;
            for (int r = 0; r < nprocs; ++r) {
                recvdispls_[static_cast<std::size_t>(r)] = rtot;
                rtot += recvcounts_[static_cast<std::size_t>(r)];
            }
            remap_recv_.assign(static_cast<std::size_t>(rtot), 0.0);
            MPI_Alltoallv(remap_send_.data(), sendcounts_.data(),
                          senddispls_.data(), MPI_DOUBLE, remap_recv_.data(),
                          recvcounts_.data(), recvdispls_.data(), MPI_DOUBLE,
                          comm);

            // Unpack into FFTW-MPI padded buffer.
            if (fft_member)
                std::memset(mpi_data_, 0,
                            static_cast<size_t>(2 * mpi_alloc_)
                                * sizeof(double));
            for (int i = 0; i + 3 < rtot; i += 4) {
                const int ix =
                    static_cast<int>(remap_recv_[static_cast<std::size_t>(i)]);
                const int iy = static_cast<int>(
                    remap_recv_[static_cast<std::size_t>(i + 1)]);
                const int iz = static_cast<int>(
                    remap_recv_[static_cast<std::size_t>(i + 2)]);
                const double val = remap_recv_[static_cast<std::size_t>(i + 3)];
                const ptrdiff_t a = static_cast<ptrdiff_t>(ix) - l0s;
                if (a < 0 || a >= ln0)
                    continue;
                const ptrdiff_t dst =
                    iz + static_cast<ptrdiff_t>(n2pad) * (iy + K1 * a);
                mpi_data_[dst] += val; // sum if multiple bricks map here
            }
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

        // ── FFT → brick remap: pack my FFT slab potential to brick owners ─
        // For each (ix,iy,iz) in my FFT slab, find brick owner by mesh partition.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/remap", ctx.stage, kTimerCategory);
            auto brick_owner = [&](int ix, int iy, int iz) -> int {
                // Inverse of setup_brick_partition_ split.
                int cx = 0, cy = 0, cz = 0;
                for (int c = 0; c < dd->dims[0]; ++c) {
                    const int lo = c * K0 / dd->dims[0];
                    const int hi = (c + 1) * K0 / dd->dims[0] - 1;
                    if (ix >= lo && ix <= hi) {
                        cx = c;
                        break;
                    }
                }
                for (int c = 0; c < dd->dims[1]; ++c) {
                    const int lo = c * K1 / dd->dims[1];
                    const int hi = (c + 1) * K1 / dd->dims[1] - 1;
                    if (iy >= lo && iy <= hi) {
                        cy = c;
                        break;
                    }
                }
                for (int c = 0; c < dd->dims[2]; ++c) {
                    const int lo = c * K2 / dd->dims[2];
                    const int hi = (c + 1) * K2 / dd->dims[2] - 1;
                    if (iz >= lo && iz <= hi) {
                        cz = c;
                        break;
                    }
                }
                return dd->rank_of(cx, cy, cz);
            };

            sendcounts_.assign(static_cast<std::size_t>(nprocs), 0);
            for (ptrdiff_t a = 0; a < ln0; ++a) {
                const int ix = static_cast<int>(l0s + a);
                for (int iy = 0; iy < K1; ++iy)
                    for (int iz = 0; iz < K2; ++iz)
                        sendcounts_[static_cast<std::size_t>(
                            brick_owner(ix, iy, iz))] += 4;
            }
            senddispls_.assign(static_cast<std::size_t>(nprocs), 0);
            stot = 0;
            for (int r = 0; r < nprocs; ++r) {
                senddispls_[static_cast<std::size_t>(r)] = stot;
                stot += sendcounts_[static_cast<std::size_t>(r)];
            }
            remap_send_.assign(static_cast<std::size_t>(stot), 0.0);
            cursor = senddispls_;
            for (ptrdiff_t a = 0; a < ln0; ++a) {
                const int ix = static_cast<int>(l0s + a);
                for (int iy = 0; iy < K1; ++iy)
                    for (int iz = 0; iz < K2; ++iz) {
                        const int r = brick_owner(ix, iy, iz);
                        int& p = cursor[static_cast<std::size_t>(r)];
                        const ptrdiff_t src =
                            iz + static_cast<ptrdiff_t>(n2pad) * (iy + K1 * a);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(ix);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(iy);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            static_cast<double>(iz);
                        remap_send_[static_cast<std::size_t>(p++)] =
                            mpi_data_[src];
                    }
            }
            MPI_Alltoall(sendcounts_.data(), 1, MPI_INT, recvcounts_.data(), 1,
                         MPI_INT, comm);
            recvdispls_.assign(static_cast<std::size_t>(nprocs), 0);
            rtot = 0;
            for (int r = 0; r < nprocs; ++r) {
                recvdispls_[static_cast<std::size_t>(r)] = rtot;
                rtot += recvcounts_[static_cast<std::size_t>(r)];
            }
            remap_recv_.assign(static_cast<std::size_t>(rtot), 0.0);
            MPI_Alltoallv(remap_send_.data(), sendcounts_.data(),
                          senddispls_.data(), MPI_DOUBLE, remap_recv_.data(),
                          recvcounts_.data(), recvdispls_.data(), MPI_DOUBLE,
                          comm);

            // Zero owned region; keep ghost for forward_comm fill later.
            for (int ix = nxlo_in_; ix <= nxhi_in_; ++ix)
                for (int iy = nylo_in_; iy <= nyhi_in_; ++iy)
                    for (int iz = nzlo_in_; iz <= nzhi_in_; ++iz)
                        brick_[static_cast<std::size_t>(
                            brick_index_(ix, iy, iz))] = 0.0;

            for (int i = 0; i + 3 < rtot; i += 4) {
                const int ix =
                    static_cast<int>(remap_recv_[static_cast<std::size_t>(i)]);
                const int iy = static_cast<int>(
                    remap_recv_[static_cast<std::size_t>(i + 1)]);
                const int iz = static_cast<int>(
                    remap_recv_[static_cast<std::size_t>(i + 2)]);
                const double val = remap_recv_[static_cast<std::size_t>(i + 3)];
                if (ix >= nxlo_in_ && ix <= nxhi_in_ && iy >= nylo_in_
                    && iy <= nyhi_in_ && iz >= nzlo_in_ && iz <= nzhi_in_)
                    brick_[static_cast<std::size_t>(brick_index_(ix, iy, iz))] =
                        val;
            }
        }

        // ── forward_comm: fill ghost potential from neighbors ─────────
        // Send owned edge → neighbor's ghost (copy, not add). Dim-by-dim
        // with matching Sendrecv; edges/corners via sequential dims.
        {
            ATV_PROFILE_SCOPE(ctx, "pppm/halo", ctx.stage, kTimerCategory);
            auto copy_slab = [&](int jx0, int jy0, int jz0, int nx, int ny,
                                 int nz, const std::vector<double>& buf) {
                int p = 0;
                for (int iz = 0; iz < nz; ++iz)
                    for (int iy = 0; iy < ny; ++iy)
                        for (int ix = 0; ix < nx; ++ix) {
                            const int gx = jx0 + ix;
                            const int gy = jy0 + iy;
                            const int gz = jz0 + iz;
                            if (gx >= nxlo_out_ && gx <= nxhi_out_
                                && gy >= nylo_out_ && gy <= nyhi_out_
                                && gz >= nzlo_out_ && gz <= nzhi_out_)
                                brick_[static_cast<std::size_t>(
                                    brick_index_(gx, gy, gz))] =
                                    buf[static_cast<std::size_t>(p)];
                            ++p;
                        }
            };
            for (int dim = 0; dim < 3; ++dim) {
                const int lo_nbr =
                    dd->face_neighbors[static_cast<std::size_t>(2 * dim)];
                const int hi_nbr =
                    dd->face_neighbors[static_cast<std::size_t>(2 * dim + 1)];

                // Owned edge to send; ghost destination after recv.
                // Perp: expand to out-range for unfinished dims so corners fill.
                int elo0[3], elo1[3], ehi0[3], ehi1[3];
                int gdst_lo[3], gdst_hi[3];
                const int x0 = (dim > 0) ? nxlo_out_ : nxlo_in_;
                const int x1 = (dim > 0) ? nxhi_out_ : nxhi_in_;
                const int y0 = (dim > 1) ? nylo_out_ : nylo_in_;
                const int y1 = (dim > 1) ? nyhi_out_ : nyhi_in_;

                if (dim == 0) {
                    elo0[0] = nxlo_in_;
                    elo1[0] = nxlo_in_ + nghost_ - 1;
                    ehi0[0] = nxhi_in_ - nghost_ + 1;
                    ehi1[0] = nxhi_in_;
                    elo0[1] = ehi0[1] = nylo_in_;
                    elo1[1] = ehi1[1] = nyhi_in_;
                    elo0[2] = ehi0[2] = nzlo_in_;
                    elo1[2] = ehi1[2] = nzhi_in_;
                    gdst_lo[0] = nxlo_out_;
                    gdst_hi[0] = nxhi_in_ + 1;
                    gdst_lo[1] = gdst_hi[1] = nylo_in_;
                    gdst_lo[2] = gdst_hi[2] = nzlo_in_;
                } else if (dim == 1) {
                    elo0[0] = ehi0[0] = x0;
                    elo1[0] = ehi1[0] = x1;
                    elo0[1] = nylo_in_;
                    elo1[1] = nylo_in_ + nghost_ - 1;
                    ehi0[1] = nyhi_in_ - nghost_ + 1;
                    ehi1[1] = nyhi_in_;
                    elo0[2] = ehi0[2] = nzlo_in_;
                    elo1[2] = ehi1[2] = nzhi_in_;
                    gdst_lo[0] = gdst_hi[0] = x0;
                    gdst_lo[1] = nylo_out_;
                    gdst_hi[1] = nyhi_in_ + 1;
                    gdst_lo[2] = gdst_hi[2] = nzlo_in_;
                } else {
                    elo0[0] = ehi0[0] = x0;
                    elo1[0] = ehi1[0] = x1;
                    elo0[1] = ehi0[1] = y0;
                    elo1[1] = ehi1[1] = y1;
                    elo0[2] = nzlo_in_;
                    elo1[2] = nzlo_in_ + nghost_ - 1;
                    ehi0[2] = nzhi_in_ - nghost_ + 1;
                    ehi1[2] = nzhi_in_;
                    gdst_lo[0] = gdst_hi[0] = x0;
                    gdst_lo[1] = gdst_hi[1] = y0;
                    gdst_lo[2] = nzlo_out_;
                    gdst_hi[2] = nzhi_in_ + 1;
                }
                if (elo1[dim] < elo0[dim] || ehi1[dim] < ehi0[dim])
                    continue;

                std::vector<double> s_lo, s_hi, r_lo, r_hi;
                const int n_lo = pack_slab(elo0[0], elo1[0], elo0[1], elo1[1],
                                           elo0[2], elo1[2], s_lo);
                const int n_hi = pack_slab(ehi0[0], ehi1[0], ehi0[1], ehi1[1],
                                           ehi0[2], ehi1[2], s_hi);
                if (n_lo == 0 || n_hi == 0)
                    continue;
                if (n_lo != n_hi)
                    throw std::runtime_error(
                        "KSpacePPPM forward_comm: edge slab size mismatch");
                r_lo.resize(static_cast<std::size_t>(n_lo));
                r_hi.resize(static_cast<std::size_t>(n_hi));
                const int tag = 200 + dim;
                // owned_lo → lo_nbr; recv from hi into ghost_hi
                MPI_Sendrecv(s_lo.data(), n_lo, MPI_DOUBLE, lo_nbr, tag,
                             r_hi.data(), n_lo, MPI_DOUBLE, hi_nbr, tag, comm,
                             MPI_STATUS_IGNORE);
                // owned_hi → hi_nbr; recv from lo into ghost_lo
                MPI_Sendrecv(s_hi.data(), n_hi, MPI_DOUBLE, hi_nbr, tag + 3,
                             r_lo.data(), n_hi, MPI_DOUBLE, lo_nbr, tag + 3,
                             comm, MPI_STATUS_IGNORE);

                const int nx_e = elo1[0] - elo0[0] + 1;
                const int ny_e = elo1[1] - elo0[1] + 1;
                const int nz_e = elo1[2] - elo0[2] + 1;
                const int nx_h = ehi1[0] - ehi0[0] + 1;
                const int ny_h = ehi1[1] - ehi0[1] + 1;
                const int nz_h = ehi1[2] - ehi0[2] + 1;
                copy_slab(gdst_hi[0], gdst_hi[1], gdst_hi[2], nx_e, ny_e, nz_e,
                          r_hi);
                copy_slab(gdst_lo[0], gdst_lo[1], gdst_lo[2], nx_h, ny_h, nz_h,
                          r_lo);
            }
        }

        // Force from local brick (same single-image map as deposit).
        ATV_PROFILE_SCOPE(ctx, "pppm/interp", ctx.stage, kTimerCategory);
        const double fscale = ATV::units::coulomb / (ATV::units::pi * vol);
        const double ic00 = ic(0, 0), ic01 = ic(0, 1), ic02 = ic(0, 2);
        const double ic10 = ic(1, 0), ic11 = ic(1, 1), ic12 = ic(1, 2);
        const double ic20 = ic(2, 0), ic21 = ic(2, 1), ic22 = ic(2, 2);
        const double K0d = static_cast<double>(K0);
        const double K1d = static_cast<double>(K1);
        const double K2d = static_cast<double>(K2);
        auto map1f = [](int g, int K, int in_lo, int in_hi, int out_lo,
                        int out_hi) -> int {
            if (g >= in_lo && g <= in_hi)
                return g;
            if (g - K >= in_lo && g - K <= in_hi)
                return g - K;
            if (g + K >= in_lo && g + K <= in_hi)
                return g + K;
            if (g >= out_lo && g <= out_hi)
                return g;
            if (g - K >= out_lo && g - K <= out_hi)
                return g - K;
            if (g + K >= out_lo && g + K <= out_hi)
                return g + K;
            return 0x7fffffff;
        };
        auto brick_val = [&](int ix, int iy, int iz) -> double {
            const int gx =
                map1f(ix, K0, nxlo_in_, nxhi_in_, nxlo_out_, nxhi_out_);
            const int gy =
                map1f(iy, K1, nylo_in_, nyhi_in_, nylo_out_, nyhi_out_);
            const int gz =
                map1f(iz, K2, nzlo_in_, nzhi_in_, nzlo_out_, nzhi_out_);
            if (gx == 0x7fffffff || gy == 0x7fffffff || gz == 0x7fffffff)
                return 0.0;
            return brick_[static_cast<std::size_t>(brick_index_(gx, gy, gz))];
        };
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
                for (int b = 0; b < n; ++b) {
                    const int g1 = wrap_k(b1 + b, K1);
                    for (int c = 0; c < n; ++c) {
                        const int g2 = wrap_k(b2 + c, K2);
                        const double cval = brick_val(g0, g1, g2);
                        g0s += cval * dthx[a] * thy[b] * thz[c];
                        g1s += cval * thx[a] * dthy[b] * thz[c];
                        g2s += cval * thx[a] * thy[b] * dthz[c];
                    }
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
