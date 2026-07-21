/// @file cpu/potentials/pair_lj_cut.cpp
/// @brief CPU PairLJCut — LJ 12-6 shifted, xsimd ket-loop, OMP force reduction.

#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/domain.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// xsimd vector batch<double> for the LJ inner kernel (Track B.2).
// Linked directly to atomiverse via target_link_libraries; -march=native is
// already gated to CXX in the top-level CMakeLists. We use the *default*
// architecture (NEON64 on Apple silicon, AVX2/AVX-512 on x86-64); on
// architectures without xsimd support the batch_t::size collapses to 1 and
// the loop degenerates into the scalar code path with no measurable
// overhead.
#include <xsimd/xsimd.hpp>

#include "atomiverse/cpu/frame/box_simd.h"

namespace ATV {
namespace cpu {

PairLJCut::PairLJCut(double epsilon, double sigma, double cutoff)
    : Pair(static_cast<AccReal>(cutoff)), epsilon_(epsilon), sigma_(sigma) {
    sigma6_ = sigma * sigma * sigma * sigma * sigma * sigma;
    cutoff_sq_ = cutoff * cutoff;

    // Compute V_shift = V(r_cutoff). The base's cutoff_ holds the
    // canonical Å value; the derived caches (cutoff_sq_, v_shift_,
    // sigma6_) are computed once from it.
    double r2inv = double(1) / cutoff_sq_;
    double r6inv = r2inv * r2inv * r2inv * sigma6_;
    v_shift_ = double(4) * epsilon_ * (r6inv * r6inv - r6inv);
}

PairLJCut::PairLJCut(std::vector<LJTypeCoeff> coeffs, double cutoff)
    : Pair(static_cast<AccReal>(cutoff)), epsilon_(0.0), sigma_(0.0) {
    if (cutoff <= 0.0)
        throw std::invalid_argument("PairLJCut: cutoff must be positive");
    if (coeffs.empty())
        throw std::invalid_argument(
            "PairLJCut: the per-type constructor needs at least one "
            "coefficient. There "
            "is no system-wide epsilon/sigma to fall back on, deliberately.");

    // epsilon_/sigma_ are never read on this path — every coefficient comes from the
    // table. But the cutoff-derived caches ARE read by the hot loop, and a constructor
    // establishes EVERY invariant of the object, not just the ones it is thinking
    // about. (Leaving cutoff_sq_ at zero made `r2 >= rc_sq` true for every pair, so the
    // kernel silently computed nothing and every energy came back exactly 0.0.)
    sigma6_ = 0.0;
    v_shift_ = 0.0;
    cutoff_sq_ = cutoff * cutoff;
    per_type_mode_ = true;

    for (const LJTypeCoeff& c : coeffs)
        set_type_coeff(c.type_i, c.type_j, c.epsilon, c.sigma);
}

PairLJCut& PairLJCut::set_type_coeff(int type_i, int type_j, double epsilon,
                                     double sigma) & {
    if (!per_type_mode_)
        throw std::logic_error(
            "PairLJCut::set_type_coeff: this PairLJCut was built with a single "
            "epsilon/sigma and has no type table. Construct it from a "
            "coefficient list "
            "instead.");
    if (type_i < 0 || type_j < 0)
        throw std::invalid_argument(
            "PairLJCut::set_type_coeff: types must be >= 0");
    if (sigma <= 0.0)
        throw std::invalid_argument(
            "PairLJCut::set_type_coeff: sigma must be > 0");

    const int need = std::max(type_i, type_j) + 1;
    if (need > n_types_) {
        // Grow the square tables, preserving what is already set.
        std::vector<double> e(static_cast<std::size_t>(need) * need, 0.0);
        std::vector<double> s(static_cast<std::size_t>(need) * need, 0.0);
        std::vector<char> f(static_cast<std::size_t>(need) * need, 0);
        for (int a = 0; a < n_types_; ++a)
            for (int b = 0; b < n_types_; ++b) {
                const std::size_t old =
                    static_cast<std::size_t>(a) * n_types_ + b;
                const std::size_t nw = static_cast<std::size_t>(a) * need + b;
                e[nw] = eps_tab_[old];
                s[nw] = sigma_tab_[old];
                f[nw] = set_tab_[old];
            }
        eps_tab_.swap(e);
        sigma_tab_.swap(s);
        set_tab_.swap(f);
        n_types_ = need;
    }

    const std::size_t ij = static_cast<std::size_t>(type_i) * n_types_ + type_j;
    const std::size_t ji = static_cast<std::size_t>(type_j) * n_types_ + type_i;
    eps_tab_[ij] = eps_tab_[ji] = epsilon;
    sigma_tab_[ij] = sigma_tab_[ji] = sigma;
    set_tab_[ij] = set_tab_[ji] = 1;
    return *this;
}

PairLJCut PairLJCut::set_type_coeff(int type_i, int type_j, double epsilon,
                                    double sigma) && {
    set_type_coeff(type_i, type_j, epsilon, sigma);
    return std::move(*this);
}

PairLJCut& PairLJCut::set_special_bonds(double s12, double s13, double s14) & {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return *this;
}

PairLJCut PairLJCut::set_special_bonds(double s12, double s13, double s14) && {
    s12_ = s12;
    s13_ = s13;
    s14_ = s14;
    return std::move(*this);
}

PairLJCut& PairLJCut::set_compute_stress(bool on) & {
    compute_stress_ = on;
    return *this;
}

PairLJCut PairLJCut::set_compute_stress(bool on) && {
    compute_stress_ = on;
    return std::move(*this);
}

void PairLJCut::build(const BuildContext& ctx) {
    // No "did you remember to set coefficients?" check is needed: the per-type
    // constructor refuses an empty list, so a PairLJCut cannot exist in that state.
    // That is the point of moving the table into the constructor.
    finalize_type_table_();
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    // Size to the backing-store capacity, not the owned count: the per-thread
    // accumulators are indexed up to n_force = n_local + n_ghost, and ghost
    // rows live in [n_local, nmax). (compute() grows them further if a halo
    // overflow later enlarges the frame.)
    int n = ctx.frame.capacity();
    fx_thr_.resize(n_threads_);
    fy_thr_.resize(n_threads_);
    fz_thr_.resize(n_threads_);
    // Single-thread (pure-MPI) fast path accumulates straight into the frame's
    // force arrays, so no per-thread scratch is needed — leave the buffers
    // empty. The multi-thread path needs one private buffer set per thread to
    // avoid the cross-thread write race on shared force rows.
    if (n_threads_ > 1) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n, double(0));
            fy_thr_[t].resize(n, double(0));
            fz_thr_[t].resize(n, double(0));
        }
    }

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    has_partial_special_ =
        (s12_ > 0.0 && s12_ < 1.0) || (s13_ > 0.0 && s13_ < 1.0)
        || (s14_ > 0.0 && s14_ < 1.0);
    special_.clear();
    if (!any_special_)
        return;
    const auto* dd = ctx.runtime.md.dd;
    const int n_owners =
        (dd && ctx.runtime.mpi_size > 1) ? dd->n_local : ctx.frame.nlocal();
    special_.rebuild(ctx.frame, n_owners);
}

void PairLJCut::rebuild_filtered_list(const Frame& f,
                                      const md::PairList& pairs) {
    // Drop scale-0 specials; keep partial/full with per-pair scale.
    // Rebuilds with every NL version advance so multi-rank Migrate stays safe.
    const int n = pairs.inum;
    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    special_.rebuild(f, n);
    filtered_inum_ = n;
    filtered_numneigh_.assign(n, 0);
    filtered_firstneigh_.assign(n, 0);
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_neighbors_.reserve(pairs.total_pairs);
    filtered_scale_.reserve(pairs.total_pairs);
    filtered_has_shifts_ = pairs.has_shifts;
    if (filtered_has_shifts_) {
        filtered_shift_x_.clear();
        filtered_shift_y_.clear();
        filtered_shift_z_.clear();
        filtered_shift_x_.reserve(pairs.total_pairs);
        filtered_shift_y_.reserve(pairs.total_pairs);
        filtered_shift_z_.reserve(pairs.total_pairs);
    }

    const int* numneigh = pairs.numneigh.data();
    const int* firstneigh = pairs.firstneigh.data();
    const int* neighbors = pairs.neighbors.data();
    const double* sx = filtered_has_shifts_ ? pairs.shift_x.data() : nullptr;
    const double* sy = filtered_has_shifts_ ? pairs.shift_y.data() : nullptr;
    const double* sz = filtered_has_shifts_ ? pairs.shift_z.data() : nullptr;
    const TagInt* __restrict tag = f.span(kAtom_tag);

    for (int i = 0; i < n; ++i) {
        const int off = firstneigh[i];
        const int jn = numneigh[i];
        filtered_firstneigh_[i] = static_cast<int>(filtered_neighbors_.size());
        int kept = 0;
        for (int jj = 0; jj < jn; ++jj) {
            const int j = neighbors[off + jj];
            const double sc = special_.scale_owner(i, tag[j]);
            if (sc == 0.0)
                continue;
            filtered_neighbors_.push_back(j);
            filtered_scale_.push_back(sc);
            if (filtered_has_shifts_) {
                filtered_shift_x_.push_back(sx[off + jj]);
                filtered_shift_y_.push_back(sy[off + jj]);
                filtered_shift_z_.push_back(sz[off + jj]);
            }
            ++kept;
        }
        filtered_numneigh_[i] = kept;
    }
}

namespace {
/// Lorentz-Berthelot: arithmetic mean of sigma, geometric mean of epsilon. The
/// standard combining rule, and the one LAMMPS uses by default (`pair_modify mix
/// geometric` is the alternative; this is `mix arithmetic`).
inline void lorentz_berthelot(double eii, double sii, double ejj, double sjj,
                              double& eij, double& sij) {
    eij = std::sqrt(eii * ejj);
    sij = 0.5 * (sii + sjj);
}
} // namespace

/// Fill the off-diagonal from the diagonal by Lorentz-Berthelot, and precompute
/// sigma^6 and the per-pair energy shift. Called once, from build().
void PairLJCut::finalize_type_table_() {
    if (n_types_ == 0)
        return;
    const double rc = static_cast<double>(this->pair_cutoff());

    sig6_tab_.assign(static_cast<std::size_t>(n_types_) * n_types_, 0.0);
    shift_tab_.assign(static_cast<std::size_t>(n_types_) * n_types_, 0.0);

    for (int a = 0; a < n_types_; ++a) {
        for (int b = 0; b < n_types_; ++b) {
            const std::size_t ab = static_cast<std::size_t>(a) * n_types_ + b;
            double e = eps_tab_[ab], s = sigma_tab_[ab];

            if (!set_tab_[ab]) {
                const std::size_t aa =
                    static_cast<std::size_t>(a) * n_types_ + a;
                const std::size_t bb =
                    static_cast<std::size_t>(b) * n_types_ + b;
                if (set_tab_[aa] && set_tab_[bb]) {
                    lorentz_berthelot(eps_tab_[aa], sigma_tab_[aa],
                                      eps_tab_[bb], sigma_tab_[bb], e, s);
                    eps_tab_[ab] = e;
                    sigma_tab_[ab] = s;
                } else {
                    // No diagonal to mix from: this type pair simply does not
                    // interact. eps = 0 is the honest encoding (TIP3P hydrogens have
                    // no LJ site at all), NOT a fallback to the single-sigma value —
                    // which would silently reintroduce the very bug this exists to fix.
                    e = 0.0;
                    s = 1.0;
                }
            }
            const double s2 = s * s;
            const double s6 = s2 * s2 * s2;
            sig6_tab_[ab] = s6;

            const double r2inv = 1.0 / (rc * rc);
            const double r6inv = r2inv * r2inv * r2inv * s6;
            shift_tab_[ab] = 4.0 * e * (r6inv * r6inv - r6inv);
        }
    }
}

void PairLJCut::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (!ctx.runtime.md.pair_list) {
        throw std::runtime_error(
            "PairLJCut::compute: runtime.md.pair_list is null. Did you add a "
            "NeighborList component to the force phase ahead of LJ?");
    }
    const auto& pairs = *ctx.runtime.md.pair_list;
    const auto& box = f.simbox;
    int n_force = f.nlocal();
    const auto* dd = ctx.runtime.md.dd;
    if (dd && ctx.runtime.mpi_size > 1) {
        n_force = dd->n_local + dd->n_ghost;
    }
    // Ensure the per-thread force accumulators cover [0, n_force). They are
    // sized at build() from the then-current atom count, but n_force grows with
    // the ghost count (and the frame capacity can itself grow on a thin-
    // subdomain halo overflow). Previously every rank kept nlocal == global N,
    // so the build-time size always dominated; with right-sized per-rank
    // storage that no longer holds, so grow here on demand (rare — only when
    // the ghost region expands; matches the neighbor-list growth contract).
    if (n_threads_ > 1
        && static_cast<int>(fx_thr_.empty() ? 0 : fx_thr_[0].size())
               < n_force) {
        for (int t = 0; t < n_threads_; ++t) {
            fx_thr_[t].resize(n_force, double(0));
            fy_thr_[t].resize(n_force, double(0));
            fz_thr_[t].resize(n_force, double(0));
        }
    }
    double energy_total = 0;

    // Exclusion-filtered CSR (built once per NL rebuild). See
    // `rebuild_filtered_list` for the invariant. When no exclusions are
    // configured we iterate the upstream CSR directly — no copy, no
    // allocation, no per-pair branch.
    const bool use_filtered = any_special_;
    if (use_filtered && pairs.version != filtered_version_) {
        rebuild_filtered_list(f, pairs);
        filtered_version_ = pairs.version;
    }
    const double* __restrict k_scales =
        use_filtered ? filtered_scale_.data() : nullptr;
    const int k_inum = use_filtered ? filtered_inum_ : pairs.inum;
    const int* __restrict k_ilist = use_filtered ? nullptr // contiguous 0..n-1
                                                 : pairs.ilist.data();
    const int* __restrict k_numneigh =
        use_filtered ? filtered_numneigh_.data() : pairs.numneigh.data();
    const int* __restrict k_firstneigh =
        use_filtered ? filtered_firstneigh_.data() : pairs.firstneigh.data();
    const int* __restrict k_neighbors =
        use_filtered ? filtered_neighbors_.data() : pairs.neighbors.data();

    // Per-pair PBC shift fast path: d = xi - (xj + shift), skipping the
    // per-pair min-image. Bit-identical to the min-image path because the
    // stored shift is, by construction, `xi - dx - xj` where `dx` is the
    // exact displacement the neighbor build computed (`shift = (shx,shy,shz)`
    // on the half-stencil path; the full min-image displacement otherwise) --
    // it is the same integer image, stable for the list's lifetime under the
    // neighbor skin. This holds under MPI as well: extended-axis ghosts carry
    // a zero shift (they are already at their real, halo-resolved position, so
    // min-image is a no-op) and only the non-decomposed wrapped pairs carry a
    // nonzero shift, which the neighbor build already encoded correctly. The
    // shift array is populated only on the serial neighbor-build path, so
    // `has_shifts` is false (and this falls back to min-image) under OMP>1.
    const bool k_has_shifts =
        (use_filtered ? filtered_has_shifts_ : pairs.has_shifts);
    const double* __restrict k_shift_x =
        !k_has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_x_.data() : pairs.shift_x.data());
    const double* __restrict k_shift_y =
        !k_has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_y_.data() : pairs.shift_y.data());
    const double* __restrict k_shift_z =
        !k_has_shifts
            ? nullptr
            : (use_filtered ? filtered_shift_z_.data() : pairs.shift_z.data());

    // Pair virial accumulators (Voigt order: xx yy zz xy xz yz). Reduced
    // across OpenMP threads via OMP reduction; cross-rank reduction is
    // handled by Reduce at end_of_step.
    double v_xx = 0, v_yy = 0, v_zz = 0, v_xy = 0, v_xz = 0, v_yz = 0;

    // LAMMPS-style vflag gating: skip the 6-component pair virial when no
    // stress consumer is active (plain NVE). Loop-invariant, so the
    // compiler unswitches the inner loop; forces/energy are unaffected.
    const bool do_virial = compute_stress_;

    // Resolve every schema column ONCE, before the OMP parallel region opens
    // (never inside it — the span resolution counter is not thread-safe, and
    // every thread would otherwise resolve the same key).
    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);
    const int* __restrict k_type = f.span(kAtom_type);
    AccReal* __restrict fx_ptr = f.span(kAtom_fx);
    AccReal* __restrict fy_ptr = f.span(kAtom_fy);
    AccReal* __restrict fz_ptr = f.span(kAtom_fz);

#pragma omp parallel reduction(+ : energy_total) reduction(+ : v_xx)           \
    reduction(+ : v_yy) reduction(+ : v_zz) reduction(+ : v_xy)                \
    reduction(+ : v_xz) reduction(+ : v_yz)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        // Single-thread (pure-MPI): accumulate straight into the frame's force
        // arrays — already cleared by ZeroForces (owned rows) +
        // HaloSend::zero_ghost_forces (ghost rows). Elides the per-step
        // zero-fill below AND the reduce sweep at the end of the region, two
        // O(n_force) passes LAMMPS never pays. The multi-thread path keeps each
        // thread's writes in a private buffer to avoid the shared-row race.
        const bool direct = (n_threads_ == 1);
        double* __restrict fxt_ptr;
        double* __restrict fyt_ptr;
        double* __restrict fzt_ptr;
        if (direct) {
            fxt_ptr = fx_ptr;
            fyt_ptr = fy_ptr;
            fzt_ptr = fz_ptr;
        } else {
            auto& fxt = fx_thr_[tid];
            auto& fyt = fy_thr_[tid];
            auto& fzt = fz_thr_[tid];
            std::fill(fxt.begin(), fxt.begin() + n_force, double(0));
            std::fill(fyt.begin(), fyt.begin() + n_force, double(0));
            std::fill(fzt.begin(), fzt.begin() + n_force, double(0));
            fxt_ptr = fxt.data();
            fyt_ptr = fyt.data();
            fzt_ptr = fzt.data();
        }

        // Hoist box edges + LJ constants to locals so the inner loop keeps
        // them in registers instead of reloading the Mat3 / member fields
        // on every pair.
        const double lx = box.length(0), ly = box.length(1), lz = box.length(2);
        const double half_lx = double(0.5) * lx;
        const double half_ly = double(0.5) * ly;
        const double half_lz = double(0.5) * lz;
        const double rc_sq = cutoff_sq_;
        const double sig6 = sigma6_;
        const double eps4 = double(4) * epsilon_;
        const double eps24 = double(24) * epsilon_;

        // Per-type coefficients. When the table is active the coefficients depend on
        // (type_i, type_j) and cannot be hoisted out of the inner loop, so the SIMD ket
        // loop is skipped entirely and every neighbour goes down the scalar path. The
        // single-epsilon path — the one benchmarked against LAMMPS — is untouched.
        const bool k_multi = (n_types_ > 0);
        // Partial special scales force scalar path (per-pair weight).
        const bool use_simd = !k_multi && !has_partial_special_;
        const int k_ntypes = n_types_;
        const double* k_eps = k_multi ? eps_tab_.data() : nullptr;
        const double* k_sig6 = k_multi ? sig6_tab_.data() : nullptr;
        const double* k_shift = k_multi ? shift_tab_.data() : nullptr;
        const double vshift = v_shift_;

        // x_ptr/y_ptr/z_ptr are the schema spans resolved once above, before
        // the parallel region opened (the 13.7x rule) — every thread shares
        // the same read-only pointers.

        // Track B.2 — xsimd ket-loop vectorization of the inner pair loop.
        //
        // Strategy: pack W=batch_t::size neighbors into vector registers,
        // run all distance / cutoff / polynomial / force math through
        // SIMD (NEON64 W=2 on Apple silicon, AVX2 W=4, AVX-512 W=8).
        // Excluded / out-of-cutoff lanes are masked to zero contributions
        // via xsimd::select — branch-free.
        //
        // Bit-identicality: per-lane SIMD ops on NEON / AVX use plain
        // (non-FMA) multiplies and adds (xsimd 11.2.0 routes operator*/+
        // to vmulq/vaddq, not vfmaq), and division uses vdivq_f64 (full
        // IEEE divide). The horizontal accumulation into per-atom force
        // (`fi_x`), per-thread force buffers (`fxt_ptr[j]`), energy and
        // virial is performed by extracting lanes one at a time and
        // adding them in the *same* linear order as the original scalar
        // loop. This preserves the exact summation tree, so per-pair
        // contributions are bit-identical iff the per-lane intermediates
        // (`r2`, `fpair`, `fxp`, etc.) are bit-identical to scalar.
        //
        // The one residual ULP risk is `-ffp-contract=on` (clang default):
        // the *scalar* baseline may emit a fused multiply-add for
        // `dx*dx + dy*dy + dz*dz` written in a single statement, while
        // xsimd's batch operator+ stays unfused. We measure and report
        // the actual delta; if it exceeds 1e-12 the spec acceptance gate
        // (ac-012) will need a tolerance refinement (caller's call).
        using batch_t = xsimd::batch<double>;
        constexpr std::size_t W = batch_t::size;

        const batch_t v_lx(lx), v_ly(ly), v_lz(lz);
        const batch_t v_half_lx(half_lx), v_half_ly(half_ly),
            v_half_lz(half_lz);
        const batch_t v_neg_half_lx(-half_lx), v_neg_half_ly(-half_ly),
            v_neg_half_lz(-half_lz);
        // Floor-form min-image constants. The legacy single-`select` pair
        // form below only handled |Δ| ≤ L; cross-PBC ghosts at ±2L are
        // silently dropped by the cutoff filter. The floor form
        // `Δ -= L · round(Δ/L)` is correct for arbitrary |Δ| in one
        // SIMD op without a branch (see Box::min_image fix).
        const batch_t v_inv_lx(1.0 / lx), v_inv_ly(1.0 / ly),
            v_inv_lz(1.0 / lz);
        const batch_t v_rc_sq(rc_sq);
        const batch_t v_sig6(sig6);
        const batch_t v_eps4(eps4), v_eps24(eps24), v_vshift(vshift);
        const batch_t v_one(1.0), v_two(2.0), v_zero(0.0);

        // Stack-resident packing buffers reused per ket iteration.
        // alignas(64) keeps store_aligned legal on AVX-512 boxes; on NEON
        // 16-byte alignment suffices but 64 is cheap and harmless.
        alignas(64) double xj_buf[16], yj_buf[16], zj_buf[16];
        alignas(64) double dx_buf[16], dy_buf[16], dz_buf[16];
        alignas(64) double fxp_buf[16], fyp_buf[16], fzp_buf[16];
        alignas(64) double r2_buf[16], r6inv_buf[16];
        alignas(64) std::int32_t mask_buf[16];

#pragma omp for schedule(static)
        for (int ii = 0; ii < k_inum; ++ii) {
            // Filtered path: i == ii (neighbors are already pruned of
            // excluded partners). Upstream path: `k_ilist[ii]` gives the
            // owner atom index when NL builds a sparse subset.
            const int i = use_filtered ? ii : k_ilist[ii];
            const int jnum = k_numneigh[i];
            const int off = k_firstneigh[i];
            // Atom-i state hoisted into registers for the inner loop.
            // `fi_*` accumulates Σ_j F_ij once; we store back to `fxt_ptr[i]`
            // a single time at the end -- removes an `fxt[i] += ...` memory
            // RMW per pair (~50 ns bubble per iter on Apple Silicon).
            const double xi = x_ptr[i];
            const double yi = y_ptr[i];
            const double zi = z_ptr[i];
            const batch_t v_xi(xi), v_yi(yi), v_zi(zi);
            double fi_x = 0, fi_y = 0, fi_z = 0;
            // Batch accumulators for the i-force, virial and energy. Masked
            // (out-of-cutoff) lanes contribute 0 (v_fpair is select-masked),
            // so we accumulate whole batches and horizontal-reduce once per
            // atom i -- far less scalar work at W=8 (AVX-512) than the old
            // lane-by-lane add. Only the Newton-3rd-law scatter to f[j]
            // (random index) stays scalar. (FP summation order differs from
            // the legacy lane-serial form: results match to tolerance, not
            // bit-for-bit.)
            batch_t acc_fix(0.0), acc_fiy(0.0), acc_fiz(0.0);
            batch_t acc_vxx(0.0), acc_vyy(0.0), acc_vzz(0.0);
            batch_t acc_vxy(0.0), acc_vxz(0.0), acc_vyz(0.0);
            batch_t acc_e(0.0);

            int jj = 0;
            // ── SIMD ket loop: W neighbors per iteration ────────────
            // Skipped wholesale under a per-type table: the coefficients vary per pair,
            // so nothing hoists and the scalar tail below handles every neighbour.
            for (; use_simd && jj + static_cast<int>(W) <= jnum;
                 jj += static_cast<int>(W)) {
                // Manual gather: pack the next W neighbor coords into the
                // aligned stack buffers. j-indices are non-contiguous (NL
                // is a permutation), so a true SIMD gather instruction
                // would be needed for 1-cycle load -- on NEON64 there is
                // no double-precision gather, so the scalar-load loop is
                // the right thing here.
                for (std::size_t lane = 0; lane < W; ++lane) {
                    const int j =
                        k_neighbors[off + jj + static_cast<int>(lane)];
                    xj_buf[lane] = x_ptr[j];
                    yj_buf[lane] = y_ptr[j];
                    zj_buf[lane] = z_ptr[j];
                }
                batch_t v_xj = batch_t::load_aligned(xj_buf);
                batch_t v_yj = batch_t::load_aligned(yj_buf);
                batch_t v_zj = batch_t::load_aligned(zj_buf);

                batch_t v_dx = v_xi - v_xj;
                batch_t v_dy = v_yi - v_yj;
                batch_t v_dz = v_zi - v_zj;
                if (k_has_shifts) {
                    // Pre-stored image: d = xi - (xj + shift). Contiguous
                    // load (shifts parallel to neighbors). Bit-identical to
                    // the floor-form below for the (stable) chosen image.
                    v_dx = v_dx - batch_t::load_unaligned(&k_shift_x[off + jj]);
                    v_dy = v_dy - batch_t::load_unaligned(&k_shift_y[off + jj]);
                    v_dz = v_dz - batch_t::load_unaligned(&k_shift_z[off + jj]);
                    (void)v_half_lx;
                    (void)v_neg_half_lx;
                    (void)v_half_ly;
                    (void)v_neg_half_ly;
                    (void)v_half_lz;
                    (void)v_neg_half_lz;
                } else {
                    // Branch-free minimum image: select(dx > +L/2, dx-L, dx);
                    // then select(dx < -L/2, dx+L, dx). Hoisted constants
                    // (v_lx, v_half_lx, ...) keep the compiler from reloading
                    // box members across the inner loop -- swapping this for
                    // box.min_image_batch costs ~7% per-step (member loads
                    // through `this` are not hoisted under aliasing analysis).
                    // Floor-form min-image (single shot, handles |Δ| up to
                    // any integer multiple of L). The legacy single-`select`
                    // pair form was correct for `|Δ| ≤ L` but silently left
                    // `|Δ| = L` for cross-PBC ghosts at ±2L (then the
                    // cutoff filter dropped legitimate pairs).
                    v_dx = v_dx - v_lx * xsimd::round(v_dx * v_inv_lx);
                    v_dy = v_dy - v_ly * xsimd::round(v_dy * v_inv_ly);
                    v_dz = v_dz - v_lz * xsimd::round(v_dz * v_inv_lz);
                    // Suppress unused-variable warnings for the legacy
                    // half-L locals which are now reserved for the v3
                    // hot-path microbench (`bench_md_step` v_half_*
                    // sensitivity).
                    (void)v_half_lx;
                    (void)v_neg_half_lx;
                    (void)v_half_ly;
                    (void)v_neg_half_ly;
                    (void)v_half_lz;
                    (void)v_neg_half_lz;
                }

                // r2 with parenthesized addition tree to keep order
                // matching scalar `(dx*dx + dy*dy) + dz*dz`.
                batch_t v_dxsq = v_dx * v_dx;
                batch_t v_dysq = v_dy * v_dy;
                batch_t v_dzsq = v_dz * v_dz;
                batch_t v_r2 = (v_dxsq + v_dysq) + v_dzsq;

                // Cutoff mask. Lanes outside the cutoff get fpair=0 so
                // they contribute nothing to forces / energy / virial; the
                // scalar code reaches those lanes via its `continue`.
                auto mask = v_r2 < v_rc_sq;
                // Guard against r2 == 0 (would-be NaN from 1/r2). All
                // lanes inside the cutoff are guaranteed > 0 (no atom
                // self-pair in the half list); padding lanes outside
                // cutoff are masked anyway. We replace masked-out r2 with
                // 1.0 to keep the divide finite.
                batch_t v_r2_safe = xsimd::select(mask, v_r2, v_one);

                batch_t v_r2inv = v_one / v_r2_safe;
                batch_t v_r6inv = ((v_r2inv * v_r2inv) * v_r2inv) * v_sig6;
                batch_t v_fpair =
                    v_eps24 * v_r2inv * (v_two * v_r6inv * v_r6inv - v_r6inv);
                v_fpair = xsimd::select(mask, v_fpair, v_zero);

                batch_t v_fxp = v_fpair * v_dx;
                batch_t v_fyp = v_fpair * v_dy;
                batch_t v_fzp = v_fpair * v_dz;

                // i-force, virial and energy: whole-batch accumulation
                // (masked lanes contribute 0). Per-pair energy is masked so
                // out-of-cutoff lanes add nothing.
                acc_fix = acc_fix + v_fxp;
                acc_fiy = acc_fiy + v_fyp;
                acc_fiz = acc_fiz + v_fzp;
                if (do_virial) {
                    acc_vxx = acc_vxx + v_dx * v_fxp;
                    acc_vyy = acc_vyy + v_dy * v_fyp;
                    acc_vzz = acc_vzz + v_dz * v_fzp;
                    acc_vxy = acc_vxy + v_dx * v_fyp;
                    acc_vxz = acc_vxz + v_dx * v_fzp;
                    acc_vyz = acc_vyz + v_dy * v_fzp;
                }
                acc_e = acc_e
                        + xsimd::select(mask,
                                        v_eps4 * (v_r6inv * v_r6inv - v_r6inv)
                                            - v_vshift,
                                        v_zero);

                // Newton 3rd-law scatter to j (random index) stays scalar;
                // masked lanes have fxp=0 so the subtract is a harmless no-op.
                v_fxp.store_aligned(fxp_buf);
                v_fyp.store_aligned(fyp_buf);
                v_fzp.store_aligned(fzp_buf);
                for (std::size_t lane = 0; lane < W; ++lane) {
                    const int j =
                        k_neighbors[off + jj + static_cast<int>(lane)];
                    fxt_ptr[j] -= fxp_buf[lane];
                    fyt_ptr[j] -= fyp_buf[lane];
                    fzt_ptr[j] -= fzp_buf[lane];
                }
                (void)mask_buf;
                (void)dx_buf;
                (void)dy_buf;
                (void)dz_buf;
                (void)r2_buf;
                (void)r6inv_buf;
            }

            // Horizontal-reduce the batch accumulators into the per-atom /
            // per-thread scalars before the scalar tail adds to them.
            fi_x += xsimd::reduce_add(acc_fix);
            fi_y += xsimd::reduce_add(acc_fiy);
            fi_z += xsimd::reduce_add(acc_fiz);
            if (do_virial) {
                v_xx += xsimd::reduce_add(acc_vxx);
                v_yy += xsimd::reduce_add(acc_vyy);
                v_zz += xsimd::reduce_add(acc_vzz);
                v_xy += xsimd::reduce_add(acc_vxy);
                v_xz += xsimd::reduce_add(acc_vxz);
                v_yz += xsimd::reduce_add(acc_vyz);
            }
            energy_total += xsimd::reduce_add(acc_e);

            // ── Scalar tail (last jnum % W neighbors) ──────────────
            for (; jj < jnum; ++jj) {
                const int j = k_neighbors[off + jj];
                double dx, dy, dz;
                if (k_has_shifts) {
                    dx = xi - (x_ptr[j] + k_shift_x[off + jj]);
                    dy = yi - (y_ptr[j] + k_shift_y[off + jj]);
                    dz = zi - (z_ptr[j] + k_shift_z[off + jj]);
                } else {
                    dx = xi - x_ptr[j];
                    dy = yi - y_ptr[j];
                    dz = zi - z_ptr[j];
                    // Inlined branchy min-image: matches the SIMD ket-loop
                    // form above. Multi-wrap to handle cross-PBC ghosts at
                    // ±2L on small Cartesian grids (see Box::min_image fix).
                    while (dx > half_lx)
                        dx -= lx;
                    while (dx < -half_lx)
                        dx += lx;
                    while (dy > half_ly)
                        dy -= ly;
                    while (dy < -half_ly)
                        dy += ly;
                    while (dz > half_lz)
                        dz -= lz;
                    while (dz < -half_lz)
                        dz += lz;
                }
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc_sq)
                    continue;

                // Per-pair coefficients under a type table; the hoisted scalars
                // otherwise. An eps of exactly 0 (a type pair with no LJ site, e.g.
                // TIP3P hydrogens) contributes no force and no energy — which is the
                // point, and is why an unset pair mixes to eps = 0 rather than silently
                // falling back to the single-sigma value.
                double p_sig6 = sig6, p_eps24 = eps24, p_eps4 = eps4,
                       p_shift = vshift;
                if (k_multi) {
                    const int ti = k_type[i], tj = k_type[j];
                    const std::size_t ab =
                        static_cast<std::size_t>(ti) * k_ntypes + tj;
                    p_sig6 = k_sig6[ab];
                    const double e = k_eps[ab];
                    p_eps24 = double(24) * e;
                    p_eps4 = double(4) * e;
                    p_shift = k_shift[ab];
                }

                const double r2inv = double(1) / r2;
                const double r6inv = r2inv * r2inv * r2inv * p_sig6;
                const double sc =
                    (k_scales != nullptr) ? k_scales[off + jj] : double(1);
                const double fpair =
                    sc * p_eps24 * r2inv * (double(2) * r6inv * r6inv - r6inv);

                const double fxp = fpair * dx;
                const double fyp = fpair * dy;
                const double fzp = fpair * dz;
                fi_x += fxp;
                fi_y += fyp;
                fi_z += fzp;
                fxt_ptr[j] -= fxp;
                fyt_ptr[j] -= fyp;
                fzt_ptr[j] -= fzp;

                if (do_virial) {
                    v_xx += dx * fxp;
                    v_yy += dy * fyp;
                    v_zz += dz * fzp;
                    v_xy += dx * fyp;
                    v_xz += dx * fzp;
                    v_yz += dy * fzp;
                }

                energy_total +=
                    sc * (p_eps4 * (r6inv * r6inv - r6inv) - p_shift);
            }
            fxt_ptr[i] += fi_x;
            fyt_ptr[i] += fi_y;
            fzt_ptr[i] += fi_z;
        }

        // Reduce per-thread forces into frame (skipped in single-thread direct
        // mode — the kernel already wrote straight into the frame arrays).
        if (!direct) {
#pragma omp for schedule(static)
            for (int i = 0; i < n_force; ++i) {
                for (int t = 0; t < n_threads_; ++t) {
                    fx_ptr[i] += fx_thr_[t][i];
                    fy_ptr[i] += fy_thr_[t][i];
                    fz_ptr[i] += fz_thr_[t][i];
                }
            }
        }
    }

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (do_virial) {
        f.meta.get(kStress)[0] += v_xx;
        f.meta.get(kStress)[1] += v_yy;
        f.meta.get(kStress)[2] += v_zz;
        f.meta.get(kStress)[3] += v_xy;
        f.meta.get(kStress)[4] += v_xz;
        f.meta.get(kStress)[5] += v_yz;
    }
}

void PairLJCut::destroy(const DestroyContext& ctx) {
    (void)ctx;
    fx_thr_.clear();
    fy_thr_.clear();
    fz_thr_.clear();
    special_.clear();
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    any_special_ = false;
    has_partial_special_ = false;
}

std::size_t PairLJCut::memory_bytes() const {
    // `size()` (live elements) ≈ resident pages — see PairList::bytes.
    std::size_t n = 0;
    for (const auto& v : fx_thr_)
        n += v.size() * sizeof(double);
    for (const auto& v : fy_thr_)
        n += v.size() * sizeof(double);
    for (const auto& v : fz_thr_)
        n += v.size() * sizeof(double);
    n += (filtered_neighbors_.size() + filtered_numneigh_.size()
          + filtered_firstneigh_.size())
         * sizeof(int);
    n += (filtered_shift_x_.size() + filtered_shift_y_.size()
          + filtered_shift_z_.size() + filtered_scale_.size())
         * sizeof(double);
    for (const auto& list : special_.by_owner)
        n += list.capacity() * sizeof(std::pair<std::int64_t, double>);
    n += special_.tag_to_owner.size()
         * (sizeof(std::int64_t) + sizeof(int) + 16);
    return n;
}

} // namespace cpu
} // namespace ATV
