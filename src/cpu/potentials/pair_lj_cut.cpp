/// @file cpu/potentials/pair_lj_cut.cpp
/// @brief CPU PairLJCut — LJ 12-6 shifted, xsimd ket-loop, OMP force reduction.

#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

void PairLJCut::grow_type_table_(int need) {
    if (need <= n_types_)
        return;
    // Grow the square tables, preserving what is already set. New entries are
    // unset: a type pair with no coefficient and no diagonal to mix from does
    // not interact (eps = 0, see finalize_type_table_).
    std::vector<double> e(static_cast<std::size_t>(need) * need, 0.0);
    std::vector<double> s(static_cast<std::size_t>(need) * need, 0.0);
    std::vector<char> f(static_cast<std::size_t>(need) * need, 0);
    for (int a = 0; a < n_types_; ++a)
        for (int b = 0; b < n_types_; ++b) {
            const std::size_t old = static_cast<std::size_t>(a) * n_types_ + b;
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

    grow_type_table_(std::max(type_i, type_j) + 1);

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

PairLJCut& PairLJCut::set_energy_shift(bool on) & {
    energy_shift_ = on;
    return *this;
}

PairLJCut PairLJCut::set_energy_shift(bool on) && {
    energy_shift_ = on;
    return std::move(*this);
}

void PairLJCut::build(const BuildContext& ctx) {
    filtered_source_ = nullptr;
    part_source_ = nullptr;
    // No "did you remember to set coefficients?" check is needed: the per-type
    // constructor refuses an empty list, so a PairLJCut cannot exist in that state.
    // That is the point of moving the table into the constructor.
    if (per_type_mode_) {
        // The kernel indexes the table by (type_i, type_j) of every pair, so
        // it must span every type the Frame carries. A type no coefficient
        // names is an unset pair like any other: it does not interact.
        const int* type = ctx.frame.span(kAtom_type);
        int max_type = -1;
        for (int i = 0; i < ctx.frame.nlocal(); ++i) {
            if (type[i] < 0)
                throw std::runtime_error(
                    "PairLJCut: atom types must be >= 0 for a per-type table");
            max_type = std::max(max_type, type[i]);
        }
        grow_type_table_(max_type + 1);
    }
    types_source_ = nullptr;
    if (!per_type_mode_) {
        const double r2inv = 1.0 / cutoff_sq_;
        const double r6inv = r2inv * r2inv * r2inv * sigma6_;
        v_shift_ =
            energy_shift_ ? 4.0 * epsilon_ * (r6inv * r6inv - r6inv) : 0.0;
    }
    finalize_type_table_();
    n_threads_ = 1;
#ifdef _OPENMP
    n_threads_ = omp_get_max_threads();
#endif
    // Chunk force buffers are sized by compute() to the rows the list
    // touches; a single-thread run never allocates them.
    chunks_.clear();

    special_.s12 = s12_;
    special_.s13 = s13_;
    special_.s14 = s14_;
    any_special_ = special_.active();
    has_partial_special_ = (s12_ > 0.0 && s12_ < 1.0)
                           || (s13_ > 0.0 && s13_ < 1.0)
                           || (s14_ > 0.0 && s14_ < 1.0);
    special_.clear();
    if (!any_special_)
        return;
    const int n_owners = ctx.frame.nlocal();
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
    special_.rebuild(f, f.nlocal());
    filtered_inum_ = n;
    filtered_numneigh_.assign(f.n_total(), 0);
    filtered_firstneigh_.assign(f.n_total(), 0);
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_neighbors_.reserve(pairs.total_pairs);
    filtered_scale_.reserve(pairs.total_pairs);
    const int* numneigh = pairs.numneigh.data();
    const int* firstneigh = pairs.firstneigh.data();
    const int* neighbors = pairs.neighbors.data();
    const TagInt* __restrict tag = f.span(kAtom_tag);

    for (int ii = 0; ii < n; ++ii) {
        const int i = pairs.ilist[ii];
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
            shift_tab_[ab] =
                energy_shift_ ? 4.0 * e * (r6inv * r6inv - r6inv) : 0.0;
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
    const int n_force = cpu::force_count(ctx);
    // Exclusion-filtered CSR (built once per NL rebuild). See
    // `rebuild_filtered_list` for the invariant. When no exclusions are
    // configured we iterate the upstream CSR directly — no copy, no
    // allocation, no per-pair branch.
    // Rows (and with them types) change only at a neighbor rebuild. A type
    // outside the table, e.g. one an MC insertion introduced after build(),
    // would index past it; refuse it at the rebuild that brings it in.
    if (n_types_ > 0
        && (types_source_ != &pairs || pairs.version != types_version_)) {
        const int* type = f.span(kAtom_type);
        for (int i = 0; i < n_force; ++i)
            if (type[i] < 0 || type[i] >= n_types_)
                throw std::runtime_error(
                    "PairLJCut: atom type " + std::to_string(type[i])
                    + " is outside the coefficient table (types 0.."
                    + std::to_string(n_types_ - 1)
                    + "); name it in a coefficient before build()");
        types_source_ = &pairs;
        types_version_ = pairs.version;
    }

    const bool use_filtered = any_special_;
    if (use_filtered
        && (filtered_source_ != &pairs || pairs.version != filtered_version_)) {
        rebuild_filtered_list(f, pairs);
        filtered_version_ = pairs.version;
        filtered_source_ = &pairs;
    }
    const double* __restrict k_scales =
        use_filtered ? filtered_scale_.data() : nullptr;
    const int k_inum = use_filtered ? filtered_inum_ : pairs.inum;
    const int* __restrict k_ilist = pairs.ilist.data();
    const int* __restrict k_numneigh =
        use_filtered ? filtered_numneigh_.data() : pairs.numneigh.data();
    const int* __restrict k_firstneigh =
        use_filtered ? filtered_firstneigh_.data() : pairs.firstneigh.data();
    const int* __restrict k_neighbors =
        use_filtered ? filtered_neighbors_.data() : pairs.neighbors.data();

    // Team size, balanced rows and the touched-row bound follow the list.
    if (part_source_ != &pairs || pairs.version != part_version_) {
        partition_.rebuild(k_inum, k_ilist, k_numneigh, k_firstneigh,
                           k_neighbors);
        part_source_ = &pairs;
        part_version_ = pairs.version;
    }
    const int team = partition_.team(n_threads_);
    // Rows past the list's highest reference receive nothing.
    const int n_touch = std::min(partition_.n_touch(), n_force);
    // Single thread (pure-MPI, or a list too small to share): accumulate
    // straight into the frame's force arrays — already cleared by ZeroForces
    // (owned rows) + CommBrick::zero_ghost_forces (ghost rows) — so neither a
    // scratch zero-fill nor a reduce sweep runs, two passes LAMMPS never
    // pays. Otherwise the rows are cut into chunks handed out dynamically;
    // each chunk owns its force buffer and its energy/virial slot (summed in
    // chunk order, so the result does not depend on the thread that ran it).
    const bool direct = (team == 1);
    const int n_chunks = partition_.chunks(team);
    chunks_.prepare(n_chunks, direct ? 0 : n_touch);

    // LAMMPS-style vflag gating: skip the 6-component pair virial when no
    // stress consumer is active (plain NVE). Loop-invariant, so the
    // compiler unswitches the inner loop; forces/energy are unaffected.
    const bool do_virial = compute_stress_ && ctx.stress_requested;
    const bool do_forces = ctx.forces_requested;

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

#pragma omp parallel num_threads(team) if (team > 1)
    {
        // Hoist box edges + LJ constants to locals so the inner loop keeps
        // them in registers instead of reloading the Mat3 / member fields
        // on every pair.
        const double rc_sq = cutoff_sq_;
        const double sig6 = sigma6_;
        const double eps4 = double(4) * epsilon_;
        const double eps24 = double(24) * epsilon_;

        // Per-type coefficients. When the table is active the coefficients depend on
        // (type_i, type_j) and cannot be hoisted out of the inner loop, so the hoisted
        // SIMD ket loop is skipped and the gathered one below packs each lane's
        // coefficients (and special weight) next to its coordinates. The
        // single-epsilon path — the one benchmarked against LAMMPS — is untouched.
        const bool k_multi = (n_types_ > 0);
        // Partial special scales also need the per-pair weight: gathered loop.
        const bool use_simd = !k_multi && !has_partial_special_;
        const bool use_gather = !use_simd;
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

        const batch_t v_rc_sq(rc_sq);
        const batch_t v_sig6(sig6);
        const batch_t v_eps4(eps4), v_eps24(eps24), v_vshift(vshift);
        const batch_t v_one(1.0), v_two(2.0), v_zero(0.0);

        // Stack-resident packing buffers reused per ket iteration.
        // alignas(64) keeps store_aligned legal on AVX-512 boxes; on NEON
        // 16-byte alignment suffices but 64 is cheap and harmless.
        alignas(64) double xj_buf[16], yj_buf[16], zj_buf[16];
        alignas(64) double fxp_buf[16], fyp_buf[16], fzp_buf[16];
        alignas(64) double r2_buf[16], r6inv_buf[16];
        alignas(64) double sig6_buf[16], eps_buf[16], shift_buf[16];
        alignas(64) std::int32_t mask_buf[16];

#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_chunks; ++c) {
            double* fxc = fx_ptr;
            double* fyc = fy_ptr;
            double* fzc = fz_ptr;
            if (!direct)
                chunks_.begin_chunk(c, fxc, fyc, fzc);
            double* __restrict fxt_ptr = fxc;
            double* __restrict fyt_ptr = fyc;
            double* __restrict fzt_ptr = fzc;
            // Chunk-local sums, kept in registers across the force scatter.
            double e_thr = 0;
            double t_xx = 0, t_yy = 0, t_zz = 0, t_xy = 0, t_xz = 0, t_yz = 0;

            const auto [ii_begin, ii_end] = partition_.range(c, n_chunks);
            for (int ii = ii_begin; ii < ii_end; ++ii) {
                // Filtered path: i == ii (neighbors are already pruned of
                // excluded partners). Upstream path: `k_ilist[ii]` gives the
                // owner atom index when NL builds a sparse subset.
                const int i = k_ilist[ii];
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
                    batch_t v_fpair = v_eps24 * v_r2inv
                                      * (v_two * v_r6inv * v_r6inv - v_r6inv);
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
                    // UNCONDITIONAL — do NOT put `if (do_virial)` back here.
                    //
                    // These six batch reduction chains used to sit inside that
                    // loop-invariant branch. Under the project's own Release
                    // flags (GCC 14.3, `-O3 -march=native`, AVX-512 ⇒ W = 8) the
                    // optimizer then dropped most of the accumulation: on a 64-ion
                    // jittered lattice the ket loop contributed −0.294 eV to the
                    // virial trace where the same registers, summed lane-serially,
                    // give −4.168 eV. Energy and forces stayed exact to 1e-17, so
                    // nothing looked wrong — the total pair virial was simply 60 %
                    // short, and every pressure, stress and barostat number taken
                    // from the default single-type LJ path with it. `-O2`,
                    // `-Os`, `-O3` without `-march=native`, and
                    // `-O3 -march=native -fno-tree-vectorize` all produced the
                    // correct value; only the shipped combination did not.
                    //
                    // The branch was a micro-optimisation (skip 6 batch FMAs in a
                    // plain NVE run). Correct stress outranks it, and a
                    // branch-free accumulation is the form the auto-vectorizer
                    // cannot reassociate away. `do_virial` still gates the
                    // horizontal reduce, the scalar tail and the flush below — all
                    // of which are outside this innermost loop.
                    //
                    // Pinned by tests/cpu/potentials/test_pair_virial.cpp
                    // (`PairLJCut.SimdKetLoopVirialMatchesLatticeSum`), which
                    // compares against an independent brute-force image sum and so
                    // catches the miscompilation whatever produces it.
                    acc_vxx = acc_vxx + v_dx * v_fxp;
                    acc_vyy = acc_vyy + v_dy * v_fyp;
                    acc_vzz = acc_vzz + v_dz * v_fzp;
                    acc_vxy = acc_vxy + v_dx * v_fyp;
                    acc_vxz = acc_vxz + v_dx * v_fzp;
                    acc_vyz = acc_vyz + v_dy * v_fzp;
                    acc_e =
                        acc_e
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
                        if (do_forces) {
                            fxt_ptr[j] -= fxp_buf[lane];
                            fyt_ptr[j] -= fyp_buf[lane];
                            fzt_ptr[j] -= fzp_buf[lane];
                        }
                    }
                    (void)mask_buf;
                    (void)r2_buf;
                    (void)r6inv_buf;
                }

                // ── Gathered SIMD ket loop: per-type table and/or per-pair special
                // weights. Same arithmetic as the scalar tail below, W lanes at a
                // time: each lane's sigma^6 / epsilon / shift is gathered from the
                // table row of type_i by type_j, and its weight read from the
                // filtered list (contiguous). Out-of-cutoff lanes are masked to 0.
                const std::size_t g_row =
                    k_multi ? static_cast<std::size_t>(k_type[i]) * k_ntypes
                            : 0;
                for (; use_gather && jj + static_cast<int>(W) <= jnum;
                     jj += static_cast<int>(W)) {
                    for (std::size_t lane = 0; lane < W; ++lane) {
                        const int j =
                            k_neighbors[off + jj + static_cast<int>(lane)];
                        xj_buf[lane] = x_ptr[j];
                        yj_buf[lane] = y_ptr[j];
                        zj_buf[lane] = z_ptr[j];
                        if (k_multi) {
                            const std::size_t ab =
                                g_row + static_cast<std::size_t>(k_type[j]);
                            sig6_buf[lane] = k_sig6[ab];
                            eps_buf[lane] = k_eps[ab];
                            shift_buf[lane] = k_shift[ab];
                        }
                    }
                    const batch_t v_dx = v_xi - batch_t::load_aligned(xj_buf);
                    const batch_t v_dy = v_yi - batch_t::load_aligned(yj_buf);
                    const batch_t v_dz = v_zi - batch_t::load_aligned(zj_buf);
                    const batch_t v_r2 =
                        (v_dx * v_dx + v_dy * v_dy) + v_dz * v_dz;
                    const auto mask = v_r2 < v_rc_sq;
                    const batch_t v_r2inv =
                        v_one / xsimd::select(mask, v_r2, v_one);

                    const batch_t v_s6 =
                        k_multi ? batch_t::load_aligned(sig6_buf) : v_sig6;
                    const batch_t v_e4 =
                        k_multi ? batch_t(4.0) * batch_t::load_aligned(eps_buf)
                                : v_eps4;
                    const batch_t v_e24 =
                        k_multi ? batch_t(24.0) * batch_t::load_aligned(eps_buf)
                                : v_eps24;
                    const batch_t v_sh =
                        k_multi ? batch_t::load_aligned(shift_buf) : v_vshift;
                    const batch_t v_sc =
                        (k_scales != nullptr)
                            ? batch_t::load_unaligned(k_scales + off + jj)
                            : v_one;

                    const batch_t v_r6inv =
                        ((v_r2inv * v_r2inv) * v_r2inv) * v_s6;
                    const batch_t v_fpair = xsimd::select(
                        mask,
                        ((v_sc * v_e24) * v_r2inv)
                            * (v_two * v_r6inv * v_r6inv - v_r6inv),
                        v_zero);
                    const batch_t v_fxp = v_fpair * v_dx;
                    const batch_t v_fyp = v_fpair * v_dy;
                    const batch_t v_fzp = v_fpair * v_dz;
                    acc_fix = acc_fix + v_fxp;
                    acc_fiy = acc_fiy + v_fyp;
                    acc_fiz = acc_fiz + v_fzp;
                    // Unconditional, as in the hoisted loop above (see there).
                    acc_vxx = acc_vxx + v_dx * v_fxp;
                    acc_vyy = acc_vyy + v_dy * v_fyp;
                    acc_vzz = acc_vzz + v_dz * v_fzp;
                    acc_vxy = acc_vxy + v_dx * v_fyp;
                    acc_vxz = acc_vxz + v_dx * v_fzp;
                    acc_vyz = acc_vyz + v_dy * v_fzp;
                    acc_e =
                        acc_e
                        + xsimd::select(
                            mask,
                            v_sc
                                * (v_e4 * (v_r6inv * v_r6inv - v_r6inv) - v_sh),
                            v_zero);
                    v_fxp.store_aligned(fxp_buf);
                    v_fyp.store_aligned(fyp_buf);
                    v_fzp.store_aligned(fzp_buf);
                    if (do_forces) {
                        for (std::size_t lane = 0; lane < W; ++lane) {
                            const int j =
                                k_neighbors[off + jj + static_cast<int>(lane)];
                            fxt_ptr[j] -= fxp_buf[lane];
                            fyt_ptr[j] -= fyp_buf[lane];
                            fzt_ptr[j] -= fzp_buf[lane];
                        }
                    }
                }

                // Horizontal-reduce the batch accumulators into the per-atom /
                // per-thread scalars before the scalar tail adds to them.
                fi_x += xsimd::reduce_add(acc_fix);
                fi_y += xsimd::reduce_add(acc_fiy);
                fi_z += xsimd::reduce_add(acc_fiz);
                double a_xx = 0, a_yy = 0, a_zz = 0, a_xy = 0, a_xz = 0,
                       a_yz = 0;
                if (do_virial) {
                    a_xx = xsimd::reduce_add(acc_vxx);
                    a_yy = xsimd::reduce_add(acc_vyy);
                    a_zz = xsimd::reduce_add(acc_vzz);
                    a_xy = xsimd::reduce_add(acc_vxy);
                    a_xz = xsimd::reduce_add(acc_vxz);
                    a_yz = xsimd::reduce_add(acc_vyz);
                }
                double e_i = xsimd::reduce_add(acc_e);

                // ── Scalar tail (last jnum % W neighbors) ──────────────
                for (; jj < jnum; ++jj) {
                    const int j = k_neighbors[off + jj];
                    const double dx = xi - x_ptr[j];
                    const double dy = yi - y_ptr[j];
                    const double dz = zi - z_ptr[j];
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
                        const std::size_t ab =
                            g_row + static_cast<std::size_t>(k_type[j]);
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
                    const double fpair = sc * p_eps24 * r2inv
                                         * (double(2) * r6inv * r6inv - r6inv);

                    const double fxp = fpair * dx;
                    const double fyp = fpair * dy;
                    const double fzp = fpair * dz;
                    fi_x += fxp;
                    fi_y += fyp;
                    fi_z += fzp;
                    if (do_forces) {
                        fxt_ptr[j] -= fxp;
                        fyt_ptr[j] -= fyp;
                        fzt_ptr[j] -= fzp;
                    }

                    if (do_virial) {
                        a_xx += dx * fxp;
                        a_yy += dy * fyp;
                        a_zz += dz * fzp;
                        a_xy += dx * fyp;
                        a_xz += dx * fzp;
                        a_yz += dy * fzp;
                    }

                    e_i += sc * (p_eps4 * (r6inv * r6inv - r6inv) - p_shift);
                }
                if (do_forces) {
                    fxt_ptr[i] += fi_x;
                    fyt_ptr[i] += fi_y;
                    fzt_ptr[i] += fi_z;
                }
                e_thr += e_i;
                t_xx += a_xx;
                t_yy += a_yy;
                t_zz += a_zz;
                t_xy += a_xy;
                t_xz += a_xz;
                t_yz += a_yz;
            }
            double* sums = chunks_.sums(c);
            sums[0] = e_thr;
            sums[1] = t_xx;
            sums[2] = t_yy;
            sums[3] = t_zz;
            sums[4] = t_xy;
            sums[5] = t_xz;
            sums[6] = t_yz;
        } // chunks (implicit barrier: every scatter has landed)

        // Gather the chunk buffers into the frame (direct mode already wrote
        // straight into it).
        if (!direct && do_forces)
            chunks_.reduce_forces(fx_ptr, fy_ptr, fz_ptr);
    }

    const double energy_total = chunks_.sum(0);
    const double v_xx = chunks_.sum(1), v_yy = chunks_.sum(2),
                 v_zz = chunks_.sum(3), v_xy = chunks_.sum(4),
                 v_xz = chunks_.sum(5), v_yz = chunks_.sum(6);

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
    chunks_.clear();
    special_.clear();
    filtered_neighbors_.clear();
    filtered_scale_.clear();
    filtered_numneigh_.clear();
    filtered_firstneigh_.clear();
    filtered_inum_ = 0;
    filtered_version_ = 0;
    filtered_source_ = nullptr;
    part_source_ = nullptr;
    types_source_ = nullptr;
    any_special_ = false;
    has_partial_special_ = false;
}

std::size_t PairLJCut::memory_bytes() const {
    // `size()` (live elements) ≈ resident pages — see PairList::bytes.
    std::size_t n = 0;
    n += chunks_.memory_bytes();
    n += (filtered_neighbors_.size() + filtered_numneigh_.size()
          + filtered_firstneigh_.size())
         * sizeof(int);
    n += filtered_scale_.size() * sizeof(double);
    for (const auto& list : special_.by_owner)
        n += list.capacity() * sizeof(std::pair<std::int64_t, double>);
    n += special_.tag_to_owner.size()
         * (sizeof(std::int64_t) + sizeof(int) + 16);
    return n;
}

} // namespace cpu
} // namespace ATV
