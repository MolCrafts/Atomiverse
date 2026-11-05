/// @file cpu/potentials/pair_lj_cut.cpp
/// @brief CPU PairLJCut — LJ 12-6 shifted, xsimd ket-loop, OMP force reduction.
///
/// The tail correction follows LAMMPS `PairLJCut::init_one`
/// (`src/pair_lj_cut.cpp`, etail_ij / ptail_ij via `core/lj_type_table.h`),
/// `src/compute_pe.cpp` (etail/V) and `ComputePressure::virial_compute`
/// (`src/compute_pressure.cpp`, ptail/V on the diagonal). LAMMPS checkout
/// `e8c4c278a8`:
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.

#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

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
    table_.emplace(std::move(coeffs), cutoff);
    refresh_cutoff_();
}

void PairLJCut::refresh_cutoff_() {
    // LAMMPS cutforce: the largest per-pair cutoff. The neighbor list must
    // cover it; the hot loop's outer test uses it before the per-pair one.
    const double rc = table_->max_cutoff();
    cutoff_ = static_cast<AccReal>(rc);
    cutoff_sq_ = rc * rc;
}

PairLJCut& PairLJCut::set_type_coeff(const LJTypeCoeff& row) & {
    if (!table_)
        throw std::logic_error(
            "PairLJCut::set_type_coeff: this PairLJCut was built with a single "
            "epsilon/sigma and has no type table. Construct it from a "
            "coefficient list "
            "instead.");
    table_->set_coeff(row);
    refresh_cutoff_();
    return *this;
}

PairLJCut PairLJCut::set_type_coeff(const LJTypeCoeff& row) && {
    set_type_coeff(row);
    return std::move(*this);
}

PairLJCut& PairLJCut::set_r4_attraction(std::vector<R4TypeCoeff> rows) & {
    if (!table_)
        throw std::logic_error(
            "PairLJCut::set_r4_attraction: this PairLJCut was built with a "
            "single epsilon/sigma and has no type table; the r^-4 term is "
            "per type pair. Construct it from a coefficient list instead.");
    table_->set_r4(std::move(rows));
    refresh_cutoff_();
    return *this;
}

PairLJCut PairLJCut::set_r4_attraction(std::vector<R4TypeCoeff> rows) && {
    set_r4_attraction(std::move(rows));
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

PairLJCut& PairLJCut::set_tail_correction(bool on) & {
    tail_ = on;
    tail_ready_ = false;
    return *this;
}

PairLJCut PairLJCut::set_tail_correction(bool on) && {
    set_tail_correction(on);
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
    if (table_) {
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
        table_->build(max_type + 1, energy_shift_);
        n_types_ = table_->n_types();
    }
    types_source_ = nullptr;
    if (!table_) {
        const double r2inv = 1.0 / cutoff_sq_;
        const double r6inv = r2inv * r2inv * r2inv * sigma6_;
        v_shift_ =
            energy_shift_ ? 4.0 * epsilon_ * (r6inv * r6inv - r6inv) : 0.0;
    }
    // The tail's type counts belong to this build's Frame.
    tail_ready_ = false;
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
    md::ForcePass forces(ctx);
    const int team = partition_.team(forces.team());
    // Rows past the list's highest reference receive nothing.
    const int n_touch = std::min(partition_.n_touch(), n_force);
    // Single thread (pure-MPI, or a list too small to share): accumulate
    // straight into the frame's force arrays — already cleared by ZeroForces
    // (owned rows) + CommBrick::zero_ghost_forces (ghost rows) — so neither a
    // scratch zero-fill nor a reduce sweep runs, two passes LAMMPS never
    // pays. Otherwise the rows are cut into chunks handed out dynamically;
    // each chunk owns its force buffer and its energy/virial slot (summed in
    // chunk order, so the result does not depend on the thread that ran it).
    const int n_chunks = partition_.chunks(team);
    forces.open(team, n_chunks, n_touch);

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
    // Only the per-type table indexes by the force-field type.
    const int* __restrict k_type =
        n_types_ > 0 ? f.span(kAtom_type).data() : nullptr;

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
        const bool k_multi = table_.has_value();
        // Partial special scales also need the per-pair weight: gathered loop.
        const bool use_simd = !k_multi && !has_partial_special_;
        // The r^-4 term has its own cutoff and formula per pair: a table that
        // carries it goes down the scalar path (`ATV::r4_attraction`).
        const bool use_gather = !use_simd && !(k_multi && table_->has_r4());
        const int k_ntypes = n_types_;
        // LAMMPS init_one records, one per (type_i, type_j) (lj1..lj4, own
        // LJ cutoff and offset, r^-4 term).
        const LJPairCoeff* k_tab = k_multi ? table_->pairs().data() : nullptr;
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
        alignas(64) double lj1_buf[16], lj2_buf[16], lj3_buf[16];
        alignas(64) double lj4_buf[16], offset_buf[16], cutsq_buf[16];
        alignas(64) std::int32_t mask_buf[16];

#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < n_chunks; ++c) {
            const md::ForceRows rows = forces.rows(c);
            double* __restrict fxt_ptr = rows.fx;
            double* __restrict fyt_ptr = rows.fy;
            double* __restrict fzt_ptr = rows.fz;
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
                // time: each lane's init_one record (lj1..lj4, offset, own LJ
                // cutoff) is gathered from the table row of type_i by type_j, and
                // its weight read from the filtered list (contiguous). Lanes
                // outside their pair's cutoff are masked to 0.
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
                            const LJPairCoeff& c =
                                k_tab[g_row
                                      + static_cast<std::size_t>(k_type[j])];
                            lj1_buf[lane] = c.lj1;
                            lj2_buf[lane] = c.lj2;
                            lj3_buf[lane] = c.lj3;
                            lj4_buf[lane] = c.lj4;
                            offset_buf[lane] = c.offset;
                            cutsq_buf[lane] = c.cut_ljsq;
                        }
                    }
                    const batch_t v_dx = v_xi - batch_t::load_aligned(xj_buf);
                    const batch_t v_dy = v_yi - batch_t::load_aligned(yj_buf);
                    const batch_t v_dz = v_zi - batch_t::load_aligned(zj_buf);
                    const batch_t v_r2 =
                        (v_dx * v_dx + v_dy * v_dy) + v_dz * v_dz;
                    // A pair with no LJ site has cut_ljsq = 0: always masked.
                    const auto mask =
                        v_r2 < (k_multi ? batch_t::load_aligned(cutsq_buf)
                                        : v_rc_sq);
                    const batch_t v_r2inv =
                        v_one / xsimd::select(mask, v_r2, v_one);
                    const batch_t v_sc =
                        (k_scales != nullptr)
                            ? batch_t::load_unaligned(k_scales + off + jj)
                            : v_one;

                    batch_t v_fpair, v_epair;
                    if (k_multi) {
                        const batch_t v_r6inv = (v_r2inv * v_r2inv) * v_r2inv;
                        v_fpair = (v_r2inv * v_r6inv)
                                  * (batch_t::load_aligned(lj1_buf) * v_r6inv
                                     - batch_t::load_aligned(lj2_buf));
                        v_epair =
                            v_r6inv
                                * (batch_t::load_aligned(lj3_buf) * v_r6inv
                                   - batch_t::load_aligned(lj4_buf))
                            - batch_t::load_aligned(offset_buf);
                        v_fpair = v_sc * v_fpair;
                        v_epair = v_sc * v_epair;
                    } else {
                        const batch_t v_r6inv =
                            ((v_r2inv * v_r2inv) * v_r2inv) * v_sig6;
                        v_fpair = ((v_sc * v_eps24) * v_r2inv)
                                  * (v_two * v_r6inv * v_r6inv - v_r6inv);
                        v_epair = v_sc
                                  * (v_eps4 * (v_r6inv * v_r6inv - v_r6inv)
                                     - v_vshift);
                    }
                    v_fpair = xsimd::select(mask, v_fpair, v_zero);
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
                    acc_e = acc_e + xsimd::select(mask, v_epair, v_zero);
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

                    const double sc =
                        (k_scales != nullptr) ? k_scales[off + jj] : double(1);
                    double fpair, epair;
                    if (k_multi) {
                        // LAMMPS init_one record of (type_i, type_j): its own
                        // LJ cutoff and offset, plus the r^-4 term. A pair with
                        // no LJ site has cut_ljsq = 0 and is skipped here, which
                        // is why an unset pair resolves to eps = 0 rather than
                        // silently falling back to the single-sigma value.
                        const LJPairCoeff& c =
                            k_tab[g_row + static_cast<std::size_t>(k_type[j])];
                        fpair = 0.0;
                        epair = 0.0;
                        if (r2 < c.cut_ljsq) {
                            const double r2inv = double(1) / r2;
                            const double r6inv = r2inv * r2inv * r2inv;
                            fpair = r2inv * r6inv * (c.lj1 * r6inv - c.lj2);
                            epair = r6inv * (c.lj3 * r6inv - c.lj4) - c.offset;
                        }
                        if (r2 < c.r4_cutsq) {
                            const R4Term<double> t =
                                r4_attraction(r2, c.r4_a, c.r4_rc5inv, c.r4_e0);
                            fpair += t.fpair;
                            epair += t.energy;
                        }
                        fpair *= sc;
                        epair *= sc;
                    } else {
                        const double r2inv = double(1) / r2;
                        const double r6inv = r2inv * r2inv * r2inv * sig6;
                        fpair = sc * eps24 * r2inv
                                * (double(2) * r6inv * r6inv - r6inv);
                        epair = sc * (eps4 * (r6inv * r6inv - r6inv) - vshift);
                    }

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

                    e_i += epair;
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
            forces.store(c, e_thr,
                         md::Voigt6{{t_xx, t_yy, t_zz, t_xy, t_xz, t_yz}});
        } // chunks
    }

    const double energy_total = forces.energy();
    const md::Voigt6 vir = forces.virial();

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyPair) += energy_total;
    if (do_virial)
        vir.flush_to(f.meta.get(kStress));

    // ── LJ tail correction (LAMMPS pair_modify tail yes) ─────────────────
    // A global constant over the fixed type counts: it lands on the first
    // rank only (the energy/virial reductions sum ranks). The volume is read
    // every step because barostats move the cell.
    if (tail_) {
#ifdef ATV_HAS_MPI
        const bool mpi = cpu::multi_rank_dd(ctx.runtime);
#else
        const bool mpi = false;
#endif
        if (!tail_ready_) {
            const int n_owned = f.nlocal();
            LJTail t;
            if (table_) {
                std::vector<double> cnt(static_cast<std::size_t>(n_types_),
                                        0.0);
                for (int i = 0; i < n_owned; ++i)
                    cnt[static_cast<std::size_t>(k_type[i])] += 1.0;
#ifdef ATV_HAS_MPI
                if (mpi)
                    MPI_Allreduce(MPI_IN_PLACE, cnt.data(), n_types_,
                                  MPI_DOUBLE, MPI_SUM,
                                  ctx.runtime.md.dd->cart_comm);
#endif
                t = table_->tail(cnt);
            } else {
                // Single-type form: every atom is the one LJ type.
                double n = static_cast<double>(n_owned);
#ifdef ATV_HAS_MPI
                if (mpi)
                    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_DOUBLE, MPI_SUM,
                                  ctx.runtime.md.dd->cart_comm);
#endif
                t = lj_tail_pair(epsilon_, sigma_, std::sqrt(cutoff_sq_), n, n);
            }
            etail_pref_ = t.energy;
            ptail_pref_ = t.virial;
            tail_ready_ = true;
        }
        const bool first_rank = !mpi || ctx.runtime.mpi_rank == 0;
        if (first_rank) {
            const double vol = f.box.volume();
            f.meta.get(kEnergy) += etail_pref_ / vol;
            f.meta.get(kEnergyPair) += etail_pref_ / vol;
            if (do_virial)
                for (int d = 0; d < 3; ++d)
                    f.meta.get(kStress)[d] += ptail_pref_ / vol;
        }
    }
}

void PairLJCut::destroy(const DestroyContext& ctx) {
    (void)ctx;
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
