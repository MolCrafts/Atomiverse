/// @file cpu/minimizer/lbfgs.cpp
/// @brief L-BFGS step implementation: two-loop recursion + in-step line
///        search + multi-criteria convergence. MPI-aware: owned-row loops +
///        cross-rank reductions.

#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/minimize/global_reduce.h"
#include "atomiverse/cpu/minimize/lbfgs_history.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ATV {
namespace cpu {
namespace minimizer {

namespace {

/// Owned-row inner product (per-rank partial; reduce with `global_sum`).
double dot_local(const std::vector<double>& a, const std::vector<double>& b,
                 int dim) {
    double s = 0;
#pragma omp parallel for schedule(static) reduction(+ : s)
    for (int j = 0; j < dim; ++j)
        s += a[j] * b[j];
    return s;
}

} // namespace

LBFGS::LBFGS(int memory, double alpha0, int max_ls)
    : memory_(memory), alpha0_(alpha0), max_ls_(max_ls) {}

double LBFGS::reduce_(double v) const {
    // `hist_` must not know MPI exists — that coupling is exactly what welded the
    // two-loop recursion to the frame. So the all-reduce is bound here, where a
    // LaunchContext is in scope, and handed to the history as a plain double->double.
    return ls_ctx_ ? global_sum(*ls_ctx_, v) : v;
}

LBFGS& LBFGS::line_search(LineSearchMode mode, double c1, double c2,
                          double max_step, double min_alpha) {
    ls_mode_ = mode;
    ls_c1_ = c1;
    ls_c2_ = c2;
    ls_max_step_ = max_step;
    ls_min_alpha_ = min_alpha;
    return *this;
}

void LBFGS::build(const cpu::BuildContext& ctx) {
    // Size every buffer to the frame's allocated capacity: under multi-rank
    // `atoms.size()` is the backing-store capacity (≥ n_local + n_ghost), so
    // the owned-row count can never outgrow it. All per-step work happens in
    // these build-time buffers — no allocation in launch().
    const int n = ctx.frame.nlocal();
    // A valid domain-decomposed rank may own zero atoms. It must still build
    // the optimizer and enter every collective in lockstep with non-empty
    // ranks. Keep one inert storage element so LbfgsHistory's pointer-based
    // zero-length operations receive a valid address; launch() continues to
    // use the exact local dimension (zero on that rank).
    const int dim = std::max(n * 3, 1);

    // The curvature history + two-loop recursion live in LbfgsHistory, which knows
    // nothing about atoms — the band optimizer in cpu/path/ is its second caller.
    // The Reducer is bound HERE, at the one place that has an MPI-aware LaunchContext:
    // the recursion itself must not know MPI exists, which is what made it extractable.
    hist_ = std::make_unique<LbfgsHistory>(
        memory_, dim, [this](double v) { return reduce_(v); });

    direction_.assign(dim, 0);
    x_ws_.assign(dim, 0);
    g_ws_.assign(dim, 0);

    x_prev_valid_ = false;
    energy_prev_ = 0;
    step_ = 0;
    ls_phi0_ = 0;
    ls_gd0_ = 0;
    ls_dir_max_ = 0;
    last_generation_ = ~std::uint64_t{0};
    trace_.clear();

    const char* mode_name = (ls_mode_ == LineSearchMode::StrongWolfe)
                                ? "strong_wolfe"
                            : (ls_mode_ == LineSearchMode::Armijo) ? "armijo"
                                                                   : "fixed";
    ATV_INFO("[lbfgs] build memory={} alpha0={:.6f} max_ls={} line_search={}",
             memory_, alpha0_, max_ls_, mode_name);
}

void LBFGS::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    // The history's Reducer closes over `this` and reads this pointer: the recursion
    // needs an all-reduce but must not know what a LaunchContext is. Re-pointed every
    // launch because the context is a per-call reference, never a stored one.
    ls_ctx_ = &ctx;
    const int n = owned_count(ctx);
    const int dim = n * 3;

    // Hoisted once for all three phases below (convergence_check, line_search,
    // search_direction) -- the launch-boundary rule.
    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    const AccReal* __restrict sp_fx = f.span(kAtom_fx);
    const AccReal* __restrict sp_fy = f.span(kAtom_fy);
    const AccReal* __restrict sp_fz = f.span(kAtom_fz);

    // Capacity growth (driver-level rebuild_frame after a migration influx)
    // is the one legal re-allocation point.
    if (dim > static_cast<int>(x_ws_.size())) {
        hist_->grow(dim);
        direction_.resize(dim, 0);
        x_ws_.resize(dim, 0);
        g_ws_.resize(dim, 0);
    }

    // Atom migration reshuffles local row indexing: curvature history and
    // previous-step state key off row order and must be discarded.
    // `neighbor_generation` is bumped in lockstep on every rank by
    // `Comm::exchange`, so this reset stays rank-uniform.
    {
        const auto* dd = ctx.runtime.md.dd;
        const std::uint64_t gen =
            (dd && ctx.runtime.mpi_size > 1) ? dd->neighbor_generation : 0;
        if (gen != last_generation_) {
            if (last_generation_ != ~std::uint64_t{0}) {
                hist_->reset();
                x_prev_valid_ = false;
                ATV_DEBUG("[lbfgs] migration detected (gen {}) — history "
                          "reset, restarting as steepest descent",
                          gen);
            }
            last_generation_ = gen;
        }
    }

    // ── Convergence check phase ──────────────────────────────────────
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        double local_max_force = 0;
        double sum_f2 = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum_f2)                \
    reduction(max : local_max_force)
        for (int i = 0; i < n; ++i) {
            const double fx = sp_fx[i], fy = sp_fy[i], fz = sp_fz[i];
            const double f2 = fx * fx + fy * fy + fz * fz;
            sum_f2 += f2;
            local_max_force = std::max(local_max_force, std::sqrt(f2));
        }

        const bool disp_valid = step_ > 0 && x_prev_valid_;
        const auto& xp = hist_->x_prev();
        double local_max_disp = 0;
        double sum_d2 = 0;
        if (disp_valid) {
#pragma omp parallel for schedule(static) reduction(+ : sum_d2)                \
    reduction(max : local_max_disp)
            for (int i = 0; i < n; ++i) {
                const double dx = sp_x[i] - xp[i * 3 + 0];
                const double dy = sp_y[i] - xp[i * 3 + 1];
                const double dz = sp_z[i] - xp[i * 3 + 2];
                sum_d2 += dx * dx + dy * dy + dz * dz;
                local_max_disp = std::max(
                    local_max_disp,
                    std::max({std::abs(dx), std::abs(dy), std::abs(dz)}));
            }
        }

        double sums[3] = {sum_f2, sum_d2, static_cast<double>(n)};
        global_sum(ctx, sums, 3);
        double maxs[2] = {local_max_force, local_max_disp};
        global_max(ctx, maxs, 2);

        const double n_global = std::max(sums[2], 1.0);
        max_force_ = maxs[0];
        const double rms_force = std::sqrt(sums[0] / (3 * n_global));
        const double max_disp = maxs[1];
        const double rms_disp =
            disp_valid ? std::sqrt(sums[1] / (3 * n_global)) : double(0);
        const double energy_change =
            (step_ > 0) ? std::abs(f.meta.get(kEnergy) - energy_prev_)
                        : double(1e30);

        OptRecord rec;
        rec.step = step_;
        rec.energy = f.meta.get(kEnergy);
        rec.max_force = max_force_;
        rec.rms_force = rms_force;
        rec.max_displacement = max_disp;
        rec.rms_displacement = rms_disp;
        rec.energy_change = energy_change;
        trace_.push_back(rec);

        // PUBLISH, do not decide. The optimizer measures; a Criterion judges
        // (cpu/driver/criteria.h). Until this line, LBFGS carried five convergence
        // tolerances in its constructor and wrote `outer_status.converged = true`
        // itself — so the criteria could not be changed without touching the algorithm,
        // CG had to duplicate the identical five-way test, and a nested optimizer's
        // convergence stopped the ENCLOSING loop. That last one shipped twice.
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = f.meta.get(kEnergy);
            m->max_force = max_force_;
            m->rms_force = rms_force;
            m->max_displacement = max_disp;
            m->rms_displacement = rms_disp;
            m->energy_change = energy_change;
            m->has_displacement = disp_valid;
        }

        ATV_INFO("[lbfgs] step={} energy_eV={:.6f} max_force_eVA={:.6f} "
                 "rms_force_eVA={:.6f} max_disp_A={:.6e} rms_disp_A={:.6e} "
                 "dE_eV={:.6e}",
                 step_, f.meta.get(kEnergy), max_force_, rms_force, max_disp,
                 rms_disp, energy_change);

        energy_prev_ = f.meta.get(kEnergy);
        return;
    }

    // ── Line-search phase ────────────────────────────────────────────
    //
    // search_direction (run first this step) has already stored `direction_`,
    // the descent slope `ls_gd0_ = g.direction`, the origin energy `ls_phi0_`,
    // and the global `ls_dir_max_`. Here we choose the step length and write
    // the accepted geometry x_prev_ + alpha*direction_. The driver's force
    // phase (next) recomputes forces/energy at that geometry for the
    // convergence check, so we do not re-evaluate after committing alpha.
    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        const auto& xp = hist_->x_prev();
        // Displacement cap: bound the trial step so a wild move cannot blow up
        // an expensive (SCF) evaluator. alpha is dimensionless on direction_.
        const double cap =
            (ls_dir_max_ > 1e-20) ? ls_max_step_ / ls_dir_max_ : 1.0;

        // Fixed mode (or no re-eval handle available): legacy single step.
        if (ls_mode_ == LineSearchMode::Fixed
            || !ctx.runtime.opt.reeval_forces) {
            double alpha = alpha0_;
            if (ls_dir_max_ > 1e-20)
                alpha =
                    std::min(alpha0_, 0.1 / ls_dir_max_); // legacy 0.1 Å cap
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                sp_x[i] = xp[i * 3 + 0] + alpha * direction_[i * 3 + 0];
                sp_y[i] = xp[i * 3 + 1] + alpha * direction_[i * 3 + 1];
                sp_z[i] = xp[i * 3 + 2] + alpha * direction_[i * 3 + 2];
            }
            return;
        }

        // No descent (step 0, or degenerate): leave the geometry unchanged;
        // the convergence check decides whether we are already at a minimum.
        if (!(ls_gd0_ < 0))
            return;

        // phi(alpha) writes the trial geometry, replays the force phase via
        // the driver's re-eval handle, and returns the globally-reduced
        // energy and directional derivative g(x+alpha p).p.
        auto phi = [&](double alpha, bool need_slope) -> LsSample {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                sp_x[i] = xp[i * 3 + 0] + alpha * direction_[i * 3 + 0];
                sp_y[i] = xp[i * 3 + 1] + alpha * direction_[i * 3 + 1];
                sp_z[i] = xp[i * 3 + 2] + alpha * direction_[i * 3 + 2];
            }
            ctx.runtime.opt.reeval_forces(need_slope);
            LsSample s;
            s.f = f.meta.get(kEnergy); // globally reduced by md::mpi::Reduce
            double gd_local = 0;
#pragma omp parallel for schedule(static) reduction(+ : gd_local)
            for (int i = 0; i < n; ++i) {
                gd_local += -sp_fx[i] * direction_[i * 3 + 0]
                            + -sp_fy[i] * direction_[i * 3 + 1]
                            + -sp_fz[i] * direction_[i * 3 + 2];
            }
            s.dphi = global_sum(ctx, gd_local);
            return s;
        };

        const double alpha_max = std::max(cap, ls_min_alpha_);
        const double alpha_init = std::min(1.0, alpha_max);
        LineSearchResult lr;
        // Suppress migration while trials replay the force phase (see
        // Context::line_search_trial); halo exchange still refreshes ghosts.
        ctx.runtime.opt.line_search_trial = true;
        if (ls_mode_ == LineSearchMode::StrongWolfe)
            lr = strong_wolfe(phi, ls_phi0_, ls_gd0_, alpha_init, ls_c1_,
                              ls_c2_, max_ls_, alpha_max);
        else // Armijo
            lr = armijo_backtrack(phi, ls_phi0_, ls_gd0_, alpha_init, ls_c1_,
                                  ls_contract_, max_ls_, ls_min_alpha_);
        ctx.runtime.opt.line_search_trial = false;

        // Commit the accepted geometry (the last trial may have been a
        // different alpha). On a failed search (e.g. machine-flat near the
        // minimum) lr.alpha may be ~0 → effectively no move, which the
        // convergence check then resolves.
        const double alpha = lr.alpha;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            sp_x[i] = xp[i * 3 + 0] + alpha * direction_[i * 3 + 0];
            sp_y[i] = xp[i * 3 + 1] + alpha * direction_[i * 3 + 1];
            sp_z[i] = xp[i * 3 + 2] + alpha * direction_[i * 3 + 2];
        }
        ATV_DEBUG("[lbfgs] line_search step={} mode={} alpha={:.6e} ok={} "
                  "n_eval={}",
                  step_, static_cast<int>(ls_mode_), alpha, lr.ok,
                  lr.n_value_evals);
        return;
    }

    // ── Search-direction phase ───────────────────────────────────────

    // Gradient = -force (we minimize energy, force = -grad E).
    auto& x = x_ws_;
    auto& g = g_ws_;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x[i * 3 + 0] = sp_x[i];
        x[i * 3 + 1] = sp_y[i];
        x[i * 3 + 2] = sp_z[i];
        g[i * 3 + 0] = -sp_fx[i];
        g[i * 3 + 1] = -sp_fy[i];
        g[i * 3 + 2] = -sp_fz[i];
    }

    ATV_DEBUG("[lbfgs] search_direction step={} stored_history={}", step_,
              hist_->stored());

    // Fold the step just taken into the curvature history, then recover the
    // quasi-Newton direction from it. Both the (s,y) update and Nocedal's two-loop
    // recursion now live in LbfgsHistory (cpu/minimizer/lbfgs_history.h) — they were
    // inlined here, reading ctx.frame directly, until the band optimizer became their
    // second caller and the algorithm had to stop being about atoms.
    hist_->push(x.data(), g.data(), dim);
    hist_->direction(g.data(), direction_.data(), dim);

    // Save the pre-step state for the next history update. It doubles as the line
    // search's base point: the trial geometry is x_prev + alpha*direction.
    hist_->save(x.data(), g.data(), dim);
    x_prev_valid_ = true;

    // Descent precheck (N&W §3.1 / Alg PR+): the quasi-Newton direction is a
    // descent direction iff g.p < 0. A bad (s,y) history or numerical noise
    // can violate it; reset to steepest descent so the line search is
    // well-posed. The dot products are globally reduced for rank-uniformity.
    double gd0 = global_sum(ctx, dot_local(g, direction_, dim));
    if (!(gd0 < 0)) {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < dim; ++j)
            direction_[j] = -g[j];
        gd0 = -global_sum(ctx, dot_local(g, g, dim));
        ATV_DEBUG("[lbfgs] non-descent direction step={} — reset to -g", step_);
    }

    // Global max|direction| component for the line-search displacement cap.
    double dir_max = 0;
#pragma omp parallel for schedule(static) reduction(max : dir_max)
    for (int j = 0; j < dim; ++j)
        dir_max = std::max(dir_max, std::abs(direction_[j]));
    ls_dir_max_ = global_max(ctx, dir_max);
    ls_gd0_ = gd0;
    ls_phi0_ =
        f.meta.get(kEnergy); // energy at x_prev_ (pre-step, this geometry)

    ++step_;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
