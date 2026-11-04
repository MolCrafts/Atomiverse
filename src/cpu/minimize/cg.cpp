/// @file cpu/minimize/cg.cpp
/// @brief Polak-Ribière+ CG step; publishes ConvergenceMetrics (mirrors LBFGS).
///        MPI-aware: owned-row loops + cross-rank reductions.

#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/iteration_status.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/minimize/global_reduce.h"
#include "atomiverse/cpu/minimize/line_search.h"
#include "atomiverse/cpu/minimize/vector_loop.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace minimizer {

CG::CG(double alpha0) : alpha0_(alpha0) {}

CG& CG::line_search(LineSearchMode mode, double c1, double c2, double max_step,
                    double min_alpha) {
    ls_mode_ = mode;
    ls_c1_ = c1;
    ls_c2_ = c2;
    ls_max_step_ = max_step;
    ls_min_alpha_ = min_alpha;
    return *this;
}

void CG::build(const cpu::BuildContext& ctx) {
    // Capacity-sized buffers (atoms.size() ≥ n_local under multi-rank) —
    // all per-step work happens here, no allocation in launch().
    const int dim = ctx.frame.nlocal() * 3;
    x_prev_.assign(dim, 0.0);
    g_prev_.assign(dim, 0.0);
    d_prev_.assign(dim, 0.0);
    x_ws_.assign(dim, 0.0);
    g_ws_.assign(dim, 0.0);
    d_ws_.assign(dim, 0.0);
    has_prev_ = false;
    x_prev_valid_ = false;
    step_ = 0;
    last_generation_ = ~std::uint64_t{0};
    trace_.clear();
}

void CG::launch(const cpu::LaunchContext& ctx) {
    require_finite_forces(ctx);
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int dim = n * 3;

    // Hoisted once for all three phases below (convergence_check, line_search,
    // search_direction) -- the launch-boundary rule.
    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    AccReal* __restrict sp_fx = f.span(kAtom_fx);
    AccReal* __restrict sp_fy = f.span(kAtom_fy);
    AccReal* __restrict sp_fz = f.span(kAtom_fz);

    // Capacity growth is the one legal re-allocation point.
    if (dim > static_cast<int>(x_ws_.size())) {
        x_prev_.resize(dim, 0.0);
        g_prev_.resize(dim, 0.0);
        d_prev_.resize(dim, 0.0);
        x_ws_.resize(dim, 0.0);
        g_ws_.resize(dim, 0.0);
        d_ws_.resize(dim, 0.0);
    }

    // Migration reshuffles local rows → previous-step vectors are keyed to
    // dead indexing; restart as steepest descent. Rank-uniform: the
    // generation bump is lockstep on every rank (`Comm::exchange`).
    {
        const auto* dd = ctx.runtime.md.dd;
        const std::uint64_t gen =
            (cpu::multi_rank_dd(ctx.runtime)) ? dd->neighbor_generation : 0;
        if (gen != last_generation_) {
            if (last_generation_ != ~std::uint64_t{0}) {
                has_prev_ = false;
                x_prev_valid_ = false;
                ATV_DEBUG("[cg] migration detected (gen {}) — restarting as "
                          "steepest descent",
                          gen);
            }
            last_generation_ = gen;
        }
    }

    // ── Convergence check ──────────────────────────────────────────────
    if (ctx.at(ATV::minimizer::Phase::convergence_check)) {
        double local_max_force = 0;
        double sum_f2 = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum_f2)                \
    reduction(max : local_max_force) if (n >= kParallelMinElements)
        for (int i = 0; i < n; ++i) {
            const double fx = sp_fx[i];
            const double fy = sp_fy[i];
            const double fz = sp_fz[i];
            const double f2 = fx * fx + fy * fy + fz * fz;
            sum_f2 += f2;
            local_max_force = std::max(local_max_force, std::sqrt(f2));
        }

        const bool disp_valid = step_ > 0 && x_prev_valid_;
        double local_max_disp = 0, sum_d2 = 0;
        if (disp_valid) {
#pragma omp parallel for schedule(static) reduction(+ : sum_d2)                \
    reduction(max : local_max_disp) if (n >= kParallelMinElements)
            for (int i = 0; i < n; ++i) {
                const double dx = sp_x[i] - x_prev_[i * 3 + 0];
                const double dy = sp_y[i] - x_prev_[i * 3 + 1];
                const double dz = sp_z[i] - x_prev_[i * 3 + 2];
                sum_d2 += dx * dx + dy * dy + dz * dz;
                local_max_disp = std::max(
                    local_max_disp,
                    std::max({std::abs(dx), std::abs(dy), std::abs(dz)}));
            }
        }

        // Batched collectives, entered unconditionally by every rank.
        double sums[3] = {sum_f2, sum_d2, static_cast<double>(n)};
        global_sum(ctx, sums, 3);
        double maxs[2] = {local_max_force, local_max_disp};
        global_max(ctx, maxs, 2);

        const double n_global = std::max(sums[2], 1.0);
        max_force_ = maxs[0];
        const double rms_force = std::sqrt(sums[0] / (3 * n_global));
        const double max_disp = maxs[1];
        const double rms_disp =
            disp_valid ? std::sqrt(sums[1] / (3 * n_global)) : 0.0;
        const double energy_change =
            (!trace_.empty()) ? std::abs(f.meta.get(kEnergy) - energy_prev_)
                              : 1e30;

        ATV::minimizer::OptRecord rec{};
        rec.step = step_;
        rec.energy = f.meta.get(kEnergy);
        rec.max_force = max_force_;
        rec.rms_force = rms_force;
        rec.max_displacement = max_disp;
        rec.rms_displacement = rms_disp;
        rec.energy_change = energy_change;
        trace_.push_back(rec);

        // PUBLISH, do not decide — see the same note in lbfgs.cpp. This five-way test
        // existed here TOO, character for character. One policy, two copies, and no way
        // for a user to supply a third: that is what a criterion replaces.
        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = f.meta.get(kEnergy);
            m->max_force = max_force_;
            m->rms_force = rms_force;
            m->max_displacement = max_disp;
            m->rms_displacement = rms_disp;
            m->energy_change = energy_change;
            m->has_displacement = disp_valid;
        }

        ATV_INFO("[cg] step={} energy_eV={:.6f} max_force_eVA={:.6f} "
                 "rms_force_eVA={:.6f} max_disp_A={:.6e} rms_disp_A={:.6e} "
                 "dE_eV={:.6e}",
                 step_, f.meta.get(kEnergy), max_force_, rms_force, max_disp,
                 rms_disp, energy_change);

        energy_prev_ = f.meta.get(kEnergy);
        return;
    }

    // ── Line-search phase ───────────────────────────────────────────────
    //
    // search_direction (this step) stored the PR+ direction in d_prev_, the
    // origin geometry in x_prev_, and the descent slope/origin energy. Here we
    // pick the step length and write x_prev_ + alpha*d_prev_; the driver's
    // force phase (next) recomputes forces/energy at that geometry.
    if (ctx.at(ATV::minimizer::Phase::line_search)) {
        // Fixed mode (or no re-eval handle): legacy single step x + alpha0*d.
        if (ls_mode_ == LineSearchMode::Fixed
            || !ctx.runtime.opt.reeval_forces) {
#pragma omp parallel for schedule(static) if (n >= kParallelMinElements)
            for (int i = 0; i < n; ++i) {
                sp_x[i] = x_prev_[i * 3 + 0] + alpha0_ * d_prev_[i * 3 + 0];
                sp_y[i] = x_prev_[i * 3 + 1] + alpha0_ * d_prev_[i * 3 + 1];
                sp_z[i] = x_prev_[i * 3 + 2] + alpha0_ * d_prev_[i * 3 + 2];
            }
            return;
        }
        if (!(ls_gd0_ < 0))
            return; // step 0 / degenerate — leave geometry to the conv. check

        // Last trial evaluated with forces (see LBFGS: trial-force reuse).
        double last_alpha = 0;
        bool last_valid = false;
        auto phi = [&](double alpha, bool need_slope) -> LsSample {
            last_valid = false;
#pragma omp parallel for schedule(static) if (n >= kParallelMinElements)
            for (int i = 0; i < n; ++i) {
                sp_x[i] = x_prev_[i * 3 + 0] + alpha * d_prev_[i * 3 + 0];
                sp_y[i] = x_prev_[i * 3 + 1] + alpha * d_prev_[i * 3 + 1];
                sp_z[i] = x_prev_[i * 3 + 2] + alpha * d_prev_[i * 3 + 2];
            }
            ctx.runtime.opt.reeval_forces(need_slope);
            if (need_slope)
                require_finite_forces(ctx, sp_fx, sp_fy, sp_fz);
            last_alpha = alpha;
            last_valid = need_slope;
            LsSample s;
            s.f = f.meta.get(kEnergy);
            double gd_local = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : gd_local) if (n >= kParallelMinElements)
            for (int i = 0; i < n; ++i) {
                gd_local += -sp_fx[i] * d_prev_[i * 3 + 0]
                            + -sp_fy[i] * d_prev_[i * 3 + 1]
                            + -sp_fz[i] * d_prev_[i * 3 + 2];
            }
            s.dphi = global_sum(ctx, gd_local);
            return s;
        };

        auto search = [&]() -> LineSearchResult {
            const double cap_now =
                (ls_dir_max_ > 1e-20) ? ls_max_step_ / ls_dir_max_ : 1.0;
            const double alpha_max = std::max(cap_now, ls_min_alpha_);
            const double alpha_init = std::min(1.0, alpha_max);
            if (ls_mode_ == LineSearchMode::StrongWolfe)
                return strong_wolfe(phi, ls_phi0_, ls_gd0_, alpha_init, ls_c1_,
                                    ls_c2_, /*max_evals=*/20, alpha_max);
            return armijo_backtrack(phi, ls_phi0_, ls_gd0_, alpha_init, ls_c1_,
                                    ls_contract_, /*max_evals=*/20,
                                    ls_min_alpha_);
        };
        // Trials never own the accepted geometry.
        auto restore_origin = [&] {
#pragma omp parallel for schedule(static) if (n >= kParallelMinElements)
            for (int i = 0; i < n; ++i) {
                sp_x[i] = x_prev_[i * 3 + 0];
                sp_y[i] = x_prev_[i * 3 + 1];
                sp_z[i] = x_prev_[i * 3 + 2];
                sp_fx[i] = -g_prev_[i * 3 + 0];
                sp_fy[i] = -g_prev_[i * 3 + 1];
                sp_fz[i] = -g_prev_[i * 3 + 2];
            }
            f.meta.get(kEnergy) = ls_phi0_;
            last_valid = false;
        };
        LineSearchResult lr;
        bool accepted = false;
        try {
            struct RestoreTrial {
                cpu::Context& context;
                bool migration, evaluation;
                ~RestoreTrial() {
                    context.opt.line_search_trial = migration;
                    context.exec.trial_evaluation = evaluation;
                }
            } restore{ctx.runtime, ctx.runtime.opt.line_search_trial,
                      ctx.runtime.exec.trial_evaluation};
            ctx.runtime.opt.line_search_trial = true;
            ctx.runtime.exec.trial_evaluation = true;
            lr = search();
            accepted = lr.ok || lr.sufficient_decrease;
            bool steepest = true;
            for (int j = 0; j < 3 * n && steepest; ++j)
                steepest = d_prev_[j] == -g_prev_[j];
            if (!accepted && global_max(ctx, steepest ? 0.0 : 1.0) > 0.0) {
                // Restart conjugacy: retry once along steepest descent.
                restore_origin();
                double gg = 0, dir_max = 0;
                for (int j = 0; j < 3 * n; ++j) {
                    d_prev_[j] = -g_prev_[j];
                    gg += g_prev_[j] * g_prev_[j];
                    dir_max = std::max(dir_max, std::abs(d_prev_[j]));
                }
                ls_gd0_ = -global_sum(ctx, gg);
                ls_dir_max_ = global_max(ctx, dir_max);
                ATV_INFO("[cg] step={} line search failed; restarting along "
                         "steepest descent",
                         step_);
                if (ls_gd0_ < 0) {
                    lr = search();
                    accepted = lr.ok || lr.sufficient_decrease;
                }
            }
        } catch (...) {
            restore_origin();
            throw;
        }
        if (!accepted) {
            // No lower energy along steepest descent either: keep the origin
            // and stop without convergence (a result, not an error).
            restore_origin();
            if (IterationStatus* status = ctx.runtime.loop.status) {
                status->stop_requested = true;
                status->reason = "cg_line_search_failed";
            }
            ATV_WARN("[cg] step={} line search failed along steepest descent; "
                     "stopping without convergence",
                     step_);
            return;
        }

        // Accepted step == last trial evaluated: keep its forces when every
        // force-phase op is trial-invariant. Otherwise the normal force phase
        // evaluates this selected geometry and stages its electronic history
        // for the enclosing physical-step transaction.
        const double alpha = lr.alpha;
        if (ctx.runtime.opt.trial_forces_reusable && last_valid
            && alpha == last_alpha) {
            ctx.runtime.opt.forces_current = true;
            return;
        }
#pragma omp parallel for schedule(static) if (n >= kParallelMinElements)
        for (int i = 0; i < n; ++i) {
            sp_x[i] = x_prev_[i * 3 + 0] + alpha * d_prev_[i * 3 + 0];
            sp_y[i] = x_prev_[i * 3 + 1] + alpha * d_prev_[i * 3 + 1];
            sp_z[i] = x_prev_[i * 3 + 2] + alpha * d_prev_[i * 3 + 2];
        }
        return;
    }

    // ── Search direction (Polak-Ribière+) ──────────────────────────────
    auto& g = g_ws_;
    auto& x = x_ws_;
#pragma omp parallel for schedule(static) if (n >= kParallelMinElements)
    for (int i = 0; i < n; ++i) {
        x[i * 3 + 0] = sp_x[i];
        x[i * 3 + 1] = sp_y[i];
        x[i * 3 + 2] = sp_z[i];
        g[i * 3 + 0] = -sp_fx[i];
        g[i * 3 + 1] = -sp_fy[i];
        g[i * 3 + 2] = -sp_fz[i];
    }

    auto& d = d_ws_;
    if (!has_prev_) {
        // Step 0: steepest descent.
#pragma omp parallel for schedule(static) if (dim >= kParallelMinElements)
        for (int k = 0; k < dim; ++k)
            d[k] = -g[k];
    } else {
        // β = max(0, (g · (g - g_prev)) / (g_prev · g_prev)) — both inner
        // products are global sums so every rank applies the identical β.
        double num = 0, den = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : num, den) if (dim >= kParallelMinElements)
        for (int k = 0; k < dim; ++k) {
            num += g[k] * (g[k] - g_prev_[k]);
            den += g_prev_[k] * g_prev_[k];
        }
        double nd[2] = {num, den};
        global_sum(ctx, nd, 2);
        const double beta = (nd[1] > 0.0) ? std::max(0.0, nd[0] / nd[1]) : 0.0;
#pragma omp parallel for schedule(static) if (dim >= kParallelMinElements)
        for (int k = 0; k < dim; ++k)
            d[k] = -g[k] + beta * d_prev_[k];
    }

    // Descent precheck: PR+ does not always yield a descent direction (N&W
    // §5.2). Reset to steepest descent if g·d ≥ 0. Globally reduced.
    double gd_local = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : gd_local) if (dim >= kParallelMinElements)
    for (int k = 0; k < dim; ++k)
        gd_local += g[k] * d[k];
    double gd0 = global_sum(ctx, gd_local);
    if (!(gd0 < 0)) {
#pragma omp parallel for schedule(static) if (dim >= kParallelMinElements)
        for (int k = 0; k < dim; ++k)
            d[k] = -g[k];
        double gg_local = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(+ : gg_local) if (dim >= kParallelMinElements)
        for (int k = 0; k < dim; ++k)
            gg_local += g[k] * g[k];
        gd0 = -global_sum(ctx, gg_local);
    }

    // Global max|d| for the displacement cap.
    double dir_max = 0;
#pragma omp parallel for schedule(static)                                      \
    reduction(max : dir_max) if (dim >= kParallelMinElements)
    for (int k = 0; k < dim; ++k)
        dir_max = std::max(dir_max, std::abs(d[k]));
    ls_dir_max_ = global_max(ctx, dir_max);
    ls_gd0_ = gd0;
    ls_phi0_ = f.meta.get(kEnergy);

    // Save state for next iteration (buffer swaps — no copies, no allocs).
    // After the swap the current direction is in d_prev_ and the origin
    // geometry in x_prev_, which the line_search phase reads.
    std::swap(x_prev_, x_ws_);
    std::swap(g_prev_, g_ws_);
    std::swap(d_prev_, d_ws_);
    has_prev_ = true;
    x_prev_valid_ = true;
    ++step_;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
