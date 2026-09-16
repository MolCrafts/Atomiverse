/// @file cpu/minimize/fire.cpp
/// @brief Bitzek FIRE 1.0 step; publishes ConvergenceMetrics (mirrors CG).
///        MPI-aware: owned-row loops + cross-rank reductions.

#include "atomiverse/cpu/minimize/fire.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/minimize/global_reduce.h"

#include <algorithm>
#include <cmath>

namespace ATV {
namespace cpu {
namespace minimizer {

void FIRE::build(const cpu::BuildContext& ctx) {
    const int dim = ctx.frame.nlocal() * 3;
    v_.assign(dim, 0.0);
    x_prev_.assign(dim, 0.0);
    dt_ = dt0;
    alpha_ = alpha0;
    n_pos_ = 0;
    step_ = 0;
    has_prev_ = false;
    x_prev_valid_ = false;
    last_generation_ = ~std::uint64_t{0};
    max_force_ = 0;
    energy_prev_ = 0;
    trace_.clear();
}

void FIRE::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const int dim = n * 3;

    StorageReal* __restrict sp_x = f.span(kAtom_x);
    StorageReal* __restrict sp_y = f.span(kAtom_y);
    StorageReal* __restrict sp_z = f.span(kAtom_z);
    const AccReal* __restrict sp_fx = f.span(kAtom_fx);
    const AccReal* __restrict sp_fy = f.span(kAtom_fy);
    const AccReal* __restrict sp_fz = f.span(kAtom_fz);

    if (dim > static_cast<int>(v_.size())) {
        v_.resize(dim, 0.0);
        x_prev_.resize(dim, 0.0);
    }

    // Migration reshuffles local rows → inertial state is invalid.
    {
        const auto* dd = ctx.runtime.md.dd;
        const std::uint64_t gen =
            (dd && ctx.runtime.mpi_size > 1) ? dd->neighbor_generation : 0;
        if (gen != last_generation_) {
            if (last_generation_ != ~std::uint64_t{0}) {
#pragma omp parallel for schedule(static)
                for (int k = 0; k < dim; ++k)
                    v_[k] = 0.0;
                dt_ = dt0;
                alpha_ = alpha0;
                n_pos_ = 0;
                has_prev_ = false;
                x_prev_valid_ = false;
                ATV_DEBUG("[fire] migration detected (gen {}) — restart "
                          "inertial state",
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
    reduction(max : local_max_force)
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
    reduction(max : local_max_disp)
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
            (step_ > 0) ? std::abs(f.meta.get(kEnergy) - energy_prev_) : 1e30;

        OptRecord rec{};
        rec.step = step_;
        rec.energy = f.meta.get(kEnergy);
        rec.max_force = max_force_;
        rec.rms_force = rms_force;
        rec.max_displacement = max_disp;
        rec.rms_displacement = rms_disp;
        rec.energy_change = energy_change;
        trace_.push_back(rec);

        if (ConvergenceMetrics* m = ctx.runtime.loop.metrics) {
            m->energy = f.meta.get(kEnergy);
            m->max_force = max_force_;
            m->rms_force = rms_force;
            m->max_displacement = max_disp;
            m->rms_displacement = rms_disp;
            m->energy_change = energy_change;
            m->has_displacement = disp_valid;
        }

        ATV_INFO(
            "[fire] step={} energy_eV={:.6f} max_force_eVA={:.6f} "
            "rms_force_eVA={:.6f} max_disp_A={:.6e} dt_fs={:.4f} alpha={:.4f}",
            step_, f.meta.get(kEnergy), max_force_, rms_force, max_disp, dt_,
            alpha_);

        energy_prev_ = f.meta.get(kEnergy);
        return;
    }

    // ── Search direction = full FIRE 1.0 step (no line_search) ─────────
    if (!ctx.at(ATV::minimizer::Phase::search_direction))
        return;

    // Snapshot geometry for next displacement metrics (before we move).
    // On step 0 this is the pre-move frame; after force, metrics see post-move.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x_prev_[i * 3 + 0] = sp_x[i];
        x_prev_[i * 3 + 1] = sp_y[i];
        x_prev_[i * 3 + 2] = sp_z[i];
    }
    x_prev_valid_ = true;

    // P = F·v (global).
    double p_local = 0;
#pragma omp parallel for schedule(static) reduction(+ : p_local)
    for (int i = 0; i < n; ++i) {
        p_local += sp_fx[i] * v_[i * 3 + 0] + sp_fy[i] * v_[i * 3 + 1]
                   + sp_fz[i] * v_[i * 3 + 2];
    }
    const double P = global_sum(ctx, p_local);

    // |v| and |F| for mixing (global Euclidean norms).
    double v2 = 0, f2 = 0;
#pragma omp parallel for schedule(static) reduction(+ : v2, f2)
    for (int i = 0; i < n; ++i) {
        v2 += v_[i * 3 + 0] * v_[i * 3 + 0] + v_[i * 3 + 1] * v_[i * 3 + 1]
              + v_[i * 3 + 2] * v_[i * 3 + 2];
        f2 += sp_fx[i] * sp_fx[i] + sp_fy[i] * sp_fy[i] + sp_fz[i] * sp_fz[i];
    }
    double norms[2] = {v2, f2};
    global_sum(ctx, norms, 2);
    const double v_norm = std::sqrt(std::max(norms[0], 0.0));
    const double f_norm = std::sqrt(std::max(norms[1], 0.0));

    // Bitzek: P>0 accelerate; P≤0 freeze. When |v|≈0 (first kick after F
    // appears, or cold start), P=0 is not a true uphill reverse — do not
    // shrink Δt (that would make the first physical steps vanish under eV/Å).
    constexpr double kVeps = 1e-30;
    if (P > 0.0 && v_norm > kVeps) {
        ++n_pos_;
        // Mix: v ← (1-α)v + α F̂ |v|
        if (f_norm > 1e-30) {
            const double scale = alpha_ * v_norm / f_norm;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                v_[i * 3 + 0] =
                    (1.0 - alpha_) * v_[i * 3 + 0] + scale * sp_fx[i];
                v_[i * 3 + 1] =
                    (1.0 - alpha_) * v_[i * 3 + 1] + scale * sp_fy[i];
                v_[i * 3 + 2] =
                    (1.0 - alpha_) * v_[i * 3 + 2] + scale * sp_fz[i];
            }
        }
        if (n_pos_ > n_min) {
            dt_ = std::min(dt_ * f_inc, dt_max);
            alpha_ *= f_alpha;
        }
    } else if (v_norm > kVeps) {
        // True uphill / reverse power: freeze, shrink Δt, reset α.
        n_pos_ = 0;
#pragma omp parallel for schedule(static)
        for (int k = 0; k < dim; ++k)
            v_[k] = 0.0;
        dt_ *= f_dec;
        alpha_ = alpha0;
    } else {
        // |v|≈0: keep α, keep Δt; symplectic Euler below builds the first kick.
        n_pos_ = 0;
        alpha_ = alpha0;
    }

    // Symplectic Euler, unit mass: v += dt F; Δx = dt v; clamp; x += Δx.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        v_[i * 3 + 0] += dt_ * sp_fx[i];
        v_[i * 3 + 1] += dt_ * sp_fy[i];
        v_[i * 3 + 2] += dt_ * sp_fz[i];

        double dx = dt_ * v_[i * 3 + 0];
        double dy = dt_ * v_[i * 3 + 1];
        double dz = dt_ * v_[i * 3 + 2];
        const double dmag = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (max_step > 0.0 && dmag > max_step) {
            const double s = max_step / dmag;
            dx *= s;
            dy *= s;
            dz *= s;
        }
        sp_x[i] += dx;
        sp_y[i] += dy;
        sp_z[i] += dz;
    }

    has_prev_ = true;
    ++step_;
}

} // namespace minimizer
} // namespace cpu
} // namespace ATV
