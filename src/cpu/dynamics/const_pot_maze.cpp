/// @file cpu/dynamics/const_pot_maze.cpp
/// @brief Mass-zero constrained (SHAKE-like) electrode charge dynamics.

#include "atomiverse/cpu/dynamics/const_pot_maze.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"

namespace ATV {
namespace cpu {
namespace md {

namespace {

void matvec(const std::vector<double>& A, int n, const std::vector<double>& x,
            std::vector<double>& y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j)
            s += A[static_cast<std::size_t>(i) * n + j]
                 * x[static_cast<std::size_t>(j)];
        y[static_cast<std::size_t>(i)] = s;
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v)
        m = std::max(m, std::abs(x));
    return m;
}

} // namespace

ConstPotMaZe::ConstPotMaZe(double eta) : bo_(eta) {
    // Foothold-twin defaults (used only when symplectic == false).
    bo_.use_cg = true;
    bo_.predictor_order = 2;
    bo_.electroneutrality = true;
}

ConstPotMaZe& ConstPotMaZe::add_electrode(int type, double psi) & {
    bo_.add_electrode(type, psi);
    return *this;
}
ConstPotMaZe ConstPotMaZe::add_electrode(int type, double psi) && {
    bo_.add_electrode(type, psi);
    return std::move(*this);
}

void ConstPotMaZe::clear_history() {
    Q_prev_.clear();
    Q_curr_.clear();
    hist_count_ = 0;
    max_dQ_ = 0.0;
}

void ConstPotMaZe::build(const BuildContext& ctx) {
    if (!(maze_tol > 0.0))
        throw std::invalid_argument("ConstPotMaZe: maze_tol must be > 0 [V]");
    if (max_maze_iters <= 0)
        throw std::invalid_argument("ConstPotMaZe: max_maze_iters must be > 0");
    bo_.build(ctx);
    clear_history();
    have_u_ = false;
    constraint_residual_ = 0.0;
    last_maze_iters_ = 0;
}

void ConstPotMaZe::launch(const LaunchContext& ctx) {
    if (symplectic) {
        launch_symplectic_(ctx);
        return;
    }
    // Adiabatic foothold twin: CG + ASPC BO solve each step.
    bo_.launch(ctx);
    ChargePacker::forward(ctx);
    const auto& Q = bo_.charges();
    max_dQ_ = 0.0;
    if (hist_count_ >= 1 && Q.size() == Q_curr_.size()) {
        for (std::size_t i = 0; i < Q.size(); ++i)
            max_dQ_ = std::max(max_dQ_, std::abs(Q[i] - Q_curr_[i]));
    }
    Q_curr_ = Q;
    hist_count_ = std::min(hist_count_ + 1, 2);
}

void ConstPotMaZe::ensure_u_() {
    if (have_u_)
        return;
    const int m = static_cast<int>(bo_.elec_atoms_.size());
    u_.assign(static_cast<std::size_t>(m), 0.0);
    // CG on A u = 1 to well below maze_tol so the χ shift cannot pollute
    // the constraint residual. Solved once per elastance build.
    const double utol = maze_tol * 1e-2;
    std::vector<double> r(static_cast<std::size_t>(m), 1.0);
    std::vector<double> p = r;
    std::vector<double> Ap(static_cast<std::size_t>(m));
    double rs = dot(r, r);
    int it = 0;
    while (inf_norm(r) > utol) {
        if (++it > max_maze_iters)
            throw std::runtime_error(
                "ConstPotMaZe: neutrality-direction CG (A u = 1) failed to "
                "converge within max_maze_iters");
        matvec(bo_.A_, m, p, Ap);
        const double pAp = dot(p, Ap);
        if (!(pAp > 1e-300))
            throw std::runtime_error(
                "ConstPotMaZe: CG breakdown on A u = 1 (p·Ap ≈ 0)");
        const double alpha = rs / pAp;
        for (int i = 0; i < m; ++i) {
            u_[static_cast<std::size_t>(i)] +=
                alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -=
                alpha * Ap[static_cast<std::size_t>(i)];
        }
        const double rsnew = dot(r, r);
        const double beta = rsnew / rs;
        for (int i = 0; i < m; ++i)
            p[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(i)]
                + beta * p[static_cast<std::size_t>(i)];
        rs = rsnew;
    }
    have_u_ = true;
}

int ConstPotMaZe::shake_cg_(std::vector<double>& Q,
                            const std::vector<double>& rhs) {
    // SHAKE for the linear constant-potential constraint in the mass-zero
    // limit: corrections along constraint gradients, Q ← Q̃ + Aγ with
    // A²γ = rhs − AQ̃. The CG residual of that system is −σ(Q̃ + Aγ), so
    // the stopping rule ‖r‖∞ ≤ maze_tol bounds the constraint directly.
    const int m = static_cast<int>(bo_.elec_atoms_.size());
    std::vector<double> r(static_cast<std::size_t>(m));
    std::vector<double> tmp(static_cast<std::size_t>(m));
    matvec(bo_.A_, m, Q, tmp); // A Q̃
    for (int i = 0; i < m; ++i)
        r[static_cast<std::size_t>(i)] =
            rhs[static_cast<std::size_t>(i)] - tmp[static_cast<std::size_t>(i)];
    if (inf_norm(r) <= maze_tol)
        return 0; // Verlet guess already on the constraint manifold.

    std::vector<double> gamma(static_cast<std::size_t>(m), 0.0);
    std::vector<double> p = r;
    std::vector<double> Bp(static_cast<std::size_t>(m));
    double rs = dot(r, r);
    int it = 0;
    for (;;) {
        if (++it > max_maze_iters)
            throw std::runtime_error(
                "ConstPotMaZe: constraint CG failed to converge within "
                "max_maze_iters");
        matvec(bo_.A_, m, p, tmp);  // A p
        matvec(bo_.A_, m, tmp, Bp); // A² p
        const double pBp = dot(p, Bp);
        if (!(pBp > 1e-300))
            throw std::runtime_error(
                "ConstPotMaZe: constraint CG breakdown (p·A²p ≈ 0)");
        const double alpha = rs / pBp;
        for (int i = 0; i < m; ++i) {
            gamma[static_cast<std::size_t>(i)] +=
                alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -=
                alpha * Bp[static_cast<std::size_t>(i)];
        }
        if (inf_norm(r) <= maze_tol)
            break;
        const double rsnew = dot(r, r);
        const double beta = rsnew / rs;
        for (int i = 0; i < m; ++i)
            p[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(i)]
                + beta * p[static_cast<std::size_t>(i)];
        rs = rsnew;
    }
    // Q ← Q̃ + Aγ (correction strictly along the constraint gradients).
    matvec(bo_.A_, m, gamma, tmp);
    for (int i = 0; i < m; ++i)
        Q[static_cast<std::size_t>(i)] += tmp[static_cast<std::size_t>(i)];
    return it;
}

void ConstPotMaZe::launch_symplectic_(const LaunchContext& ctx) {
    if (!bo_.matrix_ready_ || bo_.rebuild_matrix_every_step) {
        bo_.collect_electrodes_(ctx.frame);
        bo_.build_matrix_(ctx.frame);
        bo_.matrix_ready_ = true;
        have_u_ = false;
        clear_history(); // electrode set / elastance changed
    }
    bo_.build_rhs_(ctx.frame); // fills b_ and neutrality_target_

    const int m = static_cast<int>(bo_.elec_atoms_.size());
    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        rhs[static_cast<std::size_t>(i)] =
            bo_.psi_vec_[static_cast<std::size_t>(i)]
            - bo_.b_[static_cast<std::size_t>(i)];

    if (bo_.electroneutrality)
        ensure_u_();

    // Previous committed charges (for the Q(t) continuity metric) still
    // live in bo_.Q_ until this launch overwrites them.
    const std::vector<double> Q_before = bo_.Q_;

    // Initial guess: Verlet extrapolation of the **unconstrained** solve
    // history (the χ neutrality shift is re-applied on commit — same
    // convention as ConstPotBO's ASPC record; extrapolating committed
    // charges would leave a constant |χ| constraint violation forever).
    std::vector<double> Q(static_cast<std::size_t>(m), 0.0);
    if (hist_count_ >= 2) {
        for (int i = 0; i < m; ++i)
            Q[static_cast<std::size_t>(i)] =
                2.0 * Q_curr_[static_cast<std::size_t>(i)]
                - Q_prev_[static_cast<std::size_t>(i)];
    } else if (hist_count_ == 1) {
        Q = Q_curr_;
    }

    last_maze_iters_ = shake_cg_(Q, rhs);

    // Shift the unconstrained history (Q_{n-1} ← Q_n ← Q_unc).
    if (hist_count_ >= 1)
        Q_prev_ = Q_curr_;
    Q_curr_ = Q;

    // Simultaneous electroneutrality constraint: uniform potential shift χ
    // along the cached u = A⁻¹1 (gradient of the neutrality constraint in
    // the KKT system), preserving A Q = rhs + χ·1.
    double chi = 0.0;
    if (bo_.electroneutrality) {
        double eQ = 0.0;
        double eu = 0.0;
        for (int i = 0; i < m; ++i) {
            eQ += Q[static_cast<std::size_t>(i)];
            eu += u_[static_cast<std::size_t>(i)];
        }
        if (!(std::abs(eu) > 1e-30))
            throw std::runtime_error(
                "ConstPotMaZe: singular neutrality direction (1·A⁻¹1 ≈ 0)");
        chi = (bo_.neutrality_target_ - eQ) / eu;
        for (int i = 0; i < m; ++i)
            Q[static_cast<std::size_t>(i)] +=
                chi * u_[static_cast<std::size_t>(i)];
    }

    // Constraint residual ‖AQ − rhs − χ·1‖∞ and the BO-convention residual
    // ‖AQ − rhs‖∞ (includes the Lagrange shift) for solver() consistency.
    std::vector<double> AQ(static_cast<std::size_t>(m));
    matvec(bo_.A_, m, Q, AQ);
    constraint_residual_ = 0.0;
    bo_.residual_ = 0.0;
    for (int i = 0; i < m; ++i) {
        const double unshifted =
            AQ[static_cast<std::size_t>(i)] - rhs[static_cast<std::size_t>(i)];
        constraint_residual_ =
            std::max(constraint_residual_, std::abs(unshifted - chi));
        bo_.residual_ = std::max(bo_.residual_, std::abs(unshifted));
    }

    // Commit into the owned BO solver so its accessors stay truthful.
    bo_.Q_ = Q;
    bo_.last_iters_ = last_maze_iters_;
    bo_.last_used_predictor_ = false;
    bo_.total_Q_.assign(bo_.groups_.size(), 0.0);
    for (int a = 0; a < m; ++a) {
        const int g = bo_.elec_group_[static_cast<std::size_t>(a)];
        bo_.total_Q_[static_cast<std::size_t>(g)] +=
            Q[static_cast<std::size_t>(a)];
    }
    bo_.write_charges_(ctx.frame);
    ChargePacker::forward(ctx);
    if (bo_.write_electrode_forces)
        bo_.write_electrode_forces_(ctx.frame);

    // Q(t) continuity metric on committed charges.
    max_dQ_ = 0.0;
    if (hist_count_ >= 1 && Q_before.size() == Q.size()) {
        for (int i = 0; i < m; ++i)
            max_dQ_ = std::max(
                max_dQ_, std::abs(Q[static_cast<std::size_t>(i)]
                                  - Q_before[static_cast<std::size_t>(i)]));
    }
    hist_count_ = std::min(hist_count_ + 1, 2);
}

void ConstPotMaZe::destroy(const DestroyContext& ctx) {
    bo_.destroy(ctx);
    clear_history();
    u_.clear();
    have_u_ = false;
    constraint_residual_ = 0.0;
    last_maze_iters_ = 0;
}

} // namespace md
} // namespace cpu
} // namespace ATV
