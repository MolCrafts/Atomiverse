/// @file src/cpu/gpw/internal/ot.cpp
/// @brief OT parameterization, CP2K FULL_SINGLE_INVERSE, OT-DIIS, fixed-H.

#include "atomiverse/cpu/gpw/internal/ot.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

namespace {

using xt::linalg::dot;

double frob(const RealMatrix& a, const RealMatrix& b) {
    return xt::sum(a * b)();
}

double sinc(double u) {
    if (std::abs(u) < 1e-4) {
        const double u2 = u * u;
        return 1.0 - u2 / 6.0 + u2 * u2 / 120.0;
    }
    return std::sin(u) / u;
}

/// Dense symmetric linear solve via eigendecomposition (SPD / PSD pseudo-inv).
RealMatrix sym_inverse(const RealMatrix& M) {
    return linalg::sym_function(M, [](double t) {
        return t > 1e-10 ? 1.0 / t : 0.0;
    });
}

} // namespace

RealMatrix ot_parameterize(const RealMatrix& C0, const RealMatrix& x,
                           const RealMatrix& S) {
    const RealMatrix G = dot(xt::transpose(x), dot(S, x));
    const RealMatrix cosU = linalg::sym_function(
        G, [](double t) { return std::cos(std::sqrt(std::max(t, 0.0))); });
    const RealMatrix sincU = linalg::sym_function(
        G, [](double t) { return sinc(std::sqrt(std::max(t, 0.0))); });
    return dot(C0, cosU) + dot(x, sincU);
}

OtPrecond::OtPrecond(const RealMatrix& T, const RealMatrix& S) {
    m_inv_ = sym_inverse(T + S);
}

RealMatrix OtPrecond::apply(const RealMatrix& g) const {
    return dot(m_inv_, g);
}

// CP2K make_full_single_inverse (preconditioner_makes.F):
//   M = H
//   cThc ← Cᵀ H C;  ε_HOMO = max eig(cThc)
//   cThc ← cThc − ½ I
//   M  ← M − 2 (S C) cThc (S C)ᵀ
//   ε_min ← min generalized eig of (M, S)   (dense dsygvd; CP2K uses Arnoldi)
//   σ  ← max(1.5 (ε_min − ε_HOMO), energy_gap)
//   if ε_min < σ: M ← M + (σ − ε_min) S
// Dense inverse via symmetric eigendecomposition (PSD guard).
OtPrecondSingleInverse::OtPrecondSingleInverse(const RealMatrix& H,
                                               const RealMatrix& S,
                                               const RealMatrix& C,
                                               double energy_gap) {
    const int nb = static_cast<int>(H.shape(0));
    const int no = static_cast<int>(C.shape(1));
    // CP2K FULL_SINGLE_INVERSE default ENERGY_GAP is 0.08 Ha. No artificial
    // floor: the occupied projector + min-ev shift keep M PD on water RKS
    // and O₂ UKS (the old floor-1.0 FULL_SINGLE form was a dense substitute
    // for missing min-ev estimation).
    const double gap = std::max(energy_gap, 0.0);

    if (no <= 0 || nb <= 0) {
        m_inv_ = sym_inverse(H + std::max(gap, 0.08) * S);
        return;
    }

    const RealMatrix HC = dot(H, C);
    RealMatrix cThc = dot(xt::transpose(C), HC);
    auto [theta_occ, Vocc] = xt::linalg::eigh(cThc);
    (void)Vocc;
    double max_ev = theta_occ(0);
    for (double t : theta_occ)
        max_ev = std::max(max_ev, t);
    for (int i = 0; i < no; ++i)
        cThc(i, i) -= 0.5;

    // M = H − 2 (S C) cThc (S C)ᵀ
    const RealMatrix SC = dot(S, C);
    RealMatrix M = H - 2.0 * dot(SC, dot(cThc, xt::transpose(SC)));

    // Lowest generalized eigenvalue of (M, S) — dense stand-in for Arnoldi.
    double min_ev = 0.0;
    try {
        auto [evals, evecs] = linalg::diagonalize_fock_generalized(M, S);
        (void)evecs;
        min_ev = evals(0);
        for (double t : evals)
            min_ev = std::min(min_ev, t);
    } catch (...) {
        try {
            const RealMatrix Sh = linalg::sym_function(
                S, [](double t) { return t > 1e-12 ? 1.0 / std::sqrt(t) : 0.0; });
            const RealMatrix Morth = dot(Sh, dot(M, Sh));
            auto [evals, evecs] = xt::linalg::eigh(Morth);
            (void)evecs;
            min_ev = evals(0);
            for (double t : evals)
                min_ev = std::min(min_ev, t);
        } catch (...) {
            min_ev = max_ev;
        }
    }

    // Shift virtual floor: CP2K pre_shift logic.
    double pre_shift = std::max(1.5 * (min_ev - max_ev), gap);
    if (min_ev < pre_shift)
        pre_shift = pre_shift - min_ev;
    else
        pre_shift = 0.0;
    if (pre_shift != 0.0)
        M = M + pre_shift * S;

    m_inv_ = sym_inverse(M);
}

RealMatrix OtPrecondSingleInverse::apply(const RealMatrix& g) const {
    return dot(m_inv_, g);
}

OtFixedResult ot_minimize_fixed(const RealMatrix& H, const RealMatrix& S,
                                const RealMatrix& T, const RealMatrix& C0,
                                double tol, int max_iter, double occ) {
    // Fixed-H global min = lowest no generalized eigenvectors.
    (void)T;
    (void)tol;
    (void)max_iter;
    const int nb = static_cast<int>(H.shape(0));
    const int no = static_cast<int>(C0.shape(1));
    OtFixedResult res;
    res.iters = 0;
    res.converged = true;
    if (no <= 0) {
        res.C = C0;
        res.band_energy = 0.0;
        return res;
    }
    auto [eig, Cfull] = linalg::diagonalize_fock_generalized(H, S);
    RealMatrix C = xt::zeros<double>({nb, no});
    for (int i = 0; i < no; ++i)
        for (int u = 0; u < nb; ++u)
            C(u, i) = Cfull(u, i);
    C = linalg::lowdin_orthonormalize(C, S);
    res.C = C;
    res.band_energy = occ * frob(C, dot(H, C));
    res.iters = 1;
    (void)eig;
    return res;
}

// ── OT orbital DIIS (CP2K ot_diis_step, SAFE_DIIS) ─────────────────────────
//
// DIIS B matrix uses CP2K's default g·e metric (qs_ot_minimizer.F):
//   when adding history index j, B_ij = −s ⟨g_j, e_i⟩ for i ≤ j, then
//   B_ji = B_ij. The e·e Pulay form is available in CP2K but is the slower
//   of the two options there; product path matches the default.

OtOrbitalDiis::OtOrbitalDiis(int depth, double stepsize)
    : depth_(std::max(1, depth)), stepsize_(stepsize) {}

void OtOrbitalDiis::reset() {
    hist_x_.clear();
    hist_e_.clear();
    hist_g_.clear();
}

namespace {

/// Rebuild B from a sliding window with CP2K column-j g·e fills.
void fill_diis_B_ge(RealMatrix& B, const std::vector<RealMatrix>& hist_g,
                    const std::vector<RealMatrix>& hist_e, double s, int m) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i <= j; ++i) {
            // B_ij = −s ⟨g_j, e_i⟩  (CP2K ot_diis_step default branch)
            const double bij =
                -s * frob(hist_g[static_cast<std::size_t>(j)],
                          hist_e[static_cast<std::size_t>(i)]);
            B(i, j) = bij;
            B(j, i) = bij;
        }
    }
}

} // namespace

RealMatrix OtOrbitalDiis::step(const RealMatrix& x, const RealMatrix& g,
                               const RealMatrix& precond_g) {
    // Error vector e = −stepsize · P⁻¹ g  (CP2K matrix_h_e)
    RealMatrix e = -stepsize_ * precond_g;

    hist_x_.push_back(x);
    hist_e_.push_back(e);
    hist_g_.push_back(g);
    if (static_cast<int>(hist_x_.size()) > depth_) {
        hist_x_.erase(hist_x_.begin());
        hist_e_.erase(hist_e_.begin());
        hist_g_.erase(hist_g_.begin());
    }
    const int m = static_cast<int>(hist_x_.size());

    // First vector / m=1: pure SD step x + e
    auto sd = [&]() -> RealMatrix { return x + e; };

    if (m == 1)
        return sd();

    // Solve  [B  1] [c] = [0]  with CP2K g·e B
    //        [1ᵀ 0] [λ]   [1]
    const int nsys = m + 1;
    RealMatrix B = xt::zeros<double>({nsys, nsys});
    RealMatrix rhs = xt::zeros<double>({nsys, 1});
    fill_diis_B_ge(B, hist_g_, hist_e_, stepsize_, m);
    for (int i = 0; i < m; ++i) {
        B(i, m) = 1.0;
        B(m, i) = 1.0;
    }
    B(m, m) = 0.0;
    rhs(m, 0) = 1.0;

    // Solve B c = rhs (small dense system; depth ≤ 7 → nsys ≤ 8).
    bool do_sd = false;
    RealMatrix coeffs;
    try {
        coeffs = xt::linalg::solve(B, rhs);
    } catch (...) {
        do_sd = true;
    }

    RealMatrix x_new = xt::zeros<double>(x.shape());
    if (!do_sd) {
        for (int i = 0; i < m; ++i) {
            const double ci = coeffs(i, 0);
            if (!std::isfinite(ci)) {
                do_sd = true;
                break;
            }
            x_new = x_new + ci * (hist_x_[static_cast<std::size_t>(i)]
                                  + hist_e_[static_cast<std::size_t>(i)]);
        }
        if (!do_sd) {
            // SAFE_DIIS: step must be a descent direction: g · Δx < 0
            const double overlap = frob(g, x_new - x);
            if (overlap > 0.0)
                do_sd = true;
        }
    }
    if (do_sd)
        x_new = sd();
    return x_new;
}

RealMatrix OtOrbitalDiis::direction(const RealMatrix& g,
                                    const RealMatrix& precond_g) {
    // Error = −P⁻¹ g (unit-scale; Armijo chooses the actual step length).
    // Metric uses s=1 so B_ij = −⟨g_j, e_i⟩ = ⟨g_j, P⁻¹ g_i⟩.
    RealMatrix e = -precond_g;
    hist_e_.push_back(e);
    hist_g_.push_back(g);
    // Keep x-history as zeros so step() and direction() share the same
    // ring buffer length discipline.
    hist_x_.push_back(xt::zeros<double>(g.shape()));
    if (static_cast<int>(hist_e_.size()) > depth_) {
        hist_e_.erase(hist_e_.begin());
        hist_g_.erase(hist_g_.begin());
        hist_x_.erase(hist_x_.begin());
    }
    const int m = static_cast<int>(hist_e_.size());
    auto sd = [&]() -> RealMatrix { return e; };
    if (m == 1)
        return sd();

    const int nsys = m + 1;
    RealMatrix B = xt::zeros<double>({nsys, nsys});
    RealMatrix rhs = xt::zeros<double>({nsys, 1});
    fill_diis_B_ge(B, hist_g_, hist_e_, /*s=*/1.0, m);
    for (int i = 0; i < m; ++i) {
        B(i, m) = 1.0;
        B(m, i) = 1.0;
    }
    B(m, m) = 0.0;
    rhs(m, 0) = 1.0;

    bool do_sd = false;
    RealMatrix coeffs;
    try {
        coeffs = xt::linalg::solve(B, rhs);
    } catch (...) {
        do_sd = true;
    }

    RealMatrix d = xt::zeros<double>(g.shape());
    if (!do_sd) {
        for (int i = 0; i < m; ++i) {
            const double ci = coeffs(i, 0);
            if (!std::isfinite(ci)) {
                do_sd = true;
                break;
            }
            d = d + ci * hist_e_[static_cast<std::size_t>(i)];
        }
        if (!do_sd) {
            // SAFE_DIIS: direction must be descent: g · d < 0
            if (frob(g, d) >= 0.0)
                do_sd = true;
        }
    }
    if (do_sd)
        d = sd();
    return d;
}

// ── Complex (Bloch) OT ─────────────────────────────────────────────────────

namespace {

using Cplx = std::complex<double>;

double c_frob(const linalg::CplxMatrix& a, const linalg::CplxMatrix& b) {
    double s = 0.0;
    const int n0 = static_cast<int>(a.shape(0));
    const int n1 = static_cast<int>(a.shape(1));
    for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j)
            s += (std::conj(a(i, j)) * b(i, j)).real();
    return s;
}

double c_sinc(double u) {
    if (std::abs(u) < 1e-4) {
        const double u2 = u * u;
        return 1.0 - u2 / 6.0 + u2 * u2 / 120.0;
    }
    return std::sin(u) / u;
}

} // namespace

linalg::CplxMatrix ot_parameterize_k(const linalg::CplxMatrix& C0,
                                     const linalg::CplxMatrix& x,
                                     const linalg::CplxMatrix& S) {
    const linalg::CplxMatrix G = linalg::c_dagger_mul(x, linalg::c_mul(S, x));
    const linalg::CplxMatrix cosU = linalg::herm_function(
        G, [](double t) { return std::cos(std::sqrt(std::max(t, 0.0))); });
    const linalg::CplxMatrix sincU = linalg::herm_function(
        G, [](double t) { return c_sinc(std::sqrt(std::max(t, 0.0))); });
    return linalg::c_mul(C0, cosU) + linalg::c_mul(x, sincU);
}

OtPrecondSingleInverseK::OtPrecondSingleInverseK(const linalg::CplxMatrix& H,
                                                 const linalg::CplxMatrix& S,
                                                 const linalg::CplxMatrix& C) {
    // Simplified H + a S form for multi-k (FULL_KINETIC path preferred there).
    const linalg::CplxMatrix eps = linalg::c_dagger_mul(C, linalg::c_mul(H, C));
    std::vector<double> theta;
    linalg::CplxMatrix V;
    linalg::eigh_hermitian(eps, theta, V);
    double lambda_lo = theta.empty() ? 0.0 : theta[0];
    for (double t : theta)
        lambda_lo = std::min(lambda_lo, t);
    const double a = -lambda_lo + 1.0;
    const linalg::CplxMatrix M = H + a * S;
    m_inv_ = linalg::herm_function(
        M, [](double t) { return t > 1e-8 ? 1.0 / t : 0.0; });
}

linalg::CplxMatrix
OtPrecondSingleInverseK::apply(const linalg::CplxMatrix& g) const {
    return linalg::c_mul(m_inv_, g);
}

OtFixedResultK ot_minimize_fixed_k(const linalg::CplxMatrix& H,
                                   const linalg::CplxMatrix& S,
                                   const linalg::CplxMatrix& T,
                                   const linalg::CplxMatrix& C0, double tol,
                                   int max_iter) {
    // FULL_KINETIC M = T + S CG on complex OT manifold (multi-k product).
    const linalg::CplxMatrix M = T + S;
    const linalg::CplxMatrix Minv = linalg::herm_function(
        M, [](double t) { return t > 1e-12 ? 1.0 / t : 0.0; });

    linalg::CplxMatrix C = linalg::lowdin_orthonormalize_k(C0, S);
    auto band = [&](const linalg::CplxMatrix& Cc) {
        return 2.0 * c_frob(Cc, linalg::c_mul(H, Cc));
    };

    linalg::CplxMatrix d, pz_prev;
    double denom_prev = 1.0;
    bool have_prev = false;

    OtFixedResultK res;
    for (int it = 0; it < max_iter; ++it) {
        res.iters = it + 1;
        const linalg::CplxMatrix g = linalg::band_gradient_k(H, S, C);
        double gnorm = 0.0;
        for (const Cplx v : g)
            gnorm = std::max(gnorm, std::abs(v));
        if (gnorm < tol) {
            res.converged = true;
            break;
        }
        linalg::CplxMatrix pz =
            linalg::s_tangent_k(C, S, linalg::c_mul(Minv, g));
        if (!have_prev) {
            d = -pz;
        } else {
            const double num = c_frob(g, pz - pz_prev);
            const double beta = std::max(0.0, num / denom_prev);
            d = linalg::s_tangent_k(C, S, -pz + beta * d);
        }
        pz_prev = pz;
        denom_prev = c_frob(g, pz);
        have_prev = true;

        const double E0 = band(C);
        double slope = c_frob(g, d);
        double alpha;
        if (slope >= 0.0) {
            d = -pz;
            slope = c_frob(g, d);
            alpha = 0.5;
        } else {
            const double at = 1.0;
            const double Et = band(ot_parameterize_k(C, at * d, S));
            const double denom = 2.0 * (Et - E0 - slope * at);
            alpha = (denom > 1e-300) ? -slope * at * at / denom : at;
            for (int bt = 0; bt < 30; ++bt) {
                if (band(ot_parameterize_k(C, alpha * d, S)) < E0)
                    break;
                alpha *= 0.5;
            }
        }
        C = linalg::lowdin_orthonormalize_k(ot_parameterize_k(C, alpha * d, S),
                                            S);
    }
    res.C = C;
    res.band_energy = band(C);
    return res;
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
