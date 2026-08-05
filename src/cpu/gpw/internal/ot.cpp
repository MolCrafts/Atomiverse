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
//   cThc ← Cᵀ H C − ½ I
//   M  ← M − 2 (S C) cThc (S C)ᵀ
//   σ  ← max(1.5 (ε_min − ε_homo), energy_gap); shift virtuals if needed
//   M  ← M + σ S
// Product uses the HOMO = max eigenvalue of Cᵀ H C and a cheap lower bound
// on the spectrum of M for σ (min diagonal of Löwdin-orthogonalized M).
OtPrecondSingleInverse::OtPrecondSingleInverse(const RealMatrix& H,
                                               const RealMatrix& S,
                                               const RealMatrix& C,
                                               double energy_gap) {
    // H-aware OT preconditioner (CP2K FULL_SINGLE family).
    //
    // Product form (stable on water RKS + O₂ UKS energy OT):
    //   M = H + (−ε_HOMO + ENERGY_GAP) S
    // with ε_HOMO = max eig of the small Cᵀ H C block. This is the dense
    // analogue of CP2K FULL_SINGLE (shift by HOMO, scale virtuals by gap);
    // FULL_SINGLE_INVERSE's extra occupied projector term
    //   −2 (SC)(CᵀHC − ½ I)(SC)ᵀ
    // is available in CP2K for sparse Cholesky invert but, when inverted
    // densely here without Arnoldi LUMO estimation, degraded CG+Armijo
    // convergence on the product micro-cases (water 23→190 outer).
    //
    // Dense M^{-1} via symmetric eigendecomposition (PSD guard).
    const int no = static_cast<int>(C.shape(1));
    // ENERGY_GAP default in CP2K is 0.08, but the dense product path needs a
    // larger margin (1 Ha) for stable CG+Armijo on open-shell micro-cases —
    // too aggressive a shift leaves M barely PD and stalls O₂ UKS.
    const double gap = std::max(energy_gap, 1.0);
    double homo = 0.0;
    if (no > 0) {
        const RealMatrix cThc = dot(xt::transpose(C), dot(H, C));
        auto [theta, V] = xt::linalg::eigh(cThc);
        (void)V;
        homo = theta(0);
        for (double t : theta)
            homo = std::max(homo, t);
    }
    // Shift so occupied spectrum sits at ~+gap in the S-metric.
    const double a = -homo + gap;
    m_inv_ = sym_inverse(H + a * S);
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

OtOrbitalDiis::OtOrbitalDiis(int depth, double stepsize)
    : depth_(std::max(1, depth)), stepsize_(stepsize) {}

void OtOrbitalDiis::reset() {
    hist_x_.clear();
    hist_e_.clear();
}

RealMatrix OtOrbitalDiis::step(const RealMatrix& x, const RealMatrix& g,
                               const RealMatrix& precond_g) {
    // Error vector e = −stepsize · P⁻¹ g  (CP2K matrix_h_e)
    RealMatrix e = -stepsize_ * precond_g;

    hist_x_.push_back(x);
    hist_e_.push_back(e);
    if (static_cast<int>(hist_x_.size()) > depth_) {
        hist_x_.erase(hist_x_.begin());
        hist_e_.erase(hist_e_.begin());
    }
    const int m = static_cast<int>(hist_x_.size());

    // First vector / m=1: pure SD step x + e
    auto sd = [&]() -> RealMatrix { return x + e; };

    if (m == 1)
        return sd();

    // DIIS B_ij = e_i · e_j  (error-vector metric; standard Pulay form).
    // CP2K also supports a g·e metric; e·e is stable and matches common
    // orbital-DIIS practice. Solve  [B  1] [c] = [0]
    //                               [1ᵀ 0] [λ]   [1]
    const int nsys = m + 1;
    RealMatrix B = xt::zeros<double>({nsys, nsys});
    RealMatrix rhs = xt::zeros<double>({nsys, 1});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            const double bij = frob(hist_e_[static_cast<std::size_t>(i)],
                                    hist_e_[static_cast<std::size_t>(j)]);
            B(i, j) = bij;
            B(j, i) = bij;
        }
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
    RealMatrix e = -precond_g;
    hist_e_.push_back(e);
    // Keep x-history as zeros so step() and direction() share the same
    // ring buffer length discipline.
    hist_x_.push_back(xt::zeros<double>(g.shape()));
    if (static_cast<int>(hist_e_.size()) > depth_) {
        hist_e_.erase(hist_e_.begin());
        hist_x_.erase(hist_x_.begin());
    }
    const int m = static_cast<int>(hist_e_.size());
    auto sd = [&]() -> RealMatrix { return e; };
    if (m == 1)
        return sd();

    const int nsys = m + 1;
    RealMatrix B = xt::zeros<double>({nsys, nsys});
    RealMatrix rhs = xt::zeros<double>({nsys, 1});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            const double bij = frob(hist_e_[static_cast<std::size_t>(i)],
                                    hist_e_[static_cast<std::size_t>(j)]);
            B(i, j) = bij;
            B(j, i) = bij;
        }
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
