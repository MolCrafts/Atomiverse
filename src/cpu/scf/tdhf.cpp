/// @file cpu/scf/tdhf.cpp
/// @brief TDHFPack (TDA + TDHF/RPA lowest-roots solvers over the link-08
///        OrbitalHessian A/B kernel) and the RHF::tda()/RHF::tdhf() method
///        surface (stability pre-check, transition dipoles, oscillator
///        strengths). @see internal/tdhf_pack.h, tdhf.h.

#include "atomiverse/cpu/scf/internal/tdhf_pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"
#include "atomiverse/cpu/scf/internal/stability.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/tdhf.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

/// Negative-ω² guard [Hartree²]: a reduced-RPA eigenvalue below −kOmega2Tol
/// is a genuine channel instability (imaginary ω) and throws; the band
/// [−kOmega2Tol, +kOmega2Tol] is a numerically-zero root whose ω^{∓1/2}
/// back-substitution is ill-conditioned and throws too. No fallback.
constexpr double kOmega2Tol = 1e-10;

/// Positive-definiteness floor on (A − B) eigenvalues [Hartree]: the
/// Hermitian reduction needs (A − B)^{±1/2}.
constexpr double kAmBFloor = 1e-10;

/// Davidson residual threshold ‖r‖₂ [Hartree] per root.
constexpr double kDavidsonResTol = 1e-9;

/// Davidson expansion-vector survival threshold after orthogonalization.
constexpr double kNewDirTol = 1e-10;

double vdot(const RealVector& a, const RealVector& b) {
    double s = 0.0;
    for (std::size_t r = 0; r < a.size(); ++r)
        s += a(r) * b(r);
    return s;
}

/// Two modified-Gram-Schmidt passes of t against the orthonormal set V;
/// returns the surviving norm (0 ⇒ no new direction) and normalizes t.
double orthonormalize_against(RealVector& t, const std::vector<RealVector>& V) {
    for (int pass = 0; pass < 2; ++pass)
        for (const RealVector& v : V) {
            const double proj = vdot(v, t);
            for (std::size_t r = 0; r < t.size(); ++r)
                t(r) -= proj * v(r);
        }
    const double tn = std::sqrt(vdot(t, t));
    if (tn < kNewDirTol)
        return 0.0;
    for (std::size_t r = 0; r < t.size(); ++r)
        t(r) /= tn;
    return tn;
}

/// Deterministic Davidson seeds: coordinate vectors of the k smallest
/// entries of `diag` (ties break to the lowest index).
std::vector<int> seed_indices(const RealVector& diag, int k) {
    const int n = static_cast<int>(diag.size());
    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](int p, int q) {
        return diag(p) < diag(q);
    });
    order.resize(static_cast<std::size_t>(k));
    return order;
}

/// Symmetric eigendecomposition M = U diag(s) Uᵀ raised to the given power
/// of s (p = ±1/2 for the (A−B)^{±1/2} factors), with a positivity floor.
RealMatrix sym_power(const RealMatrix& M, double p, double floor,
                     const char* what) {
    auto [s, U] = xt::linalg::eigh(M);
    const std::size_t m = M.shape()[0];
    if (s(0) <= floor)
        throw std::runtime_error(
            std::string(what) + ": (A - B) is not positive definite "
            "(lowest eigenvalue = " + std::to_string(s(0))
            + " Ha) — the Hermitian RPA reduction is undefined for this "
            "reference (no fallback)");
    RealMatrix Us = U;
    for (std::size_t c = 0; c < m; ++c) {
        const double w = std::pow(s(c), p);
        for (std::size_t r = 0; r < m; ++r)
            Us(r, c) *= w;
    }
    RealMatrix out = xt::linalg::dot(Us, xt::transpose(U));
    // Symmetrize against round-off (U diag Uᵀ is symmetric analytically).
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = r + 1; c < m; ++c) {
            const double v = 0.5 * (out(r, c) + out(c, r));
            out(r, c) = v;
            out(c, r) = v;
        }
    return out;
}

/// The reduced-RPA ω² gate: negative beyond tolerance ⇒ channel instability
/// (imaginary ω), |ω²| ≈ 0 ⇒ ill-conditioned back-substitution. Throws.
void check_omega2(double w2, int root) {
    if (w2 < -kOmega2Tol)
        throw std::runtime_error(
            "TDHFPack::rpa: reduced RPA eigenvalue omega^2 = "
            + std::to_string(w2) + " Ha^2 (root " + std::to_string(root)
            + ") is negative — the channel's (A + B) block is unstable and "
            "the full RPA excitation energy is imaginary; use tda() or "
            "reconverge the reference along the instability mode "
            "(no fallback)");
    if (w2 <= kOmega2Tol)
        throw std::runtime_error(
            "TDHFPack::rpa: reduced RPA eigenvalue omega^2 = "
            + std::to_string(w2) + " Ha^2 (root " + std::to_string(root)
            + ") is numerically zero — the (X, Y) back-substitution is "
            "ill-conditioned (no fallback)");
}

} // namespace

void TDHFPack::check_n_states_(int n_states) const {
    if (n_states < 1 || n_states > hessian_.n_ov())
        throw std::invalid_argument(
            "TDHFPack: n_states = " + std::to_string(n_states)
            + " outside [1, n_ov = " + std::to_string(hessian_.n_ov())
            + "] — the rotation space has only n_ov roots (no fallback)");
}

TDHFPack::Roots TDHFPack::tda(int n_states) const {
    check_n_states_(n_states);
    return (hessian_.n_ov() <= dense_threshold) ? tda_dense_(n_states)
                                                : tda_davidson_(n_states);
}

TDHFPack::Roots TDHFPack::rpa(int n_states) const {
    check_n_states_(n_states);
    return (hessian_.n_ov() <= dense_threshold) ? rpa_dense_(n_states)
                                                : rpa_davidson_(n_states);
}

TDHFPack::Roots TDHFPack::tda_dense_(int n_states) const {
    const RealMatrix A =
        0.5 * (hessian_.dense_plus(ch_) + hessian_.dense_minus(ch_));
    auto [w, U] = xt::linalg::eigh(A); // ascending
    const std::size_t n = A.shape()[0];
    const std::size_t k = static_cast<std::size_t>(n_states);
    Roots out;
    out.omega = xt::zeros<double>({k});
    out.X = xt::zeros<double>({k, n});
    out.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        out.omega(s) = w(s);
        for (std::size_t r = 0; r < n; ++r)
            out.X(s, r) = U(r, s);
    }
    return out;
}

TDHFPack::Roots TDHFPack::rpa_dense_(int n_states) const {
    const RealMatrix Mp = hessian_.dense_plus(ch_);
    const RealMatrix Mm = hessian_.dense_minus(ch_);
    const RealMatrix T = sym_power(Mm, 0.5, kAmBFloor, "TDHFPack::rpa");
    const RealMatrix Ti = sym_power(Mm, -0.5, kAmBFloor, "TDHFPack::rpa");

    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    const std::size_t n = M.shape()[0];
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M); // ascending ω²

    const std::size_t k = static_cast<std::size_t>(n_states);
    Roots out;
    out.omega = xt::zeros<double>({k});
    out.X = xt::zeros<double>({k, n});
    out.Y = xt::zeros<double>({k, n});
    for (std::size_t s = 0; s < k; ++s) {
        check_omega2(w2(s), static_cast<int>(s));
        const double omega = std::sqrt(w2(s));
        out.omega(s) = omega;
        // (X+Y) = ω^{-1/2} T z, (X−Y) = ω^{1/2} T^{-1} z ⇒ XᵀX − YᵀY = zᵀz = 1.
        for (std::size_t r = 0; r < n; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                xpy += T(r, c) * Z(c, s);
                xmy += Ti(r, c) * Z(c, s);
            }
            xpy /= std::sqrt(omega);
            xmy *= std::sqrt(omega);
            out.X(s, r) = 0.5 * (xpy + xmy);
            out.Y(s, r) = 0.5 * (xpy - xmy);
        }
    }
    return out;
}

TDHFPack::Roots TDHFPack::tda_davidson_(int n_states) const {
    const int n = hessian_.n_ov();
    const int k = n_states;
    const RealVector diag = hessian_.diagonal_plus(ch_);
    const int max_subspace = std::min(n, std::max(60, 12 * k));
    const int max_iter = 300;

    std::vector<RealVector> V, W; // orthonormal subspace and A·V products
    for (int idx : seed_indices(diag, k)) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(n)});
        b(idx) = 1.0;
        V.push_back(b);
        W.push_back(0.5 * (hessian_.sigma_plus(b, ch_)
                           + hessian_.sigma_minus(b, ch_)));
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix H = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int p = 0; p < m; ++p)
            for (int q = p; q < m; ++q) {
                const double s =
                    0.5 * (vdot(V[static_cast<std::size_t>(p)],
                                W[static_cast<std::size_t>(q)])
                           + vdot(V[static_cast<std::size_t>(q)],
                                  W[static_cast<std::size_t>(p)]));
                H(p, q) = s;
                H(q, p) = s;
            }
        auto [w, S] = xt::linalg::eigh(H);

        // Ritz vectors + residuals for the k lowest roots.
        std::vector<RealVector> x(static_cast<std::size_t>(k)),
            res(static_cast<std::size_t>(k));
        bool all_conv = true;
        for (int s = 0; s < k; ++s) {
            RealVector xs = xt::zeros<double>({static_cast<std::size_t>(n)});
            RealVector as = xt::zeros<double>({static_cast<std::size_t>(n)});
            for (int p = 0; p < m; ++p) {
                const double c = S(p, s);
                for (int r = 0; r < n; ++r) {
                    xs(r) += c * V[static_cast<std::size_t>(p)](r);
                    as(r) += c * W[static_cast<std::size_t>(p)](r);
                }
            }
            RealVector rs = as - w(s) * xs;
            if (std::sqrt(vdot(rs, rs)) >= kDavidsonResTol)
                all_conv = false;
            x[static_cast<std::size_t>(s)] = std::move(xs);
            res[static_cast<std::size_t>(s)] = std::move(rs);
        }

        bool expanded = false;
        bool cap_blocked = false;
        if (!all_conv && m < n) {
            for (int s = 0; s < k; ++s) {
                if (std::sqrt(vdot(res[static_cast<std::size_t>(s)],
                                   res[static_cast<std::size_t>(s)]))
                    < kDavidsonResTol)
                    continue;
                if (static_cast<int>(V.size()) >= max_subspace) {
                    cap_blocked = true;
                    break;
                }
                RealVector t =
                    xt::zeros<double>({static_cast<std::size_t>(n)});
                for (int r = 0; r < n; ++r) {
                    double denom = w(s) - diag(r);
                    if (std::abs(denom) < 1e-8)
                        denom = (denom < 0.0) ? -1e-8 : 1e-8;
                    t(r) = res[static_cast<std::size_t>(s)](r) / denom;
                }
                if (orthonormalize_against(t, V) > 0.0) {
                    V.push_back(t);
                    W.push_back(0.5 * (hessian_.sigma_plus(t, ch_)
                                       + hessian_.sigma_minus(t, ch_)));
                    expanded = true;
                }
            }
        }
        if (!all_conv && cap_blocked && !expanded)
            throw std::runtime_error(
                "TDHFPack::tda: Davidson did not converge within the "
                "subspace cap (" + std::to_string(max_subspace)
                + " vectors, n_ov = " + std::to_string(n)
                + ") — no fallback");

        if (all_conv || m == n || !expanded) {
            Roots out;
            out.omega = xt::zeros<double>({static_cast<std::size_t>(k)});
            out.X = xt::zeros<double>({static_cast<std::size_t>(k),
                                       static_cast<std::size_t>(n)});
            out.Y = xt::zeros<double>({static_cast<std::size_t>(k),
                                       static_cast<std::size_t>(n)});
            for (int s = 0; s < k; ++s) {
                out.omega(s) = w(s);
                RealVector& xs = x[static_cast<std::size_t>(s)];
                const double xn = std::sqrt(vdot(xs, xs));
                for (int r = 0; r < n; ++r)
                    out.X(s, r) = xs(r) / xn;
            }
            return out;
        }
    }
    throw std::runtime_error(
        "TDHFPack::tda: Davidson did not converge in 300 iterations "
        "(n_ov = " + std::to_string(hessian_.n_ov()) + ") — no fallback");
}

TDHFPack::Roots TDHFPack::rpa_davidson_(int n_states) const {
    // Reduced-subspace RPA (Stratmann-Scuseria-Frisch 1998): expand ONE
    // orthonormal subspace V, project P = Vᵀ(A+B)V and Q = Vᵀ(A−B)V via σ±
    // products, and solve the ω² problem Q^{1/2} P Q^{1/2} z = ω² z densely
    // in the m-dimensional subspace only.
    const int n = hessian_.n_ov();
    const int k = n_states;
    const RealVector diag = hessian_.diagonal_plus(ch_);
    const int max_subspace = std::min(n, std::max(60, 12 * k));
    const int max_iter = 300;

    std::vector<RealVector> V, Wp, Wm; // subspace, σ₊·V, σ₋·V
    for (int idx : seed_indices(diag, k)) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(n)});
        b(idx) = 1.0;
        V.push_back(b);
        Wp.push_back(hessian_.sigma_plus(b, ch_));
        Wm.push_back(hessian_.sigma_minus(b, ch_));
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix P = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        RealMatrix Q = P;
        for (int p = 0; p < m; ++p)
            for (int q = p; q < m; ++q) {
                const double sp =
                    0.5 * (vdot(V[static_cast<std::size_t>(p)],
                                Wp[static_cast<std::size_t>(q)])
                           + vdot(V[static_cast<std::size_t>(q)],
                                  Wp[static_cast<std::size_t>(p)]));
                const double sm =
                    0.5 * (vdot(V[static_cast<std::size_t>(p)],
                                Wm[static_cast<std::size_t>(q)])
                           + vdot(V[static_cast<std::size_t>(q)],
                                  Wm[static_cast<std::size_t>(p)]));
                P(p, q) = sp;
                P(q, p) = sp;
                Q(p, q) = sm;
                Q(q, p) = sm;
            }
        const RealMatrix Qh =
            sym_power(Q, 0.5, kAmBFloor, "TDHFPack::rpa(davidson)");
        const RealMatrix Qih =
            sym_power(Q, -0.5, kAmBFloor, "TDHFPack::rpa(davidson)");
        RealMatrix M =
            xt::linalg::dot(Qh, RealMatrix(xt::linalg::dot(P, Qh)));
        for (int r = 0; r < m; ++r)
            for (int c = r + 1; c < m; ++c) {
                const double v = 0.5 * (M(r, c) + M(c, r));
                M(r, c) = v;
                M(c, r) = v;
            }
        auto [w2, Z] = xt::linalg::eigh(M);

        // Back-substituted Ritz pairs + the two residuals per root:
        //   r₁ = (A+B)(X+Y) − ω(X−Y),  r₂ = (A−B)(X−Y) − ω(X+Y).
        std::vector<double> omega(static_cast<std::size_t>(k));
        std::vector<RealVector> xpy(static_cast<std::size_t>(k)),
            xmy(static_cast<std::size_t>(k)), r1(static_cast<std::size_t>(k)),
            r2(static_cast<std::size_t>(k));
        bool all_conv = true;
        for (int s = 0; s < k; ++s) {
            check_omega2(w2(s), s);
            const double om = std::sqrt(w2(s));
            omega[static_cast<std::size_t>(s)] = om;
            // Subspace coordinates: (X+Y) = V·(Q^{1/2} z / √ω),
            // (X−Y) = V·(Q^{-1/2} z · √ω); (X+Y)ᵀ(X−Y) = zᵀz = 1.
            RealVector c = xt::zeros<double>({static_cast<std::size_t>(m)});
            RealVector d = c;
            for (int p = 0; p < m; ++p) {
                double ch_c = 0.0, ch_d = 0.0;
                for (int q = 0; q < m; ++q) {
                    ch_c += Qh(p, q) * Z(q, s);
                    ch_d += Qih(p, q) * Z(q, s);
                }
                c(p) = ch_c / std::sqrt(om);
                d(p) = ch_d * std::sqrt(om);
            }
            RealVector xp = xt::zeros<double>({static_cast<std::size_t>(n)});
            RealVector xm = xp, s1 = xp, s2 = xp;
            for (int p = 0; p < m; ++p)
                for (int r = 0; r < n; ++r) {
                    xp(r) += c(p) * V[static_cast<std::size_t>(p)](r);
                    xm(r) += d(p) * V[static_cast<std::size_t>(p)](r);
                    s1(r) += c(p) * Wp[static_cast<std::size_t>(p)](r);
                    s2(r) += d(p) * Wm[static_cast<std::size_t>(p)](r);
                }
            RealVector rr1 = s1 - om * xm;
            RealVector rr2 = s2 - om * xp;
            if (std::sqrt(vdot(rr1, rr1)) >= kDavidsonResTol
                || std::sqrt(vdot(rr2, rr2)) >= kDavidsonResTol)
                all_conv = false;
            xpy[static_cast<std::size_t>(s)] = std::move(xp);
            xmy[static_cast<std::size_t>(s)] = std::move(xm);
            r1[static_cast<std::size_t>(s)] = std::move(rr1);
            r2[static_cast<std::size_t>(s)] = std::move(rr2);
        }

        bool expanded = false;
        bool cap_blocked = false;
        if (!all_conv && m < n) {
            for (int s = 0; s < k; ++s) {
                for (const RealVector* rv :
                     {&r1[static_cast<std::size_t>(s)],
                      &r2[static_cast<std::size_t>(s)]}) {
                    if (std::sqrt(vdot(*rv, *rv)) < kDavidsonResTol)
                        continue;
                    if (static_cast<int>(V.size()) >= max_subspace) {
                        cap_blocked = true;
                        break;
                    }
                    RealVector t =
                        xt::zeros<double>({static_cast<std::size_t>(n)});
                    for (int r = 0; r < n; ++r) {
                        double denom =
                            omega[static_cast<std::size_t>(s)] - diag(r);
                        if (std::abs(denom) < 1e-8)
                            denom = (denom < 0.0) ? -1e-8 : 1e-8;
                        t(r) = (*rv)(r) / denom;
                    }
                    if (orthonormalize_against(t, V) > 0.0) {
                        V.push_back(t);
                        Wp.push_back(hessian_.sigma_plus(t, ch_));
                        Wm.push_back(hessian_.sigma_minus(t, ch_));
                        expanded = true;
                    }
                }
            }
        }

        if (!all_conv && cap_blocked && !expanded)
            throw std::runtime_error(
                "TDHFPack::rpa: Davidson did not converge within the "
                "subspace cap (" + std::to_string(max_subspace)
                + " vectors, n_ov = " + std::to_string(n)
                + ") — no fallback");

        if (all_conv || m == n || !expanded) {
            Roots out;
            out.omega = xt::zeros<double>({static_cast<std::size_t>(k)});
            out.X = xt::zeros<double>({static_cast<std::size_t>(k),
                                       static_cast<std::size_t>(n)});
            out.Y = xt::zeros<double>({static_cast<std::size_t>(k),
                                       static_cast<std::size_t>(n)});
            for (int s = 0; s < k; ++s) {
                out.omega(s) = omega[static_cast<std::size_t>(s)];
                for (int r = 0; r < n; ++r) {
                    out.X(s, r) =
                        0.5 * (xpy[static_cast<std::size_t>(s)](r)
                               + xmy[static_cast<std::size_t>(s)](r));
                    out.Y(s, r) =
                        0.5 * (xpy[static_cast<std::size_t>(s)](r)
                               - xmy[static_cast<std::size_t>(s)](r));
                }
            }
            return out;
        }
    }
    throw std::runtime_error(
        "TDHFPack::rpa: Davidson did not converge in 300 iterations "
        "(n_ov = " + std::to_string(hessian_.n_ov()) + ") — no fallback");
}

} // namespace internal

// ── RHF method surface: tda() / tdhf() beside stability() ────────────────

namespace {

internal::Channel internal_channel(ExcitationChannel ch) {
    return (ch == ExcitationChannel::triplet) ? internal::Channel::triplet
                                              : internal::Channel::singlet;
}

/// Package pack roots into the public result: eV energies for both
/// channels; singlet channel adds length-gauge transition dipoles through
/// the link-03 MultipoleIntegrals kernel (μ_0n = √2 Σ_ia (X+Y)_ia ⟨i|r|a⟩)
/// and f = (2/3)·ω[Ha]·|μ|²; the triplet channel is spin-forbidden (μ = 0,
/// f = 0). Two callers (tda / tdhf) — second-caller extraction rule.
ExcitedStatesResult excited_states_result(
    const SlaterDeterminant& det, const IntegralPack& pack,
    const internal::TDHFPack::Roots& roots, ExcitationChannel channel,
    bool tda_flag) {
    const std::size_t k = roots.omega.size();
    ExcitedStatesResult out;
    out.channel = channel;
    out.tda = tda_flag;
    out.energy_ev.resize(k);
    out.oscillator_strength.assign(k, 0.0);
    out.transition_dipole_au = xt::zeros<double>({k, std::size_t(3)});
    for (std::size_t s = 0; s < k; ++s)
        out.energy_ev[s] = roots.omega(s) * units::kHartreeInEv;
    if (channel == ExcitationChannel::triplet)
        return out; // spin-forbidden from the singlet ground state

    // AO dipole matrices (engine Cartesian convention; spherical bases
    // sandwich with the pack's c2s), origin at 0 — transition moments
    // between orthogonal states are origin-independent.
    const double origin[3] = {0.0, 0.0, 0.0};
    const MultipoleIntegrals mi{pack.shells.data(),
                                static_cast<int>(pack.shells.size()), origin};

    // occ/virt split by occupation (f = 1 ⇔ occupied) — the same
    // convention as OrbitalHessian's composite index p = a·n_occ + i.
    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const std::size_t n_ao = C.shape()[0];
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());

    RealMatrix Co = xt::zeros<double>({n_ao, static_cast<std::size_t>(no)});
    RealMatrix Cv = xt::zeros<double>({n_ao, static_cast<std::size_t>(nv)});
    for (int i = 0; i < no; ++i)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Co(mu, static_cast<std::size_t>(i)) =
                C(mu, static_cast<std::size_t>(occ[static_cast<std::size_t>(i)]));
    for (int a = 0; a < nv; ++a)
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Cv(mu, static_cast<std::size_t>(a)) =
                C(mu, static_cast<std::size_t>(virt[static_cast<std::size_t>(a)]));

    for (int axis = 0; axis < 3; ++axis) {
        xt::xtensor<double, 2> D = mi.dipole(axis);
        if (pack.spherical()) {
            const auto& c2s = pack.c2s;
            D = xt::linalg::dot(
                xt::transpose(c2s),
                xt::xtensor<double, 2>(xt::linalg::dot(D, c2s)));
        }
        // ⟨φ_i|r_axis|φ_a⟩ = (C_occᵀ D C_virt)_(i,a).
        const RealMatrix d_mo = xt::linalg::dot(
            xt::transpose(Co), RealMatrix(xt::linalg::dot(D, Cv)));
        for (std::size_t s = 0; s < k; ++s) {
            double mu_s = 0.0;
            for (int a = 0; a < nv; ++a)
                for (int i = 0; i < no; ++i) {
                    const std::size_t p =
                        static_cast<std::size_t>(a * no + i);
                    mu_s += (roots.X(s, p) + roots.Y(s, p)) * d_mo(i, a);
                }
            out.transition_dipole_au(s, axis) = std::sqrt(2.0) * mu_s;
        }
    }
    for (std::size_t s = 0; s < k; ++s) {
        const double mu2 =
            out.transition_dipole_au(s, 0) * out.transition_dipole_au(s, 0)
            + out.transition_dipole_au(s, 1) * out.transition_dipole_au(s, 1)
            + out.transition_dipole_au(s, 2) * out.transition_dipole_au(s, 2);
        out.oscillator_strength[s] = (2.0 / 3.0) * roots.omega(s) * mu2;
    }
    return out;
}

} // namespace

ExcitedStatesResult RHF::tda(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::tda: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const IntegralPack& pack = integral_workspace_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::TDHFPack tdhf_pack{hessian, internal_channel(channel)};
    return excited_states_result(det, pack, tdhf_pack.tda(n_states), channel,
                                 /*tda_flag=*/true);
}

ExcitedStatesResult RHF::tdhf(int n_states, ExcitationChannel channel) const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::tdhf: no SCF result — call run(frame) first");
    const SlaterDeterminant& det = determinant();
    const IntegralPack& pack = integral_workspace_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();
    const internal::OrbitalHessian hessian{det, eri};
    const internal::Channel ch = internal_channel(channel);

    // Link-08 stability pre-check: an unstable (A + B) block makes the full
    // RPA ω imaginary in this channel — fail loudly before solving.
    const internal::StabilityAnalysis analysis{hessian};
    const double lowest = analysis.lowest(ch);
    if (lowest < -1e-8)
        throw std::runtime_error(
            std::string("RHF::tdhf: the ")
            + (channel == ExcitationChannel::triplet ? "triplet" : "singlet")
            + " (A + B) block of the reference is unstable (lowest "
            "eigenvalue = " + std::to_string(lowest)
            + " Ha) — full RPA would give imaginary excitation energies; "
            "use tda() or reconverge the reference along the instability "
            "mode (stability().rotated_C) first (no fallback)");

    const internal::TDHFPack tdhf_pack{hessian, ch};
    return excited_states_result(det, pack, tdhf_pack.rpa(n_states), channel,
                                 /*tda_flag=*/false);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
