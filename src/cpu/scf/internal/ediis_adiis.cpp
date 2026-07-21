/// @file cpu/scf/internal/ediis_adiis.cpp
/// @brief EDIIS / ADIIS far-field Fock mixers: history bookkeeping, the
///        spin-summed trace metrics, and the shared deterministic
///        simplex-constrained quadratic minimizer (t-parameterization
///        gradient descent + exact KKT polish on the detected support).
///
/// EDIIS: Kudin, Scuseria, Cancès, J. Chem. Phys. 116, 8255 (2002),
///        DOI 10.1063/1.1470195.
/// ADIIS: Hu, Yang, J. Chem. Phys. 132, 054109 (2010), DOI 10.1063/1.3304922.

#include "atomiverse/cpu/scf/internal/ediis_adiis.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

/// @brief Tr(A·B) for symmetric real matrices = Σ A∘B (dimensionless·Hartree).
double trace_dot(const RealMatrix& A, const RealMatrix& B) {
    return xt::sum(A * B)();
}

/// @brief Minimize f(c) = gᵀc + ½ cᵀQc over the simplex {Σc = 1, c ≥ 0}.
///
/// Deterministic small-dimension solver (m ≤ the mixer history depth):
/// multi-start gradient descent in the t-parameterization c_i = t_i²/Σ_j t_j²
/// with Armijo backtracking and a fixed iteration budget, followed by an
/// exact KKT polish on the detected support (equality-constrained solve,
/// dropping negative components until feasible). All simplex vertices are
/// also evaluated as candidates. Non-finite input or a degenerate quadratic
/// falls back to c = e_latest (all weight on the newest iterate) — no NaN
/// ever escapes.
///
/// @param g Linear coefficients [Hartree], size m.
/// @param Q Symmetric quadratic form [Hartree], m × m.
/// @return Simplex coefficients, size m (Σ = 1, c ≥ 0).
std::vector<double> minimize_simplex(const std::vector<double>& g,
                                     const RealMatrix& Q) {
    const int m = static_cast<int>(g.size());
    std::vector<double> fallback(static_cast<std::size_t>(m), 0.0);
    fallback[static_cast<std::size_t>(m - 1)] = 1.0; // newest iterate
    if (m == 1)
        return fallback;
    for (int i = 0; i < m; ++i) {
        if (!std::isfinite(g[static_cast<std::size_t>(i)]))
            return fallback;
        for (int j = 0; j < m; ++j)
            if (!std::isfinite(Q(i, j)))
                return fallback;
    }

    auto value = [&](const std::vector<double>& c) {
        double f = 0.0;
        for (int i = 0; i < m; ++i) {
            f += g[static_cast<std::size_t>(i)]
                 * c[static_cast<std::size_t>(i)];
            double qc = 0.0;
            for (int j = 0; j < m; ++j)
                qc += Q(i, j) * c[static_cast<std::size_t>(j)];
            f += 0.5 * c[static_cast<std::size_t>(i)] * qc;
        }
        return f;
    };
    auto grad_c = [&](const std::vector<double>& c, std::vector<double>& gc) {
        for (int i = 0; i < m; ++i) {
            double v = g[static_cast<std::size_t>(i)];
            for (int j = 0; j < m; ++j)
                v += Q(i, j) * c[static_cast<std::size_t>(j)];
            gc[static_cast<std::size_t>(i)] = v;
        }
    };

    double best_f = value(fallback);
    std::vector<double> best_c = fallback;
    // `slack` admits an exact KKT-polished candidate that FP-ties the
    // gradient-descent iterate it refines (the polish is the higher-accuracy
    // solution of the same face).
    auto consider = [&](const std::vector<double>& c, double slack) {
        double s = 0.0;
        for (double ci : c) {
            if (!std::isfinite(ci) || ci < -1e-10)
                return;
            s += ci;
        }
        if (std::fabs(s - 1.0) > 1e-8)
            return;
        const double f = value(c);
        if (std::isfinite(f) && f < best_f + slack) {
            best_f = std::min(f, best_f);
            best_c = c;
        }
    };

    // All vertices are feasible candidates (covers concave/indefinite Q).
    for (int i = 0; i < m; ++i) {
        std::vector<double> c(static_cast<std::size_t>(m), 0.0);
        c[static_cast<std::size_t>(i)] = 1.0;
        consider(c, 0.0);
    }

    // Exact KKT polish on a support set: minimize on {c_S: Σc_S = 1},
    // dropping negative components until feasible. Accepted with a tiny
    // slack so the exact solve supersedes an FP-tied descent iterate.
    auto kkt_polish = [&](std::vector<int> support, double slack) {
        for (int round = 0; round <= m && !support.empty(); ++round) {
            const int k = static_cast<int>(support.size());
            xt::xtensor<double, 2> A =
                xt::zeros<double>({static_cast<std::size_t>(k + 1),
                                   static_cast<std::size_t>(k + 1)});
            xt::xtensor<double, 1> b =
                xt::zeros<double>({static_cast<std::size_t>(k + 1)});
            for (int a = 0; a < k; ++a) {
                for (int bb = 0; bb < k; ++bb)
                    A(a, bb) = Q(support[static_cast<std::size_t>(a)],
                                 support[static_cast<std::size_t>(bb)]);
                A(a, k) = 1.0;
                A(k, a) = 1.0;
                b(a) = -g[static_cast<std::size_t>(
                    support[static_cast<std::size_t>(a)])];
            }
            b(k) = 1.0;
            xt::xtensor<double, 1> x;
            try {
                x = xt::linalg::solve(A, b);
            } catch (...) {
                return; // singular face — vertices already cover it
            }
            int min_idx = -1;
            double min_val = -1e-12;
            for (int a = 0; a < k; ++a)
                if (x(a) < min_val) {
                    min_val = x(a);
                    min_idx = a;
                }
            if (min_idx < 0) {
                std::vector<double> c(static_cast<std::size_t>(m), 0.0);
                for (int a = 0; a < k; ++a)
                    c[static_cast<std::size_t>(
                        support[static_cast<std::size_t>(a)])] =
                        (x(a) > 0.0) ? x(a) : 0.0;
                double s = 0.0;
                for (double ci : c)
                    s += ci;
                if (s > 0.0)
                    for (double& ci : c)
                        ci /= s;
                consider(c, slack);
                return;
            }
            support.erase(support.begin() + min_idx);
        }
    };

    // Multi-start projected gradient descent in t-space (approximate), each
    // start followed by an exact KKT polish on the support it detects.
    std::vector<double> t(static_cast<std::size_t>(m));
    std::vector<double> c(static_cast<std::size_t>(m));
    std::vector<double> gc(static_cast<std::size_t>(m));
    std::vector<double> gt(static_cast<std::size_t>(m));
    std::vector<double> t_try(static_cast<std::size_t>(m));
    std::vector<double> c_try(static_cast<std::size_t>(m));
    auto to_c = [&](const std::vector<double>& tv, std::vector<double>& cv) {
        double s = 0.0;
        for (double ti : tv)
            s += ti * ti;
        if (s <= 0.0) {
            for (double& ci : cv)
                ci = 1.0 / m;
            return;
        }
        for (int i = 0; i < m; ++i)
            cv[static_cast<std::size_t>(i)] =
                tv[static_cast<std::size_t>(i)]
                * tv[static_cast<std::size_t>(i)] / s;
    };

    for (int start = 0; start <= m; ++start) {
        for (int i = 0; i < m; ++i)
            t[static_cast<std::size_t>(i)] = (start == 0) ? 1.0 : 0.3;
        if (start > 0)
            t[static_cast<std::size_t>(start - 1)] = 1.0;

        to_c(t, c);
        double f = value(c);
        double eta = 1.0;
        for (int it = 0; it < 300; ++it) {
            grad_c(c, gc);
            double s = 0.0, cg = 0.0;
            for (int i = 0; i < m; ++i) {
                s += t[static_cast<std::size_t>(i)]
                     * t[static_cast<std::size_t>(i)];
                cg += c[static_cast<std::size_t>(i)]
                      * gc[static_cast<std::size_t>(i)];
            }
            double gnorm = 0.0;
            for (int i = 0; i < m; ++i) {
                gt[static_cast<std::size_t>(i)] =
                    2.0 * t[static_cast<std::size_t>(i)] / s
                    * (gc[static_cast<std::size_t>(i)] - cg);
                gnorm = std::max(
                    gnorm, std::fabs(gt[static_cast<std::size_t>(i)]));
            }
            if (gnorm < 1e-14)
                break;
            bool accepted = false;
            for (int h = 0; h < 40; ++h) {
                for (int i = 0; i < m; ++i)
                    t_try[static_cast<std::size_t>(i)] =
                        t[static_cast<std::size_t>(i)]
                        - eta * gt[static_cast<std::size_t>(i)];
                to_c(t_try, c_try);
                const double f_try = value(c_try);
                if (std::isfinite(f_try) && f_try < f) {
                    t = t_try;
                    c = c_try;
                    f = f_try;
                    eta *= 1.6;
                    accepted = true;
                    break;
                }
                eta *= 0.5;
            }
            if (!accepted)
                break;
        }
        consider(c, 0.0);
        // Exact polish of the support this start found (tie-preferred).
        std::vector<int> support;
        for (int i = 0; i < m; ++i)
            if (c[static_cast<std::size_t>(i)] > 1e-8)
                support.push_back(i);
        if (!support.empty())
            kkt_polish(support, 1e-12 * (1.0 + std::fabs(best_f)));
    }

    // Full-support polish (the common near-interior case; tie-preferred).
    {
        std::vector<int> all(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i)
            all[static_cast<std::size_t>(i)] = i;
        kkt_polish(all, 1e-12 * (1.0 + std::fabs(best_f)));
    }

    return best_c;
}

/// @brief F̃ = Σ_i c_i F_i over a history deque.
RealMatrix combine(const std::deque<RealMatrix>& hist,
                   const std::vector<double>& c) {
    RealMatrix out = c[0] * hist[0];
    for (std::size_t i = 1; i < hist.size(); ++i)
        out += c[i] * hist[i];
    return out;
}

} // namespace

// ── EDIISMixer ────────────────────────────────────────────────────────────

void EDIISMixer::reset() {
    Fa_.clear();
    Fb_.clear();
    Da_.clear();
    Db_.clear();
    E_.clear();
    c_.clear();
}

void EDIISMixer::push_(const RealMatrix& Fa, const RealMatrix& Da,
                       const RealMatrix& Fb, const RealMatrix& Db,
                       double energy) {
    Fa_.push_back(Fa);
    Da_.push_back(Da);
    Fb_.push_back(Fb);
    Db_.push_back(Db);
    E_.push_back(energy);
    while (static_cast<int>(Fa_.size()) > max_vectors_) {
        Fa_.pop_front();
        Da_.pop_front();
        Fb_.pop_front();
        Db_.pop_front();
        E_.pop_front();
    }
}

double EDIISMixer::objective(const std::vector<double>& c) const {
    const int m = size();
    double f = 0.0;
    for (int i = 0; i < m; ++i)
        f += c[static_cast<std::size_t>(i)] * E_[static_cast<std::size_t>(i)];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            const RealMatrix dD =
                Da_[static_cast<std::size_t>(i)]
                - Da_[static_cast<std::size_t>(j)];
            const RealMatrix dF =
                Fa_[static_cast<std::size_t>(i)]
                - Fa_[static_cast<std::size_t>(j)];
            double b = trace_dot(dD, dF);
            const RealMatrix dDb = Db_[static_cast<std::size_t>(i)]
                                   - Db_[static_cast<std::size_t>(j)];
            const RealMatrix dFb = Fb_[static_cast<std::size_t>(i)]
                                   - Fb_[static_cast<std::size_t>(j)];
            b += trace_dot(dDb, dFb);
            f -= 0.5 * c[static_cast<std::size_t>(i)]
                 * c[static_cast<std::size_t>(j)] * b;
        }
    return f;
}

void EDIISMixer::solve_() {
    const int m = size();
    // Spin-summed pair traces C_ij = ⟨D_i|F_j⟩; B_ij = C_ii + C_jj − C_ij −
    // C_ji; f(c) = Eᵀc − ½ cᵀBc  ⇒  g = E, Q = −B.
    RealMatrix C = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            C(i, j) = trace_dot(Da_[static_cast<std::size_t>(i)],
                                Fa_[static_cast<std::size_t>(j)])
                      + trace_dot(Db_[static_cast<std::size_t>(i)],
                                  Fb_[static_cast<std::size_t>(j)]);
    RealMatrix Q = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    std::vector<double> g(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        g[static_cast<std::size_t>(i)] = E_[static_cast<std::size_t>(i)];
        for (int j = 0; j < m; ++j)
            Q(i, j) = -(C(i, i) + C(j, j) - C(i, j) - C(j, i));
    }
    c_ = minimize_simplex(g, Q);
}

std::pair<RealMatrix, RealMatrix>
EDIISMixer::step(const RealMatrix& F_alpha, const RealMatrix& D_alpha,
                 const RealMatrix& F_beta, const RealMatrix& D_beta,
                 double energy) {
    push_(F_alpha, D_alpha, F_beta, D_beta, energy);
    if (size() < 2) {
        c_ = {1.0};
        return {F_alpha, F_beta};
    }
    solve_();
    return {combine(Fa_, c_), combine(Fb_, c_)};
}

RealMatrix EDIISMixer::step(const RealMatrix& F, const RealMatrix& D,
                            double energy) {
    // Zero β channel: the spin-summed metric reduces to the single-channel
    // ⟨D_i−D_j|F_i−F_j⟩ over the total (factor-2) density, as in the paper.
    const RealMatrix Z = xt::zeros<double>(F.shape());
    return step(F, D, Z, Z, energy).first;
}

// ── ADIISMixer ────────────────────────────────────────────────────────────

void ADIISMixer::reset() {
    Fa_.clear();
    Fb_.clear();
    Da_.clear();
    Db_.clear();
    c_.clear();
}

void ADIISMixer::push_(const RealMatrix& Fa, const RealMatrix& Da,
                       const RealMatrix& Fb, const RealMatrix& Db) {
    Fa_.push_back(Fa);
    Da_.push_back(Da);
    Fb_.push_back(Fb);
    Db_.push_back(Db);
    while (static_cast<int>(Fa_.size()) > max_vectors_) {
        Fa_.pop_front();
        Da_.pop_front();
        Fb_.pop_front();
        Db_.pop_front();
    }
}

double ADIISMixer::objective(const std::vector<double>& c) const {
    const int m = size();
    const std::size_t n = static_cast<std::size_t>(m - 1); // latest
    double f = 0.0;
    for (int i = 0; i < m; ++i) {
        const RealMatrix dDa = Da_[static_cast<std::size_t>(i)] - Da_[n];
        const RealMatrix dDb = Db_[static_cast<std::size_t>(i)] - Db_[n];
        f += 2.0 * c[static_cast<std::size_t>(i)]
             * (trace_dot(dDa, Fa_[n]) + trace_dot(dDb, Fb_[n]));
        for (int j = 0; j < m; ++j) {
            const RealMatrix dFa = Fa_[static_cast<std::size_t>(j)] - Fa_[n];
            const RealMatrix dFb = Fb_[static_cast<std::size_t>(j)] - Fb_[n];
            f += c[static_cast<std::size_t>(i)]
                 * c[static_cast<std::size_t>(j)]
                 * (trace_dot(dDa, dFa) + trace_dot(dDb, dFb));
        }
    }
    return f;
}

void ADIISMixer::solve_() {
    const int m = size();
    const std::size_t n = static_cast<std::size_t>(m - 1); // latest
    // f(c) = 2 gᵀc + cᵀHc with g_i = ⟨D_i−D_n|F_n⟩ and
    // H_ij = ⟨D_i−D_n|F_j−F_n⟩ ⇒ standard form lin = 2g, Q = H + Hᵀ.
    std::vector<double> lin(static_cast<std::size_t>(m));
    RealMatrix H = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    for (int i = 0; i < m; ++i) {
        const RealMatrix dDa = Da_[static_cast<std::size_t>(i)] - Da_[n];
        const RealMatrix dDb = Db_[static_cast<std::size_t>(i)] - Db_[n];
        lin[static_cast<std::size_t>(i)] =
            2.0 * (trace_dot(dDa, Fa_[n]) + trace_dot(dDb, Fb_[n]));
        for (int j = 0; j < m; ++j) {
            const RealMatrix dFa = Fa_[static_cast<std::size_t>(j)] - Fa_[n];
            const RealMatrix dFb = Fb_[static_cast<std::size_t>(j)] - Fb_[n];
            H(i, j) = trace_dot(dDa, dFa) + trace_dot(dDb, dFb);
        }
    }
    RealMatrix Q = H + xt::transpose(H);
    c_ = minimize_simplex(lin, Q);
}

std::pair<RealMatrix, RealMatrix>
ADIISMixer::step(const RealMatrix& F_alpha, const RealMatrix& D_alpha,
                 const RealMatrix& F_beta, const RealMatrix& D_beta) {
    push_(F_alpha, D_alpha, F_beta, D_beta);
    if (size() < 2) {
        c_ = {1.0};
        return {F_alpha, F_beta};
    }
    solve_();
    return {combine(Fa_, c_), combine(Fb_, c_)};
}

RealMatrix ADIISMixer::step(const RealMatrix& F, const RealMatrix& D) {
    // Zero β channel: the spin-summed metric reduces to the single-channel
    // trace metric over the total (factor-2) density, as in the paper.
    const RealMatrix Z = xt::zeros<double>(F.shape());
    return step(F, D, Z, Z).first;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
