/// @file cpu/scf/casscf.cpp
/// @brief Closed-shell CASSCF: dense active-space CI + Newton orbital steps.
///
/// Spin-summed RDM convention (matches the Slater–Condon energy used by FCI /
/// CASCI in `fci.cpp`):
///   E = V_shift + Σ_pq h_pq γ_pq + ½ Σ_pqrs (pq|rs) Γ_pqrs
/// with γ_pq = Σ_σ ⟨a†_{pσ} a_{qσ}⟩ and
/// Γ_pqrs = Σ_{στ} ⟨a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}⟩.
/// Generalized Fock: F_pq = Σ_r h_pr γ_rq + Σ_rst (pr|st) Γ_rqst;
/// orbital gradient g_pq = 2 (F_pq − F_qp) for non-redundant pairs.

#include "atomiverse/cpu/scf/casscf.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/casscf_pack.h"
#include "atomiverse/cpu/scf/internal/ci_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// @brief Embed active RDMs into full MO space with closed-shell frozen core.
void embed_rdms(int n_mo, int ncore, int ncas,
                const xt::xtensor<double, 2>& act1,
                const xt::xtensor<double, 4>& act2, xt::xtensor<double, 2>& g1,
                xt::xtensor<double, 4>& g2) {
    const std::size_t n = static_cast<std::size_t>(n_mo);
    g1 = xt::zeros<double>({n, n});
    g2 = xt::zeros<double>({n, n, n, n});

    // Core: γ_ii = 2.
    for (int i = 0; i < ncore; ++i)
        g1(i, i) = 2.0;
    // Active block of 1-RDM.
    for (int u = 0; u < ncas; ++u)
        for (int v = 0; v < ncas; ++v)
            g1(ncore + u, ncore + v) = act1(u, v);

    // Core–core 2-RDM (closed shell):
    // Γ_iijj = 4, Γ_ijji = −2 for all core i,j (including i=j: Γ_iiii = 2).
    for (int i = 0; i < ncore; ++i) {
        for (int j = 0; j < ncore; ++j) {
            g2(i, i, j, j) += 4.0;
            g2(i, j, j, i) -= 2.0;
        }
    }
    // Core–active: Γ_iiuv = 2 γ_uv, Γ_iuv i = −γ_uv  (and symmetric slots).
    for (int i = 0; i < ncore; ++i) {
        for (int u = 0; u < ncas; ++u) {
            for (int v = 0; v < ncas; ++v) {
                const int U = ncore + u, V = ncore + v;
                const double d = act1(u, v);
                g2(i, i, U, V) += 2.0 * d;
                g2(U, V, i, i) += 2.0 * d;
                g2(i, V, U, i) -= d;
                g2(U, i, i, V) -= d;
            }
        }
    }
    // Active–active 2-RDM.
    for (int u = 0; u < ncas; ++u)
        for (int v = 0; v < ncas; ++v)
            for (int x = 0; x < ncas; ++x)
                for (int y = 0; y < ncas; ++y)
                    g2(ncore + u, ncore + v, ncore + x, ncore + y) +=
                        act2(u, v, x, y);
}

xt::xtensor<double, 2> generalized_fock(const xt::xtensor<double, 2>& h,
                                        const xt::xtensor<double, 4>& g,
                                        const xt::xtensor<double, 2>& rdm1,
                                        const xt::xtensor<double, 4>& rdm2) {
    const int n = static_cast<int>(h.shape(0));
    xt::xtensor<double, 2> F = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            double f = 0.0;
            for (int r = 0; r < n; ++r)
                f += h(p, r) * rdm1(r, q);
            for (int r = 0; r < n; ++r)
                for (int s = 0; s < n; ++s)
                    for (int t = 0; t < n; ++t)
                        f += g(p, r, s, t) * rdm2(r, q, s, t);
            F(p, q) = f;
        }
    }
    return F;
}

/// @brief exp(K) for antisymmetric K via scaled Taylor (same as SOSCF).
xt::xtensor<double, 2> exp_antisymmetric(const xt::xtensor<double, 2>& K) {
    const std::size_t m = K.shape(0);
    double fro = 0.0;
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < m; ++c)
            fro += K(r, c) * K(r, c);
    fro = std::sqrt(fro);
    int s = 0;
    double scaled = fro;
    while (scaled > 0.25 && s < 60) {
        scaled *= 0.5;
        ++s;
    }
    xt::xtensor<double, 2> A = K / std::pow(2.0, s);
    xt::xtensor<double, 2> U = xt::eye<double>(m);
    xt::xtensor<double, 2> term = U;
    for (int k = 1; k <= 30; ++k) {
        term = xt::linalg::dot(term, A) / static_cast<double>(k);
        U = U + term;
        double t_max = 0.0;
        for (std::size_t r = 0; r < m; ++r)
            for (std::size_t c = 0; c < m; ++c)
                t_max = std::max(t_max, std::abs(term(r, c)));
        if (t_max < 1e-17)
            break;
    }
    for (int k = 0; k < s; ++k)
        U = xt::linalg::dot(U, U);
    return U;
}

/// @brief Approximate diagonal orbital Hessian for pair (p<q).
/// H_pq,pq ≈ 2 (F_pp − F_qq)(γ_qq − γ_pp)/something — use simple
/// 4*(ε_q − ε_p) style with F diagonal as orbital energies.
double approx_hess(const xt::xtensor<double, 2>& F,
                   const xt::xtensor<double, 2>& rdm1, int p, int q) {
    // Standard approx for CASSCF: H_{pq} ≈ 2 (γ_qq − γ_pp)(F_pp/γ − …);
    // practical robust form used by many codes for tiny systems:
    // H ≈ 4 * max(0.1, |F_qq − F_pp|) when rotation mixes occupied-like with
    // virtual-like. Fall back to 1.0 to avoid division by zero.
    const double dF = F(q, q) - F(p, p);
    const double dg = rdm1(p, p) - rdm1(q, q);
    double h = 2.0 * dF * dg;
    // For core→virtual, γ_pp=2, γ_qq=0 → h = 2*(Fqq−Fpp)*(2) = 4(Fqq−Fpp)
    // which is the HF-like Hessian piece. Clamp away from zero.
    if (std::abs(h) < 0.1)
        h = (h >= 0.0) ? 0.1 : -0.1;
    return h;
}

} // namespace

CASSCFResult casscf_energy(const internal::IntegralView& ints,
                           const SlaterDeterminant& det, double e_scf_ha,
                           int ncas, int nelecas, double orb_tol,
                           int max_macro) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("casscf_energy: ERI is not materialized; "
                                    "call with compute_eri=true.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "casscf_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (ncas <= 0) {
        throw std::invalid_argument("casscf_energy: ncas must be positive (got "
                                    + std::to_string(ncas) + ")");
    }
    if (nelecas < 0 || (nelecas % 2) != 0) {
        throw std::invalid_argument(
            "casscf_energy: closed-shell CAS requires even nelecas (got "
            + std::to_string(nelecas) + ")");
    }

    // Working MO coefficients (mutable copy of the RHF reference).
    xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_alpha = det.n_electrons(Spin::alpha);
    const int n_act_a = nelecas / 2;
    const int n_act_b = n_act_a;
    const int ncore = n_alpha - n_act_a;
    const int nvirt = n_mo - ncore - ncas;

    if (n_act_a > ncas) {
        throw std::invalid_argument("casscf_energy: nelecas/2="
                                    + std::to_string(n_act_a)
                                    + " exceeds ncas=" + std::to_string(ncas));
    }
    if (ncore < 0) {
        throw std::invalid_argument(
            "casscf_energy: ncore=" + std::to_string(ncore)
            + " is negative (nelecas too large for the reference occupation)");
    }
    if (ncore + ncas > n_mo) {
        throw std::invalid_argument("casscf_energy: ncore+ncas="
                                    + std::to_string(ncore + ncas)
                                    + " exceeds n_mo=" + std::to_string(n_mo));
    }

    // Non-redundant orbital pairs: core–active, active–virtual, core–virtual.
    struct Pair {
        int p, q; // rotate p < q in the sense κ_qp = −κ_pq; store (lo, hi)
    };
    std::vector<Pair> pairs;
    pairs.reserve(
        static_cast<std::size_t>(ncore * ncas + ncas * nvirt + ncore * nvirt));
    for (int i = 0; i < ncore; ++i)
        for (int u = 0; u < ncas; ++u)
            pairs.push_back({i, ncore + u});
    for (int u = 0; u < ncas; ++u)
        for (int a = 0; a < nvirt; ++a)
            pairs.push_back({ncore + u, ncore + ncas + a});
    for (int i = 0; i < ncore; ++i)
        for (int a = 0; a < nvirt; ++a)
            pairs.push_back({i, ncore + ncas + a});

    double e_cas = 0.0;
    int n_iter = 0;
    bool converged = false;

    for (int macro = 0; macro < max_macro; ++macro) {
        ++n_iter;
        const xt::xtensor<double, 2> h_full = mo_hcore(ints.Hcore, C, n_mo);
        const xt::xtensor<double, 4> g_full = ao2mo_full(ints.ERI, C, n_mo);

        // Frozen-core energy (closed-shell).
        double e_core = ints.V_nuc;
        for (int i = 0; i < ncore; ++i) {
            e_core += 2.0 * h_full(i, i);
            for (int j = 0; j < ncore; ++j)
                e_core += 2.0 * g_full(i, i, j, j) - g_full(i, j, j, i);
        }

        // Effective active Hamiltonian.
        const std::size_t na = static_cast<std::size_t>(ncas);
        xt::xtensor<double, 2> h_eff = xt::zeros<double>({na, na});
        xt::xtensor<double, 4> g_act = xt::zeros<double>({na, na, na, na});
        for (int u = 0; u < ncas; ++u) {
            const int U = ncore + u;
            for (int v = 0; v < ncas; ++v) {
                const int V = ncore + v;
                double he = h_full(U, V);
                for (int i = 0; i < ncore; ++i)
                    he += 2.0 * g_full(U, V, i, i) - g_full(U, i, i, V);
                h_eff(u, v) = he;
                for (int r = 0; r < ncas; ++r) {
                    const int R = ncore + r;
                    for (int s = 0; s < ncas; ++s) {
                        const int S = ncore + s;
                        g_act(u, v, r, s) = g_full(U, V, R, S);
                    }
                }
            }
        }

        const CiSolution ci =
            dense_ci_solve(h_eff, g_act, e_core, n_act_a, n_act_b);
        e_cas = ci.energy;

        // No non-redundant rotations ⇒ CASSCF ≡ CASCI.
        if (pairs.empty()) {
            converged = true;
            break;
        }

        xt::xtensor<double, 2> act1;
        xt::xtensor<double, 4> act2;
        make_rdm12(ci, ncas, act1, act2);

        // Energy consistency check (debug-level; do not throw on tiny noise).
        // Embed and build Fock for orbital gradient.
        xt::xtensor<double, 2> g1;
        xt::xtensor<double, 4> g2;
        embed_rdms(n_mo, ncore, ncas, act1, act2, g1, g2);
        const xt::xtensor<double, 2> F =
            generalized_fock(h_full, g_full, g1, g2);

        double gmax = 0.0;
        std::vector<double> grad(pairs.size());
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            const int p = pairs[k].p;
            const int q = pairs[k].q;
            // g_pq = 2 (F_pq − F_qp); step rotates with κ_qp.
            grad[k] = 2.0 * (F(p, q) - F(q, p));
            gmax = std::max(gmax, std::abs(grad[k]));
        }
        if (gmax < orb_tol) {
            converged = true;
            break;
        }

        // Diagonal Newton: κ_pq = − g_pq / H_pq.
        xt::xtensor<double, 2> K = xt::zeros<double>(
            {static_cast<std::size_t>(n_mo), static_cast<std::size_t>(n_mo)});
        double knorm = 0.0;
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            const int p = pairs[k].p;
            const int q = pairs[k].q;
            const double H = approx_hess(F, g1, p, q);
            double kap = -grad[k] / H;
            // Trust: |κ| ≤ 0.5.
            if (kap > 0.5)
                kap = 0.5;
            if (kap < -0.5)
                kap = -0.5;
            // Antisymmetric generator: K_qp = κ, K_pq = −κ with κ = κ_{p→q}.
            K(q, p) = kap;
            K(p, q) = -kap;
            knorm += kap * kap;
        }
        (void)knorm;
        C = xt::linalg::dot(C, exp_antisymmetric(K));
    }

    if (!converged && pairs.empty())
        converged = true;

    return CASSCFResult{e_cas, e_cas - e_scf_ha, ncas,     nelecas,
                        ncore, n_iter,           converged};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
