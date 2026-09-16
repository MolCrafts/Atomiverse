/// @file cpu/scf/fci.cpp
/// @brief Implementation of the tiny determinant-basis full-CI solver (see
///        fci.h). The determinant bitstring enumeration and the Slater-Condon
///        Case A-D matrix elements (with Jordan-Wigner phases) now live in the
///        shared internal/ci_engine.h; this file assembles the dense CI matrix
///        via that engine and runs the dense eigh.
///
/// Combinatorics sanity check (by hand):
///   H2 / STO-3G: n_mo = 2, n_alpha = n_beta = 1
///                N_det = C(2,1) * C(2,1) = 2 * 2 = 4 determinants.
///   He / STO-3G: n_mo = 1, n_alpha = n_beta = 1
///                N_det = C(1,1) * C(1,1) = 1 * 1 = 1 determinant
///                => single determinant => FCI == RHF => e_corr == 0 exactly.

#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/casci.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/casci_pack.h"
#include "atomiverse/cpu/scf/internal/ci_engine.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"

#include <algorithm>
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

xt::xtensor<double, 2> fci_hamiltonian(const internal::IntegralView& ints,
                                       const SlaterDeterminant& det) {
    using Spin = SlaterDeterminant::Spin;
    // No-fallback guards -- throw, never return a sentinel (mirror
    // mp2_energy). Convergence of the reference is guarded by the caller.
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "fci_energy: ERI is not materialized; call internal::IntegralWorkspace{..., "
            "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "fci_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }

    // Row-major copy of the alpha MO coefficients (hoisted once).
    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_alpha = det.n_electrons(Spin::alpha);
    const int n_beta = det.n_electrons(Spin::beta);

    // TINY solvability cap: the dense CI matrix and its eigh grow factorially.
    // This is a validation/reference tool, not a production method.
    if (n_mo > 14) {
        throw std::invalid_argument(
            "fci_energy: n_mo=" + std::to_string(n_mo)
            + " exceeds the TINY cap (n_mo <= 14); this is a reference solver "
              "for "
              "tiny systems, not a production correlation method.");
    }
    const long long n_det = choose(n_mo, n_alpha) * choose(n_mo, n_beta);
    if (n_det > 2000) {
        throw std::invalid_argument(
            "fci_energy: N_det=" + std::to_string(n_det)
            + " exceeds the TINY cap (N_det <= 2000); this is a reference "
              "solver "
              "for tiny systems, not a production correlation method.");
    }

    // MO-basis one- and two-electron integrals (chemists' notation).
    const xt::xtensor<double, 2> h = mo_hcore(ints.Hcore, C, n_mo);
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);

    const std::vector<std::uint64_t> alphas = enumerate_strings(n_mo, n_alpha);
    const std::vector<std::uint64_t> betas = enumerate_strings(n_mo, n_beta);
    const std::size_t n_a = alphas.size();
    const std::size_t n_b = betas.size();
    const std::size_t nd = n_a * n_b;

    // Determinant index: I = ia * n_b + ib (alpha block major).
    xt::xtensor<double, 2> hci = xt::zeros<double>({nd, nd});
    for (std::size_t ia = 0; ia < n_a; ++ia) {
        for (std::size_t ib = 0; ib < n_b; ++ib) {
            const std::size_t idx_i = ia * n_b + ib;
            for (std::size_t ja = 0; ja < n_a; ++ja) {
                for (std::size_t jb = 0; jb < n_b; ++jb) {
                    const std::size_t idx_j = ja * n_b + jb;
                    if (idx_j < idx_i)
                        continue; // upper triangle only
                    const double e =
                        ci_element(alphas[ia], betas[ib], alphas[ja], betas[jb],
                                   h, g, ints.V_nuc);
                    hci(idx_i, idx_j) = e;
                    hci(idx_j, idx_i) = e; // symmetric fill
                }
            }
        }
    }
    return hci;
}

FCIResult fci_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                     double e_scf_ha) {
    using Spin = SlaterDeterminant::Spin;
    // No-fallback guards -- throw, never return a sentinel (mirror
    // fci_hamiltonian). Convergence of the reference is guarded by the caller.
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "fci_energy: ERI is not materialized; call internal::IntegralWorkspace{..., "
            "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "fci_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_alpha = det.n_electrons(Spin::alpha);
    const int n_beta = det.n_electrons(Spin::beta);

    if (n_mo > 14) {
        throw std::invalid_argument(
            "fci_energy: n_mo=" + std::to_string(n_mo)
            + " exceeds the TINY cap (n_mo <= 14); this is a reference solver "
              "for tiny systems, not a production correlation method.");
    }
    const long long n_det = choose(n_mo, n_alpha) * choose(n_mo, n_beta);
    if (n_det > 2000) {
        throw std::invalid_argument(
            "fci_energy: N_det=" + std::to_string(n_det)
            + " exceeds the TINY cap (N_det <= 2000); this is a reference "
              "solver for tiny systems, not a production correlation method.");
    }

    // Single-determinant boundary (e.g. He/STO-3G, no virtuals): full CI is the
    // RHF reference itself; correlation is exactly zero.
    if (n_det == 1) {
        return {e_scf_ha, 0.0};
    }

    // Full MO integrals (chemists' notation); the shared engine builds the dense
    // CI matrix from ci_element and diagonalizes it.
    const xt::xtensor<double, 2> h = mo_hcore(ints.Hcore, C, n_mo);
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);
    const CiSolution sol = dense_ci_solve(h, g, ints.V_nuc, n_alpha, n_beta);
    return {sol.energy, sol.energy - e_scf_ha};
}

CASCIResult casci_energy(const internal::IntegralView& ints, const SlaterDeterminant& det,
                         double e_scf_ha, int ncas, int nelecas) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("casci_energy: ERI is not materialized; "
                                    "call with compute_eri=true.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "casci_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (ncas <= 0) {
        throw std::invalid_argument("casci_energy: ncas must be positive (got "
                                    + std::to_string(ncas) + ")");
    }
    if (nelecas < 0 || (nelecas % 2) != 0) {
        throw std::invalid_argument(
            "casci_energy: closed-shell CAS requires even nelecas (got "
            + std::to_string(nelecas) + ")");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_alpha = det.n_electrons(Spin::alpha);
    const int n_act_a = nelecas / 2;
    const int n_act_b = n_act_a;
    const int ncore = n_alpha - n_act_a;

    if (n_act_a > ncas) {
        throw std::invalid_argument("casci_energy: nelecas/2="
                                    + std::to_string(n_act_a)
                                    + " exceeds ncas=" + std::to_string(ncas));
    }
    if (ncore < 0) {
        throw std::invalid_argument(
            "casci_energy: ncore=" + std::to_string(ncore)
            + " is negative (nelecas too large for the reference occupation)");
    }
    if (ncore + ncas > n_mo) {
        throw std::invalid_argument("casci_energy: ncore+ncas="
                                    + std::to_string(ncore + ncas)
                                    + " exceeds n_mo=" + std::to_string(n_mo));
    }

    // Full MO integrals; active block is the contiguous window [ncore, ncore+ncas).
    const xt::xtensor<double, 2> h_full = mo_hcore(ints.Hcore, C, n_mo);
    const xt::xtensor<double, 4> g_full = ao2mo_full(ints.ERI, C, n_mo);

    // Frozen-core energy (closed-shell): E_c = Σ_i 2 h_ii + Σ_ij [2 (ii|jj) − (ij|ji)].
    double e_core = ints.V_nuc;
    for (int i = 0; i < ncore; ++i) {
        e_core += 2.0 * h_full(i, i);
        for (int j = 0; j < ncore; ++j) {
            e_core += 2.0 * g_full(i, i, j, j) - g_full(i, j, j, i);
        }
    }

    // Effective one-electron active Hamiltonian:
    // h_eff_uv = h_uv + Σ_i^core [2 (uv|ii) − (ui|iv)].
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

    // When ncas==n_mo and ncore==0 this is identical to full CI.
    // Active CI shift is e_core (already includes V_nuc); pass 0 as V in the
    // diagonal and fold e_core into the constant instead via v_shift=e_core.
    const double e_casci =
        dense_ci_solve(h_eff, g_act, e_core, n_act_a, n_act_b).energy;
    return CASCIResult{e_casci, e_casci - e_scf_ha, ncas, nelecas, ncore};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
