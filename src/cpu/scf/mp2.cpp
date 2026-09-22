/// @file cpu/scf/mp2.cpp
/// @brief Implementation of ao2mo_ovov + closed-shell / UMP2 / RI MP2.

#include "atomiverse/cpu/scf/mp2.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/density_fitting.h"
#include "atomiverse/cpu/scf/internal/mp2_pack.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// @brief Four-index AO→MO transform to chemist (p q | r s) with independent
///        coefficient blocks per index (OrbitalHessian::transform_block_ pattern).
xt::xtensor<double, 4> ao2mo_mixed_(const xt::xtensor<double, 4>& eri_ao,
                                    const xt::xtensor<double, 2>& C0,
                                    const xt::xtensor<double, 2>& C1,
                                    const xt::xtensor<double, 2>& C2,
                                    const xt::xtensor<double, 2>& C3) {
    const xt::xtensor<double, 2>* Cs[4] = {&C3, &C2, &C1, &C0};
    // Round 0 reads eri_ao directly and writes a fresh work tensor; later
    // rounds iterate on `cur`. Dropping the old `cur = eri_ao` entry copy
    // saves one full N⁴ materialization per call (contraction order unchanged).
    xt::xtensor<double, 4> cur;
    for (int round = 0; round < 4; ++round) {
        const xt::xtensor<double, 4>& src = (round == 0) ? eri_ao : cur;
        const auto sh = src.shape();
        auto flat = xt::reshape_view(src, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
        xt::xtensor<double, 4> next =
            xt::reshape_view(h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
        cur = xt::transpose(next, {3, 0, 1, 2});
    }
    return cur; // (C0, C1, C2, C3) → (i, a, j, b) when C0=CoL, C1=CvL, …
}

/// @brief Same-spin UMP2 channel: ¼ Σ | (ia|jb) − (ib|ja) |² / Δ.
double same_spin_corr_(const xt::xtensor<double, 4>& ovov,
                       const xt::xtensor<double, 1>& eps, int n_occ,
                       int n_vir) {
    if (n_occ < 2 || n_vir < 2)
        return 0.0;
    double e = 0.0;
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            for (int a = 0; a < n_vir; ++a) {
                for (int b = 0; b < n_vir; ++b) {
                    const double iajb = ovov(i, a, j, b);
                    const double ibja = ovov(i, b, j, a);
                    const double t = iajb - ibja;
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(n_occ + a), eps(n_occ + b));
                    e += t * t / denom;
                }
            }
        }
    }
    return 0.25 * e;
}

} // namespace

xt::xtensor<double, 4> ao2mo_ovov(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_occ,
                                  int n_mo) {
    const std::size_t n = ERI.shape(0);
    const std::size_t no = static_cast<std::size_t>(n_occ);
    const std::size_t nv = static_cast<std::size_t>(n_mo - n_occ);

    // Occupied / virtual MO coefficient blocks, materialized [n_ao x no|nv].
    xt::xtensor<double, 2> Co = xt::view(C, xt::all(), xt::range(0, n_occ));
    xt::xtensor<double, 2> Cv = xt::view(C, xt::all(), xt::range(n_occ, n_mo));

    // Quarter 1: mu -> i (occupied).  (mu | nu lambda sigma) -> (i | nu lambda sigma)
    xt::xtensor<double, 2> eri2 = xt::reshape_view(ERI, {n, n * n * n});
    xt::xtensor<double, 2> q1 = xt::linalg::dot(xt::transpose(Co), eri2);
    xt::xtensor<double, 4> q1r =
        xt::reshape_view(q1, {no, n, n, n}); // (i,nu,lam,sig)

    // Quarter 2: nu -> a (virtual).  bring nu to the front, contract, restore.
    xt::xtensor<double, 4> q1t =
        xt::transpose(q1r, {1, 0, 2, 3}); // (nu,i,lam,sig)
    xt::xtensor<double, 2> q1tm = xt::reshape_view(q1t, {n, no * n * n});
    xt::xtensor<double, 2> q2 = xt::linalg::dot(xt::transpose(Cv), q1tm);
    xt::xtensor<double, 4> q2r =
        xt::reshape_view(q2, {nv, no, n, n}); // (a,i,lam,sig)
    xt::xtensor<double, 4> q2t =
        xt::transpose(q2r, {1, 0, 2, 3}); // (i,a,lam,sig)

    // Quarter 3: lambda -> j (occupied).  bring lambda to the front.
    xt::xtensor<double, 4> q2l =
        xt::transpose(q2t, {2, 0, 1, 3}); // (lam,i,a,sig)
    xt::xtensor<double, 2> q2lm = xt::reshape_view(q2l, {n, no * nv * n});
    xt::xtensor<double, 2> q3 = xt::linalg::dot(xt::transpose(Co), q2lm);
    xt::xtensor<double, 4> q3r =
        xt::reshape_view(q3, {no, no, nv, n}); // (j,i,a,sig)
    xt::xtensor<double, 4> q3t =
        xt::transpose(q3r, {1, 2, 0, 3}); // (i,a,j,sig)

    // Quarter 4: sigma -> b (virtual).  sigma is the trailing index already.
    xt::xtensor<double, 2> q3m = xt::reshape_view(q3t, {no * nv * no, n});
    xt::xtensor<double, 2> q4 = xt::linalg::dot(q3m, Cv);
    xt::xtensor<double, 4> ovov =
        xt::reshape_view(q4, {no, nv, no, nv}); // (i,a,j,b)
    return ovov;
}

xt::xtensor<double, 4> ao2mo_full(const xt::xtensor<double, 4>& ERI,
                                  const xt::xtensor<double, 2>& C, int n_mo) {
    const std::size_t n = ERI.shape(0);
    const std::size_t m = static_cast<std::size_t>(n_mo);

    // MO coefficient block, materialized [n_ao x m].
    xt::xtensor<double, 2> Cm = xt::view(C, xt::all(), xt::range(0, n_mo));

    // Quarter 1: mu -> p.  (mu | nu lam sig) -> (p | nu lam sig)
    xt::xtensor<double, 2> eri2 = xt::reshape_view(ERI, {n, n * n * n});
    xt::xtensor<double, 2> q1 = xt::linalg::dot(xt::transpose(Cm), eri2);
    xt::xtensor<double, 4> q1r =
        xt::reshape_view(q1, {m, n, n, n}); // (p,nu,lam,sig)

    // Quarter 2: nu -> q.  bring nu to the front, contract, restore.
    xt::xtensor<double, 4> q1t =
        xt::transpose(q1r, {1, 0, 2, 3}); // (nu,p,lam,sig)
    xt::xtensor<double, 2> q1tm = xt::reshape_view(q1t, {n, m * n * n});
    xt::xtensor<double, 2> q2 = xt::linalg::dot(xt::transpose(Cm), q1tm);
    xt::xtensor<double, 4> q2r =
        xt::reshape_view(q2, {m, m, n, n}); // (q,p,lam,sig)
    xt::xtensor<double, 4> q2t =
        xt::transpose(q2r, {1, 0, 2, 3}); // (p,q,lam,sig)

    // Quarter 3: lambda -> r.  bring lambda to the front.
    xt::xtensor<double, 4> q2l =
        xt::transpose(q2t, {2, 0, 1, 3}); // (lam,p,q,sig)
    xt::xtensor<double, 2> q2lm = xt::reshape_view(q2l, {n, m * m * n});
    xt::xtensor<double, 2> q3 = xt::linalg::dot(xt::transpose(Cm), q2lm);
    xt::xtensor<double, 4> q3r =
        xt::reshape_view(q3, {m, m, m, n}); // (r,p,q,sig)
    xt::xtensor<double, 4> q3t =
        xt::transpose(q3r, {1, 2, 0, 3}); // (p,q,r,sig)

    // Quarter 4: sigma -> s.  sigma is the trailing index already.
    xt::xtensor<double, 2> q3m = xt::reshape_view(q3t, {m * m * m, n});
    xt::xtensor<double, 2> q4 = xt::linalg::dot(q3m, Cm);
    xt::xtensor<double, 4> full =
        xt::reshape_view(q4, {m, m, m, m}); // (p,q,r,s)
    return full;
}

xt::xtensor<double, 2> mo_hcore(const xt::xtensor<double, 2>& Hcore,
                                const xt::xtensor<double, 2>& C, int n_mo) {
    // MO coefficient block, materialized [n_ao x n_mo].
    xt::xtensor<double, 2> Cm = xt::view(C, xt::all(), xt::range(0, n_mo));
    xt::xtensor<double, 2> tmp = xt::linalg::dot(Hcore, Cm); // [n_ao x n_mo]
    return xt::linalg::dot(xt::transpose(Cm), tmp);          // [n_mo x n_mo]
}

MP2Result mp2_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen, double os_scale, double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    // No-fallback guards -- throw, never return a sentinel. (Convergence of
    // the reference is guarded by the caller, not here.)
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("mp2_energy: ERI is not materialized; call "
                                    "internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "mp2_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }

    // Row-major copy of the alpha MO coefficients (hoisted once).
    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir = n_mo - n_occ;

    if (n_frozen < 0 || n_frozen > n_occ) {
        throw std::invalid_argument(
            "mp2_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    }

    // No virtuals (e.g. He/STO-3G) or all occupied frozen: correlation is
    // exactly zero. Return a finite 0.0 rather than empty-denominator loops.
    if (n_vir == 0 || n_frozen == n_occ) {
        return {0.0, e_scf_ha, /*ri=*/false, n_frozen, 0.0, 0.0};
    }

    const xt::xtensor<double, 4> ovov = ao2mo_ovov(ints.ERI, C, n_occ, n_mo);
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    // Frozen-core: skip the lowest n_frozen occupied spatial MOs in the
    // doubles sum (PySCF mp.MP2(mf, frozen=n_frozen) convention). Virtuals
    // remain fully correlated.
    //
    // Spin-component split (matches PySCF mp2.kernel):
    //   E_OS = Σ (ia|jb)² / Δ
    //   E_SS = Σ (ia|jb)[(ia|jb) − (ib|ja)] / Δ
    //   E_MP2 = E_OS + E_SS = Σ (ia|jb)[2(ia|jb) − (ib|ja)] / Δ
    // SCS-MP2: c_OS * E_OS + c_SS * E_SS (Grimme JCP 118, 9095 (2003)).
    double e_os = 0.0;
    double e_ss = 0.0;
    for (int i = n_frozen; i < n_occ; ++i) {
        for (int j = n_frozen; j < n_occ; ++j) {
            for (int a = 0; a < n_vir; ++a) {
                for (int b = 0; b < n_vir; ++b) {
                    const double iajb = ovov(i, a, j, b);
                    const double ibja = ovov(i, b, j, a);
                    // Closed-shell aufbau reference: every virtual lies above
                    // every occupied orbital, so the denominator is strictly
                    // negative (mp_denominator throws otherwise).
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(n_occ + a), eps(n_occ + b));
                    const double coul = iajb * iajb / denom;
                    e_os += coul;
                    e_ss += coul - iajb * ibja / denom;
                }
            }
        }
    }

    const double e_corr = os_scale * e_os + ss_scale * e_ss;
    return {e_corr, e_scf_ha + e_corr, /*ri=*/false, n_frozen, e_os, e_ss};
}

MP2Result ump2_energy(const internal::IntegralView& ints,
                      const SlaterDeterminant& det, double e_scf_ha,
                      double os_scale, double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("ump2_energy: ERI is not materialized; "
                                    "call internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "ump2_energy: GHF (spinor) reference is not supported.");
    }
    if (det.restricted()) {
        throw std::invalid_argument(
            "ump2_energy: restricted reference — use mp2_energy (RMP2).");
    }

    const xt::xtensor<double, 2>& Ca = det.coefficients(Spin::alpha);
    const xt::xtensor<double, 2>& Cb = det.coefficients(Spin::beta);
    const xt::xtensor<double, 1>& eps_a = det.orbital_energies(Spin::alpha);
    const xt::xtensor<double, 1>& eps_b = det.orbital_energies(Spin::beta);
    const int n_occ_a = det.n_electrons(Spin::alpha);
    const int n_occ_b = det.n_electrons(Spin::beta);
    const int n_mo_a = static_cast<int>(Ca.shape(1));
    const int n_mo_b = static_cast<int>(Cb.shape(1));
    const int n_vir_a = n_mo_a - n_occ_a;
    const int n_vir_b = n_mo_b - n_occ_b;

    if (n_vir_a < 0 || n_vir_b < 0) {
        throw std::invalid_argument(
            "ump2_energy: n_electrons exceeds MO count on a spin channel.");
    }
    // No doubles possible when either both same-spin channels lack space or
    // the opposite-spin product of virtuals/occs vanishes — still return finite 0.
    if ((n_occ_a < 2 || n_vir_a < 2) && (n_occ_b < 2 || n_vir_b < 2)
        && (n_occ_a == 0 || n_occ_b == 0 || n_vir_a == 0 || n_vir_b == 0)) {
        return {0.0, e_scf_ha, /*ri=*/false, /*frozen=*/0, 0.0, 0.0};
    }

    double e_ss = 0.0;
    double e_os = 0.0;

    // Same-spin αα / ββ.
    if (n_occ_a >= 2 && n_vir_a >= 2) {
        xt::xtensor<double, 2> Co =
            xt::view(Ca, xt::all(), xt::range(0, n_occ_a));
        xt::xtensor<double, 2> Cv =
            xt::view(Ca, xt::all(), xt::range(n_occ_a, n_mo_a));
        const xt::xtensor<double, 4> ovov =
            ao2mo_mixed_(ints.ERI, Co, Cv, Co, Cv);
        e_ss += same_spin_corr_(ovov, eps_a, n_occ_a, n_vir_a);
    }
    if (n_occ_b >= 2 && n_vir_b >= 2) {
        xt::xtensor<double, 2> Co =
            xt::view(Cb, xt::all(), xt::range(0, n_occ_b));
        xt::xtensor<double, 2> Cv =
            xt::view(Cb, xt::all(), xt::range(n_occ_b, n_mo_b));
        const xt::xtensor<double, 4> ovov =
            ao2mo_mixed_(ints.ERI, Co, Cv, Co, Cv);
        e_ss += same_spin_corr_(ovov, eps_b, n_occ_b, n_vir_b);
    }

    // Opposite-spin αβ: Σ (i_a a_a | j_b b_b)² / Δ  (no exchange).
    if (n_occ_a > 0 && n_occ_b > 0 && n_vir_a > 0 && n_vir_b > 0) {
        xt::xtensor<double, 2> Coa =
            xt::view(Ca, xt::all(), xt::range(0, n_occ_a));
        xt::xtensor<double, 2> Cva =
            xt::view(Ca, xt::all(), xt::range(n_occ_a, n_mo_a));
        xt::xtensor<double, 2> Cob =
            xt::view(Cb, xt::all(), xt::range(0, n_occ_b));
        xt::xtensor<double, 2> Cvb =
            xt::view(Cb, xt::all(), xt::range(n_occ_b, n_mo_b));
        const xt::xtensor<double, 4> ovov_ab =
            ao2mo_mixed_(ints.ERI, Coa, Cva, Cob, Cvb);
        for (int i = 0; i < n_occ_a; ++i) {
            for (int j = 0; j < n_occ_b; ++j) {
                for (int a = 0; a < n_vir_a; ++a) {
                    for (int b = 0; b < n_vir_b; ++b) {
                        const double v = ovov_ab(i, a, j, b);
                        const double denom = eps_a(i) + eps_b(j)
                                             - eps_a(n_occ_a + a)
                                             - eps_b(n_occ_b + b);
                        if (denom >= 0.0) {
                            throw std::invalid_argument(
                                "ump2_energy: non-negative opposite-spin "
                                "orbital-energy denominator; reference is "
                                "degenerate or non-aufbau.");
                        }
                        e_os += v * v / denom;
                    }
                }
            }
        }
    }

    const double e_corr = os_scale * e_os + ss_scale * e_ss;
    return {e_corr, e_scf_ha + e_corr, /*ri=*/false, /*frozen=*/0, e_os, e_ss};
}

MP2Result romp2_energy(const internal::IntegralView& ints,
                       const SlaterDeterminant& det, const RealMatrix& F_alpha,
                       const RealMatrix& F_beta, double e_scf_ha,
                       double os_scale, double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("romp2_energy: ERI is not materialized; "
                                    "call internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "romp2_energy: GHF (spinor) reference is not supported.");
    }

    // Shared spatial MOs (ROHF). Either restricted or unrestricted-form det
    // is accepted; occupations come from the per-spin electron counts.
    const RealMatrix& Cref = det.coefficients(Spin::alpha);
    const int n_ao = static_cast<int>(Cref.shape(0));
    const int n_mo = static_cast<int>(Cref.shape(1));
    const int n_occ_a = det.n_electrons(Spin::alpha);
    const int n_occ_b = det.n_electrons(Spin::beta);

    if (static_cast<int>(F_alpha.shape(0)) != n_ao
        || static_cast<int>(F_alpha.shape(1)) != n_ao
        || static_cast<int>(F_beta.shape(0)) != n_ao
        || static_cast<int>(F_beta.shape(1)) != n_ao) {
        throw std::invalid_argument(
            "romp2_energy: spin Fock matrices must be n_ao × n_ao matching the "
            "MO coefficient row dimension.");
    }
    if (n_occ_a < 0 || n_occ_b < 0 || n_occ_a > n_mo || n_occ_b > n_mo) {
        throw std::invalid_argument(
            "romp2_energy: electron counts out of range for the MO space.");
    }

    // Semi-canonical denominators (PySCF ROHF.canonicalize / to_uhf):
    //   ε_i^σ = (C† F_σ C)_ii
    // Roothaan F_eff eigenvalues on the det are intentionally NOT used.
    const RealMatrix Fmo_a =
        xt::linalg::dot(xt::transpose(Cref), xt::linalg::dot(F_alpha, Cref));
    const RealMatrix Fmo_b =
        xt::linalg::dot(xt::transpose(Cref), xt::linalg::dot(F_beta, Cref));
    RealVector eps_a = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    RealVector eps_b = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    for (int i = 0; i < n_mo; ++i) {
        eps_a(i) = Fmo_a(i, i);
        eps_b(i) = Fmo_b(i, i);
    }

    RealVector occ_a = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    RealVector occ_b = xt::zeros<double>({static_cast<std::size_t>(n_mo)});
    for (int i = 0; i < n_occ_a; ++i)
        occ_a(i) = 1.0;
    for (int i = 0; i < n_occ_b; ++i)
        occ_b(i) = 1.0;

    // Independent α/β channels with identical spatial C (UHF-shaped view of
    // the ROHF solution) for the shared ump2_energy kernel.
    SlaterDeterminant udet(Cref, std::move(eps_a), std::move(occ_a), n_occ_a,
                           Cref, std::move(eps_b), std::move(occ_b), n_occ_b);
    return ump2_energy(ints, udet, e_scf_ha, os_scale, ss_scale);
}

MP2Result ri_mp2_energy(const internal::IntegralView& ints,
                        const SlaterDeterminant& det, double e_scf_ha,
                        AuxBasis aux, int n_frozen, double os_scale,
                        double ss_scale) {
    using Spin = SlaterDeterminant::Spin;
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta))
        throw std::invalid_argument(
            "ri_mp2_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    if (ints.shells.empty())
        throw std::invalid_argument(
            "ri_mp2_energy: internal::IntegralWorkspace has no shells for "
            "density fitting.");

    const RealMatrix& C_rm = det.coefficients(Spin::alpha);
    // Working AO coefficients may be spherical; DensityFitting lives in
    // Cartesian engine AO. Bridge with pack.c2s when present.
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C_rm.shape()[1]);
    const int n_vir = n_mo - n_occ;
    if (n_frozen < 0 || n_frozen > n_occ)
        throw std::invalid_argument(
            "ri_mp2_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    if (n_vir == 0 || n_frozen == n_occ)
        return {0.0, e_scf_ha, /*ri=*/true, n_frozen, 0.0, 0.0};

    internal::DensityFitting df(ints.shells, ints.atoms, aux);
    const int n_cart = df.n_ao();
    const int n_aux = df.n_aux();
    const RealMatrix& B = df.b_tensor(); // (n_cart² × n_aux), column-major

    RealMatrix C_cart;
    if (ints.spherical()) {
        C_cart = xt::linalg::dot(ints.c2s, C_rm); // [n_cart × n_mo]
    } else {
        if (static_cast<int>(C_rm.shape()[0]) != n_cart)
            throw std::invalid_argument(
                "ri_mp2_energy: AO dimension mismatch with DensityFitting");
        C_cart = C_rm;
    }

    // B_ov(ia, P) = Σ_μν C_μi C_νa B_μν^P, assembled BLAS-shaped from the
    // frozen df.b_tensor() output. B is column-major (n_cart² × n_aux) with the
    // (μ·n_cart + ν) pair index on the rows, so each auxiliary column is one
    // contiguous n_cart² slice: the M_P matrix in row-major (μ,ν). Adapt that
    // slice as a view (no per-P element-wise M reconstruction) and half-
    // transform it with Co^T M_P Cv. The fitting metric (Coulomb V=(P|Q),
    // Cholesky-whitened W=L⁻¹ ⇒ V⁻¹) is frozen — B is consumed unchanged.
    const int nov = n_occ * n_vir;
    const RealMatrix Co =
        xt::view(C_cart, xt::all(), xt::range(0, n_occ)); // [n_cart × n_occ]
    const RealMatrix Cv =
        xt::view(C_cart, xt::all(), xt::range(n_occ, n_mo)); // [n_cart × n_vir]
    const std::size_t n_cart2 =
        static_cast<std::size_t>(n_cart) * static_cast<std::size_t>(n_cart);
    xt::xtensor<double, 2> B_ov = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(n_aux)});
    for (int P = 0; P < n_aux; ++P) {
        // Row-major (μ,ν) view of column P's contiguous n_cart² slice.
        auto M_P = xt::adapt(
            B.data() + static_cast<std::size_t>(P) * n_cart2, n_cart2,
            xt::no_ownership(),
            std::array<std::size_t, 2>{static_cast<std::size_t>(n_cart),
                                       static_cast<std::size_t>(n_cart)});
        const RealMatrix ia = xt::linalg::dot(
            xt::transpose(Co), xt::linalg::dot(M_P, Cv)); // [n_occ × n_vir]
        for (int i = 0; i < n_occ; ++i)
            for (int a = 0; a < n_vir; ++a)
                B_ov(static_cast<std::size_t>(i * n_vir + a),
                     static_cast<std::size_t>(P)) = ia(i, a);
    }

    // (ia|jb) ≈ Σ_P B_ov(ia,P) B_ov(jb,P) — one dgemm over B_ov replaces the
    // scalar inner-P accumulation; OS/SS split + frozen-core logic unchanged.
    const xt::xtensor<double, 2> iajb_table =
        xt::linalg::dot(B_ov, xt::transpose(B_ov)); // [nov × nov]

    const RealVector& eps = det.orbital_energies(Spin::alpha);
    double e_os = 0.0;
    double e_ss = 0.0;
    for (int i = n_frozen; i < n_occ; ++i) {
        for (int j = n_frozen; j < n_occ; ++j) {
            for (int a = 0; a < n_vir; ++a) {
                for (int b = 0; b < n_vir; ++b) {
                    const int ia = i * n_vir + a;
                    const int jb = j * n_vir + b;
                    const int ib = i * n_vir + b;
                    const int ja = j * n_vir + a;
                    const double iajb = iajb_table(ia, jb);
                    const double ibja = iajb_table(ib, ja);
                    const double denom = mp_denominator(
                        eps(i), eps(j), eps(n_occ + a), eps(n_occ + b));
                    const double coul = iajb * iajb / denom;
                    e_os += coul;
                    e_ss += coul - iajb * ibja / denom;
                }
            }
        }
    }
    const double e_corr = os_scale * e_os + ss_scale * e_ss;
    return {e_corr, e_scf_ha + e_corr, /*ri=*/true, n_frozen, e_os, e_ss};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
