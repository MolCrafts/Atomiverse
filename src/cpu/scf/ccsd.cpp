/// @file cpu/scf/ccsd.cpp
/// @brief Closed-shell spin-orbital CCSD (Gauss–Stanton intermediates).
///
/// Occupied spin-orbitals: α then β over spatial occupied MOs; virtuals
/// likewise. Chemist MO ERIs (pq|rs) from `ao2mo_full` expand to physicist
/// antisymmetrized ⟨pq||rs⟩. Canonical HF: f_pq = δ_pq ε_p (f_ia = 0).
///
/// References:
/// - Stanton, Gauss, Watts, Bartlett, J. Chem. Phys. 94, 4334 (1991)
/// - Gauss & Stanton, J. Chem. Phys. 103, 3561 (1995) Table III
/// Matches PySCF `cc.GCCSD` / closed-shell `cc.CCSD` energies on RHF orbitals.

#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/cc2.h"
#include "atomiverse/cpu/scf/internal/ccsd_pack.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/cc_diis.h"
#include "atomiverse/cpu/scf/internal/ccsd_spatial.h"
#include "atomiverse/cpu/scf/internal/eom_ccsd_common.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

namespace ATV {
namespace cpu {
namespace scf {

/// Shared spin-orbital Jacobi CCSD once SO integrals / ε are assembled.
static CCSDSolution
so_ccsd_kernel(int no, int nv, xt::xtensor<double, 4> oooo,
               xt::xtensor<double, 4> ooov, xt::xtensor<double, 4> oovv,
               xt::xtensor<double, 4> ovov, xt::xtensor<double, 4> ovvv,
               xt::xtensor<double, 4> vvvv, std::vector<double> eps_so,
               double e_scf_ha, double tol, int max_iter, int n_occ_sp,
               int n_vir_sp, int n_alpha, int n_beta, int n_vir_a,
               int n_vir_b) {
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    xt::xtensor<double, 2> fov = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 2> foo = xt::zeros<double>({No, No});
    xt::xtensor<double, 2> fvv = xt::zeros<double>({Nv, Nv});
    for (int i = 0; i < no; ++i)
        foo(i, i) = eps_so[static_cast<std::size_t>(i)];
    for (int a = 0; a < nv; ++a)
        fvv(a, a) = eps_so[static_cast<std::size_t>(no + a)];

    xt::xtensor<double, 2> eia = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 4> eijab = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i) {
        for (int a = 0; a < nv; ++a) {
            const double d = eps_so[static_cast<std::size_t>(i)]
                             - eps_so[static_cast<std::size_t>(no + a)];
            if (d >= 0.0) {
                throw std::invalid_argument(
                    "ccsd_energy: non-negative T1 denominator; reference is "
                    "degenerate or non-aufbau.");
            }
            eia(i, a) = d;
        }
    }
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    eijab(i, j, a, b) = eia(i, a) + eia(j, b);

    xt::xtensor<double, 2> t1 = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 4> t2 = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    t2(i, j, a, b) = oovv(i, j, a, b) / eijab(i, j, a, b);

    auto energy_of = [&](const xt::xtensor<double, 2>& T1,
                         const xt::xtensor<double, 4>& T2) -> double {
        double e = 0.0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                e += fov(i, a) * T1(i, a);
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        e += 0.25 * T2(i, j, a, b) * oovv(i, j, a, b);
                        e += 0.5 * T1(i, a) * T1(j, b) * oovv(i, j, a, b);
                    }
        return e;
    };

    // Pulay amplitude DIIS over flattened [t1; t2] with the Jacobi increment
    // Δt = R/D as the error vector (same fixed point as plain Jacobi; only the
    // convergence rate changes). Benefits every spin-orbital consumer (CCSD,
    // CCSD(T), CC2, EOM, UCCSD) via `so_ccsd_kernel` / `ccsd_solve`.
    internal::CcAmplitudeDiis diis(8);
    const std::vector<double>::size_type ndiis = No * Nv + No * No * Nv * Nv;
    std::vector<double> diis_amp(ndiis), diis_err(ndiis);

    int iters = 0;
    bool ok = false;
    for (int it = 0; it < max_iter; ++it) {
        const xt::xtensor<double, 4> tau = make_tau(t2, t1, 1.0);
        const xt::xtensor<double, 4> tau_h = make_tau(t2, t1, 0.5);

        // ── F intermediates ──────────────────────────────────────────
        xt::xtensor<double, 2> Fae = fvv;
        for (int a = 0; a < nv; ++a) {
            for (int e = 0; e < nv; ++e) {
                double s = Fae(a, e);
                for (int m = 0; m < no; ++m)
                    s -= 0.5 * fov(m, e) * t1(m, a);
                // Fae += einsum('mf,amef->ae', t1, ovvv.T(1,0,3,2))
                // amef = ovvv(m,a,f,e)
                for (int m = 0; m < no; ++m)
                    for (int f = 0; f < nv; ++f)
                        s += t1(m, f) * ovvv(m, a, f, e);
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * tau_h(m, n, a, f) * oovv(m, n, e, f);
                Fae(a, e) = s;
            }
        }
        xt::xtensor<double, 2> Fmi = foo;
        for (int m = 0; m < no; ++m) {
            for (int i = 0; i < no; ++i) {
                double s = Fmi(m, i);
                for (int e = 0; e < nv; ++e)
                    s += 0.5 * fov(m, e) * t1(i, e);
                for (int n = 0; n < no; ++n)
                    for (int e = 0; e < nv; ++e)
                        s += t1(n, e) * ooov(m, n, i, e);
                for (int n = 0; n < no; ++n)
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.5 * tau_h(i, n, e, f) * oovv(m, n, e, f);
                Fmi(m, i) = s;
            }
        }
        xt::xtensor<double, 2> Fme = fov;
        for (int m = 0; m < no; ++m)
            for (int e = 0; e < nv; ++e) {
                double s = Fme(m, e);
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s += t1(n, f) * oovv(m, n, e, f);
                Fme(m, e) = s;
            }

        // ── W intermediates ──────────────────────────────────────────
        xt::xtensor<double, 4> Wmnij = oooo;
        for (int m = 0; m < no; ++m)
            for (int n = 0; n < no; ++n)
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j) {
                        double s = Wmnij(m, n, i, j);
                        for (int e = 0; e < nv; ++e) {
                            s += t1(j, e) * ooov(m, n, i, e);
                            s -= t1(i, e) * ooov(m, n, j, e);
                        }
                        for (int e = 0; e < nv; ++e)
                            for (int f = 0; f < nv; ++f)
                                s += 0.25 * tau(i, j, e, f) * oovv(m, n, e, f);
                        Wmnij(m, n, i, j) = s;
                    }

        xt::xtensor<double, 4> Wabef = vvvv;
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                for (int e = 0; e < nv; ++e)
                    for (int f = 0; f < nv; ++f) {
                        double s = Wabef(a, b, e, f);
                        // tmp = einsum('mb,mafe->bafe', t1, ovvv)
                        // W = vvvv - tmp + tmp.transpose(1,0,2,3)
                        double tmp_baef = 0.0;
                        double tmp_abef = 0.0;
                        for (int m = 0; m < no; ++m) {
                            tmp_baef += t1(m, b) * ovvv(m, a, f, e);
                            tmp_abef += t1(m, a) * ovvv(m, b, f, e);
                        }
                        s += -tmp_baef + tmp_abef;
                        for (int m = 0; m < no; ++m)
                            for (int n = 0; n < no; ++n)
                                s += 0.25 * tau(m, n, a, b) * oovv(m, n, e, f);
                        Wabef(a, b, e, f) = s;
                    }

        // Wmbej: eris_ovvo(m,b,e,j) = -ovov(m,b,j,e)
        xt::xtensor<double, 4> Wmbej = xt::zeros<double>({No, Nv, Nv, No});
        for (int m = 0; m < no; ++m)
            for (int b = 0; b < nv; ++b)
                for (int e = 0; e < nv; ++e)
                    for (int j = 0; j < no; ++j) {
                        double s = -ovov(m, b, j, e);
                        for (int f = 0; f < nv; ++f)
                            s += t1(j, f) * ovvv(m, b, e, f);
                        // - einsum('nb,mnej->mbej', t1, eris_oovo)
                        // eris_oovo(m,n,e,j) = -ooov(m,n,j,e)
                        for (int n = 0; n < no; ++n)
                            s -= t1(n, b) * (-ooov(m, n, j, e));
                        for (int n = 0; n < no; ++n)
                            for (int f = 0; f < nv; ++f)
                                s -= 0.5 * t2(j, n, f, b) * oovv(m, n, e, f);
                        for (int n = 0; n < no; ++n)
                            for (int f = 0; f < nv; ++f)
                                s -= t1(j, f) * t1(n, b) * oovv(m, n, e, f);
                        Wmbej(m, b, e, j) = s;
                    }

        // Strip orbital energies (moved into denominators).
        xt::xtensor<double, 2> Fvv = Fae;
        xt::xtensor<double, 2> Foo = Fmi;
        for (int a = 0; a < nv; ++a)
            Fvv(a, a) -= eps_so[static_cast<std::size_t>(no + a)];
        for (int i = 0; i < no; ++i)
            Foo(i, i) -= eps_so[static_cast<std::size_t>(i)];

        // ── T1 ───────────────────────────────────────────────────────
        xt::xtensor<double, 2> t1new = fov;
        for (int i = 0; i < no; ++i) {
            for (int a = 0; a < nv; ++a) {
                double s = t1new(i, a);
                for (int e = 0; e < nv; ++e)
                    s += t1(i, e) * Fvv(a, e);
                for (int m = 0; m < no; ++m)
                    s -= t1(m, a) * Foo(m, i);
                for (int m = 0; m < no; ++m)
                    for (int e = 0; e < nv; ++e)
                        s += t2(i, m, a, e) * Fme(m, e);
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s -= t1(n, f) * ovov(n, a, i, f);
                for (int m = 0; m < no; ++m)
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * t2(i, m, e, f) * ovvv(m, a, e, f);
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        for (int e = 0; e < nv; ++e)
                            s -= 0.5 * t2(m, n, a, e) * ooov(m, n, i, e);
                t1new(i, a) = s / eia(i, a);
            }
        }

        // ── T2 ───────────────────────────────────────────────────────
        xt::xtensor<double, 4> t2new = xt::zeros<double>({No, No, Nv, Nv});
        auto ring_at = [&](int ii, int jj, int aa, int bb) -> double {
            double r = 0.0;
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    r += t2(ii, m, aa, e) * Wmbej(m, bb, e, jj);
            // tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, ovov)
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    r += t1(ii, e) * t1(m, aa) * ovov(m, bb, jj, e);
            return r;
        };

        for (int i = 0; i < no; ++i) {
            for (int j = 0; j < no; ++j) {
                for (int a = 0; a < nv; ++a) {
                    for (int b = 0; b < nv; ++b) {
                        double s = oovv(i, j, a, b);

                        // P(ab) Σ_e t2_ijae Ftmp_be
                        for (int e = 0; e < nv; ++e) {
                            double Fbe = Fvv(b, e);
                            double Fae_ = Fvv(a, e);
                            for (int m = 0; m < no; ++m) {
                                Fbe -= 0.5 * t1(m, b) * Fme(m, e);
                                Fae_ -= 0.5 * t1(m, a) * Fme(m, e);
                            }
                            s += t2(i, j, a, e) * Fbe - t2(i, j, b, e) * Fae_;
                        }
                        // − P(ij) Σ_m t2_imab Ftmp_mj
                        for (int m = 0; m < no; ++m) {
                            double Fmj = Foo(m, j);
                            double Fmi_ = Foo(m, i);
                            for (int e = 0; e < nv; ++e) {
                                Fmj += 0.5 * t1(j, e) * Fme(m, e);
                                Fmi_ += 0.5 * t1(i, e) * Fme(m, e);
                            }
                            s -= t2(i, m, a, b) * Fmj - t2(j, m, a, b) * Fmi_;
                        }
                        for (int m = 0; m < no; ++m)
                            for (int n = 0; n < no; ++n)
                                s += 0.5 * tau(m, n, a, b) * Wmnij(m, n, i, j);
                        for (int e = 0; e < nv; ++e)
                            for (int f = 0; f < nv; ++f)
                                s += 0.5 * tau(i, j, e, f) * Wabef(a, b, e, f);

                        // P(ij)P(ab) ring
                        s += ring_at(i, j, a, b) - ring_at(j, i, a, b)
                             - ring_at(i, j, b, a) + ring_at(j, i, b, a);

                        // + P(ij) einsum('ie,jeba->ijab', t1, ovvv)
                        double v1 = 0.0, v2 = 0.0;
                        for (int e = 0; e < nv; ++e) {
                            v1 += t1(i, e) * ovvv(j, e, b, a);
                            v2 += t1(j, e) * ovvv(i, e, b, a);
                        }
                        s += v1 - v2;

                        // − P(ab) einsum('ma,ijmb->ijab', t1, ooov)
                        double o1 = 0.0, o2 = 0.0;
                        for (int m = 0; m < no; ++m) {
                            o1 += t1(m, a) * ooov(i, j, m, b);
                            o2 += t1(m, b) * ooov(i, j, m, a);
                        }
                        s -= o1 - o2;

                        t2new(i, j, a, b) = s / eijab(i, j, a, b);
                    }
                }
            }
        }

        // Antisymmetrize T2.
        xt::xtensor<double, 4> t2sym = xt::zeros<double>({No, No, Nv, Nv});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        t2sym(i, j, a, b) =
                            0.25
                            * (t2new(i, j, a, b) - t2new(i, j, b, a)
                               - t2new(j, i, a, b) + t2new(j, i, b, a));
        t2new = t2sym;

        double dt1 = 0.0, dt2 = 0.0;
        std::vector<double>::size_type di = 0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a) {
                const double d = t1new(i, a) - t1(i, a);
                dt1 = std::max(dt1, std::abs(d));
                diis_amp[di] = t1new(i, a);
                diis_err[di] = d;
                ++di;
            }
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        const double d = t2new(i, j, a, b) - t2(i, j, a, b);
                        dt2 = std::max(dt2, std::abs(d));
                        diis_amp[di] = t2new(i, j, a, b);
                        diis_err[di] = d;
                        ++di;
                    }

        const std::vector<double> ext = diis.step(diis_amp, diis_err);
        di = 0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                t1(i, a) = ext[di++];
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        t2(i, j, a, b) = ext[di++];

        iters = it + 1;
        if (std::max(dt1, dt2) < tol) {
            ok = true;
            break;
        }
    }

    if (!ok) {
        throw std::invalid_argument(
            "ccsd_energy: amplitudes did not converge in "
            + std::to_string(max_iter)
            + " iterations (tol=" + std::to_string(tol) + ").");
    }

    const double e_corr = energy_of(t1, t2);
    CCSDSolution sol;
    sol.result = CCSDResult{e_corr, e_scf_ha + e_corr, iters, true};
    sol.t1 = std::move(t1);
    sol.t2 = std::move(t2);
    sol.oooo = std::move(oooo);
    sol.ooov = std::move(ooov);
    sol.oovv = std::move(oovv);
    sol.ovov = std::move(ovov);
    sol.ovvv = std::move(ovvv);
    sol.vvvv = std::move(vvvv);
    sol.foo = std::move(foo);
    sol.fvv = std::move(fvv);
    sol.fov = std::move(fov);
    sol.eps_so = std::move(eps_so);
    sol.no = no;
    sol.nv = nv;
    sol.n_occ_sp = n_occ_sp;
    sol.n_vir_sp = n_vir_sp;
    sol.n_alpha = n_alpha;
    sol.n_beta = n_beta;
    sol.n_vir_a = n_vir_a;
    sol.n_vir_b = n_vir_b;
    return sol;
}

CCSDSolution ccsd_solve(const internal::IntegralView& ints,
                        const SlaterDeterminant& det, double e_scf_ha,
                        double tol, int max_iter, bool cc2) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "ccsd_energy: ERI is not materialized; need compute_eri=true.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "ccsd_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "ccsd_energy: GHF (spinor) reference is not supported.");
    }
    if (max_iter < 1) {
        throw std::invalid_argument("ccsd_energy: max_iter must be ≥ 1.");
    }
    if (!(tol > 0.0)) {
        throw std::invalid_argument("ccsd_energy: tol must be > 0.");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ_sp = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir_sp = n_mo - n_occ_sp;
    if (n_vir_sp <= 0) {
        CCSDSolution sol;
        sol.result = CCSDResult{0.0, e_scf_ha, 0, true};
        sol.n_occ_sp = n_occ_sp;
        sol.n_vir_sp = 0;
        return sol;
    }

    const xt::xtensor<double, 4> eri_mo = ao2mo_full(ints.ERI, C, n_mo);
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    const int no = 2 * n_occ_sp;
    const int nv = 2 * n_vir_sp;
    std::vector<SpinOrbital> so(static_cast<std::size_t>(no + nv));
    for (int i = 0; i < n_occ_sp; ++i)
        so[static_cast<std::size_t>(i)] = SpinOrbital{i, 0};
    for (int i = 0; i < n_occ_sp; ++i)
        so[static_cast<std::size_t>(n_occ_sp + i)] = SpinOrbital{i, 1};
    for (int a = 0; a < n_vir_sp; ++a)
        so[static_cast<std::size_t>(no + a)] = SpinOrbital{n_occ_sp + a, 0};
    for (int a = 0; a < n_vir_sp; ++a)
        so[static_cast<std::size_t>(no + n_vir_sp + a)] =
            SpinOrbital{n_occ_sp + a, 1};

    auto g = [&](int p, int q, int r, int s) -> double {
        return antisym_eri(eri_mo, so[static_cast<std::size_t>(p)],
                           so[static_cast<std::size_t>(q)],
                           so[static_cast<std::size_t>(r)],
                           so[static_cast<std::size_t>(s)]);
    };

    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    xt::xtensor<double, 4> oooo = xt::zeros<double>({No, No, No, No});
    xt::xtensor<double, 4> ooov = xt::zeros<double>({No, No, No, Nv});
    xt::xtensor<double, 4> oovv = xt::zeros<double>({No, No, Nv, Nv});
    xt::xtensor<double, 4> ovov = xt::zeros<double>({No, Nv, No, Nv});
    xt::xtensor<double, 4> ovvv = xt::zeros<double>({No, Nv, Nv, Nv});
    xt::xtensor<double, 4> vvvv = xt::zeros<double>({Nv, Nv, Nv, Nv});

#pragma omp parallel for collapse(2) schedule(static) if (no * no > 4)
    for (int p = 0; p < no; ++p) {
        for (int q = 0; q < no; ++q) {
            for (int r = 0; r < no; ++r) {
                for (int s = 0; s < no; ++s)
                    oooo(p, q, r, s) = g(p, q, r, s);
                for (int s = 0; s < nv; ++s)
                    ooov(p, q, r, s) = g(p, q, r, no + s);
            }
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    oovv(p, q, r, s) = g(p, q, no + r, no + s);
        }
    }
#pragma omp parallel for collapse(2) schedule(static) if (no * nv > 4)
    for (int p = 0; p < no; ++p) {
        for (int q = 0; q < nv; ++q) {
            for (int r = 0; r < no; ++r)
                for (int s = 0; s < nv; ++s)
                    ovov(p, q, r, s) = g(p, no + q, r, no + s);
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    ovvv(p, q, r, s) = g(p, no + q, no + r, no + s);
        }
    }
#pragma omp parallel for collapse(2) schedule(static) if (nv > 2)
    for (int p = 0; p < nv; ++p)
        for (int q = 0; q < nv; ++q)
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    vvvv(p, q, r, s) = g(no + p, no + q, no + r, no + s);

    std::vector<double> eps_so(static_cast<std::size_t>(no + nv));
    for (int i = 0; i < n_occ_sp; ++i) {
        const double e = eps(static_cast<std::size_t>(i));
        eps_so[static_cast<std::size_t>(i)] = e;
        eps_so[static_cast<std::size_t>(n_occ_sp + i)] = e;
    }
    for (int a = 0; a < n_vir_sp; ++a) {
        const double e = eps(static_cast<std::size_t>(n_occ_sp + a));
        eps_so[static_cast<std::size_t>(no + a)] = e;
        eps_so[static_cast<std::size_t>(no + n_vir_sp + a)] = e;
    }

    xt::xtensor<double, 2> fov = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 2> foo = xt::zeros<double>({No, No});
    xt::xtensor<double, 2> fvv = xt::zeros<double>({Nv, Nv});
    for (int i = 0; i < no; ++i)
        foo(i, i) = eps_so[static_cast<std::size_t>(i)];
    for (int a = 0; a < nv; ++a)
        fvv(a, a) = eps_so[static_cast<std::size_t>(no + a)];

    xt::xtensor<double, 2> eia = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 4> eijab = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i) {
        for (int a = 0; a < nv; ++a) {
            const double d = eps_so[static_cast<std::size_t>(i)]
                             - eps_so[static_cast<std::size_t>(no + a)];
            if (d >= 0.0) {
                throw std::invalid_argument(
                    "ccsd_energy: non-negative T1 denominator; reference is "
                    "degenerate or non-aufbau.");
            }
            eia(i, a) = d;
        }
    }
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    eijab(i, j, a, b) = eia(i, a) + eia(j, b);

    xt::xtensor<double, 2> t1 = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 4> t2 = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    t2(i, j, a, b) = oovv(i, j, a, b) / eijab(i, j, a, b);

    auto energy_of = [&](const xt::xtensor<double, 2>& T1,
                         const xt::xtensor<double, 4>& T2) -> double {
        double e = 0.0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                e += fov(i, a) * T1(i, a);
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        e += 0.25 * T2(i, j, a, b) * oovv(i, j, a, b);
                        e += 0.5 * T1(i, a) * T1(j, b) * oovv(i, j, a, b);
                    }
        return e;
    };

    // Pulay amplitude DIIS over flattened [t1; t2] with the Jacobi increment
    // Δt = R/D as the error vector (same fixed point as plain Jacobi; only the
    // convergence rate changes). Benefits every spin-orbital consumer (CCSD,
    // CCSD(T), CC2, EOM, UCCSD) via `so_ccsd_kernel` / `ccsd_solve`.
    internal::CcAmplitudeDiis diis(8);
    const std::vector<double>::size_type ndiis = No * Nv + No * No * Nv * Nv;
    std::vector<double> diis_amp(ndiis), diis_err(ndiis);

    int iters = 0;
    bool ok = false;
    for (int it = 0; it < max_iter; ++it) {
        const xt::xtensor<double, 4> tau = make_tau(t2, t1, 1.0);
        const xt::xtensor<double, 4> tau_h = make_tau(t2, t1, 0.5);

        // ── F intermediates ──────────────────────────────────────────
        xt::xtensor<double, 2> Fae = fvv;
        for (int a = 0; a < nv; ++a) {
            for (int e = 0; e < nv; ++e) {
                double s = Fae(a, e);
                for (int m = 0; m < no; ++m)
                    s -= 0.5 * fov(m, e) * t1(m, a);
                // Fae += einsum('mf,amef->ae', t1, ovvv.T(1,0,3,2))
                // amef = ovvv(m,a,f,e)
                for (int m = 0; m < no; ++m)
                    for (int f = 0; f < nv; ++f)
                        s += t1(m, f) * ovvv(m, a, f, e);
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * tau_h(m, n, a, f) * oovv(m, n, e, f);
                Fae(a, e) = s;
            }
        }
        xt::xtensor<double, 2> Fmi = foo;
        for (int m = 0; m < no; ++m) {
            for (int i = 0; i < no; ++i) {
                double s = Fmi(m, i);
                for (int e = 0; e < nv; ++e)
                    s += 0.5 * fov(m, e) * t1(i, e);
                for (int n = 0; n < no; ++n)
                    for (int e = 0; e < nv; ++e)
                        s += t1(n, e) * ooov(m, n, i, e);
                for (int n = 0; n < no; ++n)
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.5 * tau_h(i, n, e, f) * oovv(m, n, e, f);
                Fmi(m, i) = s;
            }
        }
        xt::xtensor<double, 2> Fme = fov;
        for (int m = 0; m < no; ++m)
            for (int e = 0; e < nv; ++e) {
                double s = Fme(m, e);
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s += t1(n, f) * oovv(m, n, e, f);
                Fme(m, e) = s;
            }

        // ── W intermediates ──────────────────────────────────────────
        xt::xtensor<double, 4> Wmnij = oooo;
        for (int m = 0; m < no; ++m)
            for (int n = 0; n < no; ++n)
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j) {
                        double s = Wmnij(m, n, i, j);
                        for (int e = 0; e < nv; ++e) {
                            s += t1(j, e) * ooov(m, n, i, e);
                            s -= t1(i, e) * ooov(m, n, j, e);
                        }
                        for (int e = 0; e < nv; ++e)
                            for (int f = 0; f < nv; ++f)
                                s += 0.25 * tau(i, j, e, f) * oovv(m, n, e, f);
                        Wmnij(m, n, i, j) = s;
                    }

        xt::xtensor<double, 4> Wabef = vvvv;
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                for (int e = 0; e < nv; ++e)
                    for (int f = 0; f < nv; ++f) {
                        double s = Wabef(a, b, e, f);
                        // tmp = einsum('mb,mafe->bafe', t1, ovvv)
                        // W = vvvv - tmp + tmp.transpose(1,0,2,3)
                        double tmp_baef = 0.0;
                        double tmp_abef = 0.0;
                        for (int m = 0; m < no; ++m) {
                            tmp_baef += t1(m, b) * ovvv(m, a, f, e);
                            tmp_abef += t1(m, a) * ovvv(m, b, f, e);
                        }
                        s += -tmp_baef + tmp_abef;
                        for (int m = 0; m < no; ++m)
                            for (int n = 0; n < no; ++n)
                                s += 0.25 * tau(m, n, a, b) * oovv(m, n, e, f);
                        Wabef(a, b, e, f) = s;
                    }

        // Wmbej: eris_ovvo(m,b,e,j) = -ovov(m,b,j,e)
        xt::xtensor<double, 4> Wmbej = xt::zeros<double>({No, Nv, Nv, No});
        for (int m = 0; m < no; ++m)
            for (int b = 0; b < nv; ++b)
                for (int e = 0; e < nv; ++e)
                    for (int j = 0; j < no; ++j) {
                        double s = -ovov(m, b, j, e);
                        for (int f = 0; f < nv; ++f)
                            s += t1(j, f) * ovvv(m, b, e, f);
                        // - einsum('nb,mnej->mbej', t1, eris_oovo)
                        // eris_oovo(m,n,e,j) = -ooov(m,n,j,e)
                        for (int n = 0; n < no; ++n)
                            s -= t1(n, b) * (-ooov(m, n, j, e));
                        for (int n = 0; n < no; ++n)
                            for (int f = 0; f < nv; ++f)
                                s -= 0.5 * t2(j, n, f, b) * oovv(m, n, e, f);
                        for (int n = 0; n < no; ++n)
                            for (int f = 0; f < nv; ++f)
                                s -= t1(j, f) * t1(n, b) * oovv(m, n, e, f);
                        Wmbej(m, b, e, j) = s;
                    }

        // Strip orbital energies (moved into denominators).
        xt::xtensor<double, 2> Fvv = Fae;
        xt::xtensor<double, 2> Foo = Fmi;
        for (int a = 0; a < nv; ++a)
            Fvv(a, a) -= eps_so[static_cast<std::size_t>(no + a)];
        for (int i = 0; i < no; ++i)
            Foo(i, i) -= eps_so[static_cast<std::size_t>(i)];

        // ── T1 ───────────────────────────────────────────────────────
        xt::xtensor<double, 2> t1new = fov;
        for (int i = 0; i < no; ++i) {
            for (int a = 0; a < nv; ++a) {
                double s = t1new(i, a);
                for (int e = 0; e < nv; ++e)
                    s += t1(i, e) * Fvv(a, e);
                for (int m = 0; m < no; ++m)
                    s -= t1(m, a) * Foo(m, i);
                for (int m = 0; m < no; ++m)
                    for (int e = 0; e < nv; ++e)
                        s += t2(i, m, a, e) * Fme(m, e);
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s -= t1(n, f) * ovov(n, a, i, f);
                for (int m = 0; m < no; ++m)
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * t2(i, m, e, f) * ovvv(m, a, e, f);
                for (int m = 0; m < no; ++m)
                    for (int n = 0; n < no; ++n)
                        for (int e = 0; e < nv; ++e)
                            s -= 0.5 * t2(m, n, a, e) * ooov(m, n, i, e);
                t1new(i, a) = s / eia(i, a);
            }
        }

        // ── T2 ───────────────────────────────────────────────────────
        xt::xtensor<double, 4> t2new = xt::zeros<double>({No, No, Nv, Nv});
        auto ring_at = [&](int ii, int jj, int aa, int bb) -> double {
            double r = 0.0;
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    r += t2(ii, m, aa, e) * Wmbej(m, bb, e, jj);
            // tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, ovov)
            for (int m = 0; m < no; ++m)
                for (int e = 0; e < nv; ++e)
                    r += t1(ii, e) * t1(m, aa) * ovov(m, bb, jj, e);
            return r;
        };

        // CC2: T₁-only Fock dressing (no T₂ in F; PySCF rccsd Loo2/Lvv2).
        // For canonical HF (f_ia=0, diagonal f) these strip to zero.
        xt::xtensor<double, 2> Lvv2 = xt::zeros<double>({Nv, Nv});
        xt::xtensor<double, 2> Loo2 = xt::zeros<double>({No, No});
        if (cc2) {
            for (int a = 0; a < nv; ++a)
                for (int e = 0; e < nv; ++e) {
                    double s = fvv(a, e);
                    for (int m = 0; m < no; ++m)
                        s -= fov(m, e) * t1(m, a);
                    if (a == e)
                        s -= eps_so[static_cast<std::size_t>(no + a)];
                    Lvv2(a, e) = s;
                }
            for (int m = 0; m < no; ++m)
                for (int i = 0; i < no; ++i) {
                    double s = foo(m, i);
                    for (int e = 0; e < nv; ++e)
                        s += fov(m, e) * t1(i, e);
                    if (m == i)
                        s -= eps_so[static_cast<std::size_t>(m)];
                    Loo2(m, i) = s;
                }
        }
        // CC2: pure T₁·T₁ part of τ (no T₂) and T₁-only W intermediates
        // (drop ¼ τ(T₂)·⟨mn||ef⟩ quadratic pieces of full CCSD W).
        xt::xtensor<double, 4> tau_t1 = tau;
        xt::xtensor<double, 4> Wmnij_cc2 = oooo;
        xt::xtensor<double, 4> Wabef_cc2 = vvvv;
        if (cc2) {
            tau_t1 = make_tau(xt::zeros<double>({No, No, Nv, Nv}), t1, 1.0);
            for (int m = 0; m < no; ++m)
                for (int n = 0; n < no; ++n)
                    for (int i = 0; i < no; ++i)
                        for (int j = 0; j < no; ++j) {
                            double s = oooo(m, n, i, j);
                            for (int e = 0; e < nv; ++e) {
                                s += t1(j, e) * ooov(m, n, i, e);
                                s -= t1(i, e) * ooov(m, n, j, e);
                            }
                            for (int e = 0; e < nv; ++e)
                                for (int f = 0; f < nv; ++f)
                                    s += 0.25 * tau_t1(i, j, e, f)
                                         * oovv(m, n, e, f);
                            Wmnij_cc2(m, n, i, j) = s;
                        }
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f) {
                            double s = vvvv(a, b, e, f);
                            double tmp_baef = 0.0, tmp_abef = 0.0;
                            for (int m = 0; m < no; ++m) {
                                tmp_baef += t1(m, b) * ovvv(m, a, f, e);
                                tmp_abef += t1(m, a) * ovvv(m, b, f, e);
                            }
                            s += -tmp_baef + tmp_abef;
                            Wabef_cc2(a, b, e, f) = s;
                        }
        }

        for (int i = 0; i < no; ++i) {
            for (int j = 0; j < no; ++j) {
                for (int a = 0; a < nv; ++a) {
                    for (int b = 0; b < nv; ++b) {
                        double s = oovv(i, j, a, b);

                        if (cc2) {
                            // P(ab) Σ_e t2_ijae Lvv2_be  − P(ij) Σ_m t2_imab Loo2_mj
                            for (int e = 0; e < nv; ++e)
                                s += t2(i, j, a, e) * Lvv2(b, e)
                                     - t2(i, j, b, e) * Lvv2(a, e);
                            for (int m = 0; m < no; ++m)
                                s -= t2(i, m, a, b) * Loo2(m, j)
                                     - t2(j, m, a, b) * Loo2(m, i);
                            // W(T1)·τ(T1 only) — no T₂·W / ring
                            for (int m = 0; m < no; ++m)
                                for (int n = 0; n < no; ++n)
                                    s += 0.5 * tau_t1(m, n, a, b)
                                         * Wmnij_cc2(m, n, i, j);
                            for (int e = 0; e < nv; ++e)
                                for (int f = 0; f < nv; ++f)
                                    s += 0.5 * tau_t1(i, j, e, f)
                                         * Wabef_cc2(a, b, e, f);
                        } else {
                            // P(ab) Σ_e t2_ijae Ftmp_be
                            for (int e = 0; e < nv; ++e) {
                                double Fbe = Fvv(b, e);
                                double Fae_ = Fvv(a, e);
                                for (int m = 0; m < no; ++m) {
                                    Fbe -= 0.5 * t1(m, b) * Fme(m, e);
                                    Fae_ -= 0.5 * t1(m, a) * Fme(m, e);
                                }
                                s += t2(i, j, a, e) * Fbe
                                     - t2(i, j, b, e) * Fae_;
                            }
                            // − P(ij) Σ_m t2_imab Ftmp_mj
                            for (int m = 0; m < no; ++m) {
                                double Fmj = Foo(m, j);
                                double Fmi_ = Foo(m, i);
                                for (int e = 0; e < nv; ++e) {
                                    Fmj += 0.5 * t1(j, e) * Fme(m, e);
                                    Fmi_ += 0.5 * t1(i, e) * Fme(m, e);
                                }
                                s -= t2(i, m, a, b) * Fmj
                                     - t2(j, m, a, b) * Fmi_;
                            }
                            for (int m = 0; m < no; ++m)
                                for (int n = 0; n < no; ++n)
                                    s += 0.5 * tau(m, n, a, b)
                                         * Wmnij(m, n, i, j);
                            for (int e = 0; e < nv; ++e)
                                for (int f = 0; f < nv; ++f)
                                    s += 0.5 * tau(i, j, e, f)
                                         * Wabef(a, b, e, f);

                            // P(ij)P(ab) ring
                            s += ring_at(i, j, a, b) - ring_at(j, i, a, b)
                                 - ring_at(i, j, b, a) + ring_at(j, i, b, a);
                        }

                        // + P(ij) einsum('ie,jeba->ijab', t1, ovvv)
                        // (both CCSD and CC2)
                        double v1 = 0.0, v2 = 0.0;
                        for (int e = 0; e < nv; ++e) {
                            v1 += t1(i, e) * ovvv(j, e, b, a);
                            v2 += t1(j, e) * ovvv(i, e, b, a);
                        }
                        s += v1 - v2;

                        // − P(ab) einsum('ma,ijmb->ijab', t1, ooov)
                        double o1 = 0.0, o2 = 0.0;
                        for (int m = 0; m < no; ++m) {
                            o1 += t1(m, a) * ooov(i, j, m, b);
                            o2 += t1(m, b) * ooov(i, j, m, a);
                        }
                        s -= o1 - o2;

                        t2new(i, j, a, b) = s / eijab(i, j, a, b);
                    }
                }
            }
        }

        // Antisymmetrize T2.
        xt::xtensor<double, 4> t2sym = xt::zeros<double>({No, No, Nv, Nv});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        t2sym(i, j, a, b) =
                            0.25
                            * (t2new(i, j, a, b) - t2new(i, j, b, a)
                               - t2new(j, i, a, b) + t2new(j, i, b, a));
        t2new = t2sym;

        double dt1 = 0.0, dt2 = 0.0;
        std::vector<double>::size_type di = 0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a) {
                const double d = t1new(i, a) - t1(i, a);
                dt1 = std::max(dt1, std::abs(d));
                diis_amp[di] = t1new(i, a);
                diis_err[di] = d;
                ++di;
            }
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        const double d = t2new(i, j, a, b) - t2(i, j, a, b);
                        dt2 = std::max(dt2, std::abs(d));
                        diis_amp[di] = t2new(i, j, a, b);
                        diis_err[di] = d;
                        ++di;
                    }

        const std::vector<double> ext = diis.step(diis_amp, diis_err);
        di = 0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
                t1(i, a) = ext[di++];
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        t2(i, j, a, b) = ext[di++];

        iters = it + 1;
        if (std::max(dt1, dt2) < tol) {
            ok = true;
            break;
        }
    }

    if (!ok) {
        throw std::invalid_argument(
            "ccsd_energy: amplitudes did not converge in "
            + std::to_string(max_iter)
            + " iterations (tol=" + std::to_string(tol) + ").");
    }

    const double e_corr = energy_of(t1, t2);
    CCSDSolution sol;
    sol.result = CCSDResult{e_corr, e_scf_ha + e_corr, iters, true};
    sol.t1 = std::move(t1);
    sol.t2 = std::move(t2);
    sol.oooo = std::move(oooo);
    sol.ooov = std::move(ooov);
    sol.oovv = std::move(oovv);
    sol.ovov = std::move(ovov);
    sol.ovvv = std::move(ovvv);
    sol.vvvv = std::move(vvvv);
    sol.foo = std::move(foo);
    sol.fvv = std::move(fvv);
    sol.fov = std::move(fov);
    sol.eps_so = std::move(eps_so);
    sol.no = no;
    sol.nv = nv;
    sol.n_occ_sp = n_occ_sp;
    sol.n_vir_sp = n_vir_sp;
    return sol;
}

CCSDResult ccsd_energy(const internal::IntegralView& ints,
                       const SlaterDeterminant& det, double e_scf_ha,
                       double tol, int max_iter, bool use_diis) {
    // Closed-shell RHF energy entry: route to the spatial-orbital fast path
    // (n_o²·n_v² storage, no spin expansion). The spin-orbital `ccsd_solve`
    // stays the oracle + the backbone for EOM/(T)/CC2/gradient/UCCSD.
    return ccsd_energy_spatial(ints, det, e_scf_ha, tol, max_iter, use_diis);
}

CC2Result cc2_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha, double tol,
                     int max_iter) {
    const CCSDResult r =
        ccsd_solve(ints, det, e_scf_ha, tol, max_iter, /*cc2=*/true).result;
    return CC2Result{r.e_corr, r.e_tot, r.iterations, r.converged};
}

namespace {

/// Chemist (C0 C1 | C2 C3) AO→MO transform (local; matches mp2 ao2mo_mixed_).
xt::xtensor<double, 4> ao2mo_uhf_block(const xt::xtensor<double, 4>& eri_ao,
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
    return cur; // (C0,C1,C2,C3) → chemist (p q | r s)
}

} // namespace

CCSDSolution uccsd_solve(const internal::IntegralView& ints,
                         const SlaterDeterminant& det, double e_scf_ha,
                         double tol, int max_iter) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "uccsd_solve: ERI is not materialized; need compute_eri=true.");
    }
    if (det.restricted()) {
        throw std::invalid_argument(
            "uccsd_solve: requires an unrestricted (UHF) determinant — use "
            "ccsd_solve for closed-shell RHF.");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "uccsd_solve: GHF (spinor) reference is not supported.");
    }
    if (max_iter < 1) {
        throw std::invalid_argument("uccsd_solve: max_iter must be ≥ 1.");
    }
    if (!(tol > 0.0)) {
        throw std::invalid_argument("uccsd_solve: tol must be > 0.");
    }

    const xt::xtensor<double, 2> Ca = det.coefficients(Spin::alpha);
    const xt::xtensor<double, 2> Cb = det.coefficients(Spin::beta);
    const xt::xtensor<double, 1>& eps_a = det.orbital_energies(Spin::alpha);
    const xt::xtensor<double, 1>& eps_b = det.orbital_energies(Spin::beta);
    const int n_mo = static_cast<int>(Ca.shape(1));
    if (static_cast<int>(Cb.shape(1)) != n_mo) {
        throw std::invalid_argument(
            "uccsd_solve: α and β MO counts differ — unsupported.");
    }
    const int na = det.n_electrons(Spin::alpha);
    const int nb = det.n_electrons(Spin::beta);
    const int nva = n_mo - na;
    const int nvb = n_mo - nb;
    if (nva < 0 || nvb < 0) {
        throw std::invalid_argument(
            "uccsd_solve: occupation exceeds MO count.");
    }
    if (nva + nvb == 0) {
        CCSDSolution sol;
        sol.result = CCSDResult{0.0, e_scf_ha, 0, true};
        sol.n_alpha = na;
        sol.n_beta = nb;
        sol.n_vir_a = 0;
        sol.n_vir_b = 0;
        sol.no = na + nb;
        sol.nv = 0;
        return sol;
    }

    // Chemist MO ERIs for spin pairs: eri_στ(p,q,r,s) = (p_σ q_σ | r_τ s_τ).
    // ⟨pq‖rs⟩ needs (pr|qs) = eri_στ(p,r,q,s) when p,r ∈ σ and q,s ∈ τ.
    const xt::xtensor<double, 4> eri_aa =
        ao2mo_uhf_block(ints.ERI, Ca, Ca, Ca, Ca);
    const xt::xtensor<double, 4> eri_bb =
        ao2mo_uhf_block(ints.ERI, Cb, Cb, Cb, Cb);
    const xt::xtensor<double, 4> eri_ab =
        ao2mo_uhf_block(ints.ERI, Ca, Ca, Cb, Cb); // (p_a q_a | r_b s_b)
    const xt::xtensor<double, 4> eri_ba =
        ao2mo_uhf_block(ints.ERI, Cb, Cb, Ca, Ca); // (p_b q_b | r_a s_a)

    // Spin-orbital list: occ [α|β], vir [α|β]. The block-spin gather below
    // reads spin-pair ERIs directly, so it keeps its own inline
    // antisymmetrization rather than the single-tensor antisym_eri.
    const int no = na + nb;
    const int nv = nva + nvb;
    std::vector<SpinOrbital> so(static_cast<std::size_t>(no + nv));
    for (int i = 0; i < na; ++i)
        so[static_cast<std::size_t>(i)] = SpinOrbital{i, 0};
    for (int i = 0; i < nb; ++i)
        so[static_cast<std::size_t>(na + i)] = SpinOrbital{i, 1};
    for (int a = 0; a < nva; ++a)
        so[static_cast<std::size_t>(no + a)] = SpinOrbital{na + a, 0};
    for (int a = 0; a < nvb; ++a)
        so[static_cast<std::size_t>(no + nva + a)] = SpinOrbital{nb + a, 1};

    auto chemist = [&](const SpinOrbital& p, const SpinOrbital& q,
                       const SpinOrbital& r, const SpinOrbital& s) -> double {
        // (pq|rs) with p,q same spin and r,s same spin.
        if (p.spin != q.spin || r.spin != s.spin)
            return 0.0;
        if (p.spin == 0 && r.spin == 0)
            return eri_aa(p.spat, q.spat, r.spat, s.spat);
        if (p.spin == 0 && r.spin == 1)
            return eri_ab(p.spat, q.spat, r.spat, s.spat);
        if (p.spin == 1 && r.spin == 0)
            return eri_ba(p.spat, q.spat, r.spat, s.spat);
        return eri_bb(p.spat, q.spat, r.spat, s.spat);
    };

    auto g = [&](int p, int q, int r, int s) -> double {
        // ⟨pq||rs⟩ = (pr|qs) − (ps|qr)
        const SpinOrbital& P = so[static_cast<std::size_t>(p)];
        const SpinOrbital& Q = so[static_cast<std::size_t>(q)];
        const SpinOrbital& R = so[static_cast<std::size_t>(r)];
        const SpinOrbital& S = so[static_cast<std::size_t>(s)];
        double v = 0.0;
        if (P.spin == R.spin && Q.spin == S.spin)
            v += chemist(P, R, Q, S); // (pr|qs)
        if (P.spin == S.spin && Q.spin == R.spin)
            v -= chemist(P, S, Q, R); // (ps|qr)
        return v;
    };

    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    xt::xtensor<double, 4> oooo = xt::zeros<double>({No, No, No, No});
    xt::xtensor<double, 4> ooov = xt::zeros<double>({No, No, No, Nv});
    xt::xtensor<double, 4> oovv = xt::zeros<double>({No, No, Nv, Nv});
    xt::xtensor<double, 4> ovov = xt::zeros<double>({No, Nv, No, Nv});
    xt::xtensor<double, 4> ovvv = xt::zeros<double>({No, Nv, Nv, Nv});
    xt::xtensor<double, 4> vvvv = xt::zeros<double>({Nv, Nv, Nv, Nv});

    for (int p = 0; p < no; ++p) {
        for (int q = 0; q < no; ++q) {
            for (int r = 0; r < no; ++r) {
                for (int s = 0; s < no; ++s)
                    oooo(p, q, r, s) = g(p, q, r, s);
                for (int s = 0; s < nv; ++s)
                    ooov(p, q, r, s) = g(p, q, r, no + s);
            }
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    oovv(p, q, r, s) = g(p, q, no + r, no + s);
        }
    }
    for (int p = 0; p < no; ++p) {
        for (int q = 0; q < nv; ++q) {
            for (int r = 0; r < no; ++r)
                for (int s = 0; s < nv; ++s)
                    ovov(p, q, r, s) = g(p, no + q, r, no + s);
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    ovvv(p, q, r, s) = g(p, no + q, no + r, no + s);
        }
    }
    for (int p = 0; p < nv; ++p)
        for (int q = 0; q < nv; ++q)
            for (int r = 0; r < nv; ++r)
                for (int s = 0; s < nv; ++s)
                    vvvv(p, q, r, s) = g(no + p, no + q, no + r, no + s);

    std::vector<double> eps_so(static_cast<std::size_t>(no + nv));
    for (int i = 0; i < na; ++i)
        eps_so[static_cast<std::size_t>(i)] =
            eps_a(static_cast<std::size_t>(i));
    for (int i = 0; i < nb; ++i)
        eps_so[static_cast<std::size_t>(na + i)] =
            eps_b(static_cast<std::size_t>(i));
    for (int a = 0; a < nva; ++a)
        eps_so[static_cast<std::size_t>(no + a)] =
            eps_a(static_cast<std::size_t>(na + a));
    for (int a = 0; a < nvb; ++a)
        eps_so[static_cast<std::size_t>(no + nva + a)] =
            eps_b(static_cast<std::size_t>(nb + a));

    return so_ccsd_kernel(no, nv, std::move(oooo), std::move(ooov),
                          std::move(oovv), std::move(ovov), std::move(ovvv),
                          std::move(vvvv), std::move(eps_so), e_scf_ha, tol,
                          max_iter, /*n_occ_sp=*/na, /*n_vir_sp=*/nva, na, nb,
                          nva, nvb);
}

namespace {

double triples_energy(const CCSDSolution& amps) {
    const int no = amps.no;
    const int nv = amps.nv;
    if (no < 3 || nv < 3)
        return 0.0; // no unique a>b>c (or ijk) triples in the basis

    const auto& t1 = amps.t1;
    const auto& t2 = amps.t2;
    const auto& oovv = amps.oovv;
    const auto& ooov = amps.ooov;
    const auto& ovvv = amps.ovvv;
    const auto& eps = amps.eps_so;
    // f_ai = 0 for canonical HF (fov is f_ia = 0).

    // Pre-cache bcei[b,c,e,i] = ovvv(i,e,c,b) and majk[m,a,j,k] = ooov(j,k,m,a)
    // and bcjk[b,c,j,k] = oovv(j,k,b,c).
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    xt::xtensor<double, 4> bcei = xt::zeros<double>({Nv, Nv, Nv, No});
    xt::xtensor<double, 4> majk = xt::zeros<double>({No, Nv, No, No});
    xt::xtensor<double, 4> bcjk = xt::zeros<double>({Nv, Nv, No, No});
    for (int b = 0; b < nv; ++b)
        for (int c = 0; c < nv; ++c)
            for (int e = 0; e < nv; ++e)
                for (int i = 0; i < no; ++i)
                    bcei(b, c, e, i) = ovvv(i, e, c, b);
    for (int m = 0; m < no; ++m)
        for (int a = 0; a < nv; ++a)
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < no; ++k)
                    majk(m, a, j, k) = ooov(j, k, m, a);
    for (int b = 0; b < nv; ++b)
        for (int c = 0; c < nv; ++c)
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < no; ++k)
                    bcjk(b, c, j, k) = oovv(j, k, b, c);

    // get_wv(a,b,c) → (w_ijk, v_ijk) for fixed virtuals a,b,c.
    auto get_wv = [&](int a, int b, int c, xt::xtensor<double, 3>& w,
                      xt::xtensor<double, 3>& v) {
        for (int i = 0; i < no; ++i) {
            for (int j = 0; j < no; ++j) {
                for (int k = 0; k < no; ++k) {
                    double ww = 0.0;
                    // Σ_e t2(j,k,a,e) * bcei(b,c,e,i)
                    for (int e = 0; e < nv; ++e)
                        ww += t2(j, k, a, e) * bcei(b, c, e, i);
                    // − Σ_m t2(i,m,b,c) * majk(m,a,j,k)
                    for (int m = 0; m < no; ++m)
                        ww -= t2(i, m, b, c) * majk(m, a, j, k);
                    w(i, j, k) = ww;
                    // v = t1(i,a)*bcjk(b,c,j,k) + fvo(a,i)*t2(j,k,b,c) + w
                    // fvo = 0 for canonical HF
                    v(i, j, k) = t1(i, a) * bcjk(b, c, j, k) + ww;
                }
            }
        }
        // Cyclic permutation of occupied indices on w only:
        // w = w + w.transpose(2,0,1) + w.transpose(1,2,0)
        xt::xtensor<double, 3> w0 = w;
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int k = 0; k < no; ++k)
                    w(i, j, k) = w0(i, j, k) + w0(k, i, j) + w0(j, k, i);
    };

    double et = 0.0;
    xt::xtensor<double, 3> wabc = xt::zeros<double>({No, No, No});
    xt::xtensor<double, 3> vabc = xt::zeros<double>({No, No, No});
    xt::xtensor<double, 3> wcab = xt::zeros<double>({No, No, No});
    xt::xtensor<double, 3> vcab = xt::zeros<double>({No, No, No});
    xt::xtensor<double, 3> wbac = xt::zeros<double>({No, No, No});
    xt::xtensor<double, 3> vbac = xt::zeros<double>({No, No, No});

    // a > b > c restricted sum (PySCF gccsd_t)
    for (int a = 0; a < nv; ++a) {
        for (int b = 0; b < a; ++b) {
            for (int c = 0; c < b; ++c) {
                get_wv(a, b, c, wabc, vabc);
                get_wv(c, a, b, wcab, vcab);
                get_wv(b, a, c, wbac, vbac);

                const double ea = eps[static_cast<std::size_t>(no + a)];
                const double eb = eps[static_cast<std::size_t>(no + b)];
                const double ec = eps[static_cast<std::size_t>(no + c)];
                const double eabc = ea + eb + ec;

                for (int i = 0; i < no; ++i) {
                    const double ei = eps[static_cast<std::size_t>(i)];
                    for (int j = 0; j < no; ++j) {
                        const double ej = eps[static_cast<std::size_t>(j)];
                        for (int k = 0; k < no; ++k) {
                            const double ek = eps[static_cast<std::size_t>(k)];
                            const double den = ei + ej + ek - eabc;
                            if (std::abs(den) < 1e-14)
                                continue;
                            const double w =
                                (wabc(i, j, k) + wcab(i, j, k) - wbac(i, j, k))
                                / den;
                            const double v =
                                vabc(i, j, k) + vcab(i, j, k) - vbac(i, j, k);
                            et += w * v;
                        }
                    }
                }
            }
        }
    }
    return 0.5 * et;
}

} // namespace

CCSDTResult ccsd_t_energy(const internal::IntegralView& ints,
                          const SlaterDeterminant& det, double e_scf_ha,
                          double tol, int max_iter) {
    CCSDSolution sol =
        ccsd_solve(ints, det, e_scf_ha, tol, max_iter, /*cc2=*/false);
    const double e_t = triples_energy(sol);
    const double e_corr = sol.result.e_corr + e_t;
    return CCSDTResult{e_corr, e_scf_ha + e_corr,     sol.result.e_corr,
                       e_t,    sol.result.iterations, sol.result.converged};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
