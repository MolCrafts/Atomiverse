/// @file cpu/scf/ccsd_spatial.cpp
/// @brief Closed-shell spatial-orbital CCSD (Hirata / Scuseria–Janssen–Schaefer).
///
/// Spatial amplitudes t1[n_o,n_v], t2[n_o,n_o,n_v,n_v] (storage n_o²·n_v², no
/// spin expansion). The spin-summed residual is the restricted-CCSD equation
/// set of Hirata et al., J. Chem. Phys. 120, 2581 (2004) (chemist-notation MO
/// ERIs; PySCF `cc.RCCSD.update_amps`), which is the same fixed point as the
/// spin-orbital Gauss–Stanton kernel in `ccsd.cpp`. Canonical RHF reference:
/// f_ov = 0, f_oo / f_vv diagonal, so the Fock off-diagonal terms vanish.
///
/// Reference: Scuseria, Janssen, Schaefer, J. Chem. Phys. 89, 7382 (1988);
/// Hirata et al., J. Chem. Phys. 120, 2581 (2004).

#include "atomiverse/cpu/scf/internal/ccsd_spatial.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/cc_diis.h"

namespace ATV {
namespace cpu {
namespace scf {

CCSDResult ccsd_energy_spatial(const internal::IntegralView& ints,
                               const SlaterDeterminant& det, double e_scf_ha,
                               double tol, int max_iter, bool use_diis) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0)
        throw std::invalid_argument(
            "ccsd_energy_spatial: ERI is not materialized; need "
            "compute_eri=true.");
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta))
        throw std::invalid_argument(
            "ccsd_energy_spatial: closed-shell reference only (n_alpha must "
            "equal n_beta).");
    if (det.generalized())
        throw std::invalid_argument(
            "ccsd_energy_spatial: GHF (spinor) reference is not supported.");
    if (max_iter < 1)
        throw std::invalid_argument(
            "ccsd_energy_spatial: max_iter must be ≥ 1.");
    if (!(tol > 0.0))
        throw std::invalid_argument("ccsd_energy_spatial: tol must be > 0.");

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int no = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int nv = n_mo - no;
    if (nv <= 0) {
        // No virtuals → no excitations → zero correlation (e.g. He/STO-3G).
        return CCSDResult{0.0, e_scf_ha, 0, true};
    }

    const xt::xtensor<double, 4> eri = ao2mo_full(ints.ERI, C, n_mo);
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);

    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);

    // Chemist-notation MO ERI accessors (occ i∈[0,no), vir a → mo index no+a).
    auto OO = [&](int i, int j, int k, int l) -> double {
        return eri(i, j, k, l);
    };
    auto OVOV = [&](int i, int a, int j, int b) -> double {
        return eri(i, no + a, j, no + b);
    };
    auto OOVV = [&](int i, int j, int a, int b) -> double {
        return eri(i, j, no + a, no + b);
    };
    auto OVVO = [&](int i, int a, int b, int j) -> double {
        return eri(i, no + a, no + b, j);
    };
    auto OVVV = [&](int i, int a, int b, int c) -> double {
        return eri(i, no + a, no + b, no + c);
    };
    auto OVOO = [&](int i, int a, int j, int k) -> double {
        return eri(i, no + a, j, k);
    };
    auto VVVV = [&](int a, int b, int c, int d) -> double {
        return eri(no + a, no + b, no + c, no + d);
    };

    // Orbital-energy denominators (strictly negative for an aufbau reference).
    std::vector<double> eo(No), ev(Nv);
    for (int i = 0; i < no; ++i)
        eo[static_cast<std::size_t>(i)] = eps(static_cast<std::size_t>(i));
    for (int a = 0; a < nv; ++a)
        ev[static_cast<std::size_t>(a)] = eps(static_cast<std::size_t>(no + a));

    xt::xtensor<double, 2> eia = xt::zeros<double>({No, Nv});
    for (int i = 0; i < no; ++i)
        for (int a = 0; a < nv; ++a) {
            const double d = eo[static_cast<std::size_t>(i)]
                             - ev[static_cast<std::size_t>(a)];
            if (d >= 0.0)
                throw std::invalid_argument(
                    "ccsd_energy_spatial: non-negative T1 denominator; "
                    "reference is degenerate or non-aufbau.");
            eia(i, a) = d;
        }
    xt::xtensor<double, 4> eijab = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    eijab(i, j, a, b) = eia(i, a) + eia(j, b);

    // Amplitudes. Initial guess: t1 = 0, t2_ijab = (ia|jb)/D_ijab (MP2).
    xt::xtensor<double, 2> t1 = xt::zeros<double>({No, Nv});
    xt::xtensor<double, 4> t2 = xt::zeros<double>({No, No, Nv, Nv});
    for (int i = 0; i < no; ++i)
        for (int j = 0; j < no; ++j)
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b < nv; ++b)
                    t2(i, j, a, b) = OVOV(i, a, j, b) / eijab(i, j, a, b);

    // E_corr = Σ_ijab (t2 + t1⊗t1)_ijab · [2(ia|jb) − (ib|ja)].
    auto energy_of = [&](const xt::xtensor<double, 2>& T1,
                         const xt::xtensor<double, 4>& T2) -> double {
        double e = 0.0;
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        const double tau = T2(i, j, a, b) + T1(i, a) * T1(j, b);
                        const double L =
                            2.0 * OVOV(i, a, j, b) - OVOV(i, b, j, a);
                        e += tau * L;
                    }
        return e;
    };

    internal::CcAmplitudeDiis diis(8);
    const std::size_t n1 = No * Nv;
    const std::size_t n2 = No * No * Nv * Nv;
    std::vector<double> amp(n1 + n2), err(n1 + n2);

    int iters = 0;
    bool ok = false;
    for (int it = 0; it < max_iter; ++it) {
        // ── τ = t2 + t1⊗t1 ──────────────────────────────────────────────
        xt::xtensor<double, 4> tau = xt::zeros<double>({No, No, Nv, Nv});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        tau(i, j, a, b) = t2(i, j, a, b) + t1(i, a) * t1(j, b);

        // ── Fock-like intermediates (fov = 0; energy on the denominators) ─
        // Fvv_bar(a,c), Foo_bar(k,i): the correlation part only (the diagonal
        // ε is moved into the amplitude denominators).
        xt::xtensor<double, 2> Fvv = xt::zeros<double>({Nv, Nv}); // Fvv_bar
        xt::xtensor<double, 2> Foo = xt::zeros<double>({No, No}); // Foo_bar
        xt::xtensor<double, 2> Fov = xt::zeros<double>({No, Nv});
        for (int a = 0; a < nv; ++a)
            for (int c = 0; c < nv; ++c) {
                double s = 0.0;
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int d = 0; d < nv; ++d) {
                            const double t2klad = t2(k, l, a, d);
                            const double tt = t1(k, a) * t1(l, d);
                            s += -2.0 * OVOV(k, c, l, d) * t2klad
                                 + OVOV(k, d, l, c) * t2klad
                                 + -2.0 * OVOV(k, c, l, d) * tt
                                 + OVOV(k, d, l, c) * tt;
                        }
                Fvv(a, c) = s;
            }
        for (int k = 0; k < no; ++k)
            for (int i = 0; i < no; ++i) {
                double s = 0.0;
                for (int l = 0; l < no; ++l)
                    for (int c = 0; c < nv; ++c)
                        for (int d = 0; d < nv; ++d) {
                            const double t2ilcd = t2(i, l, c, d);
                            const double tt = t1(i, c) * t1(l, d);
                            s += 2.0 * OVOV(k, c, l, d) * t2ilcd
                                 - OVOV(k, d, l, c) * t2ilcd
                                 + 2.0 * OVOV(k, c, l, d) * tt
                                 - OVOV(k, d, l, c) * tt;
                        }
                Foo(k, i) = s;
            }
        for (int k = 0; k < no; ++k)
            for (int c = 0; c < nv; ++c) {
                double s = 0.0;
                for (int l = 0; l < no; ++l)
                    for (int d = 0; d < nv; ++d)
                        s += 2.0 * OVOV(k, c, l, d) * t1(l, d)
                             - OVOV(k, d, l, c) * t1(l, d);
                Fov(k, c) = s;
            }

        // Lvv_bar(a,c) = Fvv_bar + 2·(kd|ac)t1 − (kc|ad)t1;
        // Loo_bar(k,i) = Foo_bar + 2·(lc|ki)t1 − (kc|li)t1  (ovoo accessors).
        xt::xtensor<double, 2> Lvv = Fvv;
        xt::xtensor<double, 2> Loo = Foo;
        for (int a = 0; a < nv; ++a)
            for (int c = 0; c < nv; ++c) {
                double s = 0.0;
                for (int k = 0; k < no; ++k)
                    for (int d = 0; d < nv; ++d)
                        s += 2.0 * OVVV(k, d, a, c) * t1(k, d)
                             - OVVV(k, c, a, d) * t1(k, d);
                Lvv(a, c) += s;
            }
        for (int k = 0; k < no; ++k)
            for (int i = 0; i < no; ++i) {
                double s = 0.0;
                for (int l = 0; l < no; ++l)
                    for (int c = 0; c < nv; ++c)
                        s += 2.0 * OVOO(l, c, k, i) * t1(l, c)
                             - OVOO(k, c, l, i) * t1(l, c);
                Loo(k, i) += s;
            }

        // ── W intermediates ─────────────────────────────────────────────
        // Woooo(k,l,i,j)
        xt::xtensor<double, 4> Woooo = xt::zeros<double>({No, No, No, No});
        for (int k = 0; k < no; ++k)
            for (int l = 0; l < no; ++l)
                for (int i = 0; i < no; ++i)
                    for (int j = 0; j < no; ++j) {
                        double s = OO(k, i, l, j);
                        for (int c = 0; c < nv; ++c) {
                            s += OVOO(l, c, k, i) * t1(j, c);
                            s += OVOO(k, c, l, j) * t1(i, c);
                        }
                        for (int c = 0; c < nv; ++c)
                            for (int d = 0; d < nv; ++d)
                                s += OVOV(k, c, l, d)
                                     * (t2(i, j, c, d) + t1(i, c) * t1(j, d));
                        Woooo(k, l, i, j) = s;
                    }

        // Wvoov(a,k,i,c)
        xt::xtensor<double, 4> Wvoov = xt::zeros<double>({Nv, No, No, Nv});
        for (int a = 0; a < nv; ++a)
            for (int k = 0; k < no; ++k)
                for (int i = 0; i < no; ++i)
                    for (int c = 0; c < nv; ++c) {
                        double s = OVVO(k, c, a, i);
                        for (int d = 0; d < nv; ++d)
                            s += OVVV(k, c, a, d) * t1(i, d);
                        for (int l = 0; l < no; ++l)
                            s -= OVOO(k, c, l, i) * t1(l, a);
                        for (int l = 0; l < no; ++l)
                            for (int d = 0; d < nv; ++d) {
                                s -= 0.5 * OVOV(l, d, k, c) * t2(i, l, d, a);
                                s -= 0.5 * OVOV(l, c, k, d) * t2(i, l, a, d);
                                s -= OVOV(l, d, k, c) * t1(i, d) * t1(l, a);
                                s += OVOV(l, d, k, c) * t2(i, l, a, d);
                            }
                        Wvoov(a, k, i, c) = s;
                    }

        // Wvovo(a,k,c,i)
        xt::xtensor<double, 4> Wvovo = xt::zeros<double>({Nv, No, Nv, No});
        for (int a = 0; a < nv; ++a)
            for (int k = 0; k < no; ++k)
                for (int c = 0; c < nv; ++c)
                    for (int i = 0; i < no; ++i) {
                        double s = OOVV(k, i, a, c);
                        for (int d = 0; d < nv; ++d)
                            s += OVVV(k, d, a, c) * t1(i, d);
                        for (int l = 0; l < no; ++l)
                            s -= OVOO(l, c, k, i) * t1(l, a);
                        for (int l = 0; l < no; ++l)
                            for (int d = 0; d < nv; ++d) {
                                s -= 0.5 * OVOV(l, c, k, d) * t2(i, l, d, a);
                                s -= OVOV(l, c, k, d) * t1(i, d) * t1(l, a);
                            }
                        Wvovo(a, k, c, i) = s;
                    }

        // Wvvvv(a,b,c,d)
        xt::xtensor<double, 4> Wvvvv = xt::zeros<double>({Nv, Nv, Nv, Nv});
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                for (int c = 0; c < nv; ++c)
                    for (int d = 0; d < nv; ++d) {
                        double s = VVVV(a, c, b, d);
                        for (int k = 0; k < no; ++k)
                            s += -OVVV(k, d, a, c) * t1(k, b)
                                 - OVVV(k, c, b, d) * t1(k, a);
                        Wvvvv(a, b, c, d) = s;
                    }

        // ── T1 residual (before division) ───────────────────────────────
        xt::xtensor<double, 2> t1new = xt::zeros<double>({No, Nv});
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a) {
                double s = 0.0;
                for (int c = 0; c < nv; ++c)
                    s += Fvv(a, c) * t1(i, c);
                for (int k = 0; k < no; ++k)
                    s -= Foo(k, i) * t1(k, a);
                for (int k = 0; k < no; ++k)
                    for (int c = 0; c < nv; ++c)
                        s += 2.0 * Fov(k, c) * t2(k, i, c, a)
                             - Fov(k, c) * t2(i, k, c, a)
                             + Fov(k, c) * t1(i, c) * t1(k, a);
                for (int k = 0; k < no; ++k)
                    for (int c = 0; c < nv; ++c)
                        s += 2.0 * OVVO(k, c, a, i) * t1(k, c)
                             - OOVV(k, i, a, c) * t1(k, c);
                for (int k = 0; k < no; ++k)
                    for (int c = 0; c < nv; ++c)
                        for (int d = 0; d < nv; ++d) {
                            const double t2ikcd = t2(i, k, c, d);
                            s += 2.0 * OVVV(k, d, a, c) * t2ikcd
                                 - OVVV(k, c, a, d) * t2ikcd;
                            s += 2.0 * OVVV(k, d, a, c) * t1(k, d) * t1(i, c)
                                 - OVVV(k, c, a, d) * t1(k, d) * t1(i, c);
                        }
                for (int k = 0; k < no; ++k)
                    for (int l = 0; l < no; ++l)
                        for (int c = 0; c < nv; ++c) {
                            const double t2klac = t2(k, l, a, c);
                            s += -2.0 * OVOO(l, c, k, i) * t2klac
                                 + OVOO(k, c, l, i) * t2klac;
                            s += -2.0 * OVOO(l, c, k, i) * t1(l, c) * t1(k, a)
                                 + OVOO(k, c, l, i) * t1(l, c) * t1(k, a);
                        }
                t1new(i, a) = s / eia(i, a);
            }

        // ── T2 residual (before division) ───────────────────────────────
        xt::xtensor<double, 4> t2new = xt::zeros<double>({No, No, Nv, Nv});
        // Part A/B singles-driven terms need a per-(i,j,a,b) build with the
        // P(ia,jb) permutation folded in via the (j,i,b,a) transpose.
        // tmpA(i,j,a,b) = Σ_c [−(ki|bc)t1(k,a) + (ia|cb)] t1(j,c)
        // tmpB(i,j,a,b) = Σ_k [Σ_c (kc|ai)t1(j,c) + (ia|jk)] t1(k,b)
        xt::xtensor<double, 4> tmpA = xt::zeros<double>({No, No, Nv, Nv});
        xt::xtensor<double, 4> tmpB = xt::zeros<double>({No, No, Nv, Nv});
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        double sa = 0.0;
                        for (int c = 0; c < nv; ++c) {
                            double tmp2 = OVVV(i, a, c, b);
                            for (int k = 0; k < no; ++k)
                                tmp2 -= OOVV(k, i, b, c) * t1(k, a);
                            sa += tmp2 * t1(j, c);
                        }
                        tmpA(i, j, a, b) = sa;
                        double sb = 0.0;
                        for (int k = 0; k < no; ++k) {
                            double tmp2 = OVOO(i, a, j, k);
                            for (int c = 0; c < nv; ++c)
                                tmp2 += OVVO(k, c, a, i) * t1(j, c);
                            sb += tmp2 * t1(k, b);
                        }
                        tmpB(i, j, a, b) = sb;
                    }

        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        // driver (ia|jb) + singles-driven A/B (P-symmetrized)
                        double s = OVOV(i, a, j, b);
                        s += tmpA(i, j, a, b) + tmpA(j, i, b, a);
                        s -= tmpB(i, j, a, b) + tmpB(j, i, b, a);

                        // Woooo·τ  and  Wvvvv·τ
                        for (int k = 0; k < no; ++k)
                            for (int l = 0; l < no; ++l)
                                s += Woooo(k, l, i, j) * tau(k, l, a, b);
                        for (int c = 0; c < nv; ++c)
                            for (int d = 0; d < nv; ++d)
                                s += Wvvvv(a, b, c, d) * tau(i, j, c, d);

                        // Lvv / Loo dressings (P-symmetrized)
                        double fvvp = 0.0, fvvq = 0.0;
                        for (int c = 0; c < nv; ++c) {
                            fvvp += Lvv(a, c) * t2(i, j, c, b);
                            fvvq += Lvv(b, c) * t2(j, i, c, a);
                        }
                        s += fvvp + fvvq;
                        double foop = 0.0, fooq = 0.0;
                        for (int k = 0; k < no; ++k) {
                            foop += Loo(k, i) * t2(k, j, a, b);
                            fooq += Loo(k, j) * t2(k, i, b, a);
                        }
                        s -= foop + fooq;

                        // ring terms (P-symmetrized)
                        double h_ij = 0.0, h_ji = 0.0;
                        double i_ij = 0.0, i_ji = 0.0;
                        double j_ij = 0.0, j_ji = 0.0;
                        for (int k = 0; k < no; ++k)
                            for (int c = 0; c < nv; ++c) {
                                h_ij += 2.0 * Wvoov(a, k, i, c) * t2(k, j, c, b)
                                        - Wvovo(a, k, c, i) * t2(k, j, c, b);
                                h_ji += 2.0 * Wvoov(b, k, j, c) * t2(k, i, c, a)
                                        - Wvovo(b, k, c, j) * t2(k, i, c, a);
                                i_ij += Wvoov(a, k, i, c) * t2(k, j, b, c);
                                i_ji += Wvoov(b, k, j, c) * t2(k, i, a, c);
                                j_ij += Wvovo(b, k, c, i) * t2(k, j, a, c);
                                j_ji += Wvovo(a, k, c, j) * t2(k, i, b, c);
                            }
                        s += h_ij + h_ji;
                        s -= i_ij + i_ji;
                        s -= j_ij + j_ji;

                        t2new(i, j, a, b) = s / eijab(i, j, a, b);
                    }

        // ── convergence + DIIS extrapolation ────────────────────────────
        double dt1 = 0.0, dt2 = 0.0;
        std::size_t p = 0;
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a) {
                const double d = t1new(i, a) - t1(i, a);
                dt1 = std::max(dt1, std::abs(d));
                amp[p] = t1new(i, a);
                err[p] = d;
                ++p;
            }
        for (int i = 0; i < no; ++i)
            for (int j = 0; j < no; ++j)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b) {
                        const double d = t2new(i, j, a, b) - t2(i, j, a, b);
                        dt2 = std::max(dt2, std::abs(d));
                        amp[p] = t2new(i, j, a, b);
                        err[p] = d;
                        ++p;
                    }

        if (use_diis) {
            const std::vector<double> ext = diis.step(amp, err);
            p = 0;
            for (int i = 0; i < no; ++i)
                for (int a = 0; a < nv; ++a)
                    t1(i, a) = ext[p++];
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            t2(i, j, a, b) = ext[p++];
        } else {
            t1 = t1new;
            t2 = t2new;
        }

        iters = it + 1;
        if (std::max(dt1, dt2) < tol) {
            ok = true;
            break;
        }
    }

    if (!ok)
        throw std::invalid_argument(
            "ccsd_energy_spatial: amplitudes did not converge in "
            + std::to_string(max_iter)
            + " iterations (tol=" + std::to_string(tol) + ").");

    const double e_corr = energy_of(t1, t2);
    return CCSDResult{e_corr, e_scf_ha + e_corr, iters, true};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
