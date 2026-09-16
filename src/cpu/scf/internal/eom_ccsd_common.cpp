/// @file cpu/scf/internal/eom_ccsd_common.cpp
/// @brief Implementation of the shared spin-orbital EOM-CCSD pieces
///        (see internal/eom_ccsd_common.h): make_tau, the full Gauss–Stanton
///        intermediates (as the EomImds computing constructor), and the dense
///        dgeev glue.
///
/// The intermediate math is byte-identical to the former per-TU `build_imds`
/// (EE == EA verbatim; IP is a subset of the same expressions). It is now built
/// exactly once, in the EomImds constructor.

#include "atomiverse/cpu/scf/internal/eom_ccsd_common.h"

#include <cstddef>
#include <utility>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {
using T2 = xt::xtensor<double, 4>;
using T1 = xt::xtensor<double, 2>;
} // namespace

xt::xtensor<double, 4> make_tau(const xt::xtensor<double, 4>& t2,
                                const xt::xtensor<double, 2>& t1, double fac) {
    const std::size_t no = t1.shape(0);
    const std::size_t nv = t1.shape(1);
    xt::xtensor<double, 4> tau = t2;
    const double c = 0.5 * fac;
    for (std::size_t i = 0; i < no; ++i)
        for (std::size_t j = 0; j < no; ++j)
            for (std::size_t a = 0; a < nv; ++a)
                for (std::size_t b = 0; b < nv; ++b) {
                    const double t1t1_ijab = c * t1(i, a) * t1(j, b);
                    const double t1t1_jiab = c * t1(j, a) * t1(i, b);
                    const double a_ij = t1t1_ijab - t1t1_jiab;
                    const double t1t1_ijba = c * t1(i, b) * t1(j, a);
                    const double t1t1_jiba = c * t1(j, b) * t1(i, a);
                    const double a_ij_ba = t1t1_ijba - t1t1_jiba;
                    tau(i, j, a, b) += a_ij - a_ij_ba;
                }
    return tau;
}

EomImds::EomImds(const CCSDSolution& sol) {
    const int no = sol.no;
    const int nv = sol.nv;
    const std::size_t No = static_cast<std::size_t>(no);
    const std::size_t Nv = static_cast<std::size_t>(nv);
    const T1& t1 = sol.t1;
    const T2& t2 = sol.t2;
    const T2& oooo = sol.oooo;
    const T2& ooov = sol.ooov;
    const T2& oovv = sol.oovv;
    const T2& ovov = sol.ovov;
    const T2& ovvv = sol.ovvv;
    const T2& vvvv = sol.vvvv;
    const T1& foo = sol.foo;
    const T1& fvv = sol.fvv;
    const T1& fov = sol.fov;

    const T2 tau = make_tau(t2, t1, 1.0);
    const T2 tau_h = make_tau(t2, t1, 0.5);

    // --- cc_F* (section a) -------------------------------------------------
    // Fae = fvv - 0.5 fov·t1 + t1·ovvv^T - 0.5 tau_h·oovv
    T1 ccFvv = fvv;
    for (int a = 0; a < nv; ++a)
        for (int e = 0; e < nv; ++e) {
            double s = ccFvv(a, e);
            for (int m = 0; m < no; ++m)
                s -= 0.5 * fov(m, e) * t1(m, a);
            for (int m = 0; m < no; ++m)
                for (int f = 0; f < nv; ++f)
                    s += t1(m, f) * ovvv(m, a, f, e); // amef = ovvv(m,a,f,e)
            for (int m = 0; m < no; ++m)
                for (int n = 0; n < no; ++n)
                    for (int f = 0; f < nv; ++f)
                        s -= 0.5 * tau_h(m, n, a, f) * oovv(m, n, e, f);
            ccFvv(a, e) = s;
        }
    T1 ccFoo = foo;
    for (int m = 0; m < no; ++m)
        for (int i = 0; i < no; ++i) {
            double s = ccFoo(m, i);
            for (int e = 0; e < nv; ++e)
                s += 0.5 * fov(m, e) * t1(i, e);
            for (int n = 0; n < no; ++n)
                for (int e = 0; e < nv; ++e)
                    s += t1(n, e) * ooov(m, n, i, e);
            for (int n = 0; n < no; ++n)
                for (int e = 0; e < nv; ++e)
                    for (int f = 0; f < nv; ++f)
                        s += 0.5 * tau_h(i, n, e, f) * oovv(m, n, e, f);
            ccFoo(m, i) = s;
        }
    T1 ccFov = fov;
    for (int m = 0; m < no; ++m)
        for (int e = 0; e < nv; ++e) {
            double s = ccFov(m, e);
            for (int n = 0; n < no; ++n)
                for (int f = 0; f < nv; ++f)
                    s += t1(n, f) * oovv(m, n, e, f);
            ccFov(m, e) = s;
        }

    // EOM F = cc_F with extra ½ t1 contractions
    T1 Fvv = ccFvv;
    for (int a = 0; a < nv; ++a)
        for (int e = 0; e < nv; ++e) {
            double s = Fvv(a, e);
            for (int m = 0; m < no; ++m)
                s -= 0.5 * t1(m, a) * ccFov(m, e);
            Fvv(a, e) = s;
        }
    T1 Foo = ccFoo;
    for (int m = 0; m < no; ++m)
        for (int i = 0; i < no; ++i) {
            double s = Foo(m, i);
            for (int e = 0; e < nv; ++e)
                s += 0.5 * t1(i, e) * ccFov(m, e);
            Foo(m, i) = s;
        }
    T1 Fov = ccFov;

    // cc_Woooo
    T2 ccWoooo = oooo;
    for (int m = 0; m < no; ++m)
        for (int n = 0; n < no; ++n)
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j) {
                    double s = ccWoooo(m, n, i, j);
                    for (int e = 0; e < nv; ++e) {
                        s += t1(j, e) * ooov(m, n, i, e);
                        s -= t1(i, e) * ooov(m, n, j, e);
                    }
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.25 * tau(i, j, e, f) * oovv(m, n, e, f);
                    ccWoooo(m, n, i, j) = s;
                }
    // Woooo = cc_Woooo + 0.25 tau oovv
    T2 Woooo = ccWoooo;
    for (int m = 0; m < no; ++m)
        for (int n = 0; n < no; ++n)
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j) {
                    double s = Woooo(m, n, i, j);
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.25 * tau(i, j, e, f) * oovv(m, n, e, f);
                    Woooo(m, n, i, j) = s;
                }

    // cc_Wvvvv
    T2 ccWvvvv = vvvv;
    for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
            for (int e = 0; e < nv; ++e)
                for (int f = 0; f < nv; ++f) {
                    double s = ccWvvvv(a, b, e, f);
                    double tmp_baef = 0.0, tmp_abef = 0.0;
                    for (int m = 0; m < no; ++m) {
                        tmp_baef += t1(m, b) * ovvv(m, a, f, e);
                        tmp_abef += t1(m, a) * ovvv(m, b, f, e);
                    }
                    s += -tmp_baef + tmp_abef;
                    for (int m = 0; m < no; ++m)
                        for (int n = 0; n < no; ++n)
                            s += 0.25 * tau(m, n, a, b) * oovv(m, n, e, f);
                    ccWvvvv(a, b, e, f) = s;
                }
    // Wvvvv = cc_Wvvvv + 0.25 tau oovv
    T2 Wvvvv = ccWvvvv;
    for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
            for (int e = 0; e < nv; ++e)
                for (int f = 0; f < nv; ++f) {
                    double s = Wvvvv(a, b, e, f);
                    for (int m = 0; m < no; ++m)
                        for (int n = 0; n < no; ++n)
                            s += 0.25 * tau(m, n, a, b) * oovv(m, n, e, f);
                    Wvvvv(a, b, e, f) = s;
                }

    // cc_Wovvo: Wmbej = ovvo + t1·ovvv - t1·oovo - 0.5 t2 oovv - t1 t1 oovv
    // ovvo(m,b,e,j) = -ovov(m,b,j,e)
    T2 ccWovvo = xt::zeros<double>({No, Nv, Nv, No});
    for (int m = 0; m < no; ++m)
        for (int b = 0; b < nv; ++b)
            for (int e = 0; e < nv; ++e)
                for (int j = 0; j < no; ++j) {
                    double s = -ovov(m, b, j, e);
                    for (int f = 0; f < nv; ++f)
                        s += t1(j, f) * ovvv(m, b, e, f);
                    for (int n = 0; n < no; ++n)
                        s -= t1(n, b) * (-ooov(m, n, j, e)); // oovo = -ooov
                    for (int n = 0; n < no; ++n)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * t2(j, n, f, b) * oovv(m, n, e, f);
                    for (int n = 0; n < no; ++n)
                        for (int f = 0; f < nv; ++f)
                            s -= t1(j, f) * t1(n, b) * oovv(m, n, e, f);
                    ccWovvo(m, b, e, j) = s;
                }
    // Wovvo = cc_Wovvo - 0.5 t2 oovv
    T2 Wovvo = ccWovvo;
    for (int m = 0; m < no; ++m)
        for (int b = 0; b < nv; ++b)
            for (int e = 0; e < nv; ++e)
                for (int j = 0; j < no; ++j) {
                    double s = Wovvo(m, b, e, j);
                    for (int n = 0; n < no; ++n)
                        for (int f = 0; f < nv; ++f)
                            s -= 0.5 * t2(j, n, f, b) * oovv(m, n, e, f);
                    Wovvo(m, b, e, j) = s;
                }

    // Wooov = ooov + t1·oovv
    T2 Wooov = ooov;
    for (int m = 0; m < no; ++m)
        for (int n = 0; n < no; ++n)
            for (int i = 0; i < no; ++i)
                for (int e = 0; e < nv; ++e) {
                    double s = Wooov(m, n, i, e);
                    for (int f = 0; f < nv; ++f)
                        s += t1(i, f) * oovv(m, n, f, e); // mnfe with f,e
                    Wooov(m, n, i, e) = s;
                }

    // Wvovv: Wamef = -t1·oovv - ovvv.T(1,0,2,3)
    // Wamef[a,m,e,f] = -Σ_n t1(n,a) oovv(n,m,e,f) - ovvv(m,a,e,f)
    T2 Wvovv = xt::zeros<double>({Nv, No, Nv, Nv});
    for (int a = 0; a < nv; ++a)
        for (int m = 0; m < no; ++m)
            for (int e = 0; e < nv; ++e)
                for (int f = 0; f < nv; ++f) {
                    double s = -ovvv(m, a, e, f);
                    for (int n = 0; n < no; ++n)
                        s -= t1(n, a) * oovv(n, m, e, f);
                    Wvovv(a, m, e, f) = s;
                }

    // Wovoo (Wmbij) — full formula from gintermediates.Wovoo
    T2 Wovoo = xt::zeros<double>({No, Nv, No, No});
    for (int m = 0; m < no; ++m)
        for (int b = 0; b < nv; ++b)
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j) {
                    // eris.ooov.conj().T(2,3,0,1) → ooov(i,j,m,b) for real
                    double s = ooov(i, j, m, b);
                    // -Fov · t2
                    for (int e = 0; e < nv; ++e)
                        s -= Fov(m, e) * t2(i, j, b, e); // ijbe with e→ last
                    // -t1 · Woooo
                    for (int n = 0; n < no; ++n)
                        s -= t1(n, b) * Woooo(m, n, i, j);
                    // 0.5 ovvv · tau
                    for (int e = 0; e < nv; ++e)
                        for (int f = 0; f < nv; ++f)
                            s += 0.5 * ovvv(m, b, e, f) * tau(i, j, e, f);
                    // tmp1 = ooov · t2 : Σ_ne ooov(m,n,i,e) t2(j,n,b,e)  P(ij)
                    double t1a = 0.0, t1b = 0.0;
                    for (int n = 0; n < no; ++n)
                        for (int e = 0; e < nv; ++e) {
                            t1a += ooov(m, n, i, e) * t2(j, n, b, e);
                            t1b += ooov(m, n, j, e) * t2(i, n, b, e);
                        }
                    s += t1a - t1b;
                    // tmp2 = t1 · ovvo − t1 t2 oovv : P(ij)
                    // ovvo(m,b,e,j) = -ovov(m,b,j,e)
                    double u1 = 0.0, u2 = 0.0;
                    for (int e = 0; e < nv; ++e) {
                        u1 += t1(i, e) * (-ovov(m, b, j, e));
                        u2 += t1(j, e) * (-ovov(m, b, i, e));
                    }
                    for (int n = 0; n < no; ++n)
                        for (int e = 0; e < nv; ++e)
                            for (int f = 0; f < nv; ++f) {
                                u1 -= t1(i, e) * t2(n, j, b, f)
                                      * oovv(m, n, e, f);
                                u2 -= t1(j, e) * t2(n, i, b, f)
                                      * oovv(m, n, e, f);
                            }
                    s += u1 - u2;
                    Wovoo(m, b, i, j) = s;
                }

    // Wvvvo (Wabei)
    T2 Wvvvo = xt::zeros<double>({Nv, Nv, Nv, No});
    for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
            for (int e = 0; e < nv; ++e)
                for (int i = 0; i < no; ++i) {
                    // eris_vvvo = -ovvv.transpose(2,3,1,0)
                    // eris_vvvo[a,b,e,i] = -ovvv(i,e,a,b)
                    double s = -ovvv(i, e, a, b);

                    // 0.5 oovo · tau : oovo(m,n,e,i) = -ooov(m,n,i,e)
                    for (int m = 0; m < no; ++m)
                        for (int n = 0; n < no; ++n)
                            s += 0.5 * (-ooov(m, n, i, e)) * tau(m, n, a, b);
                    // -Fov · t2
                    for (int m = 0; m < no; ++m)
                        s -= Fov(m, e) * t2(m, i, a, b);
                    // -P(ab) ovvv · t2 : tmp1 = Σ_mf ovvv(m,b,e,f) t2(m,i,a,f)
                    double v1 = 0.0, v2 = 0.0;
                    for (int m = 0; m < no; ++m)
                        for (int f = 0; f < nv; ++f) {
                            v1 += ovvv(m, b, e, f) * t2(m, i, a, f);
                            v2 += ovvv(m, a, e, f) * t2(m, i, b, f);
                        }
                    s -= v1 - v2;
                    // -P(ab) (t1·ovvo − t1 t2 oovv)
                    // ovvo(m,b,e,i) = -ovov(m,b,i,e)
                    double w1 = 0.0, w2 = 0.0;
                    for (int m = 0; m < no; ++m) {
                        w1 += t1(m, a) * (-ovov(m, b, i, e));
                        w2 += t1(m, b) * (-ovov(m, a, i, e));
                    }
                    for (int m = 0; m < no; ++m)
                        for (int n = 0; n < no; ++n)
                            for (int f = 0; f < nv; ++f) {
                                w1 -= t1(m, a) * t2(n, i, b, f)
                                      * oovv(m, n, e, f);
                                w2 -= t1(m, b) * t2(n, i, a, f)
                                      * oovv(m, n, e, f);
                            }
                    s -= w1 - w2;
                    // + Wvvvv · t1
                    for (int f = 0; f < nv; ++f)
                        s += Wvvvv(a, b, e, f) * t1(i, f);
                    Wvvvo(a, b, e, i) = s;
                }

    this->Foo = std::move(Foo);
    this->Fvv = std::move(Fvv);
    this->Fov = std::move(Fov);
    this->Wovvo = std::move(Wovvo);
    this->Woovv = oovv;
    this->Woooo = std::move(Woooo);
    this->Wooov = std::move(Wooov);
    this->Wovoo = std::move(Wovoo);
    this->Wvovv = std::move(Wvovv);
    this->Wvvvv = std::move(Wvvvv);
    this->Wvvvo = std::move(Wvvvo);
    this->t1 = t1;
    this->t2 = t2;
    this->no = no;
    this->nv = nv;
}

std::vector<EigPair> eig_dense(const std::vector<std::vector<double>>& H,
                               const char* label) {
    const int n = static_cast<int>(H.size());
    if (n == 0)
        return {};
    // RealMatrix is column-major, so A(i,j) = H[i][j] reproduces the old
    // hand-rolled column-major buffer bit-for-bit under dgeev.
    linalg::RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    linalg::RealVector wr, wi;
    linalg::eig_nonsymmetric(A, wr, wi, nullptr, label);
    std::vector<EigPair> out(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        out[static_cast<std::size_t>(j)].wr = wr(static_cast<std::size_t>(j));
        out[static_cast<std::size_t>(j)].wi = wi(static_cast<std::size_t>(j));
    }
    return out;
}

std::vector<EigPair> eig_dense(const std::vector<std::vector<double>>& H,
                               const char* label, bool want_vectors) {
    const int n = static_cast<int>(H.size());
    if (n == 0)
        return {};
    // RealMatrix is column-major, so A(i,j) = H[i][j] reproduces the old
    // hand-rolled column-major buffer bit-for-bit under dgeev. eig_nonsymmetric
    // returns vr with column j = right eigenvector j, matching the old
    // vr[i + j*n] read.
    linalg::RealMatrix A = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    linalg::RealVector wr, wi;
    linalg::RealMatrix vr;
    linalg::RealMatrix* vr_ptr = want_vectors ? &vr : nullptr;
    linalg::eig_nonsymmetric(A, wr, wi, vr_ptr, label);
    std::vector<EigPair> out(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        out[static_cast<std::size_t>(j)].wr = wr(static_cast<std::size_t>(j));
        out[static_cast<std::size_t>(j)].wi = wi(static_cast<std::size_t>(j));
        if (want_vectors) {
            out[static_cast<std::size_t>(j)].vr.resize(
                static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                out[static_cast<std::size_t>(j)]
                    .vr[static_cast<std::size_t>(i)] = vr(
                    static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
