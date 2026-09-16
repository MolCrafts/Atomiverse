/// @file cpu/scf/internal/orbital_hessian.cpp
/// @brief OrbitalHessian — occ-virt MO-ERI transform (constructor) plus the
///        σ± products, dense builds, and Davidson diagonal per channel.

#include "atomiverse/cpu/scf/internal/orbital_hessian.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {
/// Channel weight of the Coulomb term 4(ai|bj) in (A+B): 4 for singlet,
/// 0 for triplet. The exchange terms −(ij|ab) − (aj|bi) are shared.
double coulomb_weight(Channel ch) {
    return (ch == Channel::singlet) ? 4.0 : 0.0;
}

/// (A−B) positivity floor [Hartree] — shared by the pre-refactor rpa.cpp /
/// g0w0.cpp file-local sym_power copies (both 1e-10, verified identical).
constexpr double kAmBFloor = 1e-10;
/// |ω²| zero / instability guard [Hartree²] — identical in rpa.cpp/g0w0.cpp.
constexpr double kOmega2Tol = 1e-10;
} // namespace

OrbitalHessian::OrbitalHessian(const SlaterDeterminant& det,
                               const xt::xtensor<double, 4>& eri_ao) {
    if (!det.restricted())
        throw std::invalid_argument(
            "OrbitalHessian: requires a restricted (closed-shell RHF) "
            "determinant — UHF-based stability is out of scope here");

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const int n = static_cast<int>(C.shape()[0]);
    const int n_mo = static_cast<int>(C.shape()[1]);

    // Integer-occupation split into occupied / virtual column lists (the
    // orbital Hessian of a fractional-occupation state is not a single-
    // determinant Hessian — refuse, no fallback).
    std::vector<int> occ, virt;
    occ.reserve(n_mo);
    virt.reserve(n_mo);
    for (int p = 0; p < n_mo; ++p) {
        const double fp = f(p);
        if (fp == 1.0)
            occ.push_back(p);
        else if (fp == 0.0)
            virt.push_back(p);
        else
            throw std::invalid_argument(
                "OrbitalHessian: fractional occupation f[" + std::to_string(p)
                + "] = " + std::to_string(fp)
                + " — the A/B kernel is defined for integer occupations");
    }
    no_ = static_cast<int>(occ.size());
    nv_ = static_cast<int>(virt.size());
    if (no_ == 0)
        throw std::invalid_argument(
            "OrbitalHessian: no occupied orbitals — empty determinant");
    if (nv_ == 0)
        throw std::invalid_argument(
            "OrbitalHessian: no virtual orbitals (n_occ=" + std::to_string(no_)
            + ", n_virt=0) — there are no occ→virt rotations, the orbital "
              "Hessian is undefined (e.g. He/STO-3G: one basis function, "
              "one doubly occupied orbital)");
    if (static_cast<int>(eri_ao.shape()[0]) != n
        || static_cast<int>(eri_ao.shape()[1]) != n
        || static_cast<int>(eri_ao.shape()[2]) != n
        || static_cast<int>(eri_ao.shape()[3]) != n)
        throw std::invalid_argument(
            "OrbitalHessian: AO ERI dimension "
            + std::to_string(eri_ao.shape()[0]) + " != determinant AO "
            + "dimension " + std::to_string(n)
            + " (the dense ERI must be in the working AO dimension — "
              "spherical count for spherical-convention bases)");

    // Occupied / virtual coefficient blocks (row-major working copies for
    // the reshape-based quarter transform) + the matching energies.
    xt::xtensor<double, 2> Co = xt::zeros<double>({n, no_});
    xt::xtensor<double, 2> Cv = xt::zeros<double>({n, nv_});
    eps_occ_ = xt::zeros<double>({static_cast<std::size_t>(no_)});
    eps_virt_ = xt::zeros<double>({static_cast<std::size_t>(nv_)});
    for (int i = 0; i < no_; ++i) {
        eps_occ_(i) = eps(occ[static_cast<std::size_t>(i)]);
        for (int mu = 0; mu < n; ++mu)
            Co(mu, i) = C(mu, occ[static_cast<std::size_t>(i)]);
    }
    for (int a = 0; a < nv_; ++a) {
        eps_virt_(a) = eps(virt[static_cast<std::size_t>(a)]);
        for (int mu = 0; mu < n; ++mu)
            Cv(mu, a) = C(mu, virt[static_cast<std::size_t>(a)]);
    }

    // The one-time O(N⁵) quarter transforms: (ai|bj) and (ij|ab).
    ovov_ = transform_block_(eri_ao, Cv, Co, Cv, Co);
    oovv_ = transform_block_(eri_ao, Co, Co, Cv, Cv);
}

xt::xtensor<double, 4> OrbitalHessian::transform_block_(
    const xt::xtensor<double, 4>& eri_ao, const xt::xtensor<double, 2>& C0,
    const xt::xtensor<double, 2>& C1, const xt::xtensor<double, 2>& C2,
    const xt::xtensor<double, 2>& C3) {
    // Four rounds, each transforming the (current) last axis and rotating it
    // to the front: after transforming with C3, C2, C1, C0 the axis order is
    // back to chemist [p, q, r, s]. Same reshape/dot pattern as the
    // Cartesian→spherical ERI sandwich in integrals.cpp.
    xt::xtensor<double, 4> cur = eri_ao;
    const xt::xtensor<double, 2>* Cs[4] = {&C3, &C2, &C1, &C0};
    for (int round = 0; round < 4; ++round) {
        const auto sh = cur.shape();
        auto flat = xt::reshape_view(cur, {sh[0] * sh[1] * sh[2], sh[3]});
        xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
        xt::xtensor<double, 4> next =
            xt::reshape_view(h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
        cur = xt::transpose(next, {3, 0, 1, 2});
    }
    return cur;
}

RealVector OrbitalHessian::sigma_plus(const RealVector& v, Channel ch) const {
    if (static_cast<int>(v.size()) != n_ov())
        throw std::invalid_argument("OrbitalHessian::sigma_plus: v.size() = "
                                    + std::to_string(v.size())
                                    + " != n_ov = " + std::to_string(n_ov()));
    const double cw = coulomb_weight(ch);
    RealVector y = xt::zeros<double>({v.size()});
    for (int a = 0; a < nv_; ++a) {
        for (int i = 0; i < no_; ++i) {
            const int p = a * no_ + i;
            double acc = (eps_virt_(a) - eps_occ_(i)) * v(p);
            for (int b = 0; b < nv_; ++b) {
                for (int j = 0; j < no_; ++j) {
                    const int q = b * no_ + j;
                    acc += (cw * ovov_(a, i, b, j) - oovv_(i, j, a, b)
                            - ovov_(a, j, b, i))
                           * v(q);
                }
            }
            y(p) = acc;
        }
    }
    return y;
}

RealVector OrbitalHessian::sigma_minus(const RealVector& v, Channel ch) const {
    (void)ch; // (A − B) is channel-independent for real closed-shell orbitals
    if (static_cast<int>(v.size()) != n_ov())
        throw std::invalid_argument("OrbitalHessian::sigma_minus: v.size() = "
                                    + std::to_string(v.size())
                                    + " != n_ov = " + std::to_string(n_ov()));
    RealVector y = xt::zeros<double>({v.size()});
    for (int a = 0; a < nv_; ++a) {
        for (int i = 0; i < no_; ++i) {
            const int p = a * no_ + i;
            double acc = (eps_virt_(a) - eps_occ_(i)) * v(p);
            for (int b = 0; b < nv_; ++b) {
                for (int j = 0; j < no_; ++j) {
                    const int q = b * no_ + j;
                    acc += (-oovv_(i, j, a, b) + ovov_(a, j, b, i)) * v(q);
                }
            }
            y(p) = acc;
        }
    }
    return y;
}

RealMatrix OrbitalHessian::dense_plus(Channel ch) const {
    const double cw = coulomb_weight(ch);
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    RealMatrix M = xt::zeros<double>({nov, nov});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            for (int b = 0; b < nv_; ++b)
                for (int j = 0; j < no_; ++j) {
                    const int p = a * no_ + i;
                    const int q = b * no_ + j;
                    double val = cw * ovov_(a, i, b, j) - oovv_(i, j, a, b)
                                 - ovov_(a, j, b, i);
                    if (a == b && i == j)
                        val += eps_virt_(a) - eps_occ_(i);
                    M(p, q) = val;
                }
    return M;
}

RealMatrix OrbitalHessian::dense_minus(Channel ch) const {
    (void)ch; // (A − B) is channel-independent for real closed-shell orbitals
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    RealMatrix M = xt::zeros<double>({nov, nov});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            for (int b = 0; b < nv_; ++b)
                for (int j = 0; j < no_; ++j) {
                    const int p = a * no_ + i;
                    const int q = b * no_ + j;
                    double val = -oovv_(i, j, a, b) + ovov_(a, j, b, i);
                    if (a == b && i == j)
                        val += eps_virt_(a) - eps_occ_(i);
                    M(p, q) = val;
                }
    return M;
}

RealMatrix OrbitalHessian::dense_A(Channel ch, double alpha_hf) const {
    // Singlet A carries the Coulomb 2(ai|jb); triplet does not. Exact
    // exchange is scaled by alpha_hf on both channels (hyb = 1 → HF/CIS).
    const double coul = (ch == Channel::singlet) ? 2.0 : 0.0;
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    RealMatrix A = xt::zeros<double>({nov, nov});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            for (int b = 0; b < nv_; ++b)
                for (int j = 0; j < no_; ++j) {
                    const int p = a * no_ + i;
                    const int q = b * no_ + j;
                    // ovov_(a,i,b,j) = (ai|bj); real orbitals ⇒ (ai|jb)=(ai|bj).
                    double val =
                        coul * ovov_(a, i, b, j) - alpha_hf * oovv_(i, j, a, b);
                    if (a == b && i == j)
                        val += eps_virt_(a) - eps_occ_(i);
                    A(p, q) = val;
                }
    return A;
}

RealMatrix OrbitalHessian::dense_B(Channel ch, double alpha_hf) const {
    // Singlet B carries Coulomb 2(ai|bj); triplet does not. Exact exchange
    // −α (aj|bi) on both channels. No orbital-energy diagonal on B.
    const double coul = (ch == Channel::singlet) ? 2.0 : 0.0;
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    RealMatrix B = xt::zeros<double>({nov, nov});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            for (int b = 0; b < nv_; ++b)
                for (int j = 0; j < no_; ++j) {
                    const int p = a * no_ + i;
                    const int q = b * no_ + j;
                    // ovov_(a,i,b,j)=(ai|bj); ovov_(a,j,b,i)=(aj|bi).
                    B(p, q) =
                        coul * ovov_(a, i, b, j) - alpha_hf * ovov_(a, j, b, i);
                }
    return B;
}

RealVector OrbitalHessian::diagonal_plus(Channel ch) const {
    const double cw = coulomb_weight(ch);
    RealVector d = xt::zeros<double>({static_cast<std::size_t>(n_ov())});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            d(a * no_ + i) = eps_virt_(a) - eps_occ_(i)
                             + (cw - 1.0) * ovov_(a, i, a, i)
                             - oovv_(i, i, a, a);
    return d;
}

DirectRpaResult OrbitalHessian::direct_rpa(Channel ch, double alpha_hf) const {
    const RealMatrix A = dense_A(ch, alpha_hf);
    const RealMatrix B = dense_B(ch, alpha_hf);
    const std::size_t n = A.shape()[0];

    double tr_A = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        tr_A += A(i, i);

    const RealMatrix Mp = A + B;
    const RealMatrix Mm = A - B;

    // Positive-definiteness guard preserved from the rpa.cpp / g0w0.cpp
    // file-local sym_power copies: they THREW when λ_min(A−B) ≤ kAmBFloor,
    // whereas linalg::sym_power CLAMPS at the floor. Keep the throw here so
    // behavior is unchanged; the (A−B)^{±1/2} math then defers to sym_power
    // (bit-identical on a healthy, PD reference).
    {
        auto [s, U] = xt::linalg::eigh(Mm);
        (void)U;
        if (s(0) <= kAmBFloor)
            throw std::runtime_error(
                "OrbitalHessian::direct_rpa: (A - B) is not positive definite "
                "(lowest eigenvalue = "
                + std::to_string(s(0))
                + " Ha) — dRPA reduction undefined (no fallback)");
    }
    const RealMatrix T =
        linalg::sym_power(Mm, 0.5, kAmBFloor, "OrbitalHessian::direct_rpa");
    const RealMatrix Ti =
        linalg::sym_power(Mm, -0.5, kAmBFloor, "OrbitalHessian::direct_rpa");

    RealMatrix M = xt::linalg::dot(T, RealMatrix(xt::linalg::dot(Mp, T)));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = r + 1; c < n; ++c) {
            const double v = 0.5 * (M(r, c) + M(c, r));
            M(r, c) = v;
            M(c, r) = v;
        }
    auto [w2, Z] = xt::linalg::eigh(M);

    DirectRpaResult out;
    out.tr_A = tr_A;
    out.omega = xt::zeros<double>({n});
    out.X = xt::zeros<double>({n, n});
    out.Y = xt::zeros<double>({n, n});
    double sum_om = 0.0;
    for (std::size_t s = 0; s < n; ++s) {
        if (w2(s) < -kOmega2Tol)
            throw std::runtime_error(
                "OrbitalHessian::direct_rpa: dRPA omega^2 = "
                + std::to_string(w2(s)) + " Ha^2 (root " + std::to_string(s)
                + ") is negative — imaginary excitation (no fallback)");
        if (w2(s) <= kOmega2Tol)
            throw std::runtime_error(
                "OrbitalHessian::direct_rpa: dRPA omega^2 = "
                + std::to_string(w2(s)) + " Ha^2 (root " + std::to_string(s)
                + ") is numerically zero (no fallback)");
        const double w = std::sqrt(w2(s));
        out.omega(s) = w;
        sum_om += w;
        // (X+Y) = ω^{-1/2} T z, (X−Y) = ω^{1/2} T^{-1} z (g0w0.cpp math).
        for (std::size_t r = 0; r < n; ++r) {
            double xpy = 0.0, xmy = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                xpy += T(r, c) * Z(c, s);
                xmy += Ti(r, c) * Z(c, s);
            }
            xpy /= std::sqrt(w);
            xmy *= std::sqrt(w);
            out.X(s, r) = 0.5 * (xpy + xmy);
            out.Y(s, r) = 0.5 * (xpy - xmy);
        }
    }
    // Furche plasmon formula (closed-shell spin-adapted singlet dRPA).
    out.e_c = 0.5 * (sum_om - tr_A);
    return out;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
