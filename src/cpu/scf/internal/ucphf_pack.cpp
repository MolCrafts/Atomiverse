/// @file cpu/scf/internal/ucphf_pack.cpp
/// @brief Unrestricted CPHF linear response + analytic UHF nuclear-Hessian
///        assembly. See ucphf_pack.h for equations and references.

#include "atomiverse/cpu/scf/internal/ucphf_pack.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/linear_solve.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

/// @brief Four-index MO transform (same pattern as OrbitalHessian /
///        SOSCFStepper::mo_transform_).
xt::xtensor<double, 4> mo_transform(const xt::xtensor<double, 4>& eri_ao,
                                    const RealMatrix& C0, const RealMatrix& C1,
                                    const RealMatrix& C2,
                                    const RealMatrix& C3) {
    xt::xtensor<double, 4> cur = eri_ao;
    const RealMatrix* Cs[4] = {&C3, &C2, &C1, &C0};
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

/// @brief UHF Coulomb J[X]_{μν} = Σ_{λσ} X_{λσ} (μν|λσ).
RealMatrix j_contraction(const xt::xtensor<double, 4>& eri,
                         const RealMatrix& X) {
    const int n = static_cast<int>(X.shape()[0]);
    RealMatrix J = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    acc += X(k, l) * eri(i, j, k, l);
            J(i, j) = acc;
        }
    return J;
}

/// @brief UHF exchange K[X]_{μν} = Σ_{λσ} X_{λσ} (μλ|νσ).
RealMatrix k_contraction(const xt::xtensor<double, 4>& eri,
                         const RealMatrix& X) {
    const int n = static_cast<int>(X.shape()[0]);
    RealMatrix K = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    acc += X(k, l) * eri(i, k, j, l);
            K(i, j) = acc;
        }
    return K;
}

} // namespace

UCphfPack::UCphfPack(const SlaterDeterminant& det,
                     const internal::IntegralView& pack,
                     const xt::xtensor<double, 4>& eri_ao,
                     const xt::xtensor<double, 4>& dS_ao,
                     const xt::xtensor<double, 4>& dFa_ao,
                     const xt::xtensor<double, 4>& dFb_ao)
    : det_(det), pack_(pack), eri_(eri_ao), dS_ao_(dS_ao), dFa_ao_(dFa_ao),
      dFb_ao_(dFb_ao) {
    if (det.restricted())
        throw std::invalid_argument(
            "UCphfPack: requires an unrestricted (UHF) determinant — use "
            "CphfPack for closed-shell RHF");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& Ca = det.coefficients(Spin::alpha);
    const RealMatrix& Cb = det.coefficients(Spin::beta);
    const RealVector& ea = det.orbital_energies(Spin::alpha);
    const RealVector& eb = det.orbital_energies(Spin::beta);
    const RealVector& fa = det.occupations(Spin::alpha);
    const RealVector& fb = det.occupations(Spin::beta);

    n_ = static_cast<int>(Ca.shape()[0]);
    natm_ = static_cast<int>(dS_ao.shape()[0]);
    const int n_mo_a = static_cast<int>(Ca.shape()[1]);
    const int n_mo_b = static_cast<int>(Cb.shape()[1]);

    if (static_cast<int>(dS_ao.shape()[2]) != n_
        || static_cast<int>(dFa_ao.shape()[2]) != n_
        || static_cast<int>(dFb_ao.shape()[2]) != n_
        || static_cast<int>(eri_ao.shape()[0]) != n_)
        throw std::invalid_argument(
            "UCphfPack: AO dimension mismatch between determinant ("
            + std::to_string(n_) + ") and derivative tensors");

    // Integer-occupation split per spin (canonical UHF).
    std::vector<int> occ_a, virt_a, occ_b, virt_b;
    for (int p = 0; p < n_mo_a; ++p)
        (fa(p) == 1.0 ? occ_a : virt_a).push_back(p);
    for (int p = 0; p < n_mo_b; ++p)
        (fb(p) == 1.0 ? occ_b : virt_b).push_back(p);
    no_a_ = static_cast<int>(occ_a.size());
    nv_a_ = static_cast<int>(virt_a.size());
    no_b_ = static_cast<int>(occ_b.size());
    nv_b_ = static_cast<int>(virt_b.size());
    if (no_a_ == 0 || nv_a_ == 0 || no_b_ == 0 || nv_b_ == 0)
        throw std::invalid_argument(
            "UCphfPack: each spin channel needs at least one occupied and "
            "one virtual orbital (n_occ_a="
            + std::to_string(no_a_) + " n_virt_a=" + std::to_string(nv_a_)
            + " n_occ_b=" + std::to_string(no_b_)
            + " n_virt_b=" + std::to_string(nv_b_) + ")");

    rot_a_ = nv_a_ * no_a_;
    rot_b_ = nv_b_ * no_b_;
    n_rot_ = rot_a_ + rot_b_;

    // Contiguous occ/virt columns (occupation order, not MO index order —
    // matches SOSCF/OrbitalHessian integer-occupation convention).
    Co_a_ = xt::zeros<double>({n_, no_a_});
    Cv_a_ = xt::zeros<double>({n_, nv_a_});
    Co_b_ = xt::zeros<double>({n_, no_b_});
    Cv_b_ = xt::zeros<double>({n_, nv_b_});
    eps_oa_ = xt::zeros<double>({static_cast<std::size_t>(no_a_)});
    eps_va_ = xt::zeros<double>({static_cast<std::size_t>(nv_a_)});
    eps_ob_ = xt::zeros<double>({static_cast<std::size_t>(no_b_)});
    eps_vb_ = xt::zeros<double>({static_cast<std::size_t>(nv_b_)});
    for (int i = 0; i < no_a_; ++i) {
        eps_oa_(i) = ea(occ_a[static_cast<std::size_t>(i)]);
        for (int mu = 0; mu < n_; ++mu)
            Co_a_(mu, i) = Ca(mu, occ_a[static_cast<std::size_t>(i)]);
    }
    for (int a = 0; a < nv_a_; ++a) {
        eps_va_(a) = ea(virt_a[static_cast<std::size_t>(a)]);
        for (int mu = 0; mu < n_; ++mu)
            Cv_a_(mu, a) = Ca(mu, virt_a[static_cast<std::size_t>(a)]);
    }
    for (int i = 0; i < no_b_; ++i) {
        eps_ob_(i) = eb(occ_b[static_cast<std::size_t>(i)]);
        for (int mu = 0; mu < n_; ++mu)
            Co_b_(mu, i) = Cb(mu, occ_b[static_cast<std::size_t>(i)]);
    }
    for (int a = 0; a < nv_b_; ++a) {
        eps_vb_(a) = eb(virt_b[static_cast<std::size_t>(a)]);
        for (int mu = 0; mu < n_; ++mu)
            Cv_b_(mu, a) = Cb(mu, virt_b[static_cast<std::size_t>(a)]);
    }

    // MO-ERI blocks for (A+B) and for the RHS occ–occ contractions.
    ovov_aa_ = mo_transform(eri_, Cv_a_, Co_a_, Cv_a_, Co_a_); // (ai|bj)
    oovv_aa_ = mo_transform(eri_, Co_a_, Co_a_, Cv_a_, Cv_a_); // (ij|ab)
    ovov_bb_ = mo_transform(eri_, Cv_b_, Co_b_, Cv_b_, Co_b_);
    oovv_bb_ = mo_transform(eri_, Co_b_, Co_b_, Cv_b_, Cv_b_);
    ovov_ab_ = mo_transform(eri_, Cv_a_, Co_a_, Cv_b_, Co_b_); // (aαiα|bβjβ)
    vooo_aa_ = mo_transform(eri_, Cv_a_, Co_a_, Co_a_, Co_a_); // (ai|kl) αα
    vooo_ab_ = mo_transform(eri_, Cv_a_, Co_a_, Co_b_, Co_b_); // (aαiα|kβlβ)
    vooo_bb_ = mo_transform(eri_, Cv_b_, Co_b_, Co_b_, Co_b_);
    vooo_ba_ = mo_transform(eri_, Cv_b_, Co_b_, Co_a_, Co_a_); // (aβiβ|kαlα)

    // Skeleton derivative MO blocks + RHS B^x.
    const int n3 = 3 * natm_;
    S_vo_a_ = xt::zeros<double>({n3, nv_a_, no_a_});
    S_oo_a_ = xt::zeros<double>({n3, no_a_, no_a_});
    S_vo_b_ = xt::zeros<double>({n3, nv_b_, no_b_});
    S_oo_b_ = xt::zeros<double>({n3, no_b_, no_b_});
    B_ = xt::zeros<double>({n3, n_rot_});

    for (int A = 0; A < natm_; ++A) {
        for (int d = 0; d < 3; ++d) {
            const int x = 3 * A + d;
            RealMatrix dS = xt::view(dS_ao_, A, d, xt::all(), xt::all());
            RealMatrix dFa = xt::view(dFa_ao_, A, d, xt::all(), xt::all());
            RealMatrix dFb = xt::view(dFb_ao_, A, d, xt::all(), xt::all());

            const RealMatrix S_vo_a = xt::linalg::dot(
                xt::transpose(Cv_a_), xt::linalg::dot(dS, Co_a_));
            const RealMatrix S_oo_a = xt::linalg::dot(
                xt::transpose(Co_a_), xt::linalg::dot(dS, Co_a_));
            const RealMatrix F_vo_a = xt::linalg::dot(
                xt::transpose(Cv_a_), xt::linalg::dot(dFa, Co_a_));
            const RealMatrix S_vo_b = xt::linalg::dot(
                xt::transpose(Cv_b_), xt::linalg::dot(dS, Co_b_));
            const RealMatrix S_oo_b = xt::linalg::dot(
                xt::transpose(Co_b_), xt::linalg::dot(dS, Co_b_));
            const RealMatrix F_vo_b = xt::linalg::dot(
                xt::transpose(Cv_b_), xt::linalg::dot(dFb, Co_b_));

            xt::view(S_vo_a_, x, xt::all(), xt::all()) = S_vo_a;
            xt::view(S_oo_a_, x, xt::all(), xt::all()) = S_oo_a;
            xt::view(S_vo_b_, x, xt::all(), xt::all()) = S_vo_b;
            xt::view(S_oo_b_, x, xt::all(), xt::all()) = S_oo_b;

            // α RHS
            for (int a = 0; a < nv_a_; ++a)
                for (int i = 0; i < no_a_; ++i) {
                    double b = F_vo_a(a, i) - eps_oa_(i) * S_vo_a(a, i);
                    for (int k = 0; k < no_a_; ++k)
                        for (int l = 0; l < no_a_; ++l)
                            b -=
                                S_oo_a(k, l)
                                * (vooo_aa_(a, i, k, l) - vooo_aa_(a, k, i, l));
                    for (int k = 0; k < no_b_; ++k)
                        for (int l = 0; l < no_b_; ++l)
                            b -= S_oo_b(k, l) * vooo_ab_(a, i, k, l);
                    B_(x, a * no_a_ + i) = b;
                }
            // β RHS
            for (int a = 0; a < nv_b_; ++a)
                for (int i = 0; i < no_b_; ++i) {
                    double b = F_vo_b(a, i) - eps_ob_(i) * S_vo_b(a, i);
                    for (int k = 0; k < no_b_; ++k)
                        for (int l = 0; l < no_b_; ++l)
                            b -=
                                S_oo_b(k, l)
                                * (vooo_bb_(a, i, k, l) - vooo_bb_(a, k, i, l));
                    for (int k = 0; k < no_a_; ++k)
                        for (int l = 0; l < no_a_; ++l)
                            b -= S_oo_a(k, l) * vooo_ba_(a, i, k, l);
                    B_(x, rot_a_ + a * no_b_ + i) = b;
                }
        }
    }
}

RealVector UCphfPack::apply_lhs(const RealVector& v) const {
    if (static_cast<int>(v.size()) != n_rot_)
        throw std::invalid_argument("UCphfPack::apply_lhs: v.size() = "
                                    + std::to_string(v.size())
                                    + " != n_rot = " + std::to_string(n_rot_));

    RealVector y = xt::zeros<double>({static_cast<std::size_t>(n_rot_)});
    // α rows
    for (int a = 0; a < nv_a_; ++a)
        for (int i = 0; i < no_a_; ++i) {
            const int p = a * no_a_ + i;
            double acc = (eps_va_(a) - eps_oa_(i)) * v(p);
            for (int b = 0; b < nv_a_; ++b)
                for (int j = 0; j < no_a_; ++j)
                    acc += (2.0 * ovov_aa_(a, i, b, j) - oovv_aa_(i, j, a, b)
                            - ovov_aa_(a, j, b, i))
                           * v(b * no_a_ + j);
            for (int b = 0; b < nv_b_; ++b)
                for (int j = 0; j < no_b_; ++j)
                    acc +=
                        2.0 * ovov_ab_(a, i, b, j) * v(rot_a_ + b * no_b_ + j);
            y(p) = acc;
        }
    // β rows
    for (int a = 0; a < nv_b_; ++a)
        for (int i = 0; i < no_b_; ++i) {
            const int p = rot_a_ + a * no_b_ + i;
            double acc = (eps_vb_(a) - eps_ob_(i)) * v(p);
            for (int b = 0; b < nv_b_; ++b)
                for (int j = 0; j < no_b_; ++j)
                    acc += (2.0 * ovov_bb_(a, i, b, j) - oovv_bb_(i, j, a, b)
                            - ovov_bb_(a, j, b, i))
                           * v(rot_a_ + b * no_b_ + j);
            for (int b = 0; b < nv_a_; ++b)
                for (int j = 0; j < no_a_; ++j)
                    acc += 2.0 * ovov_ab_(b, j, a, i) * v(b * no_a_ + j);
            y(p) = acc;
        }
    return y;
}

RealVector UCphfPack::diagonal_lhs_() const {
    RealVector d = xt::zeros<double>({static_cast<std::size_t>(n_rot_)});
    for (int a = 0; a < nv_a_; ++a)
        for (int i = 0; i < no_a_; ++i)
            d(a * no_a_ + i) = (eps_va_(a) - eps_oa_(i))
                               + (ovov_aa_(a, i, a, i) - oovv_aa_(i, i, a, a));
    for (int a = 0; a < nv_b_; ++a)
        for (int i = 0; i < no_b_; ++i)
            d(rot_a_ + a * no_b_ + i) =
                (eps_vb_(a) - eps_ob_(i))
                + (ovov_bb_(a, i, a, i) - oovv_bb_(i, i, a, a));
    // Guard against zero diagonals (near-degenerate).
    for (int p = 0; p < n_rot_; ++p)
        if (std::abs(d(p)) < 1e-10)
            d(p) = (d(p) >= 0.0 ? 1e-10 : -1e-10);
    return d;
}

RealMatrix UCphfPack::dense_lhs_() const {
    RealMatrix A = xt::zeros<double>({n_rot_, n_rot_});
    for (int p = 0; p < n_rot_; ++p) {
        RealVector e = xt::zeros<double>({static_cast<std::size_t>(n_rot_)});
        e(p) = 1.0;
        const RealVector col = apply_lhs(e);
        for (int q = 0; q < n_rot_; ++q)
            A(q, p) = col(q);
    }
    return A;
}

void UCphfPack::solve() {
    const int n3 = 3 * natm_;
    U_ = xt::zeros<double>({n3, n_rot_});
    iterations_ = 0;

    if (n_rot_ <= dense_threshold) {
        const RealMatrix Aplus = dense_lhs_();
        RealMatrix rhs = xt::zeros<double>({n_rot_, n3});
        for (int x = 0; x < n3; ++x)
            for (int p = 0; p < n_rot_; ++p)
                rhs(p, x) = -B_(x, p);
        const RealMatrix sol = xt::linalg::solve(Aplus, rhs);
        for (int x = 0; x < n3; ++x)
            for (int p = 0; p < n_rot_; ++p)
                U_(x, p) = sol(p, x);
        solved_ = true;
        return;
    }

    // Preconditioned conjugate gradient per right-hand side (shared
    // linalg::solve_pcg). solve_pcg stops at tol·‖b‖, whereas the pre-refactor
    // loop stopped at cg_tol·max(1,‖b‖); scaling tol per RHS reproduces that
    // exact absolute threshold (behavior-preserving). The null-RHS skip is
    // kept — solve_pcg would divide 0/0 on b = 0.
    const RealVector diag = diagonal_lhs_();
    for (int x = 0; x < n3; ++x) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(n_rot_)});
        for (int p = 0; p < n_rot_; ++p)
            b(p) = -B_(x, p);
        double b_norm = 0.0;
        for (int p = 0; p < n_rot_; ++p)
            b_norm += b(p) * b(p);
        b_norm = std::sqrt(b_norm);
        if (b_norm == 0.0)
            continue;

        const double tol_eff = cg_tol * std::max(1.0, b_norm) / b_norm;
        const RealVector u =
            linalg::solve_pcg([&](const RealVector& v) { return apply_lhs(v); },
                              diag, b, tol_eff, cg_max_iter);
        ++iterations_;
        for (int p = 0; p < n_rot_; ++p)
            U_(x, p) = u(p);
    }
    solved_ = true;
}

RealMatrix UCphfPack::density_response_alpha(int x) const {
    if (!solved_)
        throw std::runtime_error(
            "UCphfPack::density_response_alpha: call solve() first");
    RealMatrix Umat = xt::zeros<double>({nv_a_, no_a_});
    for (int a = 0; a < nv_a_; ++a)
        for (int i = 0; i < no_a_; ++i)
            Umat(a, i) = U_(x, a * no_a_ + i);
    const RealMatrix Pvo =
        xt::linalg::dot(xt::linalg::dot(Cv_a_, Umat), xt::transpose(Co_a_));
    RealMatrix Soo = xt::view(S_oo_a_, x, xt::all(), xt::all());
    const RealMatrix Poo =
        xt::linalg::dot(xt::linalg::dot(Co_a_, Soo), xt::transpose(Co_a_));
    // Factor-1 spin density: δP = C_v U C_oᵀ + h.c. − C_o S_oo C_oᵀ.
    return (Pvo + xt::transpose(Pvo)) - Poo;
}

RealMatrix UCphfPack::density_response_beta(int x) const {
    if (!solved_)
        throw std::runtime_error(
            "UCphfPack::density_response_beta: call solve() first");
    RealMatrix Umat = xt::zeros<double>({nv_b_, no_b_});
    for (int a = 0; a < nv_b_; ++a)
        for (int i = 0; i < no_b_; ++i)
            Umat(a, i) = U_(x, rot_a_ + a * no_b_ + i);
    const RealMatrix Pvo =
        xt::linalg::dot(xt::linalg::dot(Cv_b_, Umat), xt::transpose(Co_b_));
    RealMatrix Soo = xt::view(S_oo_b_, x, xt::all(), xt::all());
    const RealMatrix Poo =
        xt::linalg::dot(xt::linalg::dot(Co_b_, Soo), xt::transpose(Co_b_));
    return (Pvo + xt::transpose(Pvo)) - Poo;
}

void UCphfPack::fock_response_(int x, const RealMatrix& dPa,
                               const RealMatrix& dPb, RealMatrix& dFa,
                               RealMatrix& dFb) const {
    RealMatrix Fa0 = xt::view(dFa_ao_, x / 3, x % 3, xt::all(), xt::all());
    RealMatrix Fb0 = xt::view(dFb_ao_, x / 3, x % 3, xt::all(), xt::all());
    const RealMatrix dP = dPa + dPb;
    const RealMatrix dJ = j_contraction(eri_, dP);
    dFa = Fa0 + dJ - k_contraction(eri_, dPa);
    dFb = Fb0 + dJ - k_contraction(eri_, dPb);
}

xt::xtensor<double, 2> UCphfPack::hessian() const {
    if (!solved_)
        throw std::runtime_error("UCphfPack::hessian: call solve() first");
    const int n3 = 3 * natm_;

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix Pa = det_.density(Spin::alpha);
    const RealMatrix Pb = det_.density(Spin::beta);
    const RealMatrix P = Pa + Pb;

    // Converged Fock matrices F^σ = h + J[P] − K[P^σ].
    const RealMatrix J0 = j_contraction(eri_, P);
    const RealMatrix Fa = pack_.Hcore + J0 - k_contraction(eri_, Pa);
    const RealMatrix Fb = pack_.Hcore + J0 - k_contraction(eri_, Pb);

    // Energy-weighted density W = Σ_σ C_o^σ diag(ε) C_o^{σ⊤}.
    RealMatrix W = xt::zeros<double>({n_, n_});
    for (int i = 0; i < no_a_; ++i)
        for (int mu = 0; mu < n_; ++mu)
            for (int nu = 0; nu < n_; ++nu)
                W(mu, nu) += eps_oa_(i) * Co_a_(mu, i) * Co_a_(nu, i);
    for (int i = 0; i < no_b_; ++i)
        for (int mu = 0; mu < n_; ++mu)
            for (int nu = 0; nu < n_; ++nu)
                W(mu, nu) += eps_ob_(i) * Co_b_(mu, i) * Co_b_(nu, i);

    // ── Skeleton second-derivative part ──────────────────────────────
    const IntegralSecondDerivPack d2{pack_};
    xt::xtensor<double, 2> H = xt::zeros<double>({n3, n3});
    for (int A = 0; A < natm_; ++A)
        for (int di = 0; di < 3; ++di)
            for (int Bt = 0; Bt < natm_; ++Bt)
                for (int dj = 0; dj < 3; ++dj) {
                    double acc = d2.d2V_nuc(A, di, Bt, dj);
                    for (int mu = 0; mu < n_; ++mu)
                        for (int nu = 0; nu < n_; ++nu) {
                            const double h2 = d2.d2T(A, di, Bt, dj, mu, nu)
                                              + d2.d2V(A, di, Bt, dj, mu, nu);
                            acc += P(mu, nu) * h2
                                   - W(mu, nu) * d2.d2S(A, di, Bt, dj, mu, nu);
                        }
                    H(3 * A + di, 3 * Bt + dj) += acc;
                }
    accumulate_two_electron_hessian_uhf(H, pack_, Pa, Pb);

    // ── CPHF response: Σ_σ Tr[P^{σ,y} F^{σ(x)}] − Tr[W^y S^{(x)}] ──
    for (int y = 0; y < n3; ++y) {
        const RealMatrix dPa = density_response_alpha(y);
        const RealMatrix dPb = density_response_beta(y);
        RealMatrix dFa, dFb;
        fock_response_(y, dPa, dPb, dFa, dFb);
        // W^y = Σ_σ (δP^σ F^σ P^σ + P^σ δF^σ P^σ + P^σ F^σ δP^σ)
        // — derivative of W^σ = P^σ F^σ P^σ at an idempotent UHF density.
        const RealMatrix Wya = xt::linalg::dot(dPa, xt::linalg::dot(Fa, Pa))
                               + xt::linalg::dot(Pa, xt::linalg::dot(dFa, Pa))
                               + xt::linalg::dot(Pa, xt::linalg::dot(Fa, dPa));
        const RealMatrix Wyb = xt::linalg::dot(dPb, xt::linalg::dot(Fb, Pb))
                               + xt::linalg::dot(Pb, xt::linalg::dot(dFb, Pb))
                               + xt::linalg::dot(Pb, xt::linalg::dot(Fb, dPb));
        const RealMatrix Wy = Wya + Wyb;

        for (int x = 0; x < n3; ++x) {
            double resp = 0.0;
            for (int mu = 0; mu < n_; ++mu)
                for (int nu = 0; nu < n_; ++nu)
                    resp += dFa_ao_(x / 3, x % 3, mu, nu) * dPa(mu, nu)
                            + dFb_ao_(x / 3, x % 3, mu, nu) * dPb(mu, nu)
                            - dS_ao_(x / 3, x % 3, mu, nu) * Wy(mu, nu);
            H(x, y) += resp;
        }
    }

    for (int x = 0; x < n3; ++x)
        for (int y = 0; y < x; ++y) {
            const double s = 0.5 * (H(x, y) + H(y, x));
            H(x, y) = s;
            H(y, x) = s;
        }
    return H;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
