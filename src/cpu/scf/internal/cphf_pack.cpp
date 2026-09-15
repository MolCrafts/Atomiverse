/// @file cpu/scf/internal/cphf_pack.cpp
/// @brief CPHF linear response (link-08 σ₊ kernel as the LHS) + analytic
///        RHF nuclear-Hessian assembly. See cphf_pack.h for the equations
///        and references (Pople et al. 1979, DOI 10.1002/qua.560160825).

#include "atomiverse/cpu/scf/internal/cphf_pack.h"

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

/// @brief Closed-shell two-electron contraction G[X]_{μν} = Σ_{λσ} X_{λσ}
///        [(μν|λσ) − ½(μλ|νσ)] over a dense chemist-order AO ERI tensor.
/// @param eri Dense (μν|λσ) tensor [Hartree].
/// @param X   Symmetric AO matrix (density-like, dimensionless).
/// @return G[X] [Hartree · units of X].
RealMatrix g_contraction(const xt::xtensor<double, 4>& eri,
                         const RealMatrix& X) {
    const int n = static_cast<int>(X.shape()[0]);
    RealMatrix G = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    acc += X(k, l) * (eri(i, j, k, l) - 0.5 * eri(i, k, j, l));
            G(i, j) = acc;
        }
    }
    return G;
}

} // namespace

CphfPack::CphfPack(const SlaterDeterminant& det, const OrbitalHessian& oh,
                   const internal::IntegralView& pack,
                   const xt::xtensor<double, 4>& eri_ao,
                   const xt::xtensor<double, 4>& dS_ao,
                   const xt::xtensor<double, 4>& dF_ao)
    : det_(det), oh_(oh), pack_(pack), eri_(eri_ao), dS_ao_(dS_ao),
      dF_ao_(dF_ao) {
    if (!det.restricted())
        throw std::invalid_argument(
            "CphfPack: requires a restricted (closed-shell RHF) determinant "
            "— open-shell CPHF is a follow-up chain link");

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& eps =
        det.orbital_energies(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    n_ = static_cast<int>(C.shape()[0]);
    natm_ = static_cast<int>(dS_ao.shape()[0]);
    const int n_mo = static_cast<int>(C.shape()[1]);

    if (static_cast<int>(dS_ao.shape()[2]) != n_
        || static_cast<int>(dF_ao.shape()[2]) != n_
        || static_cast<int>(eri_ao.shape()[0]) != n_)
        throw std::invalid_argument(
            "CphfPack: AO dimension mismatch between determinant ("
            + std::to_string(n_) + ") and derivative tensors");

    // Same integer-occupation split convention as OrbitalHessian.
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) == 1.0 ? occ : virt).push_back(p);
    no_ = static_cast<int>(occ.size());
    nv_ = static_cast<int>(virt.size());
    if (no_ != oh.n_occ() || nv_ != oh.n_virt())
        throw std::invalid_argument(
            "CphfPack: occ/virt split disagrees with the OrbitalHessian "
            "kernel (n_occ "
            + std::to_string(no_) + " vs " + std::to_string(oh.n_occ()) + ")");

    Co_ = xt::zeros<double>({n_, no_});
    Cv_ = xt::zeros<double>({n_, nv_});
    eps_occ_ = xt::zeros<double>({static_cast<std::size_t>(no_)});
    eps_virt_ = xt::zeros<double>({static_cast<std::size_t>(nv_)});
    for (int i = 0; i < no_; ++i) {
        eps_occ_(i) = eps(occ[static_cast<std::size_t>(i)]);
        for (int mu = 0; mu < n_; ++mu)
            Co_(mu, i) = C(mu, occ[static_cast<std::size_t>(i)]);
    }
    for (int a = 0; a < nv_; ++a) {
        eps_virt_(a) = eps(virt[static_cast<std::size_t>(a)]);
        for (int mu = 0; mu < n_; ++mu)
            Cv_(mu, a) = C(mu, virt[static_cast<std::size_t>(a)]);
    }

    // (ai|kl) MO block — reshape/dot quarter transforms (transform_block_
    // pattern), one round per index, rotating the transformed axis front.
    {
        xt::xtensor<double, 4> cur = eri_;
        const RealMatrix* Cs[4] = {&Co_, &Co_, &Co_, &Cv_}; // l, k, i, a
        for (int round = 0; round < 4; ++round) {
            const auto sh = cur.shape();
            auto flat = xt::reshape_view(cur, {sh[0] * sh[1] * sh[2], sh[3]});
            xt::xtensor<double, 2> h = xt::linalg::dot(flat, *Cs[round]);
            xt::xtensor<double, 4> next = xt::reshape_view(
                h, {sh[0], sh[1], sh[2], Cs[round]->shape()[1]});
            cur = xt::transpose(next, {3, 0, 1, 2});
        }
        vooo_ = cur; // [a, i, k, l]
    }

    // Skeleton derivative MO blocks + the right-hand sides B^x.
    const int n3 = 3 * natm_;
    S_vo_ = xt::zeros<double>({n3, nv_, no_});
    S_oo_ = xt::zeros<double>({n3, no_, no_});
    B_ = xt::zeros<double>({n3, nv_ * no_});
    for (int A = 0; A < natm_; ++A) {
        for (int d = 0; d < 3; ++d) {
            const int x = 3 * A + d;
            RealMatrix dS = xt::view(dS_ao_, A, d, xt::all(), xt::all());
            RealMatrix dF = xt::view(dF_ao_, A, d, xt::all(), xt::all());
            const RealMatrix S_vo =
                xt::linalg::dot(xt::transpose(Cv_), xt::linalg::dot(dS, Co_));
            const RealMatrix S_oo =
                xt::linalg::dot(xt::transpose(Co_), xt::linalg::dot(dS, Co_));
            const RealMatrix F_vo =
                xt::linalg::dot(xt::transpose(Cv_), xt::linalg::dot(dF, Co_));
            xt::view(S_vo_, x, xt::all(), xt::all()) = S_vo;
            xt::view(S_oo_, x, xt::all(), xt::all()) = S_oo;
            for (int a = 0; a < nv_; ++a) {
                for (int i = 0; i < no_; ++i) {
                    double b = F_vo(a, i) - eps_occ_(i) * S_vo(a, i);
                    for (int k = 0; k < no_; ++k)
                        for (int l = 0; l < no_; ++l)
                            b -=
                                S_oo(k, l)
                                * (2.0 * vooo_(a, i, k, l) - vooo_(a, k, i, l));
                    B_(x, a * no_ + i) = b;
                }
            }
        }
    }
}

RealVector CphfPack::apply_lhs(const RealVector& v) const {
    return oh_.sigma_plus(v, Channel::singlet);
}

void CphfPack::solve() {
    const int n3 = 3 * natm_;
    const int nov = nv_ * no_;
    U_ = xt::zeros<double>({n3, nov});
    iterations_ = 0;

    if (nov <= dense_threshold) {
        // Dense direct solve: one factorization for all right-hand sides.
        const RealMatrix Aplus = oh_.dense_plus(Channel::singlet);
        RealMatrix rhs = xt::zeros<double>({nov, n3});
        for (int x = 0; x < n3; ++x)
            for (int p = 0; p < nov; ++p)
                rhs(p, x) = -B_(x, p);
        const RealMatrix sol = xt::linalg::solve(Aplus, rhs);
        for (int x = 0; x < n3; ++x)
            for (int p = 0; p < nov; ++p)
                U_(x, p) = sol(p, x);
        solved_ = true;
        return;
    }

    // Preconditioned conjugate gradient per right-hand side (shared
    // linalg::solve_pcg); ¹(A+B) is symmetric positive definite at an
    // internally stable RHF solution. solve_pcg stops at tol·‖b‖, whereas the
    // pre-refactor loop stopped at cg_tol·max(1,‖b‖); scaling tol per RHS
    // reproduces that exact absolute threshold (behavior-preserving). The
    // null-RHS skip is kept — solve_pcg would divide 0/0 on b = 0.
    const RealVector diag = oh_.diagonal_plus(Channel::singlet);
    for (int x = 0; x < n3; ++x) {
        RealVector b = xt::zeros<double>({static_cast<std::size_t>(nov)});
        for (int p = 0; p < nov; ++p)
            b(p) = -B_(x, p);
        double b_norm = 0.0;
        for (int p = 0; p < nov; ++p)
            b_norm += b(p) * b(p);
        b_norm = std::sqrt(b_norm);
        if (b_norm == 0.0)
            continue; // U^x = 0 for a null right-hand side.

        const double tol_eff = cg_tol * std::max(1.0, b_norm) / b_norm;
        const RealVector u = linalg::solve_pcg(
            [&](const RealVector& v) {
                return oh_.sigma_plus(v, Channel::singlet);
            },
            diag, b, tol_eff, cg_max_iter);
        ++iterations_;
        for (int p = 0; p < nov; ++p)
            U_(x, p) = u(p);
    }
    solved_ = true;
}

RealMatrix CphfPack::density_response(int x) const {
    if (!solved_)
        throw std::runtime_error(
            "CphfPack::density_response: call solve() first");
    // Virt-occ rotation part: 2 Σ_ai U^x_ai (C_a C_iᵀ + C_i C_aᵀ).
    RealMatrix Umat = xt::zeros<double>({nv_, no_});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            Umat(a, i) = U_(x, a * no_ + i);
    const RealMatrix Pvo =
        xt::linalg::dot(xt::linalg::dot(Cv_, Umat), xt::transpose(Co_));
    // Occ-occ orthonormality part: −2 Σ_ij S^{(x)}_ij C_i C_jᵀ (only the
    // symmetric combination U_ji + U_ij = −S^{(x)}_ij enters — convention
    // free).
    RealMatrix Soo = xt::view(S_oo_, x, xt::all(), xt::all());
    const RealMatrix Poo =
        xt::linalg::dot(xt::linalg::dot(Co_, Soo), xt::transpose(Co_));
    return 2.0 * (Pvo + xt::transpose(Pvo)) - 2.0 * Poo;
}

RealMatrix CphfPack::fock_response_(int x, const RealMatrix& Px) const {
    RealMatrix F = xt::view(dF_ao_, x / 3, x % 3, xt::all(), xt::all());
    return F + g_contraction(eri_, Px);
}

xt::xtensor<double, 2> CphfPack::hessian() const {
    if (!solved_)
        throw std::runtime_error("CphfPack::hessian: call solve() first");
    const int n3 = 3 * natm_;

    // Converged density, Fock, and energy-weighted density.
    const RealMatrix P = det_.density_total();
    const RealMatrix F_ao = pack_.Hcore + g_contraction(eri_, P);
    RealMatrix W = xt::zeros<double>({n_, n_});
    for (int i = 0; i < no_; ++i)
        for (int mu = 0; mu < n_; ++mu)
            for (int nu = 0; nu < n_; ++nu)
                W(mu, nu) += 2.0 * eps_occ_(i) * Co_(mu, i) * Co_(nu, i);

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
    accumulate_two_electron_hessian(H, pack_, P);

    // ── CPHF response part: Tr[P^y F^{(x)}] − Tr[W^y S^{(x)}] ─────────
    for (int y = 0; y < n3; ++y) {
        const RealMatrix Py = density_response(y);
        const RealMatrix Fy = fock_response_(y, Py);
        // W^y = ½ (P^y F P + P F^y P + P F P^y): the derivative of the
        // invariant form W = ½ P F P (see cphf_pack.h).
        const RealMatrix Wy =
            0.5
            * (xt::linalg::dot(Py, xt::linalg::dot(F_ao, P))
               + xt::linalg::dot(P, xt::linalg::dot(Fy, P))
               + xt::linalg::dot(P, xt::linalg::dot(F_ao, Py)));
        for (int x = 0; x < n3; ++x) {
            double resp = 0.0;
            for (int mu = 0; mu < n_; ++mu)
                for (int nu = 0; nu < n_; ++nu)
                    resp += dF_ao_(x / 3, x % 3, mu, nu) * Py(mu, nu)
                            - dS_ao_(x / 3, x % 3, mu, nu) * Wy(mu, nu);
            H(x, y) += resp;
        }
    }

    // Exact symmetry up to CPHF convergence — enforce it.
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
