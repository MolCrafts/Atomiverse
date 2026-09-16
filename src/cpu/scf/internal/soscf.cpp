/// @file cpu/scf/internal/soscf.cpp
/// @brief SOSCFStepper — CIAH augmented-Hessian Davidson, trust region and
///        the exp(κ) unitary orbital update.
///
/// Gradient / Hessian conventions (true derivatives of the SCF energy under
/// the real rotation C ← C·exp(K), K_ai = κ_ai, K_ia = −κ_ai; Helgaker,
/// Jørgensen, Olsen ch. 10):
///   RHF (spin-adapted spatial rotation, both spins share κ):
///     g_ai = 4 F_ai,  H = 4·¹(A+B)  — the 08-link σ-kernel supplies
///     (A+B)·v; the singlet channel is the energy-minimization Hessian.
///   UHF (independent per-spin rotations, concatenated α then β):
///     g^σ_ai = 2 F^σ_ai,
///     H^σσ_{ai,bj} = 2[δδ(ε_a−ε_i) + 2(ai|bj)^σσ − (ab|ij)^σσ − (aj|bi)^σσ]
///     H^αβ_{ai,bj} = 4 (a^α i^α | b^β j^β)
///     — contracted matrix-free from the unrestricted MO-ERI occ-virt
///     blocks held here (the 08 kernel is restricted-only; audit note in
///     the spec). H is never assembled as an n_rot × n_rot matrix.

#include "atomiverse/cpu/scf/internal/soscf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {
/// Gradient ∞-norm [Hartree] below which step() is an exact no-op (the
/// iterate is stationary to round-off; rotating on noise would only churn).
constexpr double kGradZeroTol = 1e-14;
/// Energy rise [Hartree] beyond which the previous step is rejected and the
/// trust radius halved (round-off-sized rises must not trigger rejects).
constexpr double kEnergyRiseTol = 1e-10;
/// |x₀| below which the augmented-Hessian root is a pure Hessian mode; the
/// κ direction is then taken from x[1:] and scaled to the trust radius.
constexpr double kX0Floor = 1e-10;
} // namespace

SOSCFStepper::SOSCFStepper(bool restricted, const internal::IntegralView& ints,
                           const int& n_alpha, const int& n_beta,
                           const RealMatrix& Hcore, const double& v_nuc,
                           const RealMatrix& F_alpha, const RealMatrix& F_beta,
                           const RealMatrix& P_alpha, const RealMatrix& P_beta,
                           RealMatrix& C_alpha, RealMatrix& C_beta,
                           RealVector& eig_alpha, RealVector& eig_beta,
                           int max_micro, double trust_radius)
    : restricted_(restricted), ints_(ints), n_alpha_(n_alpha), n_beta_(n_beta),
      Hcore_(Hcore), v_nuc_(v_nuc), F_alpha_(F_alpha), F_beta_(F_beta),
      P_alpha_(P_alpha), P_beta_(P_beta), C_alpha_(C_alpha), C_beta_(C_beta),
      eig_alpha_(eig_alpha), eig_beta_(eig_beta), max_micro_(max_micro),
      trust_radius0_(trust_radius), radius_(trust_radius) {
    if (max_micro < 1)
        throw std::invalid_argument(
            "SOSCFStepper: max_micro = " + std::to_string(max_micro)
            + " — the Davidson micro budget must be >= 1");
    if (!(trust_radius > 0.0))
        throw std::invalid_argument("SOSCFStepper: trust_radius = "
                                    + std::to_string(trust_radius)
                                    + " — the trust radius must be positive");
}

void SOSCFStepper::reset() {
    eri_ = xt::xtensor<double, 4>{};
    radius_ = trust_radius0_;
    have_prev_ = false;
    prev_energy_ = 0.0;
    last_step_norm_ = 0.0;
    last_lambda_ = 0.0;
    rejected_ = false;
    macro_steps_ = 0;
}

void SOSCFStepper::validate_() const {
    const auto n = Hcore_.shape()[0];
    auto check = [&](const char* name, const RealMatrix& M) {
        if (M.shape()[0] != n)
            throw std::invalid_argument(
                std::string("SOSCFStepper: ") + name + " AO dimension "
                + std::to_string(M.shape()[0]) + " != Hcore dimension "
                + std::to_string(n) + " (no fallback)");
    };
    check("C_alpha", C_alpha_);
    check("F_alpha", F_alpha_);
    check("P_alpha", P_alpha_);
    const int m_a = static_cast<int>(C_alpha_.shape()[1]);
    if (n_alpha_ < 0 || n_alpha_ > m_a)
        throw std::invalid_argument(
            "SOSCFStepper: n_alpha = " + std::to_string(n_alpha_)
            + " outside [0, n_mo = " + std::to_string(m_a) + "]");
    if (!restricted_) {
        check("C_beta", C_beta_);
        check("F_beta", F_beta_);
        check("P_beta", P_beta_);
        const int m_b = static_cast<int>(C_beta_.shape()[1]);
        if (m_b != m_a)
            throw std::invalid_argument(
                "SOSCFStepper: alpha/beta MO-space mismatch (n_mo_alpha = "
                + std::to_string(m_a) + ", n_mo_beta = " + std::to_string(m_b)
                + ") — the concatenated UHF rotation vector requires both "
                  "channels in the same MO space (no fallback)");
        if (n_beta_ < 0 || n_beta_ > m_b)
            throw std::invalid_argument(
                "SOSCFStepper: n_beta = " + std::to_string(n_beta_)
                + " outside [0, n_mo = " + std::to_string(m_b) + "]");
    }
}

double SOSCFStepper::current_energy() const {
    if (restricted_)
        return 0.5 * xt::sum((Hcore_ + F_alpha_) * P_alpha_)() + v_nuc_;
    return 0.5
               * (xt::sum((Hcore_ + F_alpha_) * P_alpha_)()
                  + xt::sum((Hcore_ + F_beta_) * P_beta_)())
           + v_nuc_;
}

RealVector SOSCFStepper::gradient() const {
    validate_();
    auto block = [](const RealMatrix& F, const RealMatrix& C, int no,
                    double scale) {
        const int m = static_cast<int>(C.shape()[1]);
        const int nv = m - no;
        const RealMatrix Fmo = xt::linalg::dot(
            xt::transpose(C), RealMatrix(xt::linalg::dot(F, C)));
        RealVector g =
            xt::zeros<double>({static_cast<std::size_t>(std::max(nv * no, 0))});
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
                g(a * no + i) = scale * Fmo(no + a, i);
        return g;
    };
    if (restricted_)
        return block(F_alpha_, C_alpha_, n_alpha_, 4.0);
    const RealVector ga = block(F_alpha_, C_alpha_, n_alpha_, 2.0);
    const RealVector gb = block(F_beta_, C_beta_, n_beta_, 2.0);
    RealVector g = xt::zeros<double>({ga.size() + gb.size()});
    for (std::size_t p = 0; p < ga.size(); ++p)
        g(p) = ga(p);
    for (std::size_t p = 0; p < gb.size(); ++p)
        g(ga.size() + p) = gb(p);
    return g;
}

RealMatrix SOSCFStepper::semicanonicalize_(const RealMatrix& F, RealMatrix& C,
                                           RealVector& eig, int n_occ) {
    const int n = static_cast<int>(C.shape()[0]);
    const int m = static_cast<int>(C.shape()[1]);
    const int nv = m - n_occ;
    const RealMatrix Fmo =
        xt::linalg::dot(xt::transpose(C), RealMatrix(xt::linalg::dot(F, C)));

    eig = xt::zeros<double>({static_cast<std::size_t>(m)});
    RealMatrix C_new = C;

    // Occupied block: C_occ ← C_occ · U_o with F_oo U_o = U_o diag(e_o).
    RealMatrix Uo = xt::eye<double>(static_cast<std::size_t>(n_occ));
    if (n_occ > 0) {
        RealMatrix Foo = xt::zeros<double>({n_occ, n_occ});
        for (int i = 0; i < n_occ; ++i)
            for (int j = 0; j < n_occ; ++j)
                Foo(i, j) = Fmo(i, j);
        auto [eo, U] = xt::linalg::eigh(Foo);
        Uo = U;
        for (int i = 0; i < n_occ; ++i)
            eig(i) = eo(i);
        for (int mu = 0; mu < n; ++mu)
            for (int i = 0; i < n_occ; ++i) {
                double s = 0.0;
                for (int j = 0; j < n_occ; ++j)
                    s += C(mu, j) * Uo(j, i);
                C_new(mu, i) = s;
            }
    }

    // Virtual block: C_virt ← C_virt · U_v.
    RealMatrix Uv = xt::eye<double>(static_cast<std::size_t>(std::max(nv, 0)));
    if (nv > 0) {
        RealMatrix Fvv = xt::zeros<double>({nv, nv});
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
                Fvv(a, b) = Fmo(n_occ + a, n_occ + b);
        auto [ev, U] = xt::linalg::eigh(Fvv);
        Uv = U;
        for (int a = 0; a < nv; ++a)
            eig(n_occ + a) = ev(a);
        for (int mu = 0; mu < n; ++mu)
            for (int a = 0; a < nv; ++a) {
                double s = 0.0;
                for (int b = 0; b < nv; ++b)
                    s += C(mu, n_occ + b) * Uv(b, a);
                C_new(mu, n_occ + a) = s;
            }
    }
    C = C_new;

    // Gradient block in the semicanonical basis: Uvᵀ F_vo Uo.
    RealMatrix g_blk = xt::zeros<double>({std::max(nv, 0), std::max(n_occ, 0)});
    for (int a = 0; a < nv; ++a)
        for (int i = 0; i < n_occ; ++i) {
            double s = 0.0;
            for (int b = 0; b < nv; ++b)
                for (int j = 0; j < n_occ; ++j)
                    s += Uv(b, a) * Fmo(n_occ + b, j) * Uo(j, i);
            g_blk(a, i) = s;
        }
    return g_blk;
}

xt::xtensor<double, 4> SOSCFStepper::mo_transform_(
    const xt::xtensor<double, 4>& eri_ao, const xt::xtensor<double, 2>& C0,
    const xt::xtensor<double, 2>& C1, const xt::xtensor<double, 2>& C2,
    const xt::xtensor<double, 2>& C3) {
    // Four rounds, each transforming the (current) last axis and rotating it
    // to the front — the OrbitalHessian::transform_block_ pattern.
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

void SOSCFStepper::rotate_channel_(const RealVector& kappa, int n_occ,
                                   RealMatrix& C) {
    const int m = static_cast<int>(C.shape()[1]);
    const int nv = m - n_occ;
    RealMatrix K = xt::zeros<double>(
        {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
    for (int a = 0; a < nv; ++a)
        for (int i = 0; i < n_occ; ++i) {
            const double k_ai = kappa(a * n_occ + i);
            K(n_occ + a, i) = k_ai;
            K(i, n_occ + a) = -k_ai;
        }
    C = xt::linalg::dot(C, exp_antisymmetric(K));
}

RealMatrix SOSCFStepper::exp_antisymmetric(const RealMatrix& K) {
    if (K.shape()[0] != K.shape()[1])
        throw std::invalid_argument(
            "SOSCFStepper::exp_antisymmetric: non-square generator ("
            + std::to_string(K.shape()[0]) + "x" + std::to_string(K.shape()[1])
            + ")");
    const std::size_t m = K.shape()[0];
    double asym = 0.0, norm2 = 0.0;
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < m; ++c) {
            asym = std::max(asym, std::abs(K(r, c) + K(c, r)));
            norm2 += K(r, c) * K(r, c);
        }
    if (asym > 1e-12)
        throw std::invalid_argument(
            "SOSCFStepper::exp_antisymmetric: generator is not antisymmetric "
            "(max|K + K^T| = "
            + std::to_string(asym) + " > 1e-12 — no fallback)");
    const double fro = std::sqrt(norm2);

    // Scaling: bring ‖K/2^s‖_F under 0.25 so the Taylor tail is < 1e-17
    // within ~20 terms; then square s times (exp of antisymmetric stays
    // exactly orthogonal in exact arithmetic).
    int s = 0;
    double scaled = fro;
    while (scaled > 0.25 && s < 60) {
        scaled *= 0.5;
        ++s;
    }
    RealMatrix A = K / std::pow(2.0, s);
    RealMatrix U = xt::eye<double>(m);
    RealMatrix term = U;
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

SOSCFStepper::AugmentedRoot SOSCFStepper::augmented_hessian_root(
    const RealVector& g,
    const std::function<RealVector(const RealVector&)>& h_op,
    const RealVector& h_diag, int max_iter) {
    if (g.size() != h_diag.size())
        throw std::invalid_argument(
            "SOSCFStepper::augmented_hessian_root: g size "
            + std::to_string(g.size()) + " != h_diag size "
            + std::to_string(h_diag.size()));
    const int n = static_cast<int>(g.size());
    const int N = n + 1;

    // Augmented product y = [[0, gᵀ], [g, H]] · u, matrix-free in H.
    auto aug = [&](const RealVector& u) {
        RealVector body = xt::zeros<double>({static_cast<std::size_t>(n)});
        for (int r = 0; r < n; ++r)
            body(r) = u(r + 1);
        const RealVector Hb = h_op(body);
        RealVector y = xt::zeros<double>({static_cast<std::size_t>(N)});
        double top = 0.0;
        for (int r = 0; r < n; ++r)
            top += g(r) * u(r + 1);
        y(0) = top;
        for (int r = 0; r < n; ++r)
            y(r + 1) = g(r) * u(0) + Hb(r);
        return y;
    };

    const int max_subspace = std::min(N, 40);
    const double res_tol = 1e-11;   // ‖r‖₂ threshold [Hartree]
    const double theta_tol = 1e-14; // Ritz stagnation [Hartree]

    std::vector<RealVector> V, W;
    RealVector b = xt::zeros<double>({static_cast<std::size_t>(N)});
    b(0) = 1.0; // deterministic seed: the scalar row of the AH matrix
    V.push_back(b);
    W.push_back(aug(b));

    double theta = 0.0;
    RealVector x = b;
    double theta_old = 0.0;
    bool have_old = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        const int m = static_cast<int>(V.size());
        RealMatrix Hs = xt::zeros<double>(
            {static_cast<std::size_t>(m), static_cast<std::size_t>(m)});
        for (int p = 0; p < m; ++p)
            for (int q = 0; q < m; ++q) {
                double sacc = 0.0;
                for (int r = 0; r < N; ++r)
                    sacc += V[static_cast<std::size_t>(p)](r)
                            * W[static_cast<std::size_t>(q)](r);
                Hs(p, q) = sacc;
            }
        for (int p = 0; p < m; ++p)
            for (int q = p + 1; q < m; ++q) {
                const double sacc = 0.5 * (Hs(p, q) + Hs(q, p));
                Hs(p, q) = sacc;
                Hs(q, p) = sacc;
            }
        auto [w, Sv] = xt::linalg::eigh(Hs);
        theta = w(0);

        x = xt::zeros<double>({static_cast<std::size_t>(N)});
        RealVector Ax = xt::zeros<double>({static_cast<std::size_t>(N)});
        for (int p = 0; p < m; ++p) {
            const double c = Sv(p, 0);
            for (int r = 0; r < N; ++r) {
                x(r) += c * V[static_cast<std::size_t>(p)](r);
                Ax(r) += c * W[static_cast<std::size_t>(p)](r);
            }
        }
        RealVector rvec = Ax - theta * x;
        double rnorm = 0.0;
        for (int r = 0; r < N; ++r)
            rnorm += rvec(r) * rvec(r);
        rnorm = std::sqrt(rnorm);

        const bool stagnant =
            have_old && std::abs(theta - theta_old) < theta_tol;
        if (rnorm < res_tol || (stagnant && rnorm < 1e-7) || m == N)
            break;
        theta_old = theta;
        have_old = true;

        // Davidson preconditioner: diagonal (0, h_diag) with clamping.
        RealVector t = xt::zeros<double>({static_cast<std::size_t>(N)});
        for (int r = 0; r < N; ++r) {
            const double d = (r == 0) ? 0.0 : h_diag(r - 1);
            double denom = theta - d;
            if (std::abs(denom) < 1e-8)
                denom = (denom < 0.0) ? -1e-8 : 1e-8;
            t(r) = rvec(r) / denom;
        }

        // Collapse to the Ritz vector when the subspace is full.
        if (m >= max_subspace) {
            RealVector Ax_kept = Ax;
            double xn = 0.0;
            for (int r = 0; r < N; ++r)
                xn += x(r) * x(r);
            xn = std::sqrt(xn);
            RealVector x_kept = x;
            for (int r = 0; r < N; ++r) {
                x_kept(r) /= xn;
                Ax_kept(r) /= xn;
            }
            V.clear();
            W.clear();
            V.push_back(x_kept);
            W.push_back(Ax_kept);
        }

        // Two MGS passes against the subspace, then expand.
        for (int pass = 0; pass < 2; ++pass)
            for (const RealVector& v : V) {
                double proj = 0.0;
                for (int r = 0; r < N; ++r)
                    proj += v(r) * t(r);
                for (int r = 0; r < N; ++r)
                    t(r) -= proj * v(r);
            }
        double tn = 0.0;
        for (int r = 0; r < N; ++r)
            tn += t(r) * t(r);
        tn = std::sqrt(tn);
        if (tn < 1e-12)
            break; // no new direction at working precision
        for (int r = 0; r < N; ++r)
            t(r) /= tn;
        V.push_back(t);
        W.push_back(aug(t));
    }

    // Normalize the Ritz vector (V orthonormal ⇒ ‖x‖ ≈ 1 already).
    double xn = 0.0;
    for (int r = 0; r < N; ++r)
        xn += x(r) * x(r);
    xn = std::sqrt(xn);
    for (int r = 0; r < N; ++r)
        x(r) /= xn;
    AugmentedRoot root;
    root.lambda = theta;
    root.x = x;
    return root;
}

void SOSCFStepper::step() {
    validate_();
    const int m = static_cast<int>(C_alpha_.shape()[1]);
    const int no_a = n_alpha_;
    const int nv_a = m - no_a;
    const int no_b = restricted_ ? no_a : n_beta_;
    const int nv_b = restricted_ ? nv_a : (m - no_b);
    const int rot_a = nv_a * no_a;
    const int rot_b = restricted_ ? 0 : nv_b * no_b;
    const int n_rot = restricted_ ? rot_a : (rot_a + rot_b);
    if (n_rot == 0)
        return; // empty rotation space — nothing to optimize
    ++macro_steps_;

    // ── Trust region: accept/reject the PREVIOUS rotation ────────────────
    const double e_cur = current_energy();
    if (have_prev_ && e_cur > prev_energy_ + kEnergyRiseTol) {
        C_alpha_ = saved_C_alpha_;
        eig_alpha_ = saved_eig_alpha_;
        if (restricted_) {
            C_beta_ = C_alpha_;
            eig_beta_ = eig_alpha_;
        } else {
            C_beta_ = saved_C_beta_;
            eig_beta_ = saved_eig_beta_;
        }
        radius_ *= 0.5;
        rejected_ = true;
        return; // the loop rebuilds density/Fock of the restored point
    }
    rejected_ = false;

    // ── Early exit on a stationary iterate (exact no-op, C untouched) ────
    const RealVector g_probe = gradient();
    double g_inf = 0.0;
    for (std::size_t p = 0; p < g_probe.size(); ++p)
        g_inf = std::max(g_inf, std::abs(g_probe(p)));
    if (g_inf < kGradZeroTol)
        return;

    // ── Semicanonicalize (density-invariant; makes ε-based σ exact) ──────
    const RealMatrix gblk_a =
        semicanonicalize_(F_alpha_, C_alpha_, eig_alpha_, no_a);
    RealMatrix gblk_b;
    if (restricted_) {
        C_beta_ = C_alpha_;
        eig_beta_ = eig_alpha_;
    } else {
        gblk_b = semicanonicalize_(F_beta_, C_beta_, eig_beta_, no_b);
    }

    // Accept the current point: snapshot it for a possible later reject.
    prev_energy_ = e_cur;
    have_prev_ = true;
    saved_C_alpha_ = C_alpha_;
    saved_eig_alpha_ = eig_alpha_;
    if (!restricted_) {
        saved_C_beta_ = C_beta_;
        saved_eig_beta_ = eig_beta_;
    }

    // Dense working-dimension ERI, cached per run (geometry is fixed).
    if (eri_.size() == 0)
        eri_ = (ints_.ERI.size() > 0) ? ints_.ERI : ints_.eri_working();

    // ── Gradient vector + matrix-free Hessian products ────────────────────
    RealVector g = xt::zeros<double>({static_cast<std::size_t>(n_rot)});
    std::function<RealVector(const RealVector&)> h_op;
    RealVector h_diag;

    // Storage that must outlive h_op (captured by reference semantics).
    std::shared_ptr<OrbitalHessian> hess;
    xt::xtensor<double, 4> ovovAA, oovvAA, ovovBB, oovvBB, ovovAB;
    RealVector eo_a, ev_a, eo_b, ev_b;

    if (restricted_) {
        for (int a = 0; a < nv_a; ++a)
            for (int i = 0; i < no_a; ++i)
                g(a * no_a + i) = 4.0 * gblk_a(a, i);
        RealVector f = xt::zeros<double>({static_cast<std::size_t>(m)});
        for (int i = 0; i < no_a; ++i)
            f(i) = 1.0;
        const SlaterDeterminant det(C_alpha_, eig_alpha_, f, no_a);
        hess = std::make_shared<OrbitalHessian>(det, eri_);
        h_op = [hess](const RealVector& v) {
            return RealVector(4.0 * hess->sigma_plus(v, Channel::singlet));
        };
        h_diag = 4.0 * hess->diagonal_plus(Channel::singlet);
    } else {
        for (int a = 0; a < nv_a; ++a)
            for (int i = 0; i < no_a; ++i)
                g(a * no_a + i) = 2.0 * gblk_a(a, i);
        for (int a = 0; a < nv_b; ++a)
            for (int i = 0; i < no_b; ++i)
                g(rot_a + a * no_b + i) = 2.0 * gblk_b(a, i);

        // Occ/virt slices per spin (row-major copies for the transform).
        const int n = static_cast<int>(C_alpha_.shape()[0]);
        auto slice = [n](const RealMatrix& C, int lo, int cnt) {
            xt::xtensor<double, 2> out = xt::zeros<double>({n, cnt});
            for (int mu = 0; mu < n; ++mu)
                for (int p = 0; p < cnt; ++p)
                    out(mu, p) = C(mu, lo + p);
            return out;
        };
        const auto Co_a = slice(C_alpha_, 0, no_a);
        const auto Cv_a = slice(C_alpha_, no_a, nv_a);
        const auto Co_b = slice(C_beta_, 0, no_b);
        const auto Cv_b = slice(C_beta_, no_b, nv_b);

        if (rot_a > 0) {
            ovovAA = mo_transform_(eri_, Cv_a, Co_a, Cv_a, Co_a);
            oovvAA = mo_transform_(eri_, Co_a, Co_a, Cv_a, Cv_a);
        }
        if (rot_b > 0) {
            ovovBB = mo_transform_(eri_, Cv_b, Co_b, Cv_b, Co_b);
            oovvBB = mo_transform_(eri_, Co_b, Co_b, Cv_b, Cv_b);
        }
        if (rot_a > 0 && rot_b > 0)
            ovovAB = mo_transform_(eri_, Cv_a, Co_a, Cv_b, Co_b);

        eo_a = xt::view(eig_alpha_, xt::range(0, no_a));
        ev_a = xt::view(eig_alpha_, xt::range(no_a, m));
        eo_b = xt::view(eig_beta_, xt::range(0, no_b));
        ev_b = xt::view(eig_beta_, xt::range(no_b, m));

        h_op = [=, &ovovAA, &oovvAA, &ovovBB, &oovvBB,
                &ovovAB](const RealVector& v) {
            RealVector y =
                xt::zeros<double>({static_cast<std::size_t>(rot_a + rot_b)});
            // α rows
            for (int a = 0; a < nv_a; ++a)
                for (int i = 0; i < no_a; ++i) {
                    const int p = a * no_a + i;
                    double acc = (ev_a(a) - eo_a(i)) * v(p);
                    for (int b = 0; b < nv_a; ++b)
                        for (int j = 0; j < no_a; ++j)
                            acc += (2.0 * ovovAA(a, i, b, j)
                                    - oovvAA(i, j, a, b) - ovovAA(a, j, b, i))
                                   * v(b * no_a + j);
                    if (rot_b > 0)
                        for (int b = 0; b < nv_b; ++b)
                            for (int j = 0; j < no_b; ++j)
                                acc += 2.0 * ovovAB(a, i, b, j)
                                       * v(rot_a + b * no_b + j);
                    y(p) = 2.0 * acc;
                }
            // β rows
            for (int a = 0; a < nv_b; ++a)
                for (int i = 0; i < no_b; ++i) {
                    const int p = rot_a + a * no_b + i;
                    double acc = (ev_b(a) - eo_b(i)) * v(p);
                    for (int b = 0; b < nv_b; ++b)
                        for (int j = 0; j < no_b; ++j)
                            acc += (2.0 * ovovBB(a, i, b, j)
                                    - oovvBB(i, j, a, b) - ovovBB(a, j, b, i))
                                   * v(rot_a + b * no_b + j);
                    if (rot_a > 0)
                        for (int b = 0; b < nv_a; ++b)
                            for (int j = 0; j < no_a; ++j)
                                acc +=
                                    2.0 * ovovAB(b, j, a, i) * v(b * no_a + j);
                    y(p) = 2.0 * acc;
                }
            return y;
        };

        h_diag = xt::zeros<double>({static_cast<std::size_t>(n_rot)});
        for (int a = 0; a < nv_a; ++a)
            for (int i = 0; i < no_a; ++i)
                h_diag(a * no_a + i) =
                    2.0
                    * (ev_a(a) - eo_a(i)
                       + (rot_a > 0 ? (ovovAA(a, i, a, i) - oovvAA(i, i, a, a))
                                    : 0.0));
        for (int a = 0; a < nv_b; ++a)
            for (int i = 0; i < no_b; ++i)
                h_diag(rot_a + a * no_b + i) =
                    2.0
                    * (ev_b(a) - eo_b(i)
                       + (rot_b > 0 ? (ovovBB(a, i, a, i) - oovvBB(i, i, a, a))
                                    : 0.0));
    }

    // ── CIAH: lowest root of the augmented Hessian, matrix-free ──────────
    const AugmentedRoot root =
        augmented_hessian_root(g, h_op, h_diag, max_micro_);
    last_lambda_ = root.lambda;

    RealVector kappa = xt::zeros<double>({static_cast<std::size_t>(n_rot)});
    const double x0 = root.x(0);
    double knorm = 0.0;
    for (int p = 0; p < n_rot; ++p) {
        kappa(p) = root.x(p + 1);
        knorm += kappa(p) * kappa(p);
    }
    knorm = std::sqrt(knorm);
    if (knorm == 0.0)
        return; // stationary root — no rotation
    if (std::abs(x0) >= kX0Floor) {
        for (int p = 0; p < n_rot; ++p)
            kappa(p) /= x0;
        knorm /= std::abs(x0);
    } else {
        // Pure Hessian mode: direction only, scaled to the trust radius.
        for (int p = 0; p < n_rot; ++p)
            kappa(p) *= radius_ / knorm;
        knorm = radius_;
    }
    if (knorm > radius_) {
        for (int p = 0; p < n_rot; ++p)
            kappa(p) *= radius_ / knorm;
        knorm = radius_;
    }
    last_step_norm_ = knorm;

    // ── Unitary update C ← C · exp(κ) per channel ─────────────────────────
    RealVector kappa_a = xt::view(kappa, xt::range(0, rot_a));
    rotate_channel_(kappa_a, no_a, C_alpha_);
    if (restricted_) {
        C_beta_ = C_alpha_;
    } else if (rot_b > 0) {
        RealVector kappa_b = xt::view(kappa, xt::range(rot_a, rot_a + rot_b));
        rotate_channel_(kappa_b, no_b, C_beta_);
    }
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
