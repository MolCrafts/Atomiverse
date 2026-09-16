/// @file cpu/scf/internal/density_fitting.cpp
/// @brief internal::DensityFitting — RI-J / RI-JK workspace construction
///        and the fitted J/K contractions. @see density_fitting.h for the
///        conventions and the DOI trail (Vahtras 1993; Weigend 2002/2006).

#include "atomiverse/cpu/scf/internal/density_fitting.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/cart2sph.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/rys_eri.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

// Unit zero-exponent s dummy shell: exp(-0·r²)·1 is the constant
// function 1, so a 4-center quartet with this shell on one ket slot
// evaluates the 3-center Coulomb integral (μν|P) exactly (the trick
// PySCF/libint use for 3c2e). Coefficient 1, deliberately UNNORMALIZED
// (a zero-exponent Gaussian has no finite norm); centered on its
// partner so every center difference the kernel forms is 0.
struct DummyS {
    double zero = 0.0;
    double one = 1.0;
    native::ShellBlock block(const double center[3]) const {
        native::ShellBlock b;
        b.n_prim = 1;
        b.alpha = &zero;
        b.coeff = &one;
        b.center[0] = center[0];
        b.center[1] = center[1];
        b.center[2] = center[2];
        b.l = 0;
        b.n_cart = 1;
        b.ao_start = 0;
        return b;
    }
};

// Contracted same-center Cartesian overlap of two components of one aux
// shell block (needed to self-normalize the spherical aux columns the
// same way internal::IntegralWorkspace::c2s does for the orbital basis).
double intra_block_overlap(const native::ShellBlock& b, const int ang_i[3],
                           const int ang_k[3]) {
    double s = 0.0;
    for (int p = 0; p < b.n_prim; ++p)
        for (int q = 0; q < b.n_prim; ++q)
            s += b.coeff[p] * b.coeff[q]
                 * native::overlap_prim_3d(ang_i, ang_k, b.alpha[p], b.center,
                                           b.alpha[q], b.center);
    return s;
}

} // namespace

DensityFitting::DensityFitting(const std::vector<native::Shell>& orbital_shells,
                               const std::vector<native::Atom>& atoms,
                               AuxBasis aux, double lindep_threshold)
    : lindep_threshold_(lindep_threshold) {
    if (orbital_shells.empty() || atoms.empty())
        throw std::runtime_error(
            "internal::DensityFitting: empty orbital shell list or atom "
            "list.");

    // ── Auxiliary shells (throws for elements without a JKFIT row) ────
    AtomList aux_atoms;
    aux_atoms.atoms = atoms;
    const ShellList aux_list(aux_atoms, aux);
    const auto aux_blocks = blocks_from_shells(aux_list.shells);
    const auto orb_blocks = blocks_from_shells(orbital_shells);

    n_ao_ = static_cast<int>(orbital_shells.size());
    n_aux_ = 0;
    for (const auto& b : aux_blocks)
        n_aux_ += 2 * b.l + 1;

    // ── Spherical transform per aux block, self-normalized in the aux
    //    Cartesian overlap metric (mirrors internal::IntegralWorkspace::c2s: the engine
    //    primitive-normalizes but does not unit-normalize contractions).
    //    def2-universal-jkfit is a spherical basis family — fitting in
    //    the raw Cartesian aux span would span extra lower-l functions
    //    and NOT reproduce PySCF's numbers (see density_fitting.h).
    const CartToSph c2s_tables;
    std::vector<RealMatrix> aux_c2s(aux_blocks.size());
    std::vector<int> aux_sph_start(aux_blocks.size());
    {
        int col = 0;
        for (std::size_t ib = 0; ib < aux_blocks.size(); ++ib) {
            const auto& b = aux_blocks[ib];
            const auto& m = c2s_tables.block(b.l);
            const int nc = b.n_cart;
            const int ns = 2 * b.l + 1;
            int ang[kMaxNCart][3];
            fill_cart_comps(b.l, ang);
            RealMatrix S_blk =
                xt::zeros<double>({std::size_t(nc), std::size_t(nc)});
            for (int i = 0; i < nc; ++i)
                for (int k = 0; k <= i; ++k) {
                    const double v = intra_block_overlap(b, ang[i], ang[k]);
                    S_blk(i, k) = v;
                    S_blk(k, i) = v;
                }
            RealMatrix C =
                xt::zeros<double>({std::size_t(nc), std::size_t(ns)});
            for (int j = 0; j < ns; ++j) {
                std::vector<double> v(static_cast<std::size_t>(nc));
                for (int i = 0; i < nc; ++i)
                    v[static_cast<std::size_t>(i)] =
                        m(i, j) / std::sqrt(S_blk(i, i));
                double nrm2 = 0.0;
                for (int i = 0; i < nc; ++i)
                    for (int k = 0; k < nc; ++k)
                        nrm2 += v[static_cast<std::size_t>(i)] * S_blk(i, k)
                                * v[static_cast<std::size_t>(k)];
                const double inv = 1.0 / std::sqrt(nrm2);
                for (int i = 0; i < nc; ++i)
                    C(i, j) = v[static_cast<std::size_t>(i)] * inv;
            }
            aux_c2s[ib] = std::move(C);
            aux_sph_start[ib] = col;
            col += ns;
        }
    }

    const DummyS dummy;
    const auto t_v = LogClock::now();

    // ── 2-center Coulomb metric V_PQ = (P|Q), spherical aux basis ─────
    V_ = xt::zeros<double>({std::size_t(n_aux_), std::size_t(n_aux_)});
    for (std::size_t ip = 0; ip < aux_blocks.size(); ++ip) {
        const auto& P = aux_blocks[ip];
        const auto Pd = dummy.block(P.center);
        for (std::size_t iq = 0; iq <= ip; ++iq) {
            const auto& Q = aux_blocks[iq];
            const auto Qd = dummy.block(Q.center);
            std::vector<double> buf(
                static_cast<std::size_t>(P.n_cart * Q.n_cart), 0.0);
            native::rys_eri_block(P, Pd, Q, Qd, buf.data());
            auto blk = xt::adapt(
                buf.data(), static_cast<std::size_t>(P.n_cart * Q.n_cart),
                xt::no_ownership(),
                std::vector<std::size_t>{std::size_t(P.n_cart),
                                         std::size_t(Q.n_cart)});
            // (P.n_cart × Q.n_cart) → spherical sandwich on both sides.
            const RealMatrix sph =
                xt::linalg::dot(xt::transpose(aux_c2s[ip]),
                                RealMatrix(xt::linalg::dot(blk, aux_c2s[iq])));
            const int np = 2 * P.l + 1;
            const int nq = 2 * Q.l + 1;
            for (int a = 0; a < np; ++a)
                for (int c = 0; c < nq; ++c) {
                    V_(aux_sph_start[ip] + a, aux_sph_start[iq] + c) =
                        sph(a, c);
                    V_(aux_sph_start[iq] + c, aux_sph_start[ip] + a) =
                        sph(a, c);
                }
        }
    }

    ATV_DEBUG("[df] 2-center metric ms={:.1f}", elapsed_ms(t_v));
    const auto t_chol = LogClock::now();

    // ── Cholesky V = L·Lᵀ → whitening factor W = L⁻¹ ──────────────────
    RealMatrix L;
    try {
        L = xt::linalg::cholesky(V_);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("internal::DensityFitting: Cholesky of the "
                        "auxiliary Coulomb metric failed (metric not "
                        "positive definite — linearly dependent auxiliary "
                        "functions?): ")
            + e.what());
    }
    {
        double dmin = L(0, 0), dmax = L(0, 0);
        for (int i = 0; i < n_aux_; ++i) {
            const double d = L(i, i);
            if (!std::isfinite(d) || d <= 0.0)
                dmin = 0.0;
            else {
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }
        }
        if (dmin * dmin < lindep_threshold_ * dmax * dmax)
            throw std::runtime_error(
                "internal::DensityFitting: auxiliary Coulomb metric is "
                "near-singular (Cholesky pivot ratio "
                + std::to_string(dmin * dmin / (dmax * dmax))
                + " below lindep_threshold " + std::to_string(lindep_threshold_)
                + ") — linearly dependent auxiliary functions.");
    }
    W_ = xt::linalg::solve(
        L, RealMatrix(xt::eye<double>(static_cast<std::size_t>(n_aux_))));
    ATV_DEBUG("[df] cholesky+whiten ms={:.1f}", elapsed_ms(t_chol));
    const auto t_3c = LogClock::now();

    // ── 3-center (μν|P) → spherical aux index → B = (μν|Q) · Wᵀ ───────
    // Layout note: RealMatrix is COLUMN-major, so with the (μν) pair
    // index on the rows every auxiliary column of M3 / B_ is one
    // contiguous n_ao² slice — exactly the M_P matrix k_fitted walks.
    const std::size_t n2 =
        static_cast<std::size_t>(n_ao_) * static_cast<std::size_t>(n_ao_);
    RealMatrix M3 = xt::zeros<double>({n2, std::size_t(n_aux_)});

    // Canonical orbital block pairs (bi >= bj); each pair writes a
    // disjoint set of (μν) rows of M3, so the pair loop is race-free.
    std::vector<std::pair<int, int>> pairs;
    const int n_orb_blocks = static_cast<int>(orb_blocks.size());
    pairs.reserve(static_cast<std::size_t>(n_orb_blocks)
                  * static_cast<std::size_t>(n_orb_blocks + 1) / 2);
    for (int bi = 0; bi < n_orb_blocks; ++bi)
        for (int bj = 0; bj <= bi; ++bj)
            pairs.emplace_back(bi, bj);

    // Cauchy-Schwarz prescreen: |(μν|P)| ≤ Q_AB·Q_P with
    // Q_AB = √max|(AB|AB)| and Q_P = √max(P|P). Exponentially decaying
    // bra pairs drop out for EVERY auxiliary block; the 1e-11 Hartree
    // product threshold perturbs the fitted energy far below the 50 μHa
    // regression gate.
    constexpr double kSchwarzThr = 1e-11;
    std::vector<double> q_pair(pairs.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ip = 0; ip < static_cast<std::ptrdiff_t>(pairs.size());
         ++ip) {
        const auto& A = orb_blocks[static_cast<std::size_t>(
            pairs[static_cast<std::size_t>(ip)].first)];
        const auto& Bb = orb_blocks[static_cast<std::size_t>(
            pairs[static_cast<std::size_t>(ip)].second)];
        const int nab = A.n_cart * Bb.n_cart;
        std::vector<double> qb(static_cast<std::size_t>(nab) * nab, 0.0);
        native::rys_eri_block(A, Bb, A, Bb, qb.data());
        double qmax = 0.0;
        for (int ab = 0; ab < nab; ++ab)
            qmax =
                std::max(qmax, std::fabs(qb[static_cast<std::size_t>(ab) * nab
                                            + static_cast<std::size_t>(ab)]));
        q_pair[static_cast<std::size_t>(ip)] = std::sqrt(qmax);
    }
    std::vector<double> q_aux(aux_blocks.size(), 0.0);
    for (std::size_t ipb = 0; ipb < aux_blocks.size(); ++ipb) {
        const auto& P = aux_blocks[ipb];
        const auto Pd = dummy.block(P.center);
        std::vector<double> qb(static_cast<std::size_t>(P.n_cart) * P.n_cart,
                               0.0);
        native::rys_eri_block(P, Pd, P, Pd, qb.data());
        double qmax = 0.0;
        for (int c = 0; c < P.n_cart; ++c)
            qmax = std::max(qmax,
                            std::fabs(qb[static_cast<std::size_t>(c) * P.n_cart
                                         + static_cast<std::size_t>(c)]));
        q_aux[ipb] = std::sqrt(qmax);
    }

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t ipair = 0;
         ipair < static_cast<std::ptrdiff_t>(pairs.size()); ++ipair) {
        const auto& A = orb_blocks[static_cast<std::size_t>(
            pairs[static_cast<std::size_t>(ipair)].first)];
        const auto& Bb = orb_blocks[static_cast<std::size_t>(
            pairs[static_cast<std::size_t>(ipair)].second)];
        const bool diag = (&A == &Bb);
        const int na = A.n_cart;
        const int nb = Bb.n_cart;
        const double q_ab = q_pair[static_cast<std::size_t>(ipair)];
        std::vector<double> buf;
        for (std::size_t ipb = 0; ipb < aux_blocks.size(); ++ipb) {
            if (q_ab * q_aux[ipb] < kSchwarzThr)
                continue;
            const auto& P = aux_blocks[ipb];
            const auto Pd = dummy.block(P.center);
            const int ncp = P.n_cart;
            buf.assign(static_cast<std::size_t>(na * nb * ncp), 0.0);
            native::rys_eri_block(A, Bb, P, Pd, buf.data());
            // Spherical transform of the aux index + scatter, fused as
            // plain loops (no per-call heap temporaries — this triple
            // loop runs once per surviving (pair, aux-block)).
            const RealMatrix& C = aux_c2s[ipb];
            const int ns = 2 * P.l + 1;
            const int q0 = aux_sph_start[ipb];
            for (int ia = 0; ia < na; ++ia) {
                const std::size_t mu =
                    static_cast<std::size_t>(A.ao_start + ia);
                for (int ib = 0; ib < nb; ++ib) {
                    const std::size_t nu =
                        static_cast<std::size_t>(Bb.ao_start + ib);
                    const double* row =
                        buf.data()
                        + static_cast<std::size_t>(ia * nb + ib) * ncp;
                    const std::size_t row_mn =
                        mu * static_cast<std::size_t>(n_ao_) + nu;
                    const std::size_t row_nm =
                        nu * static_cast<std::size_t>(n_ao_) + mu;
                    for (int q = 0; q < ns; ++q) {
                        double v = 0.0;
                        for (int c = 0; c < ncp; ++c)
                            v += row[c] * C(c, q);
                        M3(row_mn, static_cast<std::size_t>(q0 + q)) = v;
                        if (!diag)
                            M3(row_nm, static_cast<std::size_t>(q0 + q)) = v;
                    }
                }
            }
        }
    }

    ATV_DEBUG("[df] 3-center integrals ms={:.1f}", elapsed_ms(t_3c));
    const auto t_gemm = LogClock::now();

    // B_{μν,P} = Σ_Q (μν|Q) W_PQ — one GEMM; V⁻¹ = Wᵀ·W makes
    // J = B·(Bᵀ·p) and K = Σ_P B^P C (B^P C)ᵀ the exact robust-fit
    // contractions. Wᵀ is materialized first: a lazy transpose view
    // would push xt::linalg::dot off the dgemm fast path onto its
    // generic fallback (measured 2.3 s vs 30 ms on glycine/def2-SVP).
    const RealMatrix Wt = xt::transpose(W_);
    B_ = xt::linalg::dot(M3, Wt);
    ATV_DEBUG("[df] B contraction ms={:.1f}", elapsed_ms(t_gemm));
}

RealMatrix DensityFitting::j_fitted(const RealMatrix& P_total) const {
    const std::size_t n = static_cast<std::size_t>(n_ao_);
    // vec(P) in the row index convention of B_ (r = μ·n + ν); explicit
    // element loops keep this layout-agnostic (RealMatrix is
    // column-major).
    RealVector pvec = xt::zeros<double>({n * n});
    for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
            pvec(mu * n + nu) = P_total(mu, nu);
    const RealVector gamma = xt::linalg::dot(xt::transpose(B_), pvec);
    const RealVector jv = xt::linalg::dot(B_, gamma);
    RealMatrix J = xt::zeros<double>({n, n});
    for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
            J(mu, nu) = jv(mu * n + nu);
    return J;
}

RealMatrix DensityFitting::k_fitted(const RealMatrix& C_occ) const {
    const std::size_t n = static_cast<std::size_t>(n_ao_);
    const std::size_t naux = static_cast<std::size_t>(n_aux_);
    const std::size_t nocc = C_occ.shape()[1];
    if (nocc == 0)
        return xt::zeros<double>({n, n});
    // Per-P half transform T^P_μi = Σ_ν B^P_μν C_νi followed by the
    // outer-product accumulation K += T^P·(T^P)ᵀ. B_ is column-major
    // (n², n_aux), so column P is one contiguous n² slice bp with
    // bp[μ·n + ν] = B^P_μν. OpenMP over P with thread-local
    // accumulators, then a lower-triangle reduce + symmetrize.
    RealMatrix K = xt::zeros<double>({n, n});
    const double* bd = B_.data();
#pragma omp parallel
    {
        std::vector<double> t(n * nocc);
        std::vector<double> k_loc(n * n, 0.0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(naux); ++p) {
            const double* bp = bd + static_cast<std::size_t>(p) * n * n;
            for (std::size_t mu = 0; mu < n; ++mu)
                for (std::size_t i = 0; i < nocc; ++i) {
                    double s = 0.0;
                    for (std::size_t nu = 0; nu < n; ++nu)
                        s += bp[mu * n + nu] * C_occ(nu, i);
                    t[mu * nocc + i] = s;
                }
            for (std::size_t mu = 0; mu < n; ++mu)
                for (std::size_t nu = 0; nu <= mu; ++nu) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < nocc; ++i)
                        s += t[mu * nocc + i] * t[nu * nocc + i];
                    k_loc[mu * n + nu] += s;
                }
        }
#pragma omp critical(atv_df_k_reduce)
        for (std::size_t mu = 0; mu < n; ++mu)
            for (std::size_t nu = 0; nu <= mu; ++nu)
                K(mu, nu) += k_loc[mu * n + nu];
    }
    for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = mu + 1; nu < n; ++nu)
            K(mu, nu) = K(nu, mu);
    return K;
}

RealMatrix DensityFitting::k_fitted_from_density(const RealMatrix& P) const {
    const std::size_t n = static_cast<std::size_t>(n_ao_);
    auto ew = xt::linalg::eigh(P);
    const auto& lam = std::get<0>(ew);
    const auto& U = std::get<1>(ew);
    double amax = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        amax = std::max(amax, std::fabs(lam(i)));
    const double eps = 1e-14 * amax;
    std::vector<std::size_t> pos, neg;
    for (std::size_t i = 0; i < n; ++i) {
        if (lam(i) > eps)
            pos.push_back(i);
        else if (lam(i) < -eps)
            neg.push_back(i);
    }
    RealMatrix K = xt::zeros<double>({n, n});
    const auto accumulate = [&](const std::vector<std::size_t>& idx,
                                double sign) {
        if (idx.empty())
            return;
        RealMatrix C = xt::zeros<double>({n, idx.size()});
        for (std::size_t c = 0; c < idx.size(); ++c) {
            const double s = std::sqrt(std::fabs(lam(idx[c])));
            for (std::size_t r = 0; r < n; ++r)
                C(r, c) = s * U(r, idx[c]);
        }
        K += sign * k_fitted(C);
    };
    accumulate(pos, 1.0);
    accumulate(neg, -1.0);
    return K;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
