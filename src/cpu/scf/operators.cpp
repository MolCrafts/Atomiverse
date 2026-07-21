/// @file cpu/scf/operators.cpp
/// @brief Two-electron Fock operator members: stored-ERI
///        (IntegralPack::coulomb_exchange_fused) and direct-SCF
///        (ShellPairList / ShellBlockPairList members).

#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace scf {

// ── Stored-ERI operator (BLAS via xtensor-blas) ─────────────────────────
//
// `IntegralPack::coulomb_exchange_fused` reshapes the 4-index ERI tensor into
// an [n²×n²] matrix, turning the Coulomb double-sum into a single DGEMV call.
// `HamiltonianBuilder` selects this path whenever `IntegralPack::ERI` is a
// materialized dense tensor; otherwise the direct-SCF
// `ShellBlockPairList::g_rhf` path runs without ever materializing ERI.

void IntegralPack::coulomb_exchange_fused(const RealMatrix& P_total,
                                          RealMatrix& G) const {
    const int n = n_basis;
    // PySCF approach: J via BLAS DGEMV, K via einsum-equivalent contraction.
    //   J(i,j) = einsum('ijkl,kl->', ERI, P)  → reshape(n²,n²) @ flatten(P)
    //   K(i,j) = einsum('ikjl,kl->', ERI, P)  → explicit contraction (no xt::einsum)
    //   G = J − 0.5 K
    //
    // K cannot use DGEMV without materializing a 227 MB transposed copy.
    // numpy.einsum avoids this internally — xtensor has no einsum, so we
    // write the equivalent contraction loop. This IS what einsum does.
    const auto ns = static_cast<std::size_t>(n);
    const auto nn = ns * ns;

    // J: single DGEMV. ERI is already contiguous row-major, so adapt its
    // storage as an [n²×n²] matrix zero-copy — the previous
    // `xt::eval(xt::reshape_view(...))` materialized the full n⁴ tensor
    // (the 227 MB of the running example) on every SCF iteration.
    auto P_flat = xt::eval(xt::reshape_view(P_total, {nn}));
    auto ERI_mat = xt::adapt(ERI.data(), nn * nn, xt::no_ownership(),
                             std::array<std::size_t, 2>{nn, nn});
    xt::xtensor<double, 1> J_flat = xt::linalg::dot(ERI_mat, P_flat);

    G = xt::reshape_view(J_flat, {ns, ns});

    // K: einsum('ikjl,kl->ij', ERI, P) — contiguous l-access, no allocation.
    const double* Pp = P_total.data(); // column-major
    double* Gp = G.data();             // column-major
    const double* Ep = ERI.data();     // row-major ERI(i,j,k,l)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double k_val = 0.0;
            for (int k = 0; k < n; ++k) {
                const double* eri_ikj = Ep + ((i * n + k) * n + j) * n;
                const double* p_k = Pp + k;
                for (int l = 0; l < n; ++l)
                    k_val += p_k[l * n] * eri_ikj[l];
            }
            Gp[i + j * n] -= 0.5 * k_val;
            if (i != j)
                Gp[j + i * n] -= 0.5 * k_val;
        }
    }
}

// ── Direct-SCF implementations ────────────────────────────────────────────

namespace {

// Accumulate v into M(p,q) and — only if p!=q — also into M(q,p).
// Called on thread-local matrices, so no race conditions.
inline void sym_add(RealMatrix& M, int p, int q, double v) {
    M(p, q) += v;
    if (p != q)
        M(q, p) += v;
}

// Number of OpenMP threads available (1 when OpenMP is disabled).
inline int n_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

} // anonymous namespace

void ShellPairList::g_rhf(const RealMatrix& P, RealMatrix& G) const {
    const ShellPairList& sp = *this;
    const int n = sp.n_basis;
    const int nT = n_threads();
    const double thr = sp.threshold;
    const int npairs = static_cast<int>(sp.pairs.size());

    std::vector<RealMatrix> Gv(nT, xt::zeros<double>({n, n}));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < npairs; ++left) {
        auto& Gl = Gv[thread_id()];
        const ShellPair& pij = sp.pairs[left];
        const int a = pij.i, b = pij.j;

        for (int right = 0; right <= left; ++right) {
            const ShellPair& pkl = sp.pairs[right];
            const int c = pkl.i, d = pkl.j;

            // Cauchy-Schwarz screening: |(ab|cd)| <= Q_ab * Q_cd
            if (pij.schwarz * pkl.schwarz < thr)
                continue;

            const double val = native::eri(sp.shells[a], sp.shells[b],
                                           sp.shells[c], sp.shells[d]);
            if (std::abs(val) < thr)
                continue;

            // ── Coulomb ──────────────────────────────────────────────
            // J(a,b) += fcd * P(c,d) * val   [from all (ab|kl) terms]
            // J(c,d) += fab * P(a,b) * val   [from (kl|ab) = val, if pair differs]
            const double fcd = (c == d) ? 1.0 : 2.0;
            const double fab = (a == b) ? 1.0 : 2.0;
            sym_add(Gl, a, b, fcd * P(c, d) * val);
            if (left != right)
                sym_add(Gl, c, d, fab * P(a, b) * val);

            // ── Exchange (G = J - 0.5 K) ─────────────────────────────
            // When left != right, each K(p,q) element receives contributions
            // from two of the 8 ERI permutations (e.g. perm (ba|cd) and its
            // mirror (cd|ba) both target K(b,c)).  sym_add correctly fills
            // both M(p,q) and M(q,p) when p != q, but misses the second hit
            // when p == q (diagonal).  Fix: use explicit double-add for the
            // left != right case so the diagonal is also doubled.
            // When left == right the "mirror" quartet is the same entry — no
            // doubling, and sym_add with the existing guard is correct.
            if (left != right) {
                Gl(a, c) -= 0.5 * P(b, d) * val;
                Gl(c, a) -= 0.5 * P(b, d) * val;
                if (a != b) {
                    Gl(b, c) -= 0.5 * P(a, d) * val;
                    Gl(c, b) -= 0.5 * P(a, d) * val;
                }
                if (c != d) {
                    Gl(a, d) -= 0.5 * P(b, c) * val;
                    Gl(d, a) -= 0.5 * P(b, c) * val;
                    if (a != b) {
                        Gl(b, d) -= 0.5 * P(a, c) * val;
                        Gl(d, b) -= 0.5 * P(a, c) * val;
                    }
                }
            } else {
                // left == right (a=c, b=d): mirror permutation is the same
                // canonical quartet — sym_add avoids double-counting correctly.
                sym_add(Gl, a, c, -0.5 * P(b, d) * val);
                if (a != b)
                    sym_add(Gl, b, c, -0.5 * P(a, d) * val);
                if (c != d) {
                    // K(a,d) already covered by sym_add(b,c,...) above
                    if (a != b)
                        sym_add(Gl, b, d, -0.5 * P(a, c) * val);
                }
            }
        }
    }

    // Reduce thread-local matrices into G.
    G = xt::zeros<double>({n, n});
    for (auto& gl : Gv)
        G += gl;
}

void ShellPairList::jk_uhf(const RealMatrix& P_total, const RealMatrix& P_alpha,
                           const RealMatrix& P_beta, RealMatrix& J,
                           RealMatrix& K_alpha, RealMatrix& K_beta) const {
    const ShellPairList& sp = *this;
    const int n = sp.n_basis;
    const int nT = n_threads();
    const double thr = sp.threshold;
    const int npairs = static_cast<int>(sp.pairs.size());

    // Thread-local matrices for J, Ka, Kb.
    std::vector<RealMatrix> Jv(nT, xt::zeros<double>({n, n}));
    std::vector<RealMatrix> Kav(nT, xt::zeros<double>({n, n}));
    std::vector<RealMatrix> Kbv(nT, xt::zeros<double>({n, n}));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < npairs; ++left) {
        const int tid = thread_id();
        auto& Jl = Jv[tid];
        auto& Kal = Kav[tid];
        auto& Kbl = Kbv[tid];

        const ShellPair& pij = sp.pairs[left];
        const int a = pij.i, b = pij.j;

        for (int right = 0; right <= left; ++right) {
            const ShellPair& pkl = sp.pairs[right];
            const int c = pkl.i, d = pkl.j;

            if (pij.schwarz * pkl.schwarz < thr)
                continue;

            const double val = native::eri(sp.shells[a], sp.shells[b],
                                           sp.shells[c], sp.shells[d]);
            if (std::abs(val) < thr)
                continue;

            // ── Coulomb (uses P_total) ────────────────────────────────
            const double fcd = (c == d) ? 1.0 : 2.0;
            const double fab = (a == b) ? 1.0 : 2.0;
            sym_add(Jl, a, b, fcd * P_total(c, d) * val);
            if (left != right)
                sym_add(Jl, c, d, fab * P_total(a, b) * val);

            // ── Exchange for alpha and beta spin (no 0.5 factor) ──────
            // Same mirror-permutation fix as ShellPairList::g_rhf: when
            // left != right use explicit double-add to handle diagonal K(p,p).
            if (left != right) {
                Kal(a, c) += P_alpha(b, d) * val;
                Kal(c, a) += P_alpha(b, d) * val;
                Kbl(a, c) += P_beta(b, d) * val;
                Kbl(c, a) += P_beta(b, d) * val;
                if (a != b) {
                    Kal(b, c) += P_alpha(a, d) * val;
                    Kal(c, b) += P_alpha(a, d) * val;
                    Kbl(b, c) += P_beta(a, d) * val;
                    Kbl(c, b) += P_beta(a, d) * val;
                }
                if (c != d) {
                    Kal(a, d) += P_alpha(b, c) * val;
                    Kal(d, a) += P_alpha(b, c) * val;
                    Kbl(a, d) += P_beta(b, c) * val;
                    Kbl(d, a) += P_beta(b, c) * val;
                    if (a != b) {
                        Kal(b, d) += P_alpha(a, c) * val;
                        Kal(d, b) += P_alpha(a, c) * val;
                        Kbl(b, d) += P_beta(a, c) * val;
                        Kbl(d, b) += P_beta(a, c) * val;
                    }
                }
            } else {
                sym_add(Kal, a, c, P_alpha(b, d) * val);
                sym_add(Kbl, a, c, P_beta(b, d) * val);
                if (a != b) {
                    sym_add(Kal, b, c, P_alpha(a, d) * val);
                    sym_add(Kbl, b, c, P_beta(a, d) * val);
                }
                if (c != d) {
                    if (a != b) {
                        sym_add(Kal, b, d, P_alpha(a, c) * val);
                        sym_add(Kbl, b, d, P_beta(a, c) * val);
                    }
                }
            }
        }
    }

    // Reduce thread-local matrices.
    J = xt::zeros<double>({n, n});
    K_alpha = xt::zeros<double>({n, n});
    K_beta = xt::zeros<double>({n, n});
    for (int t = 0; t < nT; ++t) {
        J += Jv[t];
        K_alpha += Kav[t];
        K_beta += Kbv[t];
    }
}

void ShellPairList::j(const RealMatrix& P_total, RealMatrix& J) const {
    const ShellPairList& sp = *this;
    const int n = sp.n_basis;
    const int nT = n_threads();
    const double thr = sp.threshold;
    const int npairs = static_cast<int>(sp.pairs.size());

    std::vector<RealMatrix> Jv(nT, xt::zeros<double>({n, n}));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < npairs; ++left) {
        auto& Jl = Jv[thread_id()];
        const ShellPair& pij = sp.pairs[left];
        const int a = pij.i, b = pij.j;

        for (int right = 0; right <= left; ++right) {
            const ShellPair& pkl = sp.pairs[right];
            const int c = pkl.i, d = pkl.j;

            if (pij.schwarz * pkl.schwarz < thr)
                continue;

            const double val = native::eri(sp.shells[a], sp.shells[b],
                                           sp.shells[c], sp.shells[d]);
            if (std::abs(val) < thr)
                continue;

            const double fcd = (c == d) ? 1.0 : 2.0;
            const double fab = (a == b) ? 1.0 : 2.0;
            sym_add(Jl, a, b, fcd * P_total(c, d) * val);
            if (left != right)
                sym_add(Jl, c, d, fab * P_total(a, b) * val);
        }
    }

    J = xt::zeros<double>({n, n});
    for (auto& jl : Jv)
        J += jl;
}

// ── Shell-block direct-SCF (libcint/PySCF approach) ──────────────────────

// Raw-pointer helpers for column-major G and P access.
// RealMatrix is column_major: M(i,j) = data[i + j*n] (Fortran order).
// Using raw pointers eliminates xtensor's strided dispatch overhead
// in the inner loop (~150× speedup on aspirin STO-3G).

namespace {
inline void raw_sym_add(double* g, const int n, int p, int q, double v) {
    g[p + q * n] += v;
    if (p != q)
        g[q + p * n] += v;
}
} // namespace

void ShellBlockPairList::k(const RealMatrix& P, RealMatrix& K) const {
    const ShellBlockPairList& sp = *this;
    // Exchange-only twin of ShellBlockPairList::g_rhf: same shell-block ERI
    // amortization and 8-fold symmetry scatter, but accumulates only the
    // exchange contribution with NO 0.5 factor and no Coulomb term. The
    // result K satisfies the RHF convention F = Hcore + J - 0.5*K.
    const int n = sp.n_basis;
    const int nT = n_threads();
    const double thr = sp.threshold;
    const int npairs = static_cast<int>(sp.pairs.size());

    std::vector<std::vector<double>> Kraw(nT, std::vector<double>(n * n, 0.0));
    const double* Pp = P.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < npairs; ++left) {
        double* Kp = Kraw[thread_id()].data();
        const ShellBlockPair& bp_l = sp.pairs[left];
        const native::ShellBlock& Bsi = sp.blocks[bp_l.si];
        const native::ShellBlock& Bsj = sp.blocks[bp_l.sj];
        const int na = Bsi.n_cart, nb = Bsj.n_cart;

        for (int right = 0; right <= left; ++right) {
            const ShellBlockPair& bp_r = sp.pairs[right];

            if (bp_l.schwarz * bp_r.schwarz < thr)
                continue;

            const native::ShellBlock& Bsk = sp.blocks[bp_r.si];
            const native::ShellBlock& Bsl = sp.blocks[bp_r.sj];
            const int nc = Bsk.n_cart, nd = Bsl.n_cart;

            double blk[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                       * kMaxNCartOrbital] = {};
            native::eri_block(Bsi, Bsj, Bsk, Bsl, blk);

            if (left != right) {
                const bool diag_bra = (bp_l.si == bp_l.sj);
                const bool diag_ket = (bp_r.si == bp_r.sj);

                for (int ia = 0; ia < na; ++ia) {
                    const int a = Bsi.ao_start + ia;
                    for (int ib = 0; ib < nb; ++ib) {
                        if (diag_bra && ib > ia)
                            continue;
                        const int b = Bsj.ao_start + ib;
                        const double* blk_ab = blk + (ia * nb + ib) * (nc * nd);

                        for (int ic = 0; ic < nc; ++ic) {
                            const int c = Bsk.ao_start + ic;
                            for (int id = 0; id < nd; ++id) {
                                if (diag_ket && id > ic)
                                    continue;
                                const int d = Bsl.ao_start + id;
                                const double val = blk_ab[ic * nd + id];
                                if (std::abs(val) < thr)
                                    continue;

                                // Exchange only — no 0.5 factor.
                                const double Pbd = Pp[b + d * n],
                                             Pad = Pp[a + d * n];
                                const double Pbc = Pp[b + c * n],
                                             Pac = Pp[a + c * n];
                                Kp[a + c * n] += val * Pbd;
                                Kp[c + a * n] += val * Pbd;
                                if (a != b) {
                                    Kp[b + c * n] += val * Pad;
                                    Kp[c + b * n] += val * Pad;
                                }
                                if (c != d) {
                                    Kp[a + d * n] += val * Pbc;
                                    Kp[d + a * n] += val * Pbc;
                                    if (a != b) {
                                        Kp[b + d * n] += val * Pac;
                                        Kp[d + b * n] += val * Pac;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                const bool diag_block = (bp_l.si == bp_l.sj);
                struct AOpair {
                    int ia, ib;
                };
                // Max AO pair count per block pair: f-f non-diag =
                // 10*10 = 100 (kMaxNCart^2); diag pairs are fewer.
                AOpair ao_pairs[kMaxNCartOrbital * kMaxNCartOrbital];
                int n_ao_pairs = 0;
                for (int ia = 0; ia < na; ++ia)
                    for (int ib = 0; ib < nb; ++ib) {
                        if (diag_block && ib > ia)
                            continue;
                        ao_pairs[n_ao_pairs++] = {ia, ib};
                    }

                for (int left_ao = 0; left_ao < n_ao_pairs; ++left_ao) {
                    const int ia = ao_pairs[left_ao].ia;
                    const int ib = ao_pairs[left_ao].ib;
                    const int a = Bsi.ao_start + ia;
                    const int b = Bsj.ao_start + ib;
                    const double* blk_ab = blk + (ia * nb + ib) * (nc * nd);

                    for (int right_ao = 0; right_ao <= left_ao; ++right_ao) {
                        const int ic = ao_pairs[right_ao].ia;
                        const int id = ao_pairs[right_ao].ib;
                        const int c = Bsk.ao_start + ic;
                        const int d = Bsl.ao_start + id;
                        const double val = blk_ab[ic * nd + id];
                        if (std::abs(val) < thr)
                            continue;

                        if (left_ao != right_ao) {
                            const double Pbd = Pp[b + d * n],
                                         Pad = Pp[a + d * n];
                            const double Pbc = Pp[b + c * n],
                                         Pac = Pp[a + c * n];
                            Kp[a + c * n] += val * Pbd;
                            Kp[c + a * n] += val * Pbd;
                            if (a != b) {
                                Kp[b + c * n] += val * Pad;
                                Kp[c + b * n] += val * Pad;
                            }
                            if (c != d) {
                                Kp[a + d * n] += val * Pbc;
                                Kp[d + a * n] += val * Pbc;
                                if (a != b) {
                                    Kp[b + d * n] += val * Pac;
                                    Kp[d + b * n] += val * Pac;
                                }
                            }
                        } else {
                            raw_sym_add(Kp, n, a, c, val * Pp[b + d * n]);
                            if (a != b)
                                raw_sym_add(Kp, n, b, c, val * Pp[a + d * n]);
                            if (c != d && a != b)
                                raw_sym_add(Kp, n, b, d, val * Pp[a + c * n]);
                        }
                    }
                }
            }
        }
    }

    K = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    double* Kout = K.data();
    for (auto& kraw : Kraw)
        for (int i = 0; i < n * n; ++i)
            Kout[i] += kraw[i];
}

void ShellBlockPairList::g_rhf(const RealMatrix& P, RealMatrix& G) const {
    const ShellBlockPairList& sp = *this;
    const int n = sp.n_basis;
    const int nT = n_threads();
    const double thr = sp.threshold;
    const int npairs = static_cast<int>(sp.pairs.size());

    // Thread-local G matrices: use raw double arrays for inner-loop speed.
    // Column-major layout matching RealMatrix convention.
    std::vector<std::vector<double>> Graw(nT, std::vector<double>(n * n, 0.0));
    const double* Pp = P.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < npairs; ++left) {
        double* Gp = Graw[thread_id()].data();
        const ShellBlockPair& bp_l = sp.pairs[left];
        const native::ShellBlock& Bsi = sp.blocks[bp_l.si];
        const native::ShellBlock& Bsj = sp.blocks[bp_l.sj];
        const int na = Bsi.n_cart, nb = Bsj.n_cart;

        for (int right = 0; right <= left; ++right) {
            const ShellBlockPair& bp_r = sp.pairs[right];

            // Block-level Cauchy-Schwarz screening
            if (bp_l.schwarz * bp_r.schwarz < thr)
                continue;

            const native::ShellBlock& Bsk = sp.blocks[bp_r.si];
            const native::ShellBlock& Bsl = sp.blocks[bp_r.sj];
            const int nc = Bsk.n_cart, nd = Bsl.n_cart;

            // Max block size is (ff|ff) = n_cart_of(kMaxL)^4 — 10^4 =
            // 10000 doubles for f-shells.
            double blk[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                       * kMaxNCartOrbital] = {};
            native::eri_block(Bsi, Bsj, Bsk, Bsl, blk);

            if (left != right) {
                const bool diag_bra = (bp_l.si == bp_l.sj);
                const bool diag_ket = (bp_r.si == bp_r.sj);

                for (int ia = 0; ia < na; ++ia) {
                    const int a = Bsi.ao_start + ia;
                    for (int ib = 0; ib < nb; ++ib) {
                        if (diag_bra && ib > ia)
                            continue;
                        const int b = Bsj.ao_start + ib;
                        const double fab = (a == b) ? 1.0 : 2.0;
                        const double* blk_ab = blk + (ia * nb + ib) * (nc * nd);

                        for (int ic = 0; ic < nc; ++ic) {
                            const int c = Bsk.ao_start + ic;
                            for (int id = 0; id < nd; ++id) {
                                if (diag_ket && id > ic)
                                    continue;
                                const int d = Bsl.ao_start + id;
                                const double val = blk_ab[ic * nd + id];
                                if (std::abs(val) < thr)
                                    continue;

                                const double fcd = (c == d) ? 1.0 : 2.0;
                                // Coulomb
                                raw_sym_add(Gp, n, a, b,
                                            fcd * Pp[c + d * n] * val);
                                raw_sym_add(Gp, n, c, d,
                                            fab * Pp[a + b * n] * val);

                                // Exchange
                                const double hval = 0.5 * val;
                                const double Pbd = Pp[b + d * n],
                                             Pad = Pp[a + d * n];
                                const double Pbc = Pp[b + c * n],
                                             Pac = Pp[a + c * n];
                                Gp[a + c * n] -= hval * Pbd;
                                Gp[c + a * n] -= hval * Pbd;
                                if (a != b) {
                                    Gp[b + c * n] -= hval * Pad;
                                    Gp[c + b * n] -= hval * Pad;
                                }
                                if (c != d) {
                                    Gp[a + d * n] -= hval * Pbc;
                                    Gp[d + a * n] -= hval * Pbc;
                                    if (a != b) {
                                        Gp[b + d * n] -= hval * Pac;
                                        Gp[d + b * n] -= hval * Pac;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                const bool diag_block = (bp_l.si == bp_l.sj);
                struct AOpair {
                    int ia, ib;
                };
                // Max AO pair count per block pair: f-f non-diag =
                // 10*10 = 100 (kMaxNCart^2); diag pairs are fewer. The old
                // d-sized 36 overflowed the stack for f blocks (silent UB
                // that skewed AuH/def2-SVP by ~7e-6 Ha).
                AOpair ao_pairs[kMaxNCartOrbital * kMaxNCartOrbital];
                int n_ao_pairs = 0;
                for (int ia = 0; ia < na; ++ia)
                    for (int ib = 0; ib < nb; ++ib) {
                        if (diag_block && ib > ia)
                            continue;
                        ao_pairs[n_ao_pairs++] = {ia, ib};
                    }

                for (int left_ao = 0; left_ao < n_ao_pairs; ++left_ao) {
                    const int ia = ao_pairs[left_ao].ia;
                    const int ib = ao_pairs[left_ao].ib;
                    const int a = Bsi.ao_start + ia;
                    const int b = Bsj.ao_start + ib;
                    const double fab = (a == b) ? 1.0 : 2.0;
                    const double* blk_ab = blk + (ia * nb + ib) * (nc * nd);

                    for (int right_ao = 0; right_ao <= left_ao; ++right_ao) {
                        const int ic = ao_pairs[right_ao].ia;
                        const int id = ao_pairs[right_ao].ib;
                        const int c = Bsk.ao_start + ic;
                        const int d = Bsl.ao_start + id;
                        const double val = blk_ab[ic * nd + id];
                        if (std::abs(val) < thr)
                            continue;

                        const double fcd = (c == d) ? 1.0 : 2.0;
                        raw_sym_add(Gp, n, a, b, fcd * Pp[c + d * n] * val);
                        if (left_ao != right_ao)
                            raw_sym_add(Gp, n, c, d, fab * Pp[a + b * n] * val);

                        const double hval = 0.5 * val;
                        if (left_ao != right_ao) {
                            const double Pbd = Pp[b + d * n],
                                         Pad = Pp[a + d * n];
                            const double Pbc = Pp[b + c * n],
                                         Pac = Pp[a + c * n];
                            Gp[a + c * n] -= hval * Pbd;
                            Gp[c + a * n] -= hval * Pbd;
                            if (a != b) {
                                Gp[b + c * n] -= hval * Pad;
                                Gp[c + b * n] -= hval * Pad;
                            }
                            if (c != d) {
                                Gp[a + d * n] -= hval * Pbc;
                                Gp[d + a * n] -= hval * Pbc;
                                if (a != b) {
                                    Gp[b + d * n] -= hval * Pac;
                                    Gp[d + b * n] -= hval * Pac;
                                }
                            }
                        } else {
                            raw_sym_add(Gp, n, a, c, -hval * Pp[b + d * n]);
                            if (a != b)
                                raw_sym_add(Gp, n, b, c, -hval * Pp[a + d * n]);
                            if (c != d && a != b)
                                raw_sym_add(Gp, n, b, d, -hval * Pp[a + c * n]);
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local raw arrays into xtensor G matrix.
    G = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n)});
    double* Gout = G.data();
    for (auto& graw : Graw)
        for (int i = 0; i < n * n; ++i)
            Gout[i] += graw[i];
}

} // namespace scf
} // namespace cpu
} // namespace ATV
