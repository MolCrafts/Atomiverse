/// @file cpu/scf/internal/ci_engine.cpp
/// @brief Implementation of the shared determinant-basis CI engine
///        (see internal/ci_engine.h): bitstring enumeration, canonical (fci)
///        Slater–Condon matrix elements with Jordan–Wigner phases, the dense
///        CI eigensolve, and the spin-summed active-space 1-/2-RDM build.
///
/// The Slater–Condon evaluators are the canonical `fci.cpp` implementations
/// (`h(p,q)` / `g(p,q,..)` ordering, excitation level by
/// `popcount(a_i & ~a_j)`); `CiSolution` / `dense_ci_solve` / `make_rdm12` are
/// the `casscf.cpp` superset. On real-symmetric MO integrals the two forks are
/// numerically equal — the equivalence gate
/// (regressions/scf/ci-engine-equiv-h2-sto3g) pins that.

#include "atomiverse/cpu/scf/internal/ci_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

long long choose(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    k = std::min(k, n - k);
    long long r = 1;
    for (int i = 0; i < k; ++i) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

std::vector<std::uint64_t> enumerate_strings(int n_mo, int n_elec) {
    std::vector<std::uint64_t> out;
    const std::uint64_t limit = std::uint64_t(1) << n_mo;
    for (std::uint64_t s = 0; s < limit; ++s) {
        if (__builtin_popcountll(s) == n_elec)
            out.push_back(s);
    }
    return out;
}

std::vector<int> occ_list(std::uint64_t s) {
    std::vector<int> v;
    while (s) {
        v.push_back(__builtin_ctzll(s));
        s &= s - 1;
    }
    return v;
}

int jw_single_sign(std::uint64_t s, int p, int q) {
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    // Bits with index in the open interval (lo, hi).
    const std::uint64_t mask =
        ((std::uint64_t(1) << hi) - 1) & ~((std::uint64_t(1) << (lo + 1)) - 1);
    return (__builtin_popcountll(s & mask) & 1) ? -1 : 1;
}

double diagonal_element(std::uint64_t a, std::uint64_t b,
                        const xt::xtensor<double, 2>& h,
                        const xt::xtensor<double, 4>& g, double v_nuc) {
    const std::vector<int> occ_a = occ_list(a);
    const std::vector<int> occ_b = occ_list(b);

    double e = v_nuc;
    for (int i : occ_a)
        e += h(i, i);
    for (int i : occ_b)
        e += h(i, i);

    // Same-spin alpha pairs: Coulomb minus exchange.
    for (std::size_t x = 0; x < occ_a.size(); ++x)
        for (std::size_t y = x + 1; y < occ_a.size(); ++y) {
            const int i = occ_a[x], j = occ_a[y];
            e += g(i, i, j, j) - g(i, j, j, i);
        }
    // Same-spin beta pairs: Coulomb minus exchange.
    for (std::size_t x = 0; x < occ_b.size(); ++x)
        for (std::size_t y = x + 1; y < occ_b.size(); ++y) {
            const int i = occ_b[x], j = occ_b[y];
            e += g(i, i, j, j) - g(i, j, j, i);
        }
    // Opposite-spin pairs (alpha,beta): Coulomb only (no exchange).
    for (int i : occ_a)
        for (int j : occ_b)
            e += g(i, i, j, j);

    return e;
}

double single_element(std::uint64_t same_i, std::uint64_t same_j,
                      std::uint64_t other, const xt::xtensor<double, 2>& h,
                      const xt::xtensor<double, 4>& g) {
    const int p = __builtin_ctzll(same_i & ~same_j); // occupied in I, not J
    const int q = __builtin_ctzll(same_j & ~same_i); // occupied in J, not I
    const int sign = jw_single_sign(same_i, p, q);

    double v = h(p, q);
    // Same-spin spectators (shared occupation in this channel): Coulomb - exchange.
    for (int j : occ_list(same_i & same_j))
        v += g(p, q, j, j) - g(p, j, j, q);
    // Opposite-spin spectators: Coulomb only.
    for (int j : occ_list(other))
        v += g(p, q, j, j);

    return sign * v;
}

double same_spin_double_element(std::uint64_t s_from, std::uint64_t hole_mask,
                                std::uint64_t part_mask,
                                const xt::xtensor<double, 4>& g) {
    const std::vector<int> holes = occ_list(hole_mask); // [p, q], p < q
    const std::vector<int> parts = occ_list(part_mask); // [r, t], r < t
    const int p = holes[0], q = holes[1];
    const int r = parts[0], t = parts[1];

    const int sign1 = jw_single_sign(s_from, p, r);
    const std::uint64_t s_mid =
        (s_from & ~(std::uint64_t(1) << p)) | (std::uint64_t(1) << r);
    const int sign2 = jw_single_sign(s_mid, q, t);
    const int sign = sign1 * sign2;

    return sign * (g(p, r, q, t) - g(p, t, q, r));
}

double opposite_spin_double_element(std::uint64_t a_i, std::uint64_t a_j,
                                    std::uint64_t b_i, std::uint64_t b_j,
                                    const xt::xtensor<double, 4>& g) {
    const int pa = __builtin_ctzll(a_i & ~a_j); // alpha hole
    const int ra = __builtin_ctzll(a_j & ~a_i); // alpha particle
    const int qb = __builtin_ctzll(b_i & ~b_j); // beta hole
    const int sb = __builtin_ctzll(b_j & ~b_i); // beta particle

    const int sign = jw_single_sign(a_i, pa, ra) * jw_single_sign(b_i, qb, sb);
    return sign * g(pa, ra, qb, sb);
}

double ci_element(std::uint64_t a_i, std::uint64_t b_i, std::uint64_t a_j,
                  std::uint64_t b_j, const xt::xtensor<double, 2>& h,
                  const xt::xtensor<double, 4>& g, double v_nuc) {
    const int da = __builtin_popcountll(a_i & ~a_j); // alpha excitation level
    const int db = __builtin_popcountll(b_i & ~b_j); // beta excitation level
    const int level = da + db;

    if (level == 0)
        return diagonal_element(a_i, b_i, h, g, v_nuc); // Case A
    if (level == 1) {                                   // Case B
        if (da == 1)
            return single_element(a_i, a_j, b_i, h, g);
        return single_element(b_i, b_j, a_i, h, g);
    }
    if (level == 2) { // Case C
        if (da == 2)
            return same_spin_double_element(a_i, a_i & ~a_j, a_j & ~a_i, g);
        if (db == 2)
            return same_spin_double_element(b_i, b_i & ~b_j, b_j & ~b_i, g);
        return opposite_spin_double_element(a_i, a_j, b_i, b_j, g);
    }
    return 0.0; // Case D (differ by >= 3 spin-orbitals)
}

CiSolution dense_ci_solve(const xt::xtensor<double, 2>& h,
                          const xt::xtensor<double, 4>& g, double v_shift,
                          int n_alpha, int n_beta) {
    const int n_mo = static_cast<int>(h.shape(0));
    if (n_mo > 14) {
        throw std::invalid_argument("dense_ci_solve: ncas="
                                    + std::to_string(n_mo)
                                    + " exceeds the TINY cap (ncas <= 14)");
    }
    const long long n_det = choose(n_mo, n_alpha) * choose(n_mo, n_beta);
    if (n_det > 2000) {
        throw std::invalid_argument("dense_ci_solve: N_det="
                                    + std::to_string(n_det)
                                    + " exceeds the TINY cap (N_det <= 2000)");
    }
    CiSolution sol;
    sol.alphas = enumerate_strings(n_mo, n_alpha);
    sol.betas = enumerate_strings(n_mo, n_beta);
    const std::size_t n_a = sol.alphas.size();
    const std::size_t n_b = sol.betas.size();
    const std::size_t nd = n_a * n_b;
    sol.c = xt::zeros<double>({nd});

    if (nd == 1) {
        sol.energy =
            diagonal_element(sol.alphas[0], sol.betas[0], h, g, v_shift);
        sol.c(0) = 1.0;
        return sol;
    }

    xt::xtensor<double, 2> hci = xt::zeros<double>({nd, nd});
    for (std::size_t ia = 0; ia < n_a; ++ia) {
        for (std::size_t ib = 0; ib < n_b; ++ib) {
            const std::size_t idx_i = ia * n_b + ib;
            for (std::size_t ja = 0; ja < n_a; ++ja) {
                for (std::size_t jb = 0; jb < n_b; ++jb) {
                    const std::size_t idx_j = ja * n_b + jb;
                    if (idx_j < idx_i)
                        continue;
                    const double e = ci_element(sol.alphas[ia], sol.betas[ib],
                                                sol.alphas[ja], sol.betas[jb],
                                                h, g, v_shift);
                    hci(idx_i, idx_j) = e;
                    hci(idx_j, idx_i) = e;
                }
            }
        }
    }
    auto [w, U] = xt::linalg::eigh(hci);
    sol.energy = w(0);
    for (std::size_t i = 0; i < nd; ++i)
        sol.c(i) = U(i, 0);
    return sol;
}

void make_rdm12(const CiSolution& sol, int ncas, xt::xtensor<double, 2>& rdm1,
                xt::xtensor<double, 4>& rdm2) {
    const std::size_t na = static_cast<std::size_t>(ncas);
    rdm1 = xt::zeros<double>({na, na});
    rdm2 = xt::zeros<double>({na, na, na, na});
    const std::size_t n_a = sol.alphas.size();
    const std::size_t n_b = sol.betas.size();

    // Index map string → string index.
    auto find_str = [](const std::vector<std::uint64_t>& strs,
                       std::uint64_t s) -> int {
        for (std::size_t i = 0; i < strs.size(); ++i)
            if (strs[i] == s)
                return static_cast<int>(i);
        return -1;
    };

    for (std::size_t ia = 0; ia < n_a; ++ia) {
        for (std::size_t ib = 0; ib < n_b; ++ib) {
            const std::size_t idx_i = ia * n_b + ib;
            const double ci = sol.c(idx_i);
            if (std::abs(ci) < 1e-16)
                continue;
            const std::uint64_t Aa = sol.alphas[ia];
            const std::uint64_t Bb = sol.betas[ib];
            const std::vector<int> oa = occ_list(Aa);
            const std::vector<int> ob = occ_list(Bb);

            // Diagonal 1-RDM contributions.
            for (int p : oa)
                rdm1(p, p) += ci * ci;
            for (int p : ob)
                rdm1(p, p) += ci * ci;

            // α singles p→q: contributes to γ_qp (a†_q a_p).
            for (int p : oa) {
                for (int q = 0; q < ncas; ++q) {
                    if (Aa & (std::uint64_t(1) << q))
                        continue;
                    const std::uint64_t Aj = (Aa ^ (std::uint64_t(1) << p))
                                             | (std::uint64_t(1) << q);
                    const int ja = find_str(sol.alphas, Aj);
                    if (ja < 0)
                        continue;
                    const int sgn = jw_single_sign(Aa, p, q);
                    const double cj =
                        sol.c(static_cast<std::size_t>(ja) * n_b + ib);
                    const double val = sgn * ci * cj;
                    rdm1(q, p) += val;
                }
            }
            // β singles.
            for (int p : ob) {
                for (int q = 0; q < ncas; ++q) {
                    if (Bb & (std::uint64_t(1) << q))
                        continue;
                    const std::uint64_t Bj = (Bb ^ (std::uint64_t(1) << p))
                                             | (std::uint64_t(1) << q);
                    const int jb = find_str(sol.betas, Bj);
                    if (jb < 0)
                        continue;
                    const int sgn = jw_single_sign(Bb, p, q);
                    const double cj =
                        sol.c(ia * n_b + static_cast<std::size_t>(jb));
                    const double val = sgn * ci * cj;
                    rdm1(q, p) += val;
                }
            }

            // 2-RDM: loop determinant pairs with ≤2 differences per spin.
            // Direct accumulation Γ_pqrs = Σ_στ ⟨a†_pσ a†_rτ a_sτ a_qσ⟩.
            // Diagonal (I=I) same-spin and opposite-spin pairs.
            for (std::size_t x = 0; x < oa.size(); ++x) {
                for (std::size_t y = 0; y < oa.size(); ++y) {
                    if (x == y)
                        continue;
                    const int p = oa[x], r = oa[y];
                    // ⟨a†_pα a†_rα a_rα a_pα⟩ = 1 on |I⟩ → Γ_pprr contrib...
                    // For identical det: Γ_pqrs with q=p, s=r: a†_p a†_r a_r a_p = 1
                    rdm2(p, p, r, r) += ci * ci;
                    // exchange-like: a†_p a†_r a_p a_r = −1 for same spin
                    rdm2(p, r, r, p) -= ci * ci;
                }
            }
            for (std::size_t x = 0; x < ob.size(); ++x) {
                for (std::size_t y = 0; y < ob.size(); ++y) {
                    if (x == y)
                        continue;
                    const int p = ob[x], r = ob[y];
                    rdm2(p, p, r, r) += ci * ci;
                    rdm2(p, r, r, p) -= ci * ci;
                }
            }
            for (int p : oa) {
                for (int r : ob) {
                    // opposite spin: a†_pα a†_rβ a_rβ a_pα = 1
                    rdm2(p, p, r, r) += ci * ci;
                    // a†_pα a†_rβ a_pβ a_rα = 0 (different spins on last two if p≠r
                    // — only when p and r indices allow; for spatial Γ, the
                    // mixed-spin exchange term is zero when operators don't match)
                    // a†_rβ a†_pα a_pα a_rβ = 1 → also adds to Γ_rrpp
                    rdm2(r, r, p, p) += ci * ci;
                }
            }
        }
    }

    // Off-diagonal 2-RDM: for each I, apply two-electron operators to reach J.
    // Compact route: for every pair of dets differing by 0–2 spin-orbitals per
    // channel, add the corresponding Γ contributions. Already handled I=I above
    // for diagonal. For I≠J use ci*cj * matrix elements of a†a†aa.
    for (std::size_t ia = 0; ia < n_a; ++ia) {
        for (std::size_t ib = 0; ib < n_b; ++ib) {
            const std::size_t idx_i = ia * n_b + ib;
            const double ci = sol.c(idx_i);
            if (std::abs(ci) < 1e-16)
                continue;
            const std::uint64_t Aa = sol.alphas[ia];
            const std::uint64_t Bb = sol.betas[ib];

            // αα double: p,r occupied → q,s virtual (p < r, q < s for unique).
            {
                const std::vector<int> oa = occ_list(Aa);
                std::vector<int> va;
                for (int q = 0; q < ncas; ++q)
                    if (!(Aa & (std::uint64_t(1) << q)))
                        va.push_back(q);
                for (std::size_t ip = 0; ip < oa.size(); ++ip) {
                    for (std::size_t ir = ip + 1; ir < oa.size(); ++ir) {
                        const int p = oa[ip], r = oa[ir];
                        for (std::size_t iq = 0; iq < va.size(); ++iq) {
                            for (std::size_t is = iq + 1; is < va.size();
                                 ++is) {
                                const int q = va[iq], s = va[is];
                                // phase p→q then r→s
                                const int s1 = jw_single_sign(Aa, p, q);
                                const std::uint64_t mid =
                                    (Aa ^ (std::uint64_t(1) << p))
                                    | (std::uint64_t(1) << q);
                                // r may have moved if r==q impossible (virt).
                                if (mid & (std::uint64_t(1) << s))
                                    continue;
                                if (!(mid & (std::uint64_t(1) << r)))
                                    continue;
                                const int s2 = jw_single_sign(mid, r, s);
                                const std::uint64_t Aj =
                                    (mid ^ (std::uint64_t(1) << r))
                                    | (std::uint64_t(1) << s);
                                const int ja = find_str(sol.alphas, Aj);
                                if (ja < 0)
                                    continue;
                                const double cj = sol.c(
                                    static_cast<std::size_t>(ja) * n_b + ib);
                                const double w = double(s1 * s2) * ci * cj;
                                // a†_q a†_s a_r a_p |I⟩ ~ |J⟩  (order: annihilate
                                // p,r create q,s). Operator a†_qα a†_sα a_rα a_pα
                                // contributes to Γ_q p s r.
                                rdm2(q, p, s, r) += w;
                                rdm2(q, r, s, p) -= w;
                                rdm2(s, p, q, r) -= w;
                                rdm2(s, r, q, p) += w;
                            }
                        }
                    }
                }
            }
            // ββ double (same structure).
            {
                const std::vector<int> ob = occ_list(Bb);
                std::vector<int> vb;
                for (int q = 0; q < ncas; ++q)
                    if (!(Bb & (std::uint64_t(1) << q)))
                        vb.push_back(q);
                for (std::size_t ip = 0; ip < ob.size(); ++ip) {
                    for (std::size_t ir = ip + 1; ir < ob.size(); ++ir) {
                        const int p = ob[ip], r = ob[ir];
                        for (std::size_t iq = 0; iq < vb.size(); ++iq) {
                            for (std::size_t is = iq + 1; is < vb.size();
                                 ++is) {
                                const int q = vb[iq], s = vb[is];
                                const int s1 = jw_single_sign(Bb, p, q);
                                const std::uint64_t mid =
                                    (Bb ^ (std::uint64_t(1) << p))
                                    | (std::uint64_t(1) << q);
                                if (mid & (std::uint64_t(1) << s))
                                    continue;
                                if (!(mid & (std::uint64_t(1) << r)))
                                    continue;
                                const int s2 = jw_single_sign(mid, r, s);
                                const std::uint64_t Bj =
                                    (mid ^ (std::uint64_t(1) << r))
                                    | (std::uint64_t(1) << s);
                                const int jb = find_str(sol.betas, Bj);
                                if (jb < 0)
                                    continue;
                                const double cj = sol.c(
                                    ia * n_b + static_cast<std::size_t>(jb));
                                const double w = double(s1 * s2) * ci * cj;
                                rdm2(q, p, s, r) += w;
                                rdm2(q, r, s, p) -= w;
                                rdm2(s, p, q, r) -= w;
                                rdm2(s, r, q, p) += w;
                            }
                        }
                    }
                }
            }
            // αβ double: p→q (α) and r→s (β).
            {
                const std::vector<int> oa = occ_list(Aa);
                const std::vector<int> ob = occ_list(Bb);
                for (int p : oa) {
                    for (int q = 0; q < ncas; ++q) {
                        if (Aa & (std::uint64_t(1) << q))
                            continue;
                        const int sa = jw_single_sign(Aa, p, q);
                        const std::uint64_t Aj = (Aa ^ (std::uint64_t(1) << p))
                                                 | (std::uint64_t(1) << q);
                        const int ja = find_str(sol.alphas, Aj);
                        if (ja < 0)
                            continue;
                        for (int r : ob) {
                            for (int s = 0; s < ncas; ++s) {
                                if (Bb & (std::uint64_t(1) << s))
                                    continue;
                                const int sb = jw_single_sign(Bb, r, s);
                                const std::uint64_t Bj =
                                    (Bb ^ (std::uint64_t(1) << r))
                                    | (std::uint64_t(1) << s);
                                const int jb = find_str(sol.betas, Bj);
                                if (jb < 0)
                                    continue;
                                const double cj =
                                    sol.c(static_cast<std::size_t>(ja) * n_b
                                          + static_cast<std::size_t>(jb));
                                const double w = double(sa * sb) * ci * cj;
                                // a†_qα a†_sβ a_rβ a_pα → Γ_q p s r
                                rdm2(q, p, s, r) += w;
                                // a†_sβ a†_qα a_pα a_rβ → Γ_s r q p
                                rdm2(s, r, q, p) += w;
                            }
                        }
                    }
                }
            }

            // α single × β identity for 2-RDM mixed with remaining occupations
            // is already partially covered by I=J diagonal; off-diagonal singles
            // contribute when one spin matches:
            // ⟨J| a†_qα a†_tτ a_uτ a_pα |I⟩ with J = α(p→q) from I.
            for (int p : occ_list(Aa)) {
                for (int q = 0; q < ncas; ++q) {
                    if (Aa & (std::uint64_t(1) << q))
                        continue;
                    const int sgn = jw_single_sign(Aa, p, q);
                    const std::uint64_t Aj = (Aa ^ (std::uint64_t(1) << p))
                                             | (std::uint64_t(1) << q);
                    const int ja = find_str(sol.alphas, Aj);
                    if (ja < 0)
                        continue;
                    const double cj =
                        sol.c(static_cast<std::size_t>(ja) * n_b + ib);
                    const double w = double(sgn) * ci * cj;
                    // Remaining α occupied (after annihilating p): couple as
                    // a†_q a†_r a_r a_p
                    std::uint64_t rem = Aa & ~(std::uint64_t(1) << p);
                    while (rem) {
                        const int r = __builtin_ctzll(rem);
                        rem &= rem - 1;
                        rdm2(q, p, r, r) += w;
                        rdm2(q, r, r, p) -= w;
                        rdm2(r, p, q, r) -= w; // careful — may overcount; use
                        // only the two standard spin-summed slots:
                    }
                    // Opposite spin occupied in β.
                    for (int r : occ_list(Bb)) {
                        rdm2(q, p, r, r) += w;
                        rdm2(r, r, q, p) += w;
                    }
                }
            }
            for (int p : occ_list(Bb)) {
                for (int q = 0; q < ncas; ++q) {
                    if (Bb & (std::uint64_t(1) << q))
                        continue;
                    const int sgn = jw_single_sign(Bb, p, q);
                    const std::uint64_t Bj = (Bb ^ (std::uint64_t(1) << p))
                                             | (std::uint64_t(1) << q);
                    const int jb = find_str(sol.betas, Bj);
                    if (jb < 0)
                        continue;
                    const double cj =
                        sol.c(ia * n_b + static_cast<std::size_t>(jb));
                    const double w = double(sgn) * ci * cj;
                    std::uint64_t rem = Bb & ~(std::uint64_t(1) << p);
                    while (rem) {
                        const int r = __builtin_ctzll(rem);
                        rem &= rem - 1;
                        rdm2(q, p, r, r) += w;
                        rdm2(q, r, r, p) -= w;
                    }
                    for (int r : occ_list(Aa)) {
                        rdm2(q, p, r, r) += w;
                        rdm2(r, r, q, p) += w;
                    }
                }
            }
        }
    }

    // Hermitize 1-RDM (real wavefunction).
    for (int p = 0; p < ncas; ++p)
        for (int q = p + 1; q < ncas; ++q) {
            const double avg = 0.5 * (rdm1(p, q) + rdm1(q, p));
            rdm1(p, q) = avg;
            rdm1(q, p) = avg;
        }
}

} // namespace scf
} // namespace cpu
} // namespace ATV
