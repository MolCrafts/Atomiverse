/// @file cpu/scf/fci.cpp
/// @brief Implementation of the tiny determinant-basis full-CI solver (see
///        fci.h): determinant bitstring enumeration, Slater-Condon Case A-D
///        matrix assembly with Jordan-Wigner phases, dense eigh.
///
/// Combinatorics sanity check (by hand):
///   H2 / STO-3G: n_mo = 2, n_alpha = n_beta = 1
///                N_det = C(2,1) * C(2,1) = 2 * 2 = 4 determinants.
///   He / STO-3G: n_mo = 1, n_alpha = n_beta = 1
///                N_det = C(1,1) * C(1,1) = 1 * 1 = 1 determinant
///                => single determinant => FCI == RHF => e_corr == 0 exactly.

#include "atomiverse/cpu/scf/fci.h"
#include "atomiverse/cpu/scf/internal/fci_pack.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/scf/internal/fci_pack.h"
#include "atomiverse/cpu/scf/mp2.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// @brief Binomial coefficient C(n, k) (dimensionless); 0 when out of range.
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

/// @brief All occupation bitstrings of `n_mo` orbitals with popcount == n_elec,
///        in ascending integer order.
std::vector<std::uint64_t> enumerate_strings(int n_mo, int n_elec) {
    std::vector<std::uint64_t> out;
    const std::uint64_t limit = std::uint64_t(1) << n_mo;
    for (std::uint64_t s = 0; s < limit; ++s) {
        if (__builtin_popcountll(s) == n_elec)
            out.push_back(s);
    }
    return out;
}

/// @brief Occupied spatial-orbital indices in a bitstring, ascending.
std::vector<int> occ_list(std::uint64_t s) {
    std::vector<int> v;
    while (s) {
        v.push_back(__builtin_ctzll(s));
        s &= s - 1;
    }
    return v;
}

/// @brief Jordan-Wigner parity sign for moving one electron p->q within a single
///        spin string. Counts occupied orbitals strictly between p and q.
int jw_single_sign(std::uint64_t s, int p, int q) {
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    // Bits with index in the open interval (lo, hi).
    const std::uint64_t mask =
        ((std::uint64_t(1) << hi) - 1) & ~((std::uint64_t(1) << (lo + 1)) - 1);
    return (__builtin_popcountll(s & mask) & 1) ? -1 : 1;
}

/// @brief Diagonal element (Slater-Condon Case A): <I|H|I> including V_nuc.
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

/// @brief Single-excitation element (Case B) for an excitation p->q in one spin
///        channel; `same` is that channel's string in det I (also det J minus q),
///        `other` is the untouched opposite-spin channel string. Sign uses the
///        source channel string `same`.
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

/// @brief Same-spin double-excitation element (Case C, both electrons in one
///        channel). Holes p<q -> particles r<t. Sign = product of two sequential
///        single-excitation signs (p->r, then q->t on the updated string).
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

/// @brief Opposite-spin double-excitation element (Case C, one alpha + one beta
///        excitation). Only the direct Coulomb term survives; the two channel
///        signs multiply with no cross Jordan-Wigner string (block order).
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

/// @brief Full Slater-Condon matrix element <I|H|J> for two determinants.
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

} // namespace

xt::xtensor<double, 2> fci_hamiltonian(const IntegralPack& ints,
                                       const SlaterDeterminant& det) {
    using Spin = SlaterDeterminant::Spin;
    // No-fallback guards -- throw, never return a sentinel (mirror
    // mp2_energy). Convergence of the reference is guarded by the caller.
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument(
            "fci_energy: ERI is not materialized; call IntegralPack{..., "
            "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "fci_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }

    // Row-major copy of the alpha MO coefficients (hoisted once).
    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_alpha = det.n_electrons(Spin::alpha);
    const int n_beta = det.n_electrons(Spin::beta);

    // TINY solvability cap: the dense CI matrix and its eigh grow factorially.
    // This is a validation/reference tool, not a production method.
    if (n_mo > 14) {
        throw std::invalid_argument(
            "fci_energy: n_mo=" + std::to_string(n_mo)
            + " exceeds the TINY cap (n_mo <= 14); this is a reference solver "
              "for "
              "tiny systems, not a production correlation method.");
    }
    const long long n_det = choose(n_mo, n_alpha) * choose(n_mo, n_beta);
    if (n_det > 2000) {
        throw std::invalid_argument(
            "fci_energy: N_det=" + std::to_string(n_det)
            + " exceeds the TINY cap (N_det <= 2000); this is a reference "
              "solver "
              "for tiny systems, not a production correlation method.");
    }

    // MO-basis one- and two-electron integrals (chemists' notation).
    const xt::xtensor<double, 2> h = mo_hcore(ints.Hcore, C, n_mo);
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);

    const std::vector<std::uint64_t> alphas = enumerate_strings(n_mo, n_alpha);
    const std::vector<std::uint64_t> betas = enumerate_strings(n_mo, n_beta);
    const std::size_t n_a = alphas.size();
    const std::size_t n_b = betas.size();
    const std::size_t nd = n_a * n_b;

    // Determinant index: I = ia * n_b + ib (alpha block major).
    xt::xtensor<double, 2> hci = xt::zeros<double>({nd, nd});
    for (std::size_t ia = 0; ia < n_a; ++ia) {
        for (std::size_t ib = 0; ib < n_b; ++ib) {
            const std::size_t idx_i = ia * n_b + ib;
            for (std::size_t ja = 0; ja < n_a; ++ja) {
                for (std::size_t jb = 0; jb < n_b; ++jb) {
                    const std::size_t idx_j = ja * n_b + jb;
                    if (idx_j < idx_i)
                        continue; // upper triangle only
                    const double e =
                        ci_element(alphas[ia], betas[ib], alphas[ja], betas[jb],
                                   h, g, ints.V_nuc);
                    hci(idx_i, idx_j) = e;
                    hci(idx_j, idx_i) = e; // symmetric fill
                }
            }
        }
    }
    return hci;
}

FCIResult fci_energy(const IntegralPack& ints, const SlaterDeterminant& det,
                     double e_scf_ha) {
    const xt::xtensor<double, 2> hci = fci_hamiltonian(ints, det);
    const std::size_t nd = hci.shape(0);

    // Single-determinant boundary (e.g. He/STO-3G, no virtuals): full CI is the
    // RHF reference itself; correlation is exactly zero.
    if (nd == 1) {
        return {e_scf_ha, 0.0};
    }

    auto [w, U] = xt::linalg::eigh(hci);
    (void)U;
    const double e_fci =
        w(0); // eigenvalues ascending; lowest is the ground state
    return {e_fci, e_fci - e_scf_ha};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
