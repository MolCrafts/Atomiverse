/// @file cpu/scf/internal/ci_engine.h
/// @brief **PRIVATE** shared determinant-basis CI engine for the TINY reference
///        solvers (FCI / CASCI / CASSCF / NEVPT2).
///
/// One home for the determinant combinatorics, Slater–Condon matrix elements
/// (with Jordan–Wigner phases), the dense CI eigensolve, and the spin-summed
/// active-space 1-/2-RDM build shared by `fci.cpp` and `casscf.cpp`. The
/// Slater–Condon evaluators are the canonical `fci.cpp` implementations. The
/// integral index convention is `h(p,q)` one-electron and CHEMIST-notation
/// two-electron `g(p,q,r,s) = (pq|rs)` (Coulomb = g(i,i,j,j),
/// exchange = g(i,j,j,i)) — this is the axis on which the deleted casscf
/// transposed fork differed. Excitation level is
/// `popcount(a_i & ~a_j) ∈ {0,1,2}`; the container/solver superset is adopted
/// from `casscf.cpp`. Reference: Szabo & Ostlund, *Modern Quantum Chemistry*
/// §2.3.
///
/// Not a public API — only `cpu/scf` internals and kernel unit tests consume it.
/// TINY caps (n_mo ≤ 14, N_det ≤ 2000) throw with no fallback.
#pragma once

#include <cstdint>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Binomial coefficient C(n, k) (dimensionless); 0 when out of range.
/// @param n Set size (dimensionless).
/// @param k Chosen count (dimensionless).
/// @return C(n, k) (dimensionless); 0 when k < 0 or k > n.
long long choose(int n, int k);

/// @brief All occupation bitstrings of `n_mo` orbitals with popcount == n_elec,
///        in ascending integer order.
/// @param n_mo   Spatial orbitals in this spin channel (dimensionless).
/// @param n_elec Electrons in this spin channel (dimensionless).
/// @return Occupation bitstrings (dimensionless), ascending integer order.
std::vector<std::uint64_t> enumerate_strings(int n_mo, int n_elec);

/// @brief Occupied spatial-orbital indices in a bitstring, ascending.
/// @param s Occupation bitstring (dimensionless).
/// @return Occupied spatial-orbital indices (dimensionless), ascending.
std::vector<int> occ_list(std::uint64_t s);

/// @brief Jordan–Wigner parity sign for moving one electron p->q within a
///        single spin string. Counts occupied orbitals strictly between p, q.
/// @param s Occupation bitstring of the spin channel (dimensionless).
/// @param p Source orbital index (dimensionless).
/// @param q Destination orbital index (dimensionless).
/// @return +1 or -1 parity sign (dimensionless).
int jw_single_sign(std::uint64_t s, int p, int q);

/// @brief Diagonal element (Slater–Condon Case A): <I|H|I> including V_nuc.
/// @param a     Alpha occupation bitstring of determinant I (dimensionless).
/// @param b     Beta occupation bitstring of determinant I (dimensionless).
/// @param h     One-electron MO integrals h(p,q) [Hartree].
/// @param g     Two-electron MO integrals g(p,q,r,s) = (pq|rs) [Hartree].
/// @param v_nuc Nuclear-repulsion (+ frozen-core) shift [Hartree].
/// @return <I|H|I> [Hartree].
double diagonal_element(std::uint64_t a, std::uint64_t b,
                        const xt::xtensor<double, 2>& h,
                        const xt::xtensor<double, 4>& g, double v_nuc);

/// @brief Single-excitation element (Case B) for an excitation p->q in one spin
///        channel; `same_i`/`same_j` are that channel's strings, `other` the
///        untouched opposite-spin channel. Sign uses the source string.
/// @param same_i Excited channel's bitstring in bra I (dimensionless).
/// @param same_j Excited channel's bitstring in ket J (dimensionless).
/// @param other  Untouched opposite-spin channel bitstring (dimensionless).
/// @param h      One-electron MO integrals h(p,q) [Hartree].
/// @param g      Two-electron MO integrals g(p,q,r,s) = (pq|rs) [Hartree].
/// @return <I|H|J> for the single excitation [Hartree].
double single_element(std::uint64_t same_i, std::uint64_t same_j,
                      std::uint64_t other, const xt::xtensor<double, 2>& h,
                      const xt::xtensor<double, 4>& g);

/// @brief Same-spin double-excitation element (Case C, both electrons in one
///        channel). Holes p<q -> particles r<t; sign is the product of the two
///        sequential single-excitation signs.
/// @param s_from    Source bitstring of the excited channel (dimensionless).
/// @param hole_mask Two-bit mask of vacated orbitals p<q (dimensionless).
/// @param part_mask Two-bit mask of filled orbitals r<t (dimensionless).
/// @param g         Two-electron MO integrals g(p,q,r,s) = (pq|rs) [Hartree].
/// @return <I|H|J> for the same-spin double excitation [Hartree].
double same_spin_double_element(std::uint64_t s_from, std::uint64_t hole_mask,
                                std::uint64_t part_mask,
                                const xt::xtensor<double, 4>& g);

/// @brief Opposite-spin double-excitation element (Case C, one alpha + one beta
///        excitation). Only the direct Coulomb term survives; returns the
///        canonical g(pa,ra,qb,sb) ordering.
/// @param a_i Alpha bitstring of bra I (dimensionless).
/// @param a_j Alpha bitstring of ket J (dimensionless).
/// @param b_i Beta bitstring of bra I (dimensionless).
/// @param b_j Beta bitstring of ket J (dimensionless).
/// @param g   Two-electron MO integrals g(p,q,r,s) = (pq|rs) [Hartree].
/// @return <I|H|J> for the opposite-spin double excitation [Hartree].
double opposite_spin_double_element(std::uint64_t a_i, std::uint64_t a_j,
                                    std::uint64_t b_i, std::uint64_t b_j,
                                    const xt::xtensor<double, 4>& g);

/// @brief Full Slater–Condon matrix element <I|H|J> for two determinants. The
///        `_i` bitstrings are the bra determinant I, the `_j` bitstrings the ket
///        determinant J.
/// @param a_i   Alpha occupation bitstring of bra I (dimensionless).
/// @param b_i   Beta occupation bitstring of bra I (dimensionless).
/// @param a_j   Alpha occupation bitstring of ket J (dimensionless).
/// @param b_j   Beta occupation bitstring of ket J (dimensionless).
/// @param h     One-electron MO integrals h(p,q) [Hartree].
/// @param g     Two-electron MO integrals g(p,q,r,s) = (pq|rs) [Hartree].
/// @param v_nuc Nuclear-repulsion (+ frozen-core) shift [Hartree].
/// @return <I|H|J> [Hartree].
double ci_element(std::uint64_t a_i, std::uint64_t b_i, std::uint64_t a_j,
                  std::uint64_t b_j, const xt::xtensor<double, 2>& h,
                  const xt::xtensor<double, 4>& g, double v_nuc);

/// @brief Dense CI ground state: energy [Hartree], CI vector, and the enumerated
///        alpha/beta strings (block index I = ia*n_b + ib).
struct CiSolution {
    double energy = 0.0;      ///< Ground-state energy [Hartree].
    xt::xtensor<double, 1> c; ///< CI vector, length n_det.
    std::vector<std::uint64_t>
        alphas; ///< Enumerated α (resp. β) occupation bitstrings; block index I = ia*n_b + ib.
    std::vector<std::uint64_t>
        betas; ///< Enumerated α (resp. β) occupation bitstrings; block index I = ia*n_b + ib.
};

/// @brief Build the dense CI matrix from active-space h, g and a constant shift
///        (V_nuc + frozen-core energy) and diagonalize for the ground state.
/// @param h       Active one-electron MO integrals (chemist), n_mo × n_mo [Hartree].
/// @param g       Active two-electron MO integrals (chemist), n_mo⁴ [Hartree].
/// @param v_shift Constant energy shift [Hartree] (V_nuc + frozen-core).
/// @param n_alpha Active alpha electrons (dimensionless).
/// @param n_beta  Active beta electrons (dimensionless).
/// @return Ground-state CiSolution (energy [Hartree], normalized CI vector,
///         enumerated α/β strings).
/// @throws std::invalid_argument above the TINY cap (no fallback).
CiSolution dense_ci_solve(const xt::xtensor<double, 2>& h,
                          const xt::xtensor<double, 4>& g, double v_shift,
                          int n_alpha, int n_beta);

/// @brief Spin-summed active-space 1- and 2-RDMs from a CI solution.
///        γ_pq = Σ_σ⟨a†_pσ a_qσ⟩, Γ_pqrs = Σ_στ⟨a†_pσ a†_rτ a_sτ a_qσ⟩ (matches
///        casscf.cpp).
/// @param sol       Dense CI ground state (`dense_ci_solve` output).
/// @param ncas      Active spatial orbitals (dimensionless).
/// @param[out] rdm1 Spin-summed 1-RDM γ, ncas×ncas (dimensionless).
/// @param[out] rdm2 Spin-summed 2-RDM Γ, ncas⁴ (dimensionless).
void make_rdm12(const CiSolution& sol, int ncas, xt::xtensor<double, 2>& rdm1,
                xt::xtensor<double, 4>& rdm2);

} // namespace scf
} // namespace cpu
} // namespace ATV
