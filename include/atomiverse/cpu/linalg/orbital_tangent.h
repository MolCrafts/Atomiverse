/// @file cpu/linalg/orbital_tangent.h
/// @brief S-metric orbital primitives: projected band-energy gradient, tangent
///        projection, and Löwdin re-orthonormalization of an occupied
///        coefficient block under a non-unit overlap metric `S`.
///
/// **Why a metric appears at all.** Atom-centred basis functions are not
/// mutually orthogonal, so their overlap matrix `S` (entries
/// \(S_{\mu\nu} = \int \chi_\mu^{*}\chi_\nu\,d\mathbf{r}\), dimensionless,
/// symmetric/Hermitian and positive definite) is not the identity. Orbitals
/// expanded in such a basis are orthonormal only in the sense
/// `C^T S C = I` (real coefficients, Γ-point — the single \(k = 0\) sampling
/// point of a periodic cell) or `C† S C = I` (complex Bloch states at a sampled
/// reciprocal-space wavevector \(k\), with `†` the conjugate transpose). Here
/// `C` is the [nb x no] coefficient block, `nb` = number of basis functions,
/// `no` = number of occupied (or otherwise active) orbitals. Everything in this
/// header is one piece of the algebra of that constraint surface: the projector
/// onto its tangent directions, the constrained gradient of the band energy,
/// and the projection back onto it after roundoff.
///
/// Method-agnostic dense algebra — no method, basis or unit system is assumed.
/// Consumed today by the GPW orbital-transformation (OT) minimizer
/// (`cpu/gpw/internal/ot.h`, which adds the OT parameterization, the
/// preconditioners and the fixed-H minimizers on top of these) and by the
/// Car–Parrinello electronic propagators `cpu::electronic::Cpmd` /
/// `cpu::electronic::CpmdCplx`, which integrate the same constrained manifold
/// in time. (`cpu::electronic::XlBomd` does **not** use these primitives: it
/// propagates a density matrix and never touches the overlap metric.) Extracted
/// from `cpu/gpw/internal/ot` so a propagator needs no GPW dependency.
///
/// References:
///   - VandeVondele & Hutter, J. Chem. Phys. 118, 4365 (2003)
///     (DOI 10.1063/1.1543154) — the original OT method these primitives are
///     the constraint-manifold algebra of.
///   - Kühne et al., J. Chem. Phys. 152, 194103 (2020) §VII C, Eqs. 85-104
///     (DOI 10.1063/5.0007045) — the formulation followed here.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h" // eigh_hermitian
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace linalg {

/// @brief Apply a scalar function through a symmetric eigendecomposition:
///        `f(G) = V diag(f(θ)) V^T` for `G = V diag(θ) V^T`.
///
/// A *matrix function* is defined by acting on the spectrum: eigendecompose the
/// symmetric input `G` into eigenvalues θ (a vector of length n, ascending) and
/// orthonormal eigenvectors `V` (columns), apply the scalar `f` to each θ, and
/// reassemble. This is how `cos(U)`, `sinc(U)`, `M^{-1}` and `M^{-1/2}` are
/// formed on the small symmetric blocks of the OT manifold — `x^T S x` and
/// `C^T S C` (dimensionless overlap blocks), `T + S` and `H + a S` (the
/// preconditioner blocks in `cpu/gpw/internal/ot.h`).
/// Header-inline free math with no owning type.
///
/// @param G Real symmetric matrix [n x n] (units are matrix-dependent:
///          dimensionless for overlap blocks, Hartree for Hamiltonian blocks).
///          Assumed symmetric; only one triangle is referenced by the
///          eigensolver, so an asymmetric input is silently symmetrized.
/// @param f Scalar callable on one eigenvalue θ_k (argument in the units of
///          `G`'s eigenvalues). Guarding singular spectra (θ_k ≈ 0 under a
///          negative power) is the caller's job — nothing here checks that
///          `f(θ_k)` is finite; a callable may throw.
/// @return `V diag(f(θ)) V^T` [n x n], units of `f`'s result.
template <typename F>
inline RealMatrix sym_function(const RealMatrix& G, F f) {
    auto [theta, V] = xt::linalg::eigh(G);
    const std::size_t n = theta.size();
    RealMatrix D = xt::zeros<double>({n, n});
    for (std::size_t k = 0; k < n; ++k)
        D(k, k) = f(theta(k));
    return xt::linalg::dot(xt::linalg::dot(V, D), xt::transpose(V));
}

/// @brief Hermitian twin of `sym_function`: `f(G) = V diag(f(θ)) V†`
///        (LAPACK zheevd via `eigh_hermitian`).
///
/// A Hermitian matrix has real eigenvalues, so the same spectral construction as
/// `sym_function` applies with the transpose replaced by the conjugate transpose
/// `†`. Used for the complex Bloch metric blocks (`C† S C`).
///
/// @param G Hermitian matrix [n x n] (dimensionless overlap blocks or Hartree
///          Hamiltonian blocks, matrix-dependent). Only the lower triangle is
///          referenced by the eigensolver.
/// @param f Scalar callable on one real eigenvalue θ_k (argument in the units of
///          `G`'s eigenvalues). Singular-spectrum guarding is the caller's job;
///          a callable may throw (as `lowdin_orthonormalize_k` does).
/// @return `V diag(f(θ)) V†` [n x n], units of `f`'s result.
/// @throws std::runtime_error when the underlying `eigh_hermitian` (LAPACK
///         zheevd) fails.
template <typename F>
inline CplxMatrix herm_function(const CplxMatrix& G, F f) {
    std::vector<double> theta;
    CplxMatrix V;
    eigh_hermitian(G, theta, V);
    const int n = static_cast<int>(theta.size());
    CplxMatrix out = xt::zeros<std::complex<double>>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            std::complex<double> s(0, 0);
            for (int k = 0; k < n; ++k) {
                const double fk = f(theta[static_cast<std::size_t>(k)]);
                s += V(i, k) * fk * std::conj(V(j, k));
            }
            out(i, j) = s;
        }
    return out;
}

/// @brief Row-major complex matrix product `A B`, computed by BLAS `zgemm`
///        (a row-major matrix is bitwise its column-major transpose, so the
///        product is evaluated as `C^T = B^T A^T` with no copies).
/// @param A Left factor [n x m] (matrix-dependent units).
/// @param B Right factor [m x p] (matrix-dependent units).
/// @return `A B` [n x p], units of the product.
CplxMatrix c_mul(const CplxMatrix& A, const CplxMatrix& B);

/// @brief Row-major complex product with the conjugate transpose: `A† B`
///        (`†` = transpose plus complex conjugation).
/// @param A Left factor [n x m], used as `A†` (matrix-dependent units).
/// @param B Right factor [n x p] (matrix-dependent units).
/// @return `A† B` [m x p], units of the product.
CplxMatrix c_dagger_mul(const CplxMatrix& A, const CplxMatrix& B);

/// @brief S-tangent projection of a direction: `(I - C C^T S) d`.
///
/// "Tangent" means tangent to the constraint surface `C^T S C = I`: the result
/// `t` satisfies `C^T S t = 0`, so moving along `t` does not, to first order,
/// break orthonormality — and for the OT parameterization
/// (`cpu/gpw/internal/ot.h`) it does not break it at all. The subtracted piece
/// is the component of `d` that lies inside the occupied subspace itself, which
/// only re-mixes orbitals that are already there.
///
/// @param C Reference occupied coefficients [nb x no] (dimensionless),
///          `C^T S C = I`.
/// @param S Overlap matrix [nb x nb] (dimensionless).
/// @param d Unconstrained direction [nb x no] (units of `d`).
/// @return S-tangent direction [nb x no], units of `d`.
RealMatrix s_tangent(const RealMatrix& C, const RealMatrix& S,
                     const RealMatrix& d);

/// @brief Projected band-energy gradient `g = (I - S C C^T)(2·occ·H C)`.
///
/// The band energy of the occupied orbitals is
/// \(E_{\mathrm{band}} = \mathrm{occ}\cdot\mathrm{tr}(C^\top H C)\) [Hartree]
/// (each of the `no` orbitals holds `occ` electrons), whose unconstrained
/// derivative with respect to `C` is \(2\,\mathrm{occ}\,H C\); the factor
/// `(I - S C C^T)` removes the part of that derivative which merely rotates the
/// occupied orbitals among themselves and would break `C^T S C = I`.
///
/// This is the self-consistent-field (SCF) error vector: `||g|| -> 0` at the
/// self-consistent ground state of a gapped system, where
/// `H C = S C (C^T H C)` holds — substituting it makes the two terms cancel
/// identically. Closed-shell default `occ=2` recovers the historical factor 4;
/// unrestricted Kohn–Sham (UKS) spin channels use `occ=1` (single occupation,
/// `P = C C^T`).
///
/// @param H Kohn–Sham (KS) matrix — the effective one-electron Hamiltonian in
///          the basis — [nb x nb] [Hartree].
/// @param S Overlap matrix [nb x nb] (dimensionless).
/// @param C Occupied coefficients [nb x no] (dimensionless), `C^T S C = I`.
/// @param occ Orbital occupation (dimensionless; 2 closed-shell, 1 per UKS spin).
/// @return Gradient [nb x no] [Hartree].
RealMatrix band_gradient(const RealMatrix& H, const RealMatrix& S,
                         const RealMatrix& C, double occ = 2.0);

/// @brief Re-orthonormalize occupied coefficients so `C^T S C = I` (Löwdin).
///
/// Cleans roundoff after a rotation or a time step: `C <- C (C^T S C)^{-1/2}`,
/// the symmetric (Löwdin) orthogonalization — it mixes the columns of `C` as
/// little as possible, unlike a Gram–Schmidt sweep which privileges the first
/// column. The inverse square root is taken through the eigendecomposition of
/// the small [no x no] metric block `C^T S C`.
///
/// @param C Occupied coefficients [nb x no] (dimensionless).
/// @param S Overlap matrix [nb x nb] (dimensionless).
/// @return S-orthonormal coefficients [nb x no] (dimensionless).
/// @note A singular metric block is **not** diagnosed: a non-positive
///       eigenvalue of `C^T S C` yields non-finite entries rather than a throw.
///       The complex twin `lowdin_orthonormalize_k` does throw — the asymmetry
///       is real, not a documentation shortcut.
RealMatrix lowdin_orthonormalize(const RealMatrix& C, const RealMatrix& S);

// ── Complex (Bloch / multi-k) twins ────────────────────────────────────────
// Hermitian generalization of the real S-metric manifold for k-space matrices.

/// @brief S-tangent projection `(I − C C† S) d` (Hermitian metric) — the
///        Bloch twin of `s_tangent`, with the same meaning and proof.
/// @param C Reference occupied Bloch coefficients [nb x no] (dimensionless),
///          `C† S C = I`.
/// @param S Hermitian overlap [nb x nb] (dimensionless).
/// @param d Unconstrained direction [nb x no] (units of `d`).
/// @return S-tangent direction [nb x no] satisfying `C† S t = 0`, units of `d`.
CplxMatrix s_tangent_k(const CplxMatrix& C, const CplxMatrix& S,
                       const CplxMatrix& d);

/// @brief Projected band-energy gradient `g = (I − S C C†)(2·occ·H C)` at one
///        sampled k-point — the Bloch twin of `band_gradient` (see it for the
///        derivation and the self-consistency limit).
/// @param H Kohn–Sham matrix at one k [nb x nb] [Hartree], Hermitian.
/// @param S Hermitian overlap at one k [nb x nb] (dimensionless).
/// @param C Occupied coefficients [nb x no] (dimensionless), `C† S C = I`.
/// @param occ Closed-shell band occupation (dimensionless, default 2).
/// @return Gradient [nb x no] [Hartree].
CplxMatrix band_gradient_k(const CplxMatrix& H, const CplxMatrix& S,
                           const CplxMatrix& C, double occ = 2.0);

/// @brief Projected gradient with **non-uniform** band occupations \(f_i\).
///
/// For \(P = C \mathrm{diag}(f) C^\dagger\) the band-energy derivative is
/// \(2 H C f\) (column \(i\) scaled by \(f_i\)); project off the S-occupied
/// subspace. This is the scalar `occ` overload with a per-column occupation:
/// setting every \(f_i = \mathrm{occ}\) reproduces it exactly. Used by multi-k
/// OT with *metallic smearing* — replacing the sharp occupied/empty step by a
/// smooth occupation function so that bands crossing the Fermi level do not make
/// the energy discontinuous, either Fermi–Dirac (FD) or Methfessel–Paxton (MP).
///
/// @param H Kohn–Sham matrix at one k [nb x nb] [Hartree], Hermitian.
/// @param S Hermitian overlap [nb x nb] (dimensionless).
/// @param C Active coefficients [nb x no] (dimensionless), `C† S C = I`.
/// @param f Band occupations [no] (dimensionless), closed-shell form
///          \(f_i \in [0,2]\); entries beyond `f.size()` count as 0, which
///          zeroes those gradient columns.
/// @return Gradient [nb x no] [Hartree].
CplxMatrix band_gradient_k(const CplxMatrix& H, const CplxMatrix& S,
                           const CplxMatrix& C, const std::vector<double>& f);

/// @brief Löwdin re-orthonormalize so `C† S C = I`:
///        `C <- C (C† S C)^{-1/2}`, the Bloch twin of `lowdin_orthonormalize`.
/// @param C Occupied Bloch coefficients [nb x no] (dimensionless).
/// @param S Hermitian overlap [nb x nb] (dimensionless).
/// @return S-orthonormal coefficients [nb x no] (dimensionless).
/// @throws std::runtime_error on a non-positive metric eigenvalue of `C† S C`
///         (≤ 1e-16) — the basis has gone linearly dependent; no fallback value.
CplxMatrix lowdin_orthonormalize_k(const CplxMatrix& C, const CplxMatrix& S);

} // namespace linalg
} // namespace cpu
} // namespace ATV
