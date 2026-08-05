/// @file cpu/gpw/internal/ot.h
/// @brief Orbital Transformation (OT) inner minimizer for the GPW SCF.
///
/// **Internal GPW helper.** Does NOT inherit `cpu::Component`; owned by
/// `cpu::gpw::GPW` (and by tests via direct include). Replaces the dense
/// `O(nb^3)` generalized diagonalization with a direct minimization of the
/// closed-shell energy over the occupied coefficients `C` under the
/// orthonormality constraint `C^T S C = I`, following CP2K's OT method.
///
/// Parameterization (orthonormality-preserving for any S-tangent `x`,
/// `C0^T S x = 0`):
///   `C(x) = C0 cos(U) + x U^{-1} sin(U)`,  `U = (x^T S x)^{1/2}`.
/// Band-energy gradient for orbital occupation `occ` (matches
/// `build_density_matrix` factor; closed-shell `occ=2` → `P = 2 C C^T`,
/// open-shell / UKS spin channel `occ=1` → `P = C C^T`):
///   `g = (I - S C0 C0^T) (2 * occ * H C0)`.
/// Minimized by preconditioned conjugate gradient with a FULL_KINETIC
/// preconditioner `M = T + S` (geometry-fixed; inverse built once).
///
/// References:
///   - Kühne et al., J. Chem. Phys. 152, 194103 (2020) §VII C, Eqs. 85-104
///     (DOI 10.1063/5.0007045) — the fetched OT formulation followed here.
///   - VandeVondele & Hutter, J. Chem. Phys. 118, 4365 (2003)
///     (DOI 10.1063/1.1543154) — the original OT method.
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/linalg/orbital_tangent.h" // s_tangent, band_gradient

#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

using cpu::linalg::RealMatrix;

/// @brief Evaluate the OT parameterization `C(x) = C0 cos(U) + x sinc(U)`.
///
/// `U = (x^T S x)^{1/2}` via the symmetric eigendecomposition of the small
/// `no x no` matrix `x^T S x`; `sinc(0) = 1` (Taylor-guarded). For any `x`
/// with `C0^T S x = 0` the result satisfies `C(x)^T S C(x) = I` exactly.
///
/// @param C0 Reference occupied coefficients [nb x no], `C0^T S C0 = I`.
/// @param x  S-tangent step [nb x no] (project with `linalg::s_tangent` first).
/// @param S  Overlap matrix [nb x nb].
/// @return Orthonormal occupied coefficients `C(x)` [nb x no].
RealMatrix ot_parameterize(const RealMatrix& C0, const RealMatrix& x,
                           const RealMatrix& S);

/// @brief FULL_KINETIC OT preconditioner `M = T + S`, applied as `M^{-1}`.
///
/// `M` is geometry-fixed, so its inverse is built once and reused across all
/// OT steps. Dense inverse is `O(nb^3)` once per geometry (the documented
/// dense-OT bottleneck; true linear scaling needs sparse M).
class OtPrecond {
  public:
    /// @brief Build `M^{-1}` from the (geometry-fixed) kinetic + overlap.
    /// @param T Kinetic matrix [nb x nb].
    /// @param S Overlap matrix [nb x nb].
    OtPrecond(const RealMatrix& T, const RealMatrix& S);

    /// @brief Apply the preconditioner: `z = M^{-1} g`.
    /// @param g Gradient [nb x no].
    /// @return Preconditioned direction [nb x no].
    RealMatrix apply(const RealMatrix& g) const;

  private:
    RealMatrix m_inv_; ///< Dense `(T+S)^{-1}` [nb x nb].
};

/// @brief FULL_SINGLE_INVERSE-style OT preconditioner `M = H + a S`, `M^{-1}`.
///
/// CP2K's default OT preconditioner for gapped systems: H-aware (unlike the
/// geometry-fixed FULL_KINETIC `T+S`), so the orbitals near the band gap are
/// most active and convergence is far faster on hard condensed systems. The
/// level shift `a = -lambda_lo + margin` (lambda_lo = lowest occupied orbital
/// energy from the small `C^T H C` block) puts the occupied region at `~margin`
/// in the S-metric, keeping `M` positive-definite so `M^{-1} g` is a descent
/// direction. Rebuilt each OT step from the current `H, C` (dense `O(nb^3)`
/// per step — the documented dense-OT cost; sparse is a later optimization).
class OtPrecondSingleInverse {
  public:
    /// @brief Build `M^{-1}` from the current KS matrix, overlap, and occupied set.
    /// @param H KS matrix [nb x nb].
    /// @param S Overlap matrix [nb x nb].
    /// @param C Occupied coefficients [nb x no], `C^T S C = I`.
    OtPrecondSingleInverse(const RealMatrix& H, const RealMatrix& S,
                           const RealMatrix& C);

    /// @brief Apply the preconditioner: `z = M^{-1} g`.
    RealMatrix apply(const RealMatrix& g) const;

  private:
    RealMatrix m_inv_; ///< Dense `(H + a S)^{-1}` [nb x nb].
};

/// @brief Result of the fixed-H OT minimization (decoupled unit-test entry).
struct OtFixedResult {
    RealMatrix C;       ///< Converged occupied coefficients [nb x no].
    double band_energy; ///< `occ · Tr(C^T H C)` [Hartree].
    int iters;          ///< OT micro-iters (exact path: 1).
    bool converged;     ///< exact path always true; CG path: `||g|| < tol`.
};

/// @brief Minimize the band energy `occ · Tr(C^T H C)` for a FIXED H.
///
/// Returns the `no` lowest generalized eigenvectors of `(H, S)` — the global
/// minimum of the fixed-H band energy on the S-Stiefel manifold, identical to
/// the fixed point of OT-CG. Product implementation is one dense generalized
/// eigen (`diagonalize_fock_generalized`); iterative FULL_SINGLE_INVERSE CG
/// costs ≥1× O(nb³) per micro-step just to rebuild M^{-1} and leaves enough
/// residual noise to inflate outer Pulay DIIS counts on open-shell systems.
///
/// @param H KS matrix [nb x nb] (fixed).
/// @param S Overlap matrix [nb x nb].
/// @param T Kinetic matrix [nb x nb] (API; unused on the exact product path).
/// @param C0 Initial occupied guess [nb x no] (shape only; unused for exact).
/// @param tol Projected-gradient-norm threshold (API; unused for exact).
/// @param max_iter Iteration cap (API; unused for exact).
/// @param occ Orbital occupation (2 closed-shell RKS; 1 per UKS spin channel).
/// @return OtFixedResult (`iters=1`, `converged=true` on the exact path).
OtFixedResult ot_minimize_fixed(const RealMatrix& H, const RealMatrix& S,
                                const RealMatrix& T, const RealMatrix& C0,
                                double tol, int max_iter, double occ = 2.0);

// ── Complex (Bloch / multi-k) OT ───────────────────────────────────────────
// Hermitian generalization of the real OT manifold for k-space matrices, on
// `linalg::CplxMatrix` (row-major complex; same type as
// `gpw::internal::CplxMatrix`). The S-metric primitives (tangent, gradient,
// Löwdin) live in `cpu/linalg/orbital_tangent.h`.

/// @brief Complex OT parameterization `C(x) = C0 cos(U) + x sinc(U)`.
/// @param C0 Occupied Bloch coefficients [nb x no], `C0† S C0 = I`.
/// @param x  S-tangent step [nb x no] (`C0† S x = 0`).
/// @param S  Hermitian overlap [nb x nb].
linalg::CplxMatrix ot_parameterize_k(const linalg::CplxMatrix& C0,
                                     const linalg::CplxMatrix& x,
                                     const linalg::CplxMatrix& S);

/// @brief FULL_SINGLE_INVERSE preconditioner for complex Hermitian H,S.
class OtPrecondSingleInverseK {
  public:
    OtPrecondSingleInverseK(const linalg::CplxMatrix& H,
                            const linalg::CplxMatrix& S,
                            const linalg::CplxMatrix& C);
    linalg::CplxMatrix apply(const linalg::CplxMatrix& g) const;

  private:
    linalg::CplxMatrix m_inv_;
};

/// @brief Fixed-H complex OT band minimization (unit-test entry).
struct OtFixedResultK {
    linalg::CplxMatrix C;
    double band_energy; ///< `2 Re Tr(C† H C)` [Hartree].
    int iters = 0;
    bool converged = false;
};

/// @brief Minimize closed-shell band energy on a FIXED complex Hermitian H.
/// @param T Kinetic (FULL_KINETIC precond: M = T+S; may equal H for tests).
OtFixedResultK ot_minimize_fixed_k(const linalg::CplxMatrix& H,
                                   const linalg::CplxMatrix& S,
                                   const linalg::CplxMatrix& T,
                                   const linalg::CplxMatrix& C0, double tol,
                                   int max_iter);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
