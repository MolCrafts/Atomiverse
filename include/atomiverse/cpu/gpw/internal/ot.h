/// @file cpu/gpw/internal/ot.h
/// @brief Orbital Transformation (OT) inner minimizer for the GPW SCF.
///
/// **Internal GPW helper.** Does NOT inherit `cpu::Component`; owned by
/// `cpu::gpw::GPWSolver` (and by tests via direct include). Replaces the dense
/// `O(nb^3)` generalized diagonalization with a direct minimization of the
/// closed-shell energy over the occupied coefficients `C` under the
/// orthonormality constraint `C^T S C = I`, following CP2K's OT method.
///
/// Parameterization (orthonormality-preserving for any S-tangent `x`,
/// `C0^T S x = 0`):
///   `C(x) = C0 cos(U) + x U^{-1} sin(U)`,  `U = (x^T S x)^{1/2}`.
/// Closed-shell band-energy gradient (matches `build_density_matrix` factor
/// 2, i.e. `P = 2 C C^T`):
///   `g = (I - S C0 C0^T) (4 H C0)`.
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
/// @param x  S-tangent step [nb x no] (project with ot_tangent first).
/// @param S  Overlap matrix [nb x nb].
/// @return Orthonormal occupied coefficients `C(x)` [nb x no].
RealMatrix ot_parameterize(const RealMatrix& C0, const RealMatrix& x,
                           const RealMatrix& S);

/// @brief S-tangent projection of a direction: `(I - C0 C0^T S) d`.
///
/// The result `t` satisfies `C0^T S t = 0`, so `ot_parameterize(C0, alpha*t, S)`
/// stays on the constraint manifold for any step `alpha`.
///
/// @param C0 Reference occupied coefficients [nb x no], `C0^T S C0 = I`.
/// @param S  Overlap matrix [nb x nb].
/// @param d  Unconstrained direction [nb x no].
/// @return S-tangent direction [nb x no].
RealMatrix ot_tangent(const RealMatrix& C0, const RealMatrix& S,
                      const RealMatrix& d);

/// @brief Projected closed-shell band-energy gradient `g = (I - S C0 C0^T)(4 H C0)`.
///
/// This is the SCF error vector; `||g|| -> 0` at the self-consistent
/// (gapped) ground state, where `H C0 = S C0 (C0^T H C0)`.
///
/// @param H KS matrix [nb x nb].
/// @param S Overlap matrix [nb x nb].
/// @param C0 Occupied coefficients [nb x no], `C0^T S C0 = I`.
/// @return Gradient [nb x no].
RealMatrix ot_gradient(const RealMatrix& H, const RealMatrix& S,
                       const RealMatrix& C0);

/// @brief Re-orthonormalize occupied coefficients so `C^T S C = I` (Löwdin).
///
/// Cleans roundoff after a rotation; `C <- C (C^T S C)^{-1/2}`.
///
/// @param C Occupied coefficients [nb x no].
/// @param S Overlap matrix [nb x nb].
/// @return S-orthonormal coefficients [nb x no].
RealMatrix ot_orthonormalize(const RealMatrix& C, const RealMatrix& S);

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
    double band_energy; ///< `2 Tr(C^T H C)` [Hartree].
    int iters;          ///< OT iterations taken.
    bool converged;     ///< `||g|| < tol` reached.
};

/// @brief Minimize the closed-shell band energy `2 Tr(C^T H C)` for a FIXED H.
///
/// Diagonalization-free: preconditioned CG (FULL_KINETIC) on the OT manifold.
/// Converges to the `no` lowest generalized eigenvectors of `(H, S)` — i.e.
/// the same occupied subspace `diagonalize_fock_generalized` would give.
/// Used to validate the OT machinery against diagonalization with no SCF/grid.
///
/// @param H KS matrix [nb x nb] (fixed).
/// @param S Overlap matrix [nb x nb].
/// @param T Kinetic matrix [nb x nb] (preconditioner).
/// @param C0 Initial occupied guess [nb x no], `C0^T S C0 = I`.
/// @param tol Projected-gradient-norm convergence threshold.
/// @param max_iter Iteration cap.
/// @return OtFixedResult.
OtFixedResult ot_minimize_fixed(const RealMatrix& H, const RealMatrix& S,
                                const RealMatrix& T, const RealMatrix& C0,
                                double tol, int max_iter);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
