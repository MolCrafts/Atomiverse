/// @file cpu/gpw/internal/ot.h
/// @brief Orbital Transformation (OT) for GPW SCF — CP2K-aligned.
///
/// **Internal GPW helper.** Does NOT inherit `cpu::Component`. Product Γ path
/// Product Γ OT (oracle twin class):
///   - MINIMIZER DIIS (CP2K g·e SAFE metric) + fixed STEPSIZE geodesic (1 KS
///     per outer); PR+ CG fallback on energy-ascent / stall
///   - PRECONDITIONER FULL_SINGLE_INVERSE (occupied projector + min-ev shift,
///     ENERGY_GAP 0.08 Ha) dense SPD inverse
///   - OUTER period rebuilds precond without wiping DIIS history
///   - OUTER_SCF polish (no DFT+U): once projected residual is moderate,
///     hand residual SCF to Pulay DIIS on H (closes energy+err)
///
/// Fair twin `ot-minimizer-water` (wave-8 re-measure, OMP=4 bind): control
/// path **~2.72× → ~1.20×** (OT-only min **2.986 s** / CP2K pin **2.496 s**;
/// 25→10 it). Residual to ≤1.1× (~2.75 s) is **per-KS/setup** on ngrids=1
/// (cbg + GGA XC ~150 ms/KS; polish ~7 Pulay) — follow-on
/// `p0-ot-match-residual`. Pure fixed-STEPSIZE OT-DIIS plateaus ~3.8e-3;
/// keep OUTER Pulay polish.
///
/// Parameterization (orthonormality-preserving for any S-tangent `x`,
/// `C0^T S x = 0`):
///   `C(x) = C0 cos(U) + x U^{-1} sin(U)`,  `U = (x^T S x)^{1/2}`.
///
/// References:
///   - VandeVondele & Hutter, J. Chem. Phys. 118, 4365 (2003)
///   - Kühne et al., J. Chem. Phys. 152, 194103 (2020) §VII C
///   - CP2K `qs_ot*.F` / `preconditioner_makes.F` (FULL_SINGLE_INVERSE)
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/linalg/orbital_tangent.h"

#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

using cpu::linalg::RealMatrix;

/// @brief OT parameterization `C(x) = C0 cos(U) + x sinc(U)`.
RealMatrix ot_parameterize(const RealMatrix& C0, const RealMatrix& x,
                           const RealMatrix& S);

/// @brief FULL_KINETIC OT preconditioner `M = T + S` (geometry-fixed).
class OtPrecond {
  public:
    OtPrecond(const RealMatrix& T, const RealMatrix& S);
    RealMatrix apply(const RealMatrix& g) const;

  private:
    RealMatrix m_inv_;
};

/// @brief H-aware OT preconditioner (CP2K FULL_SINGLE_INVERSE).
///
/// Dense product form matching `make_full_single_inverse`:
///   `M = H − 2 (SC)(CᵀHC − ½ I)(SC)ᵀ`, then shift virtuals so the lowest
/// generalized eigenvalue sits at `max(1.5(ε_min−ε_HOMO), energy_gap)`.
/// Dense SPD inverse via eigendecomposition. Default `energy_gap = 0.08` Ha
/// (CP2K FULL_SINGLE_INVERSE default). Built once per OUTER_SCF period.
class OtPrecondSingleInverse {
  public:
    /// @param H KS matrix [nb×nb] [Ha]
    /// @param S Overlap [nb×nb]
    /// @param C Occupied coeffs [nb×no], `Cᵀ S C = I`
    /// @param energy_gap Minimum virtual shift [Ha] (CP2K ENERGY_GAP)
    OtPrecondSingleInverse(const RealMatrix& H, const RealMatrix& S,
                           const RealMatrix& C, double energy_gap = 0.08);

    RealMatrix apply(const RealMatrix& g) const;

  private:
    RealMatrix m_inv_;
};

/// @brief Result of fixed-H OT (unit tests / multi-k OUTER_SCF+U).
struct OtFixedResult {
    RealMatrix C;
    double band_energy; ///< `occ · Tr(Cᵀ H C)` [Ha]
    int iters = 0;
    bool converged = false;
};

/// @brief Fixed-H occupied eigenspace of `(H,S)` (global band-energy min).
///
/// Used where H is frozen (multi-k OUTER_SCF+U). One generalized eigen —
/// same fixed point as OT-CG on the band energy.
OtFixedResult ot_minimize_fixed(const RealMatrix& H, const RealMatrix& S,
                                const RealMatrix& T, const RealMatrix& C0,
                                double tol, int max_iter, double occ = 2.0);

/// @brief CP2K-style OT orbital DIIS (SAFE_DIIS) on S-tangent variables.
///
/// History stores `(x_i, e_i)` with `e = −stepsize · P⁻¹ g`. New point
/// `x ← Σ c_i (x_i + e_i)`; if the step is not a descent direction under
/// SAFE_DIIS, fall back to steepest descent `x_cur + e_cur`.
class OtOrbitalDiis {
  public:
    /// @param depth DIIS history length (CP2K default 7)
    /// @param stepsize Fixed OT step (CP2K STEPSIZE, default 0.15)
    explicit OtOrbitalDiis(int depth = 7, double stepsize = 0.15);

    void reset();

    /// @brief One DIIS / SAFE_DIIS step on OT variables (fixed stepsize).
    /// @param x Current OT variables [nb×no] (S-tangent to C0)
    /// @param g Current OT gradient [nb×no]
    /// @param precond_g Preconditioned gradient `P⁻¹ g` (not yet scaled)
    /// @return Updated OT variables [nb×no]
    RealMatrix step(const RealMatrix& x, const RealMatrix& g,
                    const RealMatrix& precond_g);

    /// @brief DIIS search direction for Armijo (re-anchored product path).
    ///
    /// Stores error vectors `e = −P⁻¹ g`, mixes them with Pulay DIIS, and
    /// returns a descent direction `d` (SAFE_DIIS → `−P⁻¹ g` if the mix is
    /// not a descent). Caller applies Armijo / geodesic step on `d`.
    /// @param g OT gradient [nb×no]
    /// @param precond_g `P⁻¹ g` [nb×no]
    /// @return Search direction [nb×no]
    RealMatrix direction(const RealMatrix& g, const RealMatrix& precond_g);

    int depth() const { return depth_; }
    double stepsize() const { return stepsize_; }
    int n_hist() const { return static_cast<int>(hist_e_.size()); }

  private:
    int depth_;
    double stepsize_;
    std::vector<RealMatrix> hist_x_;
    std::vector<RealMatrix> hist_e_;
    std::vector<RealMatrix> hist_g_; ///< gradients for CP2K g·e DIIS metric
};

// ── Complex (Bloch / multi-k) OT ───────────────────────────────────────────

linalg::CplxMatrix ot_parameterize_k(const linalg::CplxMatrix& C0,
                                     const linalg::CplxMatrix& x,
                                     const linalg::CplxMatrix& S);

class OtPrecondSingleInverseK {
  public:
    OtPrecondSingleInverseK(const linalg::CplxMatrix& H,
                            const linalg::CplxMatrix& S,
                            const linalg::CplxMatrix& C);
    linalg::CplxMatrix apply(const linalg::CplxMatrix& g) const;

  private:
    linalg::CplxMatrix m_inv_;
};

struct OtFixedResultK {
    linalg::CplxMatrix C;
    double band_energy = 0.0;
    int iters = 0;
    bool converged = false;
};

OtFixedResultK ot_minimize_fixed_k(const linalg::CplxMatrix& H,
                                   const linalg::CplxMatrix& S,
                                   const linalg::CplxMatrix& T,
                                   const linalg::CplxMatrix& C0, double tol,
                                   int max_iter);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
