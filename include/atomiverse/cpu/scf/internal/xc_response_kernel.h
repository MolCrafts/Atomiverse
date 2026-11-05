/// @file cpu/scf/internal/xc_response_kernel.h
/// @brief **PRIVATE** adiabatic semi-local \(f^{\mathrm{xc}}\) response kernel
///        of a closed-shell KS reference, spin-adapted per excitation channel.
///
/// Closed-shell Casida blocks (Casida 1995; Dreuw & Head-Gordon, Chem. Rev.
/// 105, 4009 (2005), DOI 10.1021/cr0505627), composite index
/// \(p = a\,n_{\mathrm{occ}} + i\):
///
///   singlet: \(K_{ia,jb} = (ia|f_{\alpha\alpha}+f_{\alpha\beta}|jb)\)
///   triplet: \(K_{ia,jb} = (ia|f_{\alpha\alpha}-f_{\alpha\beta}|jb)\)
///
/// K enters both A and B, so (A+B) gains 2K and (A−B) is unchanged. The
/// kernel is the directional derivative of \(V^{\mathrm{xc}}_\alpha\) along
/// the transition density (\(\delta P_\beta = \pm\delta P_\alpha\)), which
/// is what PySCF `tdscf.rks` contracts; every built-in semi-local functional
/// (LDA and GGA, including the semi-local part of a global hybrid) uses the
/// same path. The AO-on-grid basis is built once per kernel.
#pragma once

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/xc_grid_engine.h"
#include "atomiverse/cpu/xc/xc_functional.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Matrix-free \(f^{\mathrm{xc}}\) kernel K of one excitation channel.
class XcResponseKernel {
  public:
    /// @brief Bind a converged closed-shell determinant and its KS grid.
    /// @param det  Restricted determinant with integer occupations.
    /// @param pack Integral view (Cartesian shells; optional c2s).
    /// @param xc   Built-in functional whose semi-local part is linearized.
    /// @param grid Molecular quadrature of the reference (copied).
    /// @param ch   Singlet or triplet spin adaptation.
    /// @throws std::invalid_argument on an unrestricted determinant, an
    ///         empty shell list or a custom functional (no fallback).
    XcResponseKernel(const SlaterDeterminant& det, const IntegralView& pack,
                     xc::XCType xc, const MolecularGrid& grid, Channel ch);

    /// @brief Rotation-space dimension n_occ·n_virt.
    /// @return n_ov (dimensionless count).
    int n_ov() const { return no_ * nv_; }

    /// @brief K·v for one trial vector (composite index a·n_occ + i).
    /// @param v Trial amplitudes, length n_ov (dimensionless).
    /// @return K·v [Hartree].
    /// @throws std::invalid_argument when v.size() != n_ov().
    RealVector apply(const RealVector& v) const;

    /// @brief Explicit symmetric K (n_ov applications) for dense solves.
    /// @return n_ov × n_ov matrix [Hartree].
    RealMatrix dense() const;

  private:
    xc::BuiltinXc functional_;
    MolecularGrid grid_;
    XCGridBasis basis_;
    RealMatrix Co_; ///< Occupied coefficients in the Cartesian engine basis.
    RealMatrix Cv_; ///< Virtual coefficients in the Cartesian engine basis.
    RealMatrix P0_; ///< Per-spin reference density (engine basis).
    Channel ch_;
    int no_ = 0;
    int nv_ = 0;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
