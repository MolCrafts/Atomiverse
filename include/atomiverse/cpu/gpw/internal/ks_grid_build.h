/// @file cpu/gpw/internal/ks_grid_build.h
/// @brief Shared fine-grid KS build: ρ → Poisson → XC → V_eff [+ nonlocal vdW].
///
/// Collocation of P → ρ is path-specific (Γ cbg / multigrid products / k χ^k);
/// integrating V_eff into H is also path-specific. What is *one* algorithm is
/// the fine-grid effective potential from a total (or dual-spin) density on
/// the SCF grid. Callers collocate, call these helpers, then integrate.
///
/// Used by Γ RKS/UKS and multi-k RKS/UKS energy builders (DIIS + OT).
#pragma once

#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// Forward: full definition in `gpw_solver.h` (underlying type `int`).
enum class XcKind : int;

namespace internal {

/// @brief RKS fine-grid KS result (one V_eff).
struct KsGridRks {
    std::vector<double> veff; ///< V_es + V_xc + V_loc^SR [+ V_nl] [Hartree].
    /// Poisson electrostatic potential of \(n=\rho-n_{\mathrm{ion}}\) [Hartree]
    /// (ionic Hellmann–Feynman forces need this separately from V_eff).
    std::vector<double> v_es;
    double e_xc = 0.0; ///< XC energy [Hartree].
    double e_hartree =
        0.0; ///< Poisson Hartree energy before −e_self [Hartree].
    double e_locgrid = 0.0; ///< ∫ ρ V_loc^SR [Hartree] (0 if no grid local PP).
    double e_nl_vdw = 0.0;  ///< Nonlocal E_c^nl when vdw flags set [Hartree].
};

/// @brief UKS fine-grid KS result (dual V_eff).
struct KsGridUks {
    std::vector<double> veff_a; ///< α channel [Hartree].
    std::vector<double> veff_b; ///< β channel [Hartree].
    std::vector<double>
        v_es; ///< Poisson V of total \(n=\rho-n_{\mathrm{ion}}\).
    double e_xc = 0.0;
    double e_hartree = 0.0;
    double e_locgrid = 0.0;
    double e_nl_vdw = 0.0;
};

/// @brief Closed-shell XC on the fine grid (PADE / PBE family / OptB88-vdW).
/// @param grid Density grid. @param rho Total density. @param xc Functional.
XcOnGrid evaluate_xc_rks(const UniformGrid& grid,
                         const std::vector<double>& rho, XcKind xc);

/// @brief Collinear spin XC on the fine grid.
XcOnGridSpin evaluate_xc_uks(const UniformGrid& grid,
                             const std::vector<double>& rho_a,
                             const std::vector<double>& rho_b, XcKind xc);

/// @brief ρ → n=ρ−n_ion → Poisson → XC → V_eff (+ optional nonlocal vdW).
///
/// @param grid         Fine density grid (Poisson / XC / vdW).
/// @param rho          Total electron density on the fine grid.
/// @param n_ion        Ionic Gaussian core density (same length as rho).
/// @param v_locsr      Local short-range PP on the grid (empty ⇒ skip).
/// @param xc           Exchange-correlation kind.
/// @param nonperiodic  Use Martyna–Tuckerman Poisson when true.
/// @param vdw_df       DRSLL nonlocal correlation.
/// @param vdw_df2      LMKLL nonlocal correlation.
/// @param rvv10        rVV10 nonlocal correlation.
/// @return V_eff and energy pieces (Hartree energy is raw Poisson; subtract
///         e_self at the call site as today).
KsGridRks ks_grid_build_rks(const UniformGrid& grid,
                            const std::vector<double>& rho,
                            const std::vector<double>& n_ion,
                            const std::vector<double>& v_locsr, XcKind xc,
                            bool nonperiodic, bool vdw_df, bool vdw_df2,
                            bool rvv10);

/// @brief Dual-spin variant: Poisson on ρ=ρα+ρβ; spin XC; V_nl on total ρ.
KsGridUks ks_grid_build_uks(const UniformGrid& grid,
                            const std::vector<double>& rho_a,
                            const std::vector<double>& rho_b,
                            const std::vector<double>& n_ion,
                            const std::vector<double>& v_locsr, XcKind xc,
                            bool nonperiodic, bool vdw_df, bool vdw_df2,
                            bool rvv10);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
