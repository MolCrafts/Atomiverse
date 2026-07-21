/// @file cpu/observer/esp_charges.h
/// @brief ESP-fitted atomic charges: `MKCharges` (Merz-Singh-Kollman nested
///        vdW shells) and `ChelpgCharges` (CHELPG cubic grid).
///
/// Both classes sample the molecular electrostatic potential
///   V(r) = Σ_A Z_A/|r−R_A| − Σ_{μν} P_μν ⟨μ|1/|r−r′||ν⟩
/// on a deterministic grid and solve the Lagrange-constrained least squares
///   min Σ_k (V_k − Σ_A q_A/|r_k−R_A|)²   s.t.   Σ_A q_A = q_tot
/// (KKT linear system; SVD minimum-norm fallback when the system is
/// degenerate — buried / coincident atoms must not diverge). The electron
/// term reuses `native::nuclear_block` with a unit point charge placed at
/// each grid point; spherical-convention packs sandwich with the handle's
/// `c2s`.
///
/// GRID CONVENTIONS (deterministic; Atomiverse's own, replicated verbatim
/// by the external pinning script `tools/pin_charges_oracle.py` — grids are
/// defined in Å in both implementations):
///
/// - **MK** (Besler, Merz, Kollman, J. Comput. Chem. 11, 431 (1990),
///   DOI 10.1002/jcc.540110404): per atom A and shell scale s in
///   `shell_scales`, a golden-angle Fibonacci sphere of radius
///   r = s·r_Bondi(A) with N = ceil(density_per_ang2 · 4πr²) points:
///   for k = 0..N−1:  z = 1 − 2(k+0.5)/N,  φ = k·π(3−√5),
///   ρ = sqrt(1−z²),  p = R_A + r·(ρcosφ, ρsinφ, z).
///   A point is discarded when |p − R_B| < s·r_Bondi(B) for any B ≠ A.
///
/// - **CHELPG** (Breneman, Wiberg, J. Comput. Chem. 11, 361 (1990),
///   DOI 10.1002/jcc.540110311): cubic grid of spacing `spacing_ang` over
///   the atom bounding box padded by `band_ang` per axis (points at
///   corner + i·spacing while ≤ max + band + 1e-9 Å); a point p is kept
///   iff d_A ≥ r_Bondi(A) for EVERY atom (outside every vdW sphere) and
///   d_A ≤ r_Bondi(A) + band for SOME atom (within the band of at least
///   one).
///
/// Bondi vdW radii (Bondi 1964) are a static table in the implementation;
/// an unsupported element throws (no fallback). Note the fitted charges of
/// symmetry-equivalent atoms agree only to the grid-discretization level
/// (~1e-3 e at the default densities): neither deterministic point set is
/// exactly invariant under the molecular point group.
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {

/// @brief Merz-Singh-Kollman ESP charges on nested scaled-Bondi shells.
class MKCharges {
  public:
    /// @brief vdW shell scale factors (dimensionless); the literature MK
    ///        set 1.4/1.6/1.8/2.0 by default.
    std::vector<double> shell_scales{1.4, 1.6, 1.8, 2.0};

    /// @brief Target surface point density [points/Å²] per shell.
    double density_per_ang2 = 1.0;

    /// @brief Generate the MK sampling grid for a geometry.
    /// @param frame Geometry (positions [Å], atomic numbers in `type`).
    /// @return Grid points [Å], in the documented deterministic order.
    /// @throws std::invalid_argument for an element without a Bondi radius.
    std::vector<std::array<double, 3>> grid(const Frame& frame) const;

    /// @brief Fit ESP charges from a converged mean-field wavefunction.
    /// @param outputs A method's `result_handle()` (determinant + shells +
    ///        c2s), read after the method ran.
    /// @param frame   Geometry (positions [Å], atomic numbers in `type`).
    /// @return Per-atom fitted charges [e], Frame atom order; the total
    ///         charge constraint Σq = Z_tot − N_el holds by construction.
    /// @throws std::invalid_argument on a null / not-ready handle or an
    ///         element without a Bondi radius.
    std::vector<double> fit(const scf::MeanFieldOutputs& outputs,
                            const Frame& frame) const;

    /// @brief Constrained LSQ fit from an externally supplied potential —
    ///        the shared kernel behind `fit()`, public so the fit path is
    ///        testable without a determinant.
    /// @param V     Potential samples [Hartree/e], one per grid point.
    /// @param grid_ang Grid points [Å] (e.g. from `grid()`).
    /// @param frame Geometry — supplies the charge sites R_A.
    /// @param q_tot Total-charge constraint Σq [e].
    /// @return Per-atom fitted charges [e], Frame atom order.
    /// @throws std::invalid_argument when V and grid sizes differ.
    std::vector<double>
    fit_potential(const std::vector<double>& V,
                  const std::vector<std::array<double, 3>>& grid_ang,
                  const Frame& frame, double q_tot) const;
};

/// @brief CHELPG ESP charges on a cubic grid over the padded bounding box.
class ChelpgCharges {
  public:
    /// @brief Cubic grid spacing [Å]; 0.3 Å per Breneman-Wiberg.
    double spacing_ang = 0.3;

    /// @brief Band width beyond the vdW surface [Å]; 2.8 Å per
    ///        Breneman-Wiberg (also the bounding-box padding).
    double band_ang = 2.8;

    /// @brief Generate the CHELPG sampling grid for a geometry.
    /// @param frame Geometry (positions [Å], atomic numbers in `type`).
    /// @return Grid points [Å], in the documented deterministic order.
    /// @throws std::invalid_argument for an element without a Bondi radius.
    std::vector<std::array<double, 3>> grid(const Frame& frame) const;

    /// @brief Fit ESP charges from a converged mean-field wavefunction.
    /// @param outputs A method's `result_handle()`.
    /// @param frame   Geometry (positions [Å], atomic numbers in `type`).
    /// @return Per-atom fitted charges [e], Frame atom order; Σq equals the
    ///         molecular net charge by construction.
    /// @throws std::invalid_argument on a null / not-ready handle or an
    ///         element without a Bondi radius.
    std::vector<double> fit(const scf::MeanFieldOutputs& outputs,
                            const Frame& frame) const;

    /// @brief Constrained LSQ fit from an externally supplied potential
    ///        (same kernel and contract as `MKCharges::fit_potential`).
    /// @param V     Potential samples [Hartree/e], one per grid point.
    /// @param grid_ang Grid points [Å] (e.g. from `grid()`).
    /// @param frame Geometry — supplies the charge sites R_A.
    /// @param q_tot Total-charge constraint Σq [e].
    /// @return Per-atom fitted charges [e], Frame atom order.
    /// @throws std::invalid_argument when V and grid sizes differ.
    std::vector<double>
    fit_potential(const std::vector<double>& V,
                  const std::vector<std::array<double, 3>>& grid_ang,
                  const Frame& frame, double q_tot) const;
};

} // namespace cpu
} // namespace ATV
