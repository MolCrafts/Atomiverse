/// @file cpu/potentials/pim_induced_dipole.h
/// @brief Classical PIM self-consistent induced dipoles (bulk) — CPU.
///
/// Polarizable Ion Model for bulk electrolytes / molten salts: each atom type
/// carries isotropic polarizability \(\alpha\) [Å³]. Induced dipoles obey
/// \(\boldsymbol{\mu}_i=\alpha_i\mathbf{E}_i^{\mathrm{tot}}\) with permanent
/// field from Frame charges and mutual induction via a Thole-damped (and
/// optionally Tang–Toennies-damped) dipole tensor. Polarization energy
/// \(E_{\mathrm{pol}}=-\frac12\sum_i\boldsymbol{\mu}_i\cdot\mathbf{E}_i^{\mathrm{perm}}\)
/// [eV] and Hellmann–Feynman forces (charge–dipole + dipole–dipole) are
/// written to Frame, together with the matching configurational virial
/// \(W=\tfrac12\sum_{i\neq j}\mathbf r_{ij}\otimes\mathbf f_{ij}\) into
/// `frame.meta.get(kStress)` (Voigt xx yy zz xy xz yz, eV, LAMMPS sign) so
/// barostats (NPT/NPH) see the polarization pressure. Permanent
/// charge–charge Coulomb is **not** included — use `PairCoulCut` /
/// long-range Coulomb separately.
///
/// MetalWalls capability map **E4**. Refs: Thole, *Chem. Phys.* **59**, 341
/// (1981); Applequist; MetalWalls JOSS **5**, 2373 (2020) bulk PIM form.
/// Units: e, Å, eV, eV/Å.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

/// @brief Bulk PIM self-consistent induced-dipole Potential.
///
/// Many-body (SCC) — inherits `Potential`, not `Pair`. O(N²) all-pairs
/// foothold (NeighborConsumer withdrawn); production neighbor-list path
/// deferred.
class PimInducedDipole : public Potential {
  public:
    /// @param cutoff Real-space multipole cutoff [Å] (must be > 0).
    /// @param alpha_by_type Isotropic polarizability [Å³] per atom type index
    ///        (`kAtom_type`). Types outside the vector → α = 0 (non-polarizable).
    /// @param thole_a Thole exponential damping factor (dimensionless; 0 = off;
    ///        typical 0.39).
    /// @param pol_tol Induced-dipole SCC threshold on max |\Deltaμ| [e·Å].
    /// @param pol_max_iter Maximum SCC iterations.
    PimInducedDipole(double cutoff, std::vector<double> alpha_by_type,
                     double thole_a = 0.39, double pol_tol = 1e-10,
                     int pol_max_iter = 100);

    /// Cutoff [Å] — public (no Config bag).
    double cutoff = 0.0;
    /// Thole a (dimensionless); 0 disables Thole damping.
    double thole_a = 0.39;
    /// Tang–Toennies range parameter b [Å⁻¹]; 0 disables TT damping.
    double tt_b = 0.0;
    /// SCC tolerance [e·Å].
    double pol_tol = 1e-10;
    /// SCC iteration budget.
    int pol_max_iter = 100;
    /// α [Å³] by type index.
    std::vector<double> alpha_by_type;
    /// @brief Initial SCC guess for the next `launch` (packed xyz, e·Å).
    ///
    /// Empty (default) ⇒ cold start from μ = 0. When its size equals
    /// 3·nlocal it seeds the SCC; any other size is ignored. Consumed and
    /// cleared by the next `launch` — set it again per step (the
    /// extended-Lagrangian owner `PimXlDipole` does exactly that).
    std::vector<double> mu_guess;

    std::string name() const override { return "pim_induced_dipole"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief Polarization energy of the last launch [eV] (≤ 0 for α ≥ 0).
    double last_pol_energy() const { return e_pol_; }
    /// @brief SCC iterations used on the last launch.
    int last_pol_iters() const { return last_pol_iters_; }
    /// @brief Whether induced-dipole SCC converged on the last launch.
    bool last_pol_converged() const { return last_pol_converged_; }
    /// @brief Induced dipoles of the last launch (packed xyz, e·Å).
    const std::vector<double>& last_mu() const { return last_mu_; }

  private:
    double alpha_of(int type) const;

    double e_pol_ = 0;
    int last_pol_iters_ = 0;
    bool last_pol_converged_ = false;
    std::vector<double> last_mu_;
};

} // namespace cpu
} // namespace ATV
