/// @file cpu/scf/rhf.h
/// @brief Restricted Hartree–Fock (RHF) concrete mean-field method.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted Hartree–Fock — a mean-field electronic-structure method
///        (a single Slater determinant in which every electron moves in the
///        averaged field of all the others) whose spin-up (α) and spin-down (β)
///        electrons share one common set of spatial orbitals: the closed-shell,
///        fully paired case.
class RHF : public MeanFieldMethod<RHF> {
  public:
    /// @brief Construct RHF (MINAO guess, no density damping).
    /// @param basis Basis set (e.g. `BasisSet::STO3G`).
    /// @param charge Molecular charge [e] (electrons = ΣZ − charge).
    /// @param tol Energy / residual tolerance [Ha] (see base `tol`).
    /// @param max_iter Maximum SCF iterations [dimensionless].
    explicit RHF(BasisSet basis, int charge = 0, double tol = 1e-9,
                 int max_iter = 100);

    std::string name() const { return "rhf"; }

    /// @brief Γ-point periodic RHF. Default false keeps the molecular
    ///        engine bit-identical. When true, Frame.box must be
    ///        orthorhombic; J, K, S, T, V and E_nn share one truncated
    ///        lattice-image window (no Ewald). A cell larger than the
    ///        Gaussian pair reach has only T = 0 and recovers molecular RHF.
    bool periodic = false;

  protected:
    /// @brief Closed-shell occupation: n_alpha = n_beta = n_electrons / 2
    ///        (every occupied spatial orbital holds one α and one β electron).
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
