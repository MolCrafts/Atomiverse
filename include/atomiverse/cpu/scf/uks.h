/// @file cpu/scf/uks.h
/// @brief Unrestricted Kohn–Sham (UKS) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Unrestricted Kohn–Sham — Kohn–Sham density-functional theory with
///        independent spatial orbitals for the spin-up (α) and spin-down (β)
///        electrons, for open-shell / spin-polarized systems.
class UKS : public MeanFieldMethod<UKS> {
  public:
    const int n_alpha; ///< Alpha (spin-up) electrons.
    const int n_beta;  ///< Beta (spin-down) electrons.
    const XCType xc;   ///< Exchange–correlation functional.

    /// @brief Construct UKS (Hcore guess, 0.5 damping, DIIS 6 vectors,
    ///        energy tolerance 1e-8 Ha).
    /// @param basis Basis set.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons.
    /// @param xc      Exchange–correlation functional.
    /// @param charge  Molecular charge [e].
    UKS(BasisSet basis, int n_alpha, int n_beta, XCType xc = XCType::LSDA,
        int charge = 0);
    std::string name() const { return "uks"; }

  protected:
    /// @brief Occupation is taken directly from the `n_alpha` / `n_beta`
    ///        members; also latches the selected `xc` functional. The
    ///        electron-count argument is unused for the open-shell case.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
