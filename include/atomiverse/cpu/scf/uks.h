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
    int n_alpha = 0; ///< Alpha (spin-up) electrons (set before run).
    int n_beta = 0;  ///< Beta (spin-down) electrons (set before run).
    XCType xc =
        XCType::LSDA; ///< Exchange–correlation functional (default LSDA).

    /// @brief Construct with the UKS defaults (unrestricted KS; LSDA functional;
    ///        Hcore initial guess; 0.5 density damping; DIIS capped at 6
    ///        vectors; energy tolerance 1e-8 Ha). Set `n_alpha` / `n_beta`
    ///        before `run`.
    UKS();
    /// @brief Construct UKS with explicit α / β counts and functional.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons.
    /// @param xc      Exchange–correlation functional (default LSDA).
    UKS(int n_alpha, int n_beta, XCType xc = XCType::LSDA);
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
