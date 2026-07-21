/// @file cpu/scf/uhf.h
/// @brief Unrestricted Hartree–Fock (UHF) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Unrestricted Hartree–Fock — the same mean-field, single-determinant
///        method as RHF but with independent spatial orbitals for the spin-up
///        (α) and spin-down (β) electrons, so open-shell / spin-polarized
///        systems (unequal α and β counts) are representable.
class UHF : public MeanFieldMethod<UHF> {
  public:
    int n_alpha = 0; ///< Alpha (spin-up) electrons (must be set before run).
    int n_beta = 0;  ///< Beta (spin-down) electrons (must be set before run).

    /// @brief Construct with the UHF defaults (unrestricted HF; Hcore initial
    ///        guess; 0.7 density damping; DIIS capped at 6 vectors and started
    ///        at iteration 5; energy tolerance 1e-8 Ha). Set `n_alpha` /
    ///        `n_beta` before `run`.
    UHF();
    /// @brief Construct with an explicit α / β electron count.
    /// @param n_alpha Spin-up electrons.
    /// @param n_beta  Spin-down electrons.
    explicit UHF(int n_alpha, int n_beta);
    std::string name() const { return "uhf"; }

  protected:
    /// @brief Occupation is taken directly from the `n_alpha` / `n_beta`
    ///        members; the electron-count argument is unused for the open-shell
    ///        case.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
