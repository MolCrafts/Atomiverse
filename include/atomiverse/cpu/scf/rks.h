/// @file cpu/scf/rks.h
/// @brief Restricted Kohn–Sham (RKS) concrete mean-field method.
#pragma once

#include <string>

#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Restricted Kohn–Sham — Kohn–Sham density-functional theory (a
///        non-interacting orbital reference plus an exchange–correlation
///        functional that supplies the many-body effects) in its closed-shell
///        form, with spin-up (α) and spin-down (β) electrons sharing one set of
///        spatial orbitals. Defaults to the PBE functional on the MatchPySCF
///        integration grid.
class RKS : public MeanFieldMethod<RKS> {
  public:
    const XCType xc; ///< Exchange–correlation functional.

    /// @brief Construct RKS (MINAO guess, no damping, MatchPySCF grid,
    ///        energy tolerance 1e-9 Ha).
    /// @param basis Basis set.
    /// @param xc Functional (e.g. LSDA, PBE, BLYP, B3LYP, PBE0, CAMB3LYP,
    ///           B2PLYP). For double hybrids, `MP2{rks}` with
    ///           `os_scale = ss_scale = hybrid_pt2_fraction(xc)` supplies the
    ///           PT2 term (KS SCF alone is not the full energy).
    /// @param charge Molecular charge [e].
    RKS(BasisSet basis, XCType xc, int charge = 0);
    std::string name() const { return "rks"; }

  protected:
    /// @brief Closed-shell occupation (n_alpha = n_beta = n_electrons / 2);
    ///        also latches the selected `xc` functional into the method state.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
