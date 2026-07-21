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
    XCType xc = XCType::PBE; ///< Exchange–correlation functional (default PBE).

    /// @brief Construct with the RKS defaults (restricted KS; PBE functional;
    ///        MINAO initial guess; no damping; MatchPySCF grid; energy
    ///        tolerance 1e-9 Ha).
    RKS();
    /// @brief Construct RKS with an explicit exchange–correlation functional.
    /// @param xc Functional to use (e.g. LSDA, PBE, B3LYP, PBE0).
    explicit RKS(XCType xc);
    std::string name() const { return "rks"; }

  protected:
    /// @brief Closed-shell occupation (n_alpha = n_beta = n_electrons / 2);
    ///        also latches the selected `xc` functional into the method state.
    void resolve_occupation(int n_electrons) override;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
