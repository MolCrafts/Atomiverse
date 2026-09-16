/// @file cpu/semiempirical/basis_data/dftb_spin_constants.h
/// @brief Published collinear DFTB spin constants \(W^A_{ll'}\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
///
/// In-tree scientific parameter data — **nothing is parsed at build, test, or
/// run time**. Consumed only when the caller assigns
/// `DFTB2::spin_w_ha = basis_data::kDftbSpinConstantsPbeHa` (or the LDA
/// table); never applied silently.
///
/// Provenance:
///   * Source artifact: DFTB+ manual appendix `app:spinconst`
///     (`doc/dftb+/manual/spin_constants.tex`), citing C. Köhler, PhD thesis,
///     University of Paderborn (2004).
///   * Peer-reviewed statement: Köhler, Frauenheim, Hourahine, Seifert,
///     Sternberg, *J. Phys. Chem. A* **111**, 5622–5629 (2007),
///     DOI 10.1021/jp068802p.
///   * Units: Hartree (atomic units), matching DFTB+ `SpinConstants`.
///     Input order is DFTB+ (second shell index varies fastest:
///     \(w_{ss}, w_{sp}, w_{ps}, w_{pp}\)).
///
/// Unverified: DOI 10.1021/jp068802p was not opened. The defining equations
/// are taken from Moradi et al., arXiv:2405.05761 eqs. (2)–(3) and the DFTB+
/// input convention, not from a transcribed paper table.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief One element's shell-resolved collinear spin constants \(W^A_{ll'}\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
struct DftbSpinConstant {
    int z;       ///< atomic number \(Z\) (dimensionless)
    int n_shell; ///< valence shells (dimensionless): 1 (s only) or 2 (s+p)
    /// @brief Shell-resolved \(W_{ll'}\)
    ///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
    ///
    /// Indexed \(\{s,p\}\times\{s,p\}\); second index \(l'\) varies fastest,
    /// matching DFTB+ `SpinConstants` input order
    /// \(w_{ss}, w_{sp}, w_{ps}, w_{pp}\). Converted Ha→eV only when
    /// `DFTB2` collapses the table onto a per-atom scalar.
    double w_ha[2][2];
};

/// @brief PBE-functional collinear spin constants \(W^A_{ll'}\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
///
/// Mio / 3ob SK files are PBE-parameterized; DFTB+ requires constants
/// computed with the same functional as the SK set. Assign explicitly onto
/// `DFTB2::spin_w_ha`.
inline const std::vector<DftbSpinConstant> kDftbSpinConstantsPbeHa = {
    {1, 1, {{-0.072, 0.0}, {0.0, 0.0}}},
    {6, 2, {{-0.031, -0.025}, {-0.025, -0.023}}},
    {7, 2, {{-0.033, -0.027}, {-0.027, -0.026}}},
    {8, 2, {{-0.035, -0.030}, {-0.030, -0.028}}},
};

/// @brief LDA-functional collinear spin constants \(W^A_{ll'}\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
///
/// Provided for completeness; not a default. The source notes that only the
/// first two decimal places are numerically meaningful.
inline const std::vector<DftbSpinConstant> kDftbSpinConstantsLdaHa = {
    {1, 1, {{-0.064, 0.0}, {0.0, 0.0}}},
    {6, 2, {{-0.028, -0.024}, {-0.024, -0.022}}},
    {7, 2, {{-0.030, -0.026}, {-0.026, -0.025}}},
    {8, 2, {{-0.032, -0.028}, {-0.028, -0.027}}},
};

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
