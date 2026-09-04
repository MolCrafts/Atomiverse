/// @file cpu/semiempirical/basis_data/dftb_3ob_hubbard_derivs.h
/// @brief Published 3ob-3-1 Hubbard derivatives \(U^{\mathrm d}=dU/dq\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
///
/// In-tree scientific parameter data — **nothing is parsed at build, test, or
/// run time**. Consumed only when the caller assigns
/// `DFTB3::hubbard_du_ha = basis_data::k3obHubbardDerivsHa` (never applied
/// silently).
///
/// Provenance:
///   * Source artifact: DFTB.org **3ob-3-1** parameter-release README,
///     "List of all atomic Hubbard derivatives (atomic units)",
///     https://github.com/dftbparams/3ob (not a journal table number).
///   * License / attribution: CC-BY-SA-4.0, (c) 2017 M. Elstner (KIT).
///   * Method paper (DFTB3): M. Gaus, Q. Cui, M. Elstner, *J. Chem. Theory
///     Comput.* **7**, 931–948 (2011), DOI 10.1021/ct100684s.
///   * Parameter-set paper (3OB): M. Gaus, A. Goez, M. Elstner, *J. Chem.
///     Theory Comput.* **9**, 338–354 (2013), DOI 10.1021/ct300849w.
///     These are distinct papers; the author lists must not be merged.
///   * Units: Hartree (atomic units), matching DFTB+ `HubbardDerivs`.
///   * ζ = 4.00 (DFTB+ `DampXHExponent`, the γ^h X–H damping exponent) is
///     **not implemented** (`dftb-gamma-h-damping`).
///   * No published mio-1-1 \(U^{\mathrm d}\) set exists; 3ob supersedes MIO
///     and ships its own derivatives.
///
/// Unverified: DOI 10.1021/ct300849w is paywalled and was not opened; the
/// numbers below are taken from the parameter release (the values DFTB+ users
/// actually feed to `HubbardDerivs`), not from a paper table. This header
/// does not claim a paper table number.
#pragma once

#include <array>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief 3ob-3-1 Hubbard derivatives \(U^{\mathrm d}\)
///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV),
///        index = atomic number \(Z\).
///
/// Only \(Z = 1, 6, 7, 8\) are nonzero (H −0.1857, C −0.1492, N −0.1535,
/// O −0.1575). All other entries are 0.0 = unset. Assign explicitly onto
/// `DFTB3::hubbard_du_ha`; never applied by default. These are **not** eV
/// — Hubbard \(U\) in the SK tables is eV; \(U^{\mathrm d}\) matches DFTB+
/// `HubbardDerivs` (Hartree).
inline constexpr std::array<double, 96> k3obHubbardDerivsHa = [] {
    std::array<double, 96> u{};
    u[1] = -0.1857;
    u[6] = -0.1492;
    u[7] = -0.1535;
    u[8] = -0.1575;
    return u;
}();

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
