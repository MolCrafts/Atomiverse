/// @file electronic/smearing.h
/// @brief Metallic occupation smearing (Fermi–Dirac / Gaussian / Methfessel–Paxton).
///
/// Spin-unpolarized convention: spinless f is doubled to a band occupation.
/// FD/Gaussian f lies in [0,1]; MP is a signed quasi-occupation and may
/// overshoot these bounds. Width `degauss` is in the same energy unit as the
/// eigenvalues (Hartree inside the PW SCF).
#pragma once

#include "atomiverse/core/precision.h"

#include <utility>
#include <vector>

namespace ATV {
namespace electronic {

/// @brief Occupation smearing shared by plane-wave and GPW.
///
/// `none` is fixed integer filling. The three metallic laws are the same
/// function for both methods. GPW does not implement `gaussian` and throws
/// if a run asks for it.
enum class Smearing : int {
    none = 0,              ///< Fixed occupations.
    fermi_dirac = 1,       ///< Fermi–Dirac.
    methfessel_paxton = 2, ///< Methfessel–Paxton of order `mp_order`.
    gaussian = 3,          ///< Gaussian (cold smearing is not this).
};

/// @brief Spinless (quasi-)occupation f(x), \(x=(\varepsilon-\mu)/\sigma\).
/// @param x        Reduced energy (dimensionless).
/// @param kind     Smearing kind.
/// @param mp_order Methfessel–Paxton order \(N\ge 1\) (ignored otherwise).
/// @return FD/Gaussian occupancy in [0,1], or an unclamped MP quasi-occupancy.
AccReal smear_f_spinless(AccReal x, Smearing kind, int mp_order = 1);

/// @brief Closed-shell band (quasi-)occupation: \(2\,f(x)\).
AccReal smear_f_band(AccReal eig, AccReal mu, AccReal degauss, Smearing kind,
                     int mp_order = 1);

/// @brief Collinear spin-orbital (quasi-)occupation: \(f(x)\) (no ×2).
AccReal smear_f_spinor(AccReal eig, AccReal mu, AccReal degauss, Smearing kind,
                       int mp_order = 1);

/// @brief Entropy weight \(s(x)\) such that \(TS = \sigma\sum w\,s\) per band
///        with spin degeneracy already folded into \(s\) (band form).
///
/// For Fermi–Dirac: \(s = -2[f\ln f+(1-f)\ln(1-f)]\) with spinless \(f\).
/// For Gaussian/MP: QE-style \(w1gauss\) contribution \(\times 2\).
AccReal smear_entropy_band(AccReal eig, AccReal mu, AccReal degauss,
                           Smearing kind, int mp_order = 1);

/// @brief Spin-orbital entropy weight (half the closed-shell band form).
///
/// For Fermi–Dirac: \(s = -[f\ln f+(1-f)\ln(1-f)]\).
AccReal smear_entropy_spinor(AccReal eig, AccReal mu, AccReal degauss,
                             Smearing kind, int mp_order = 1);

/// @brief Find \(\mu\) so \(\sum_i w_i f(\varepsilon_i,\mu)=n_{\mathrm{elec}}\).
///
/// @param evals   Eigenvalues (same unit as degauss).
/// @param weights Per-eigenvalue weight (k-point weight, one entry per eval).
/// @param n_elec  Target electron count (dimensionless).
/// @param degauss Smearing width (>0).
/// @param kind    Smearing kind.
/// @param mp_order MP order.
/// @param spin_degen 2 for closed-shell band form, 1 for collinear spin-orbitals.
/// @return Fermi level \(\mu\).
AccReal find_fermi_level(const std::vector<AccReal>& evals,
                         const std::vector<AccReal>& weights, AccReal n_elec,
                         AccReal degauss, Smearing kind, int mp_order = 1,
                         AccReal spin_degen = AccReal(2));

/// @brief Occupations, electron sum, and entropy sum for a spectrum at fixed \(\mu\).
/// @return `{occupations, n_occ, entropy_sum}` where
///         \(E_{\mathrm{free}} = E_{\mathrm{KS}} - \sigma\cdot\mathrm{entropy\_sum}\).
///         FD/Gaussian occupations lie in [0,spin_degen]. MP occupations
///         may be negative or exceed spin_degen; its entropy may be signed.
struct SmearFill {
    std::vector<AccReal> occ; ///< Band / spin-orbital occupations.
    AccReal n_occ = 0;        ///< \(\sum w f\).
    AccReal entropy = 0;      ///< Dimensionless \(\sum w s\).
};

SmearFill fill_occupations(const std::vector<AccReal>& evals,
                           const std::vector<AccReal>& weights, AccReal mu,
                           AccReal degauss, Smearing kind, int mp_order = 1,
                           AccReal spin_degen = AccReal(2));

} // namespace electronic
} // namespace ATV
