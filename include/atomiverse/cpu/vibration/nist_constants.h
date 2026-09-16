/// @file cpu/vibration/nist_constants.h
/// @brief CODATA-2014 fundamental constants for vibrational analysis.
///
/// Mirrors `pyscf/data/nist.py` verbatim (CODATA-2014). The reference-port
/// fidelity rule applies: the raw constants below are the exact literals
/// PySCF uses, and the derived quantities (`HARTREE2J`, `au2hz`, the
/// `freq_au -> cm^-1` factor) are composed from them at compile time —
/// never a rounded magic literal. Any rounding drift in a constant shifts
/// computed frequencies by ~0.1 cm^-1, so the verbatim mirror is mandatory.
///
/// Reference: Mohr, Newell, Taylor, *Rev. Mod. Phys.* 88 (2016) 035009,
/// DOI 10.1103/RevModPhys.88.035009.
#pragma once

#include <cmath>

namespace ATV {
namespace cpu {
namespace vib {
namespace nist {

/// @brief Pi to double precision (matches PySCF's literal in nist.py HBAR).
inline constexpr double PI = 3.141592653589793;

/// @brief Planck constant [J*s] — CODATA-2014.
inline constexpr double PLANCK = 6.626070040e-34;

/// @brief Reduced Planck constant `hbar = h / (2*pi)` [J*s].
inline constexpr double HBAR = PLANCK / (2.0 * PI);

/// @brief Electron rest mass [kg] — CODATA-2014.
inline constexpr double E_MASS = 9.10938356e-31;

/// @brief Avogadro constant [1/mol] — CODATA-2014.
inline constexpr double AVOGADRO = 6.022140857e23;

/// @brief Atomic mass unit `1e-3 / N_A` [kg] (1 amu in kg).
inline constexpr double ATOMIC_MASS = 1.0e-3 / AVOGADRO;

/// @brief Bohr radius [Angstrom] — CODATA-2014.
inline constexpr double BOHR = 0.52917721092;

/// @brief Bohr radius [m].
inline constexpr double BOHR_SI = BOHR * 1.0e-10;

/// @brief Hartree energy in joules, `HBAR^2 / (E_MASS * BOHR_SI^2)` [J].
///
/// Composed exactly as PySCF's `nist.HARTREE2J`, not the tabulated
/// `4.359744650e-18` literal, so derived quantities stay self-consistent.
inline constexpr double HARTREE2J = HBAR * HBAR / (E_MASS * BOHR_SI * BOHR_SI);

/// @brief Speed of light [m/s] — exact (CODATA-2014).
inline constexpr double LIGHT_SPEED_SI = 299792458.0;

/// @brief Elementary charge [C] — CODATA-2014 (`nist.E_CHARGE`).
inline constexpr double E_CHARGE = 1.6021766208e-19;

/// @brief Hartree energy expressed in electron-volts, `HARTREE2J / E_CHARGE`.
///
/// Composed from the nist-mirrored constants so the eV<->Hartree
/// conversion in `MassWeighting` is self-consistent with `au2hz`.
inline constexpr double HARTREE2EV = HARTREE2J / E_CHARGE;

/// @brief One eV/Angstrom^2 expressed in Hartree/Bohr^2.
///
/// `H[Hartree/Bohr^2] = H[eV/Angstrom^2] * (eV in Hartree) * (Bohr in
/// Angstrom)^2`. Link 1 stores the numerical Hessian in eV/Angstrom^2;
/// `MassWeighting` multiplies by this factor before mass-weighting.
inline constexpr double EV_PER_ANG2_TO_HARTREE_PER_BOHR2 =
    (1.0 / HARTREE2EV) * (BOHR * BOHR);

/// @brief Boltzmann constant [J/K] — CODATA-2014.
inline constexpr double BOLTZMANN = 1.38064852e-23;

/// @brief Atomic-unit angular-frequency-to-Hz factor.
///
/// `au2hz = sqrt(HARTREE2J / (ATOMIC_MASS * BOHR_SI^2)) / (2*pi)` — PySCF's
/// exact two-factor form in `harmonic_analysis`. `std::sqrt` is not
/// `constexpr` before C++26, so this is an `inline` accessor function
/// rather than a `constexpr` variable.
/// @return The au-frequency-to-Hz conversion factor.
inline double au2hz() {
    return std::sqrt(HARTREE2J / (ATOMIC_MASS * BOHR_SI * BOHR_SI))
           / (2.0 * PI);
}

/// @brief Conversion factor: a unit `freq_au` -> wavenumber [cm^-1].
///
/// `freq_wavenumber = freq_au * au2hz / LIGHT_SPEED_SI * 1e-2` — the
/// scalar factor `au2hz / LIGHT_SPEED_SI * 1e-2` as PySCF
/// `harmonic_analysis` computes it. It is always derived here from the
/// CODATA-2014 constants above, never written as a magic literal.
/// @return The freq_au-to-wavenumber conversion factor [cm^-1].
inline double au2wavenumber() { return au2hz() / LIGHT_SPEED_SI * 1.0e-2; }

} // namespace nist
} // namespace vib
} // namespace cpu
} // namespace ATV
