/// @file cpu/vibration/phonon_result.h
/// @brief cpu::vib::PhononResult — Γ-point FD phonon spectrum of a supercell.
#pragma once

#include <string>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Finite-displacement phonon result at the supercell Γ point.
///
/// Frame purity: frequencies and force constants live here, not on Frame.
struct PhononResult {
    /// @brief Wavenumbers [cm⁻¹], length `3 * n_atoms`, ascending.
    ///        Imaginary modes (λ < 0) are negative.
    xt::xtensor<double, 1> frequencies;

    /// @brief Cartesian force-constant matrix [eV/Å²], shape `[3N, 3N]`.
    xt::xtensor<double, 2> hessian;

    /// @brief Normal-mode eigenvectors, shape `[3N, 3N]` (columns).
    xt::xtensor<double, 2> normal_modes;

    /// @brief Atoms in the supercell that was diagonalized.
    int n_atoms = 0;

    /// @brief Unit-cell atom count before expansion.
    int n_unit = 0;

    /// @brief Supercell factors along a, b, c.
    int na = 1;
    int nb = 1;
    int nc = 1;

    /// @brief Unit tag for `frequencies`.
    std::string frequency_units = "cm^-1";

    /// @brief Unit tag for `hessian`.
    std::string hessian_units = "eV/Angstrom^2";
};

} // namespace vib
} // namespace cpu
} // namespace ATV
