/// @file core/longrange_math.h
/// @brief Scalar Ewald-family energy laws shared by the CPU and CUDA
///        reciprocal solvers (each backend keeps its own parallel kernels).
#pragma once

#include <cmath>

#include "atomiverse/core/units.h"

namespace ATV::longrange {

/// @brief Ewald self-energy prefactor; multiply by Σ q_i².
/// @param alpha Ewald splitting parameter [1/length].
/// @param qqrd2e The run's Coulomb constant (LAMMPS `force->qqrd2e`).
/// @return −qqrd2e α / √π [energy/charge²].
inline double self_energy_prefactor(double alpha, double qqrd2e) {
    return -qqrd2e * alpha / std::sqrt(ATV::units::pi);
}

/// @brief Energy of the uniform neutralizing background of a charged cell.
///
/// E = −k_e π Q² / (2 V α²). It scales as 1/V, so its virial
/// −∂E/∂ε_aa equals E on each diagonal component and vanishes off-diagonal.
/// @param net_charge Total charge Q of the cell [e].
/// @param volume Cell volume [Angstrom³].
/// @param alpha Ewald splitting parameter [1/length].
/// @param qqrd2e The run's Coulomb constant (LAMMPS `force->qqrd2e`).
/// @return Background energy [energy].
inline double background_energy(double net_charge, double volume, double alpha,
                                double qqrd2e) {
    return -qqrd2e * ATV::units::pi * net_charge * net_charge
           / (2.0 * volume * alpha * alpha);
}

} // namespace ATV::longrange
