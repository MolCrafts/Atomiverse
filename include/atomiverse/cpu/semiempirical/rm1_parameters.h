/// @file cpu/semiempirical/rm1_parameters.h
/// @brief RM1 parameter accessors (same `AM1Parameter` shape as AM1).
///
/// Coverage v1: H, C, N, O, P, S. Unsupported Z returns null / throws on heat.
#pragma once

#include "atomiverse/cpu/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Return RM1 parameters for atomic number `z`, or null when absent.
const AM1Parameter* rm1_parameter(int z) noexcept;

/// @brief True when this build carries RM1 parameters for atomic number `z`.
bool rm1_has_parameter(int z) noexcept;

/// @brief Experimental gas-phase atomic heat of formation [kcal/mol] at 298 K.
/// @param z Atomic number (must have RM1 electronic parameters).
double rm1_atomic_heat_kcal_mol(int z);

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
