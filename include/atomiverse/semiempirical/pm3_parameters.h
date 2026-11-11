/// @file semiempirical/pm3_parameters.h
/// @brief PM3 CHNO fitted parameters and independently derived multipoles.
#pragma once
#include "atomiverse/semiempirical/am1_parameters.h"
namespace ATV::semiempirical {
/// Return the PM3 row for H, C, N, O; null for every other atomic number.
const AM1Parameter* pm3_parameter(int z) noexcept;
} // namespace ATV::semiempirical
