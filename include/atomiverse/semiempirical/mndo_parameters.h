/// @file semiempirical/mndo_parameters.h
/// @brief MNDO CHNO fitted parameters and independently derived multipoles.
#pragma once
#include "atomiverse/semiempirical/am1_parameters.h"
namespace ATV::semiempirical {
/// Return the MNDO row for H, C, N, O; null for every other atomic number.
const AM1Parameter* mndo_parameter(int z) noexcept;
} // namespace ATV::semiempirical
