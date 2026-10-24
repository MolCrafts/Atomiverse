/// @file semiempirical/nddo_parameters.h
/// @brief Method-qualified access to the unique NDDO parameter data.
#pragma once
#include "atomiverse/semiempirical/am1_parameters.h"
#include "atomiverse/semiempirical/nddo_result.h"

namespace ATV::semiempirical {
/// Explicit method lookup; unknown methods and unsupported elements return null.
const AM1Parameter* nddo_parameter(SemiEmpiricalMethod method, int z) noexcept;
bool nddo_has_parameter(SemiEmpiricalMethod method, int z) noexcept;
/// Experimental atomic heat [kcal/mol]; requires a supported method/element row.
double nddo_atomic_heat_kcal_mol(SemiEmpiricalMethod method, int z);
/// Method label, or "unknown" for an invalid identity.
const char* nddo_method_name(SemiEmpiricalMethod method) noexcept;
} // namespace ATV::semiempirical
