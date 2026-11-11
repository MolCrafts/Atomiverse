/// @file cpu/planewave/internal/upf_reader.h
/// @brief UPF parser admitting scalar norm-conserving data in atomic units.
#pragma once

#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include <string>

namespace ATV::cpu::pw::internal {

/// Parse the supported XML UPF payload, converting V_local and D from Ry to
/// Ha once. Unknown/augmentation/core-corrected/spin-orbit models reject.
NcPotentialData read_upf_v1(const std::string& path);

} // namespace ATV::cpu::pw::internal
