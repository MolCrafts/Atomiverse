/// @file cpu/component/thermostat.h
/// @brief cpu::Thermostat — temperature control.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Temperature control (Nose–Hoover, Langevin, CSVR, …).
class Thermostat : public Component {};

} // namespace cpu
} // namespace ATV
