/// @file cpu/component/bias.h
/// @brief cpu::Bias — collective-variable / external bias on energy and forces.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Collective-variable or external bias (energy + forces).
class Bias : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
