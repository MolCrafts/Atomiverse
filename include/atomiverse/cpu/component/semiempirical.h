/// @file cpu/component/semiempirical.h
/// @brief cpu::Semiempirical — NDDO / AM1-style energy.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Semiempirical electronic structure (AM1, …).
class Semiempirical : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
