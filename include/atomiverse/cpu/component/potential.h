/// @file cpu/component/potential.h
/// @brief cpu::Potential — classical force term (pair, bond, coul, kspace, …).
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Classical potential energy / force term.
class Potential : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
