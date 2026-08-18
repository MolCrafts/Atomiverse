/// @file cpu/component/gpw.h
/// @brief cpu::GPW — Gaussian-and-plane-waves DFT energy.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Gaussian-and-plane-waves DFT electronic structure.
class GPW : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
