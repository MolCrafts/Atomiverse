/// @file cpu/component/plane_wave.h
/// @brief cpu::PlaneWave — plane-wave DFT energy.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Plane-wave DFT electronic structure.
class PlaneWave : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
