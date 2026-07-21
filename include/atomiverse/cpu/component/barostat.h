/// @file cpu/component/barostat.h
/// @brief cpu::Barostat — pressure / cell control.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Pressure / cell control. Declares system-stress requirement.
class Barostat : public Component {
  public:
    /// @brief NPT assembly needs a system-stress Observer in the Plan.
    bool requires_system_stress() const override { return true; }
};

} // namespace cpu
} // namespace ATV
