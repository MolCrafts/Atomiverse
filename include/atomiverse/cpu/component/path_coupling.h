/// @file cpu/component/path_coupling.h
/// @brief cpu::PathCoupling — path-band force coupling (NEB spring / climb).
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Path-band force coupling (NEB spring / climbing image, …).
class PathCoupling : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cpu
} // namespace ATV
