/// @file cpu/component/observer.h
/// @brief cpu::Observer — read-only derived quantities on the Frame.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Derived quantities; does not mutate x / v / f / box / topology.
class Observer : public Component {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred;
    }
};

} // namespace cpu
} // namespace ATV
