/// @file cuda/component/observer.cuh
/// @brief cuda::Observer — read-only derived quantities. Kind tag
///        mirroring cpu::Observer.
#pragma once

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Derived quantities; does not mutate x / v / f / box / topology.
class Observer : public Component {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred;
    }
};

} // namespace cuda
} // namespace ATV
