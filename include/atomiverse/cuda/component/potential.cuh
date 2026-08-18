/// @file cuda/component/potential.cuh
/// @brief cuda::Potential — classical force term (pair, bond, coul, kspace, …).
///        Kind tag mirroring cpu::Potential. Energy/force come from the
///        kind-default probes, not a preset type list.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Classical potential energy / force term.
class Potential : public Component {
  public:
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
};

} // namespace cuda
} // namespace ATV
