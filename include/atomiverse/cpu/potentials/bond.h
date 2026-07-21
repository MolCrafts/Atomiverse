/// @file cpu/potentials/bond.h
/// @brief Bond Potential base (BondHarmonic → BondPotential → Component).
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

/// @brief Bonded 2-body Potential.
class BondPotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Bond = BondPotential;

} // namespace cpu
} // namespace ATV
