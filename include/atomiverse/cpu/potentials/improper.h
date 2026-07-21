/// @file cpu/potentials/improper.h
/// @brief Improper Potential base (4-body out-of-plane styles).
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

class ImproperPotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Improper = ImproperPotential;

} // namespace cpu
} // namespace ATV
