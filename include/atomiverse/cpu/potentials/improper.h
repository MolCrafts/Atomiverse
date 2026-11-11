/// @file cpu/potentials/improper.h
/// @brief Improper Potential base (4-body out-of-plane styles).
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

class ImproperPotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }
    /// A classical bonded term is a pure function of the geometry.
    bool trial_invariant() const override { return true; }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Improper = ImproperPotential;

} // namespace cpu
} // namespace ATV
