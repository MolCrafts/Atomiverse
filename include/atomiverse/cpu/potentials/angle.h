/// @file cpu/potentials/angle.h
/// @brief Angle Potential base (AngleHarmonic → AnglePotential → Component).
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

class AnglePotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Angle = AnglePotential;

} // namespace cpu
} // namespace ATV
