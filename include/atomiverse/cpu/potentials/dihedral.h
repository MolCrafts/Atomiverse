/// @file cpu/potentials/dihedral.h
/// @brief Dihedral Potential base.
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

class DihedralPotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Dihedral = DihedralPotential;

} // namespace cpu
} // namespace ATV
