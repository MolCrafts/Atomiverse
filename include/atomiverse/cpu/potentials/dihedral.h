/// @file cpu/potentials/dihedral.h
/// @brief Dihedral Potential base.
#pragma once

#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

class DihedralPotential : public Potential {
  public:
    void launch(const LaunchContext& ctx) final { compute(ctx); }
    /// A classical bonded term is a pure function of the geometry.
    bool trial_invariant() const override { return true; }

  protected:
    virtual void compute(const LaunchContext& ctx) = 0;
};

using Dihedral = DihedralPotential;

} // namespace cpu
} // namespace ATV
