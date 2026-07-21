/// @file cpu/semiempirical/am1_solver.h
/// @brief AM1 inner solver facade.
#pragma once

#include "atomiverse/cpu/semiempirical/nddo_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using AM1Config = NDDOConfig;
using AM1Result = NDDOResult;

/// @brief AM1 specialization of the NDDO semiempirical solver.
class AM1Solver : public NDDOSolver {
  public:
    explicit AM1Solver(AM1Config config = {})
        : NDDOSolver(normalize_(config)) {}

  private:
    static AM1Config normalize_(AM1Config config) {
        config.method = SemiEmpiricalMethod::AM1;
        return config;
    }
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
