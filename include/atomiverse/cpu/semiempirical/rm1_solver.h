/// @file cpu/semiempirical/rm1_solver.h
/// @brief RM1 inner solver facade over NDDOSolver.
#pragma once

#include "atomiverse/cpu/semiempirical/nddo_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using RM1Config = NDDOConfig;
using RM1Result = NDDOResult;

/// @brief RM1 specialization of the NDDO semiempirical solver.
class RM1Solver : public NDDOSolver {
  public:
    explicit RM1Solver(RM1Config config = {})
        : NDDOSolver(normalize_(config)) {}

  private:
    static RM1Config normalize_(RM1Config config) {
        config.method = SemiEmpiricalMethod::RM1;
        return config;
    }
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
