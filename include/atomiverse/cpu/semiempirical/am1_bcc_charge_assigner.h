/// @file cpu/semiempirical/am1_bcc_charge_assigner.h
/// @brief Atomiverse AM1 plus molrs reference AM1-BCC charge assignment.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Assign reference AM1-BCC charges to a molrs molecular graph frame.
///
/// Atomiverse supplies AM1 Mulliken base charges; molrs owns BCC typing /
/// corrections. No `AM1BCCChargeConfig` bag — parameters on the constructor.
class AM1BCCChargeAssigner {
  public:
    /// @param am1 Concrete AM1 method (parameters already set).
    /// @param parameter_set molrs BCC parameter set name (default "bcc").
    explicit AM1BCCChargeAssigner(AM1 am1 = {},
                                  std::string parameter_set = "bcc");

    std::vector<double> assign(molrs::FrameRef& frame);

    const AM1Result& last_am1_result() const { return solver_.last_result(); }
    const std::vector<double>& last_charges() const { return last_charges_; }
    const std::string& parameter_set() const { return parameter_set_; }

  private:
    AM1 solver_;
    std::string parameter_set_;
    std::vector<double> last_charges_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
