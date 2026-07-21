/// @file cpu/semiempirical/am1_bcc_charge_assigner.h
/// @brief Atomiverse AM1 plus molrs reference AM1-BCC charge assignment.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief Configuration for AM1-BCC charge assignment.
///
/// `am1` controls the Atomiverse AM1 single-point calculation. BCC atom typing,
/// correction lookup, and charge-column writes are delegated to molrs.
struct AM1BCCChargeConfig {
    AM1Config am1{};
    std::string parameter_set = "bcc";
};

/// @brief Assign reference AM1-BCC charges to a molrs molecular graph frame.
///
/// This is intentionally not an electronic-structure Driver. It is a charge
/// assignment facade: Atomiverse supplies AM1 Mulliken base charges, while
/// molrs owns the BCC typifier/correction machinery and writes `atoms.charge`.
class AM1BCCChargeAssigner {
  public:
    explicit AM1BCCChargeAssigner(AM1BCCChargeConfig config = {});

    /// @brief Compute AM1 base charges and apply molrs AM1-BCC corrections.
    ///
    /// The input frame must contain atom coordinates plus integer `atoms.type`
    /// atomic numbers (the exact Atomiverse exchange schema). BCC typing also
    /// requires canonical `atoms.element` symbols and molrs graph topology;
    /// neither field substitutes for the other.
    std::vector<double> assign(molrs::FrameRef& frame);

    const AM1Result& last_am1_result() const { return solver_.last_result(); }
    const std::vector<double>& last_charges() const { return last_charges_; }
    const AM1BCCChargeConfig& config() const { return config_; }

  private:
    AM1BCCChargeConfig config_;
    AM1Solver solver_;
    std::vector<double> last_charges_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
