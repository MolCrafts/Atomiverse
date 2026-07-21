/// @file cpu/minimizer/async_observer.h
/// @brief `cpu::minimizer::AsyncObserver` -- minimizer-bound convenience
/// wrapper over the driver-agnostic `cpu::AsyncComponentBase`.
///
/// Same machinery as `cpu::AsyncComponentBase`; adds a default binding at
/// `minimizer::Phase::end_of_step @ +20`. Use for observers that only run
/// under geometry-optimization drivers (e.g. an async energy-history logger
/// during L-BFGS). Cross-driver observers should inherit directly from
/// `cpu::AsyncComponentBase` and bind explicitly per driver.
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/async/async_component_base.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Minimizer-specific async observer base
/// (binds `minimizer::Phase::end_of_step @ +20`).
class AsyncObserver : public cpu::AsyncComponentBase {
  public:
    /* role via Observer base */
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::end_of_step, 20)};
    }
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
