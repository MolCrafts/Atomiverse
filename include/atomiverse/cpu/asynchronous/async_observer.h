/// @file cpu/asynchronous/async_observer.h
/// @brief `cpu::md::AsyncObserver` -- MD-bound convenience wrapper over
/// the driver-agnostic `cpu::AsyncComponentBase`.
///
/// Inherits the full async-dispatch machinery from `AsyncComponentBase` and
/// adds a sole behaviour: a default binding at
/// `md::Phase::end_of_step @ +20`. Users writing MD observers should inherit
/// from this class so `sim.set_observer_ref(obs)` just works; users wanting the same
/// observer logic reusable across minimizer/SCF should inherit directly
/// from `cpu::AsyncComponentBase` and bind explicitly per driver.
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/async_component_base.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief MD-specific async observer base (binds `md::Phase::end_of_step @ +20`).
class AsyncObserver : public cpu::AsyncComponentBase {
  public:
    /* role via Observer base */
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 20)};
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
