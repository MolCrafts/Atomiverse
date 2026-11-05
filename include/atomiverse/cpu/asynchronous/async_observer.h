/// @file cpu/asynchronous/async_observer.h
/// @brief `cpu::md::AsyncObserver` -- MD-bound asynchronous observer: a
///        `cpu::AsyncConsumer` with a default binding at
///        `md::Phase::end_of_step @ +20`.
///
/// Users writing MD observers derive from this class so
/// `sim.set_observer_ref(obs)` just works; an observer reused across drivers
/// derives from `cpu::AsyncConsumer` and binds explicitly per driver.
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/async_consumer.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief MD asynchronous observer (binds `md::Phase::end_of_step @ +20`).
class AsyncObserver : public cpu::AsyncConsumer {
  public:
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, 20)};
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
