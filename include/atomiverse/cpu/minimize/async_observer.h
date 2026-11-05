/// @file cpu/minimize/async_observer.h
/// @brief `cpu::minimizer::AsyncObserver` -- minimizer-bound asynchronous
///        observer: a `cpu::AsyncConsumer` with a default binding at
///        `minimizer::Phase::end_of_step @ +20`.
///
/// For observers that only run under geometry optimization (e.g. an
/// energy-history logger during L-BFGS); a step is an iteration. Cross-driver
/// observers derive from `cpu::AsyncConsumer` and bind explicitly.
#pragma once

#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/async_consumer.h"

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Minimizer asynchronous observer
///        (binds `minimizer::Phase::end_of_step @ +20`).
class AsyncObserver : public cpu::AsyncConsumer {
  public:
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::end_of_step, 20)};
    }
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
