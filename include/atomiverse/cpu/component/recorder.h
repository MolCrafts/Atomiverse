/// @file cpu/component/recorder.h
/// @brief cpu::Recorder — external I/O sinks.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Writes outside the process (files, logs, …).
class Recorder : public Component {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred
               | RoleFlag::ExternalIO;
    }
};

} // namespace cpu
} // namespace ATV
