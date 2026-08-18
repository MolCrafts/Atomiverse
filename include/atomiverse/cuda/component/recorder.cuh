/// @file cuda/component/recorder.cuh
/// @brief cuda::Recorder — external I/O sinks. Kind tag mirroring
///        cpu::Recorder. The XYZ writer is `cuda::XyzRecorder`.
#pragma once

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Writes outside the process (files, logs, …).
class Recorder : public Component {
  public:
    Role role() const override {
        return RoleFlag::FrameImmutable | RoleFlag::Deferred
               | RoleFlag::ExternalIO;
    }
};

} // namespace cuda
} // namespace ATV
