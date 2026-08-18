/// @file cuda/component/optimizer.cuh
/// @brief cuda::Optimizer — geometry optimization step. Kind tag mirroring
///        cpu::Optimizer; `require_optimizer()` probes this kind.
#pragma once

#include "atomiverse/cuda/component/component.cuh"

namespace ATV {
namespace cuda {

/// @brief Proposes a geometry-optimization search direction / step.
class Optimizer : public Component {};

} // namespace cuda
} // namespace ATV
