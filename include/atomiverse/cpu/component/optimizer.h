/// @file cpu/component/optimizer.h
/// @brief cpu::Optimizer — geometry optimizer step (L-BFGS, CG, …).
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief One geometry-optimizer iteration step.
class Optimizer : public Component {};

} // namespace cpu
} // namespace ATV
