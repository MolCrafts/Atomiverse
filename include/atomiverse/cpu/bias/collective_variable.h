/// @file cpu/bias/collective_variable.h
/// @brief CollectiveVariable — value/gradient protocol (not a Component).
#pragma once

#include <cstddef>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief Narrow CV protocol: scalar ξ(R) and optional ∇_R ξ into caller storage.
class CollectiveVariable {
  public:
    virtual ~CollectiveVariable() = default;

    /// @brief One-time resolution (tag → local index, etc.).
    virtual void build(const cpu::Frame& frame) = 0;

    /// @brief Evaluate ξ; if `grad` non-null, write per-atom gradients (3*n).
    /// @param frame Current geometry.
    /// @param grad Caller-owned buffer of size ≥ 3*nlocal, or nullptr for value-only.
    /// @return ξ [Å] (or other CV unit declared by the concrete type).
    virtual double evaluate(const cpu::Frame& frame, double* grad) const = 0;

    virtual const char* name() const = 0;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
