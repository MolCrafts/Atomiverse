/// @file cpu/bias/distance_cv.h
/// @brief Minimum-image distance CV between two stable atom tags.
#pragma once

#include <cstdint>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/bias/collective_variable.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief ξ = |r_i − r_j|_minimage [Å]; gradients equal and opposite.
class DistanceCV : public CollectiveVariable {
  public:
    /// @param tag_i Stable global tag of atom i (1-based Frame tags).
    /// @param tag_j Stable global tag of atom j.
    DistanceCV(TagInt tag_i, TagInt tag_j);

    void build(const cpu::Frame& frame) override;
    double evaluate(const cpu::Frame& frame, double* grad) const override;
    const char* name() const override { return "distance_cv"; }

  private:
    TagInt tag_i_, tag_j_;
    int row_i_ = -1;
    int row_j_ = -1;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
