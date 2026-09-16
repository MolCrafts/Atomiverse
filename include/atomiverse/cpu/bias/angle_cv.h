/// @file cpu/bias/angle_cv.h
/// @brief Three-atom angle collective variable ξ = ∠ijk [rad].
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/bias/collective_variable.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief ξ = angle at vertex j between atoms i–j–k [rad]; Wilson B-matrix grads.
class AngleCV : public CollectiveVariable {
  public:
    /// @param tag_i Tag of end atom i.
    /// @param tag_j Tag of vertex atom j.
    /// @param tag_k Tag of end atom k.
    AngleCV(TagInt tag_i, TagInt tag_j, TagInt tag_k);

    void build(const cpu::Frame& frame) override;
    double evaluate(const cpu::Frame& frame, double* grad) const override;
    const char* name() const override { return "angle_cv"; }

  private:
    TagInt tag_i_, tag_j_, tag_k_;
    int row_i_ = -1, row_j_ = -1, row_k_ = -1;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
