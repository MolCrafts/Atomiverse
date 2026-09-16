/// @file cpu/bias/dihedral_cv.h
/// @brief Four-atom dihedral collective variable ξ = φ [rad].
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/bias/collective_variable.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief ξ = torsion φ(i–j–k–l) ∈ (−π, π] [rad]; gradients via dihedral_geom.
class DihedralCV : public CollectiveVariable {
  public:
    /// @param tag_i Tag of atom i.
    /// @param tag_j Tag of atom j.
    /// @param tag_k Tag of atom k.
    /// @param tag_l Tag of atom l.
    DihedralCV(TagInt tag_i, TagInt tag_j, TagInt tag_k, TagInt tag_l);

    void build(const cpu::Frame& frame) override;
    double evaluate(const cpu::Frame& frame, double* grad) const override;
    const char* name() const override { return "dihedral_cv"; }

  private:
    TagInt tag_i_, tag_j_, tag_k_, tag_l_;
    int row_i_ = -1, row_j_ = -1, row_k_ = -1, row_l_ = -1;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
