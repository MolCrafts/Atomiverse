/// @file cpu/component/mover.h
/// @brief cpu::Mover — MC trial proposal.
#pragma once

#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Monte Carlo trial proposal (translate, volume, insert/delete, …).
class Mover : public Component {
  public:
    /// True for proposal generators that can change the bond graph.
    virtual bool changes_connectivity() const { return false; }
};

} // namespace cpu
} // namespace ATV
