/// @file cuda/component/long_range_protocol.cuh
/// @brief Declarative real/reciprocal long-range electrostatics contract
///        (CUDA mirror of cpu/component/long_range_protocol.h).
#pragma once

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cuda {

/// @brief Which half of a split long-range electrostatics method is provided.
enum class LongRangeRole {
    RealSpace,
    ReciprocalSpace,
};

/// @brief Build-time metadata for one long-range participant.
struct LongRangeContract {
    LongRangeRole role;
    AccReal alpha;           ///< Shared Ewald splitting parameter [1/Å].
    AccReal real_cutoff = 0; ///< Real-space cutoff [Å], or zero.
    AccReal accuracy = 0;    ///< Reciprocal target accuracy, or zero.
};

/// @brief Opt-in protocol implemented by both halves of split electrostatics.
class LongRangeParticipant {
  public:
    virtual ~LongRangeParticipant() = default;

    /// @return Immutable metadata used only while compiling the Plan.
    virtual LongRangeContract long_range_contract() const = 0;
};

} // namespace cuda
} // namespace ATV
