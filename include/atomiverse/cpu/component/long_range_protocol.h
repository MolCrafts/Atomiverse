/// @file cpu/component/long_range_protocol.h
/// @brief Declarative real/reciprocal long-range electrostatics contract.
///
/// Real-space screened Coulomb and reciprocal-space Ewald/PME remain ordinary
/// evaluator components.  This opt-in protocol exposes only the metadata a
/// Driver needs to reject an incomplete or inconsistent pair before the Plan
/// is built; it performs no computation and introduces no evaluator hierarchy.
#pragma once

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

/// @brief Which half of a split long-range electrostatics method is provided.
enum class LongRangeRole {
    RealSpace,
    ReciprocalSpace,
};

/// @brief Build-time metadata for one long-range participant.
struct LongRangeContract {
    LongRangeRole role;
    AccReal alpha;           ///< Shared Ewald splitting parameter [1/Angstrom].
    AccReal real_cutoff = 0; ///< Real-space cutoff [Angstrom], or zero.
    AccReal accuracy = 0;    ///< Reciprocal target accuracy, or zero.
};

/// @brief Opt-in protocol implemented by both halves of split electrostatics.
class LongRangeParticipant {
  public:
    virtual ~LongRangeParticipant() = default;

    /// @return Immutable metadata used only while compiling the Plan.
    virtual LongRangeContract long_range_contract() const = 0;
};

} // namespace cpu
} // namespace ATV
