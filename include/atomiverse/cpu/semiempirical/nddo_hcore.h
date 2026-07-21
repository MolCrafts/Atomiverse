/// @file cpu/semiempirical/nddo_hcore.h
/// @brief NDDO reference-shaped AM1/NDDO one-electron core matrix builder.
#pragma once

#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOHCoreResult {
    std::vector<double> packed_hcore;
    double nuclear_repulsion_ev = 0.0;
    bool includes_two_center_terms = false;
};

/// @brief Builds the NDDO one-electron matrix in NDDO reference hcore.F90 order.
class NDDOHCoreBuilder {
  public:
    /// @brief Port of hcore.F90 for AM1 one-center plus s/p two-center terms.
    ///
    /// Resolves x/y/z once then delegates to the raw-pointer overload (13.7×).
    NDDOHCoreResult build(const NDDOSystem& system,
                          const cpu::Frame& frame) const;

    /// @brief Same as `build(system, frame)` with caller-hoisted coords [Å].
    NDDOHCoreResult build(const NDDOSystem& system, const StorageReal* sp_x,
                          const StorageReal* sp_y,
                          const StorageReal* sp_z) const;

    /// @brief Port of hcore.F90's same-atom `uspd` diagonal initialization.
    NDDOHCoreResult build_one_center(const NDDOSystem& system) const;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
