/// @file cpu/scf/internal/level_shift.cpp
/// @brief Saunders–Hillier virtual-orbital level shift implementation.

#include "atomiverse/cpu/scf/internal/level_shift.h"

#include <cmath>

#include <xtensor-blas/xlinalg.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

RealMatrix apply_level_shift(const RealMatrix& F, const RealMatrix& S,
                             const RealMatrix& D_occ, double lambda) {
    if (std::abs(lambda) <= kLevelShiftSkipThreshold) // PySCF skip rule.
        return F;
    RealMatrix SD(xt::linalg::dot(S, D_occ));
    RealMatrix SDS(xt::linalg::dot(SD, S));
    return F + lambda * (S - SDS);
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
