/// @file cpu/semiempirical/nddo_density.h
/// @brief NDDO reference-shaped density matrix construction for NDDO methods.
#pragma once

#include <vector>

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDODensityInput {
    int mdim = 0;
    int norbs = 0;
    int n_occupied = 0;
    double full_occupancy = 2.0;
    int n_fractional = 0;
    double fractional_occupancy = 0.0;
    int mode = 1;
};

/// @brief Builds packed lower-triangle density matrices using NDDO reference densit.F90.
///
/// Coefficients are row-major C(i, j) values. The indexing and branch structure
/// mirror AM1 reference's `src/matrix/densit.F90`.
class NDDODensityBuilder {
  public:
    /// @brief Overwrite the packed density, reusing caller-owned capacity.
    /// @param coefficients Row-major orbitals; must not alias packed_density.
    /// @param input Dimensions and occupations.
    /// @param packed_density Packed lower-triangle output; resized as necessary.
    /// Invalid dimensions, occupations, storage or aliasing leave output unchanged.
    void build_packed(const std::vector<double>& coefficients,
                      const NDDODensityInput& input,
                      std::vector<double>& packed_density) const;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
