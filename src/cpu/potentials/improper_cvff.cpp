/// @file cpu/potentials/improper_cvff.cpp
/// @brief CPU ImproperCvff — V = K (1 + d cos(n φ)).
///
/// Functional form of LAMMPS `src/MOLECULE/improper_cvff.cpp` (n = 0 … 6,
/// d = ±1); the loop is the shared `md::PeriodicTorsionSet` with δ = 0 / π.

#include "atomiverse/cpu/potentials/improper_cvff.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {

ImproperCvff::ImproperCvff(std::vector<double> K, std::vector<int> d,
                           std::vector<int> n)
    : set_(cvff_torsion_terms(K, d, n),
           md::PeriodicTorsionSet::Topology::Improper, "ImproperCvff") {}

void ImproperCvff::build(const BuildContext& ctx) { set_.build(ctx); }

void ImproperCvff::compute(const LaunchContext& ctx) { set_.compute(ctx); }

void ImproperCvff::destroy(const DestroyContext& /*ctx*/) { set_.destroy(); }

} // namespace cpu
} // namespace ATV
