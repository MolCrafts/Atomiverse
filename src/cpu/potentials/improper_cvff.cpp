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

namespace {

std::vector<std::vector<TorsionTerm>> cvff_terms(const std::vector<double>& K,
                                                 const std::vector<int>& d,
                                                 const std::vector<int>& n) {
    if (K.empty() || K.size() != d.size() || K.size() != n.size())
        throw std::invalid_argument(
            "ImproperCvff: K, d, n must be non-empty and same size");
    std::vector<std::vector<TorsionTerm>> terms(K.size());
    for (std::size_t t = 0; t < K.size(); ++t) {
        if (d[t] != 1 && d[t] != -1)
            throw std::invalid_argument("ImproperCvff: d must be +1 or -1 "
                                        "(type "
                                        + std::to_string(t) + ")");
        terms[t].push_back(TorsionTerm{K[t], n[t], d[t] == 1 ? 0.0 : M_PI});
    }
    return terms;
}

} // namespace

ImproperCvff::ImproperCvff(std::vector<double> K, std::vector<int> d,
                           std::vector<int> n)
    : set_(cvff_terms(K, d, n), md::PeriodicTorsionSet::Topology::Improper,
           "ImproperCvff") {}

void ImproperCvff::build(const BuildContext& ctx) { set_.build(ctx); }

void ImproperCvff::compute(const LaunchContext& ctx) { set_.compute(ctx); }

void ImproperCvff::destroy(const DestroyContext& /*ctx*/) { set_.destroy(); }

} // namespace cpu
} // namespace ATV
