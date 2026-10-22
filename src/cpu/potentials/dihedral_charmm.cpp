/// @file cpu/potentials/dihedral_charmm.cpp
/// @brief CPU DihedralCharmm — V = K(1 + cos(nφ − δ)).

#include "atomiverse/cpu/potentials/dihedral_charmm.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {

namespace {

std::vector<std::vector<TorsionTerm>>
charmm_terms(const std::vector<double>& K, const std::vector<int>& n,
             const std::vector<double>& delta) {
    if (K.empty() || K.size() != n.size() || K.size() != delta.size())
        throw std::invalid_argument(
            "DihedralCharmm: K, n, delta must be non-empty and same size");
    std::vector<std::vector<TorsionTerm>> terms(K.size());
    for (std::size_t t = 0; t < K.size(); ++t) {
        if (n[t] < 1)
            throw std::invalid_argument(
                "DihedralCharmm: multiplicity n must be >= 1 (type "
                + std::to_string(t) + ")");
        terms[t].push_back(TorsionTerm{K[t], n[t], delta[t]});
    }
    return terms;
}

} // namespace

DihedralCharmm::DihedralCharmm(std::vector<double> K, std::vector<int> n,
                               std::vector<double> delta)
    : set_(charmm_terms(K, n, delta),
           md::PeriodicTorsionSet::Topology::Dihedral, "DihedralCharmm") {}

void DihedralCharmm::build(const BuildContext& ctx) { set_.build(ctx); }

void DihedralCharmm::compute(const LaunchContext& ctx) { set_.compute(ctx); }

void DihedralCharmm::destroy(const DestroyContext& /*ctx*/) { set_.destroy(); }

} // namespace cpu
} // namespace ATV
