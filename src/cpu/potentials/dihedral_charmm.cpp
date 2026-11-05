/// @file cpu/potentials/dihedral_charmm.cpp
/// @brief CPU DihedralCharmm — V = K(1 + cos(nφ − δ)).

#include "atomiverse/cpu/potentials/dihedral_charmm.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {

DihedralCharmm::DihedralCharmm(std::vector<double> K, std::vector<int> n,
                               std::vector<double> delta)
    : set_(charmm_torsion_terms(K, n, delta),
           md::PeriodicTorsionSet::Topology::Dihedral, "DihedralCharmm") {}

void DihedralCharmm::build(const BuildContext& ctx) { set_.build(ctx); }

void DihedralCharmm::compute(const LaunchContext& ctx) { set_.compute(ctx); }

void DihedralCharmm::destroy(const DestroyContext& /*ctx*/) { set_.destroy(); }

} // namespace cpu
} // namespace ATV
