/// @file cpu/potentials/dihedral_fourier.cpp
/// @brief CPU DihedralFourier — V = Σ_m K_m (1 + cos(n_m φ − δ_m)).
///
/// Functional form of LAMMPS `src/EXTRA-MOLECULE/dihedral_fourier.cpp`; the
/// loop is the shared `md::PeriodicTorsionSet`.

#include "atomiverse/cpu/potentials/dihedral_fourier.h"

namespace ATV {
namespace cpu {

DihedralFourier::DihedralFourier(std::vector<std::vector<TorsionTerm>> terms)
    : set_(std::move(terms), md::PeriodicTorsionSet::Topology::Dihedral,
           "DihedralFourier") {}

void DihedralFourier::build(const BuildContext& ctx) { set_.build(ctx); }

void DihedralFourier::compute(const LaunchContext& ctx) { set_.compute(ctx); }

void DihedralFourier::destroy(const DestroyContext& /*ctx*/) { set_.destroy(); }

} // namespace cpu
} // namespace ATV
