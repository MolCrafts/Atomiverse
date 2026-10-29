/// GPW adapter for the shared immutable Gaussian-core pair geometry.
#pragma once
#include "atomiverse/cpu/gpw/internal/gpw_atom.h"
#include "atomiverse/electronic/gaussian_core.h"

namespace ATV::cpu::gpw::internal {
inline gaussian_core::Model
make_core_overlap_model(const std::vector<GpwAtom>& atoms, const double* cell,
                        bool periodic) {
    std::vector<gaussian_core::Site> sites;
    sites.reserve(atoms.size());
    for (const auto& atom : atoms)
        sites.push_back(
            {{atom.pos[0], atom.pos[1], atom.pos[2]}, atom.z_val, atom.rloc});
    return gaussian_core::make_model(sites, {cell[0], cell[1], cell[2]},
                                     periodic);
}
inline double core_overlap_energy(const gaussian_core::Model& model) {
    double energy = 0;
    for (const auto& pair : model.pairs())
        energy += gaussian_core::evaluate_pair(pair).energy;
    return energy;
}
inline double core_overlap_energy(const std::vector<GpwAtom>& atoms,
                                  const double* cell, bool periodic) {
    return core_overlap_energy(make_core_overlap_model(atoms, cell, periodic));
}
inline void
add_core_overlap_forces(const gaussian_core::Model& model,
                        std::vector<std::array<double, 3>>& forces) {
    for (const auto& pair : model.pairs()) {
        // Moving one basis-cell atom moves all its periodic replicas. Its
        // self-image separations are unchanged: no atomic force, but nonzero W.
        if (pair.atom_a == pair.atom_b)
            continue;
        const auto value = gaussian_core::evaluate_pair(pair);
        for (int k = 0; k < 3; ++k) {
            const double f = value.force_over_r * pair.displacement[k];
            forces[pair.atom_a][k] += f;
            forces[pair.atom_b][k] -= f;
        }
    }
}
inline std::array<double, 6>
core_overlap_virial(const gaussian_core::Model& model) {
    constexpr int aa[6] = {0, 1, 2, 0, 0, 1};
    constexpr int bb[6] = {0, 1, 2, 1, 2, 2};
    std::array<double, 6> virial{};
    for (const auto& pair : model.pairs()) {
        const auto value = gaussian_core::evaluate_pair(pair);
        for (int k = 0; k < 6; ++k)
            virial[k] += value.force_over_r * pair.displacement[aa[k]]
                         * pair.displacement[bb[k]];
    }
    return virial;
}
} // namespace ATV::cpu::gpw::internal
