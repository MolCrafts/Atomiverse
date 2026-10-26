/// @file cpu/planewave/internal/upf_reader.h
/// @brief UPF v1 pseudopotential file parser.
#pragma once

#include "atomiverse/core/precision.h"
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct UPFData {
    /// UPF `pseudo_type` attribute verbatim ("NC", "US", "USPP", "PAW", ...).
    /// Empty when the file predates the attribute (UPF v1). Only "NC" is
    /// supported; `read_upf_v1` rejects anything else rather than treating an
    /// augmentation-charge pseudopotential as norm-conserving.
    std::string pseudo_type;
    int z_val = 0;
    int l_max = 0;
    std::vector<AccReal> r;        // radial mesh [Bohr]
    std::vector<AccReal> rab;      // PP_RAB: dr/di mesh Jacobian [Bohr]
    std::vector<AccReal> v_loc;    // raw UPF local potential [Rydberg]
    std::vector<AccReal> rho_atom; // atomic pseudocharge density
    struct KBProjector {
        int l;
        std::vector<AccReal> beta; // radial beta function
        /// Declared PP_BETA radial integration domain, in mesh points.
        /// Raw beta storage may extend beyond this compact support.
        std::size_t radial_points = 0;
    };
    std::vector<KBProjector> kb; // Kleinman-Bylander projectors
    std::vector<AccReal> d_ij; // Full D_ij matrix (n_proj × n_proj, row-major)
    std::vector<AccReal>
        q_ij; // Augmentation charge matrix (n_proj × n_proj, row-major)
    int n_proj() const { return static_cast<int>(kb.size()); }

    // Atomic wavefunctions for initial guess (atomic_wfc_mod.f90)
    std::vector<std::vector<AccReal>>
        chi;                 // chi[nb][ir]: atomic wfc radial part
    std::vector<int> lchi;   // angular momentum for each chi
    std::vector<AccReal> oc; // occupation numbers
    int nwfc = 0;            // number of atomic wavefunctions

    /// @brief Validate the radial coordinates and PP_RAB Jacobian.
    /// @throws std::invalid_argument for missing, nonfinite or unordered data.
    void validate_radial_mesh() const;

    /// @brief Integrate sampled f(r) dr with the QE Simpson rule.
    /// @param values Samples on the full mesh or an explicitly chosen prefix.
    /// @return Integral, with one additional factor of Bohr in its units.
    /// @throws std::invalid_argument for an invalid mesh, shape or sample.
    AccReal integrate_radial(std::span<const AccReal> values) const;
};

UPFData read_upf_v1(const std::string& path);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
