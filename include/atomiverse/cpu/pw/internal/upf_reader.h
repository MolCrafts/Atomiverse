/// @file cpu/pw/internal/upf_reader.h
/// @brief UPF v1 pseudopotential file parser.
#pragma once

#include "atomiverse/core/precision.h"
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct UPFData {
    int z_val = 0;
    int l_max = 0;
    std::vector<AccReal> r;        // radial mesh [Bohr]
    std::vector<AccReal> v_loc;    // local potential [Hartree]
    std::vector<AccReal> rho_atom; // atomic pseudocharge density
    struct KBProjector {
        int l;
        std::vector<AccReal> beta; // radial beta function
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
};

UPFData read_upf_v1(const std::string& path);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
