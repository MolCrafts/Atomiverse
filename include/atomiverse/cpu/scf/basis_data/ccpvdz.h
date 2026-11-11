/// @file cpu/scf/basis_data/ccpvdz.h
/// @brief cc-pvdz basis-set table (auto-generated; do not edit).
///
/// Emitted by `tools/gen_basis_data.py` from PySCF's `mol._basis`
/// dictionary, so numerical values match the PySCF ground-truth path
/// bit-for-bit. Contraction coefficients use the unnormalized
/// convention (physical GTO normalization is applied by
/// `ShellList` constructor).
///
/// Supported elements: H.
///
/// Regenerate:
///   python tools/gen_basis_data.py cc-pvdz ccpvdz H
#pragma once

#include <vector>

#include "atomiverse/cpu/scf/basis_data/basis_table.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace basis_data {

/// @brief Get the cc-pvdz table, indexed by atomic number Z.
/// @return Reference to a static table with AtomBasis entries for
///         Z=1..1; position 0 and any unsupported Z are empty.
inline const std::vector<AtomBasis>& ccpvdz() {
    static const std::vector<AtomBasis> data = []() {
        return std::vector<AtomBasis>{
            AtomBasis{}, // Z=0 unused
            // Z=1 (H): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {13.01, 1.962, 0.4446},
                                {0.019685, 0.137977, 0.478148}},
                ContractedShell{/*l=*/0 /* s */, {0.122}, {1.0}},
                ContractedShell{/*l=*/1 /* p */, {0.727}, {1.0}},
            }},
        };
    }();
    return data;
}

} // namespace basis_data
} // namespace scf
} // namespace cpu
} // namespace ATV
