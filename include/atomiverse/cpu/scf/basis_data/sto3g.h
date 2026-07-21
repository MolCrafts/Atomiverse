/// @file cpu/scf/basis_data/sto3g.h
/// @brief sto-3g basis-set table (auto-generated; do not edit).
///
/// Emitted by `tools/gen_basis_data.py` from PySCF's `mol._basis`
/// dictionary, so numerical values match the PySCF ground-truth path
/// bit-for-bit. Contraction coefficients use the unnormalized
/// convention (physical GTO normalization is applied by
/// `ShellList` constructor).
///
/// Supported elements: H, He, Li, Be, B, C, N, O.
///
/// Regenerate:
///   python tools/gen_basis_data.py sto-3g sto3g H He Li Be B C N O
#pragma once

#include <vector>

#include "atomiverse/cpu/scf/basis_data/basis_table.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace basis_data {

/// @brief Get the sto-3g table, indexed by atomic number Z.
/// @return Reference to a static table with AtomBasis entries for
///         Z=1..8; position 0 and any unsupported Z are empty.
inline const std::vector<AtomBasis>& sto3g() {
    static const std::vector<AtomBasis> data = []() {
        return std::vector<AtomBasis>{
            AtomBasis{}, // Z=0 unused
            // Z=1 (H): 1 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {3.42525091, 0.62391373, 0.1688554},
                                {0.15432897, 0.53532814, 0.44463454}},
            }},
            // Z=2 (He): 1 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {6.36242139, 1.158923, 0.31364979},
                                {0.15432897, 0.53532814, 0.44463454}},
            }},
            // Z=3 (Li): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {16.119575, 2.9362007, 0.7946505},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {0.6362897, 0.1478601, 0.0480887},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {0.6362897, 0.1478601, 0.0480887},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
            // Z=4 (Be): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {30.167871, 5.4951153, 1.4871927},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {1.3148331, 0.3055389, 0.0993707},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {1.3148331, 0.3055389, 0.0993707},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
            // Z=5 (B): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {48.791113, 8.8873622, 2.405267},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {2.2369561, 0.5198205, 0.1690618},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {2.2369561, 0.5198205, 0.1690618},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
            // Z=6 (C): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {71.616837, 13.045096, 3.5305122},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {2.9412494, 0.6834831, 0.2222899},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {2.9412494, 0.6834831, 0.2222899},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
            // Z=7 (N): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {99.106169, 18.052312, 4.8856602},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {3.7804559, 0.8784966, 0.2857144},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {3.7804559, 0.8784966, 0.2857144},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
            // Z=8 (O): 3 shells
            AtomBasis{{
                ContractedShell{/*l=*/0 /* s */,
                                {130.70932, 23.808861, 6.4436083},
                                {0.15432897, 0.53532814, 0.44463454}},
                ContractedShell{/*l=*/0 /* s */,
                                {5.0331513, 1.1695961, 0.380389},
                                {-0.09996723, 0.39951283, 0.70011547}},
                ContractedShell{/*l=*/1 /* p */,
                                {5.0331513, 1.1695961, 0.380389},
                                {0.15591627, 0.60768372, 0.39195739}},
            }},
        };
    }();
    return data;
}

} // namespace basis_data
} // namespace scf
} // namespace cpu
} // namespace ATV
