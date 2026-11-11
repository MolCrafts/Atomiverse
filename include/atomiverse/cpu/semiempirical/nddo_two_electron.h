/// @file cpu/semiempirical/nddo_two_electron.h
/// @brief NDDO reference-shaped two-center two-electron integral blocks.
#pragma once

#include <array>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOTwoCenterIntegralBlock {
    int atom_i = -1;
    int atom_j = -1;
    int n_pairs_i = 0;
    int n_pairs_j = 0;
    // Only the first n_pairs_i * n_pairs_j entries are active. Reusing a
    // smaller block leaves the inactive tail unspecified.
    std::array<double, 100> values{};

    double at(int pair_i, int pair_j) const;
};

/// @brief Port of AM1 reference `reppd/rotatd/w2mat` for AM1 s/p atoms.
///
/// Takes the caller-hoisted coordinate spans (x/y/z [Angstrom]) instead of
/// the Frame: this runs per atom PAIR per SCF iteration, and resolving
/// frame keys here blew the per-launch resolution budget (the 13.7x rule
/// — resolution belongs at the launch boundary). Callers hoist
/// `frame.span(kAtom_x)` etc. once per build.
/// @param out Caller-owned s/p block, overwritten in its active region.
void build_two_center_integral_block(const NDDOSystem& system,
                                     const StorageReal* sp_x,
                                     const StorageReal* sp_y,
                                     const StorageReal* sp_z, int atom_i,
                                     int atom_j,
                                     NDDOTwoCenterIntegralBlock& out);

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
