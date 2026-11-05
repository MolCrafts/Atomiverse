/// @file core/neighbor_check.h
/// @brief LAMMPS `Neighbor::check_distance` box-change trigger, shared by the
///        CPU Verlet decision and the CUDA device skin gate.
///
/// Ported from LAMMPS (https://www.lammps.org/, Sandia National
/// Laboratories), `src/neighbor.cpp` `Neighbor::check_distance` and
/// `src/domain.cpp` `Domain::box_corners` / `lamda_box_corners`, SHA
/// e8c4c278a8599165a0321ebfcb4e4046ef79ad79. Copyright (2003) Sandia
/// Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia
/// Corporation, the U.S. Government retains certain rights in this
/// software. Distributed under the GNU General Public License, version 2.
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define ATV_MD_HD __host__ __device__
#else
#define ATV_MD_HD
#endif

namespace ATV::md {

/// @brief Squared per-atom displacement that triggers a Verlet rebuild of a
///        list built in cell `ref` and checked in cell `now`.
///
/// LAMMPS shrinks the half-skin trigger by the motion of the box: with
/// `delta1`, `delta2` the two corner displacements it keeps (lo/hi corners
/// for an orthogonal box, all eight corners otherwise, with LAMMPS's
/// running "largest / else second" update), the trigger is
/// `0.5 (skin - (delta1 + delta2))`. An unchanged cell gives `skin / 2`.
/// The cell origin is the lower corner, fixed at 0 (ATV `Box`), so the
/// orthogonal lo corner never moves.
///
/// One deviation: when the box motion alone consumes the skin
/// (`delta1 + delta2 >= skin`) LAMMPS clamps the trigger to 0 and still
/// requires an atom to move (`rsq > 0`); a fully frozen system would then
/// keep a list the box has outrun. This returns a negative value instead,
/// which callers treat as an unconditional rebuild.
/// @param ref  Cell at the last rebuild, row-major 3x3, lattice vectors in
///             columns [Angstrom].
/// @param now  Current cell, same layout [Angstrom].
/// @param skin Verlet skin [Angstrom].
/// @return Trigger distance squared [Angstrom^2], or -1 to rebuild.
ATV_MD_HD inline double skin_trigger_sq(const double* ref, const double* now,
                                        double skin) {
    double dh[9];
    bool triclinic = false;
    for (int k = 0; k < 9; ++k) {
        dh[k] = now[k] - ref[k];
        if (k != 0 && k != 4 && k != 8)
            triclinic |= now[k] != 0.0 || ref[k] != 0.0;
    }
    double delta1 = 0.0, delta2 = 0.0;
    if (!triclinic) {
        // bboxlo is the fixed origin; bboxhi moves by the edge changes.
        delta2 = sqrt(dh[0] * dh[0] + dh[4] * dh[4] + dh[8] * dh[8]);
    } else {
        // Corners ordered x fastest, then y, then z (lamda_box_corners).
        for (int c = 0; c < 8; ++c) {
            const double l0 = (c & 1) ? 1.0 : 0.0;
            const double l1 = (c & 2) ? 1.0 : 0.0;
            const double l2 = (c & 4) ? 1.0 : 0.0;
            const double delx = dh[0] * l0 + dh[1] * l1 + dh[2] * l2;
            const double dely = dh[3] * l0 + dh[4] * l1 + dh[5] * l2;
            const double delz = dh[6] * l0 + dh[7] * l1 + dh[8] * l2;
            const double delta = sqrt(delx * delx + dely * dely + delz * delz);
            if (delta > delta1)
                delta1 = delta;
            else if (delta > delta2)
                delta2 = delta;
        }
    }
    if (delta1 + delta2 >= skin && (delta1 + delta2) > 0.0)
        return -1.0;
    const double delta = 0.5 * (skin - (delta1 + delta2));
    return delta * delta;
}

} // namespace ATV::md

#undef ATV_MD_HD
