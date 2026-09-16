/// @file cpu/gpw/internal/xdm_dispersion.h
/// @brief Free-atom exchange-hole dipole moment (XDM) pairwise dispersion.
///
/// Geometry-only free-atom XDM energy + analytic forces matching the
/// Becke–Johnson combination rules and BJ damping used by postg / QE
/// (`PW/src/xdm_dispersion.f90`, Otero-de-la-Roza & Johnson):
///
///   fac = α_i α_j / (M1_i α_j + M1_j α_i)
///   C6  = fac M1_i M1_j
///   C8  = (3/2) fac (M1_i M2_j + M2_i M1_j)
///   C10 = 2 fac (M1_i M3_j + M3_i M1_j) + (21/5) fac M2_i M2_j
///   R_c = (√(C8/C6) + (C10/C6)^{1/4} + √(C10/C8)) / 3
///   R_vdW = a1 R_c + a2
///   E = −Σ_{i≤j} Σ_T' [ C6/(R⁶+R_vdW⁶) + C8/(R⁸+R_vdW⁸) + C10/(R¹⁰+R_vdW¹⁰) ]
///
/// Free-atom α, M1–M3 from in-tree `basis_data/xdm_tables.h`. Default damping
/// is QE NC-PBE (a1=0.4283, a2=2.4690 Å). Periodic lattice sums are pure
/// real-space (orthorhombic) out to `rcut_bohr` via `pair_image_counts`.
/// Density-dependent Hirshfeld / BR hole moments are out of scope (foothold).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/basis_data/xdm_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of an XDM evaluation.
struct XdmResult {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces F_A = −∂E/∂R_A [Hartree/Bohr].
    std::vector<std::array<double, 3>> forces;
};

/// @brief Free-atom XDM pairwise dispersion (PBE BJ damping by default).
///
/// @param Z         Atomic numbers (1-based identity), length N.
/// @param pos_bohr  Positions [Bohr], length N.
/// @param cell_bohr Orthorhombic a,b,c [Bohr] (used iff periodic).
/// @param periodic  Lattice images within `rcut_bohr` when true.
/// @param rcut_bohr Pair cutoff [Bohr] (default 94.2 ≈ 50 Å).
/// @param a1        BJ a1 (default QE NC-PBE 0.4283).
/// @param a2_bohr   BJ a2 [Bohr] (default 2.4690 Å).
/// @return Energy [Hartree] and forces [Hartree/Bohr].
XdmResult xdm_pbe(const std::vector<int>& Z,
                  const std::vector<std::array<double, 3>>& pos_bohr,
                  const double cell_bohr[3], bool periodic,
                  double rcut_bohr = 94.2, double a1 = kXdmPbeA1,
                  double a2_bohr = kXdmPbeA2Bohr);

} // namespace gpw
} // namespace cpu
} // namespace ATV
