/// @file cpu/gpw/internal/ts_dispersion.h
/// @brief Tkatchenko–Scheffler (TS) pairwise dispersion for GPW.
///
/// Free-atom (volume-ratio = 1) TS energy + analytic forces matching libMBD's
/// `get_ts_energy` real-space path (Fermi damping, PBE s_R = 0.94, d = 20):
///
///   E = −Σ_{i≤j} Σ_T' C6_{ij} f_damp(R) / R⁶
///   C6_{ij} = 2 C6_i C6_j / [(α_j/α_i) C6_i + (α_i/α_j) C6_j]
///   f_damp = 1 / (1 + exp(−d (R/(s_R R0_{ij}) − 1))),  R0_{ij} = R_i + R_j
///
/// Optional per-atom volume ratios scale free-atom α, C6, R as in TS:
/// α ← α_free · v, C6 ← C6_free · v², R ← R_free · v^{1/3} (v = V/V_free).
/// Default empty ratios ⇒ free-atom limit (all v = 1). Hirshfeld volumes from
/// the GPW density (`GpwDft::hirshfeld`) feed energy, forces, and stress.
///
/// Periodic lattice sums are pure real-space (orthorhombic) out to `rcut_bohr`;
/// three-body / MBD is out of scope (see `gpw-mbd-ice`).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/basis_data/ts_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a TS evaluation.
struct TsResult {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces F_A = −∂E/∂R_A [Hartree/Bohr].
    std::vector<std::array<double, 3>> forces;
};

/// @brief PBE-TS free-atom (or volume-scaled) pairwise dispersion.
///
/// @param Z         Atomic numbers (1-based identity), length N.
/// @param pos_bohr  Positions [Bohr], length N.
/// @param cell_bohr Orthorhombic a,b,c [Bohr] (used iff periodic).
/// @param periodic  Lattice images within `rcut_bohr` when true.
/// @param rcut_bohr Pair cutoff [Bohr] (default 94.2 ≈ 50 Å).
/// @param s_r       Range scaling s_R (default 0.94 for PBE).
/// @param d_fermi   Fermi steepness d (default 20).
/// @param vol_ratio Optional V/V_free per atom; empty ⇒ all 1 (free-atom).
/// @return Energy [Hartree] and forces [Hartree/Bohr].
TsResult ts_pbe(const std::vector<int>& Z,
                const std::vector<std::array<double, 3>>& pos_bohr,
                const double cell_bohr[3], bool periodic,
                double rcut_bohr = 94.2, double s_r = 0.94,
                double d_fermi = 20.0,
                const std::vector<double>& vol_ratio = {});

} // namespace gpw
} // namespace cpu
} // namespace ATV
