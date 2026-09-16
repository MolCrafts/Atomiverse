/// @file cpu/gpw/internal/mbd_dispersion.h
/// @brief Many-body dispersion (MBD@rsSCS) for GPW.
///
/// Free-atom (volume-ratio = 1) and periodic Ewald + k-mesh MBD@rsSCS energy
/// matching libMBD `cmbd_mbd_scs_energy` / pure-Python `pymbd.mbd_energy`
/// (Ambrosetti et al., J. Chem. Phys. 140, 18A508 (2014); Hermann et al.,
/// J. Chem. Phys. 159, 174802 (2023)):
///
///   1. Range-separated self-consistent screening of dynamical polarizabilities
///      (Fermi × Gaussian dipole tensor, β damping; lattice images when periodic).
///   2. Long-range MBD Hamiltonian diagonalization → ZPE difference
///      (Ewald dipole tensor + Brillouin-zone k-mesh average when periodic).
///
/// PBE defaults: β = 0.83, Fermi a = 6. Free-atom α₀ / C₆ / R_vdW from
/// `basis_data/ts_tables.h`.
///
/// Ionic forces \(F_A = -\partial E_{\mathrm{MBD}}/\partial R_A\) are the
/// negative gradient of this geometry-only energy (product path uses high-order
/// central finite differences of the same kernel that matches libMBD; free-atom
/// and fixed volume-ratio paths only — density-chain \(\partial v/\partial R\)
/// Hirshfeld is out of scope; `GpwDft::hirshfeld` holds \(v\) at the SCF density).
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/gpw/basis_data/ts_tables.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of an MBD evaluation.
struct MbdResult {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces \(F_A = -\partial E/\partial R_A\) [Hartree/Bohr].
    std::vector<std::array<double, 3>> forces;
};

/// @brief PBE-MBD@rsSCS free-atom (or volume-scaled) many-body dispersion.
///
/// @param Z         Atomic numbers, length N.
/// @param pos_bohr  Positions [Bohr], length N.
/// @param cell_bohr Orthorhombic a,b,c [Bohr] (required when periodic).
/// @param periodic  false → isolated free-atom MBD; true → Ewald + k-mesh.
/// @param beta      Range-separation β (default 0.83 for PBE).
/// @param a_fermi   Fermi steepness a (default 6).
/// @param vol_ratio Optional V/V_free per atom; empty ⇒ all 1 (free-atom).
/// @param k_mesh    Monkhorst–Pack-like mesh (libMBD shift ½) when periodic
///                  (length-3 pointer; nullptr ⇒ {2,2,2}); ignored when
///                  isolated. Each component must be ≥ 1.
/// @return Energy [Hartree] and forces [Hartree/Bohr].
MbdResult mbd_rsscs_pbe(const std::vector<int>& Z,
                        const std::vector<std::array<double, 3>>& pos_bohr,
                        const double cell_bohr[3], bool periodic,
                        double beta = 0.83, double a_fermi = 6.0,
                        const std::vector<double>& vol_ratio = {},
                        const int* k_mesh = nullptr);

} // namespace gpw
} // namespace cpu
} // namespace ATV
