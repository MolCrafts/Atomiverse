/// @file cpu/scf/delta_scf.h
/// @brief Result type for ΔSCF / MOM excited-state energy differences.
///
/// User path: `UHF::delta_scf(frame, from, to, MomStyle::IMOM)` (see uhf.h).
/// Occupation entry on a free-standing determinant is
/// `SlaterDeterminant::with_occupation` (Gilbert–Besley–Gill ΔSCF/MOM).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Ground / excited mean-field energies and the ΔSCF gap from a
///        MOM/IMOM-locked non-Aufbau re-SCF.
struct DeltaScfResult {
    /// @brief Ground-state total energy [Hartree].
    double e_ground = 0.0;
    /// @brief Excited (non-Aufbau) total energy [Hartree].
    double e_excited = 0.0;
    /// @brief Vertical excitation energy (E_ex − E_gs) [eV].
    double delta_e_ev = 0.0;
    /// @brief Whether the excited-state SCF converged.
    bool converged = false;
    /// @brief Excited-state SCF iteration count (dimensionless).
    int iterations = 0;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
