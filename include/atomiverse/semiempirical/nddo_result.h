/// @file semiempirical/nddo_result.h
/// @brief Backend-independent NDDO method identity and physical results.
#pragma once

#include <string>
#include <vector>

namespace ATV::semiempirical {

/// @brief NDDO parameter family; the fitted Hamiltonian identity.
enum class SemiEmpiricalMethod { AM1, RM1 };

/// @brief Physical results of an NDDO method; no execution workspace.
struct NDDOResult {
    SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1;
    bool converged = false;
    int iterations = 0;
    /// @brief Total electronic + nuclear energy [Hartree].
    double electronic_energy_hartree = 0.0;
    /// @brief Total electronic + nuclear energy [eV] (same quantity as Hartree).
    double total_energy_ev = 0.0;
    double heat_of_formation_kcal_mol = 0.0;
    std::vector<double> mulliken_charges;
    /// @brief Nuclear Cartesian forces [eV/Å], layout (fx0,fy0,fz0, fx1, …).
    /// Empty unless the method's force evaluation succeeds.
    std::vector<double> forces_ev_a;
    std::string reference;
};

} // namespace ATV::semiempirical
