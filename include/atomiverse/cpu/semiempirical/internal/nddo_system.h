/// @file cpu/semiempirical/internal/nddo_system.h
/// @brief **PRIVATE WORKSPACE** for `NDDOSolver` — AO indexing + valence
///        charges. Not a parallel "System" beside Frame. Users never
///        construct this; Frame is the only I/O currency.
#pragma once

#include <cstddef>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief NDDO method family (parameter table selector).
enum class SemiEmpiricalMethod {
    AM1,
    RM1,
};

struct NDDOAtom {
    int atomic_number = 0;
    int first_ao = 0; ///< 0-based analogue of NDDO reference nfirst(i)-1.
    int last_ao = -1; ///< 0-based analogue of NDDO reference nlast(i)-1.
    int n_orbitals = 0;
    double core_charge =
        0.0; ///< NDDO reference `tore`: valence electron/core charge.
    /// Non-owning pointer into the static AM1/RM1 parameter table.
    const AM1Parameter* params = nullptr;
};

struct NDDOSystem {
    std::vector<NDDOAtom> atoms;
    int n_orbitals = 0;
    int n_electrons = 0;
    int n_alpha = 0;
    int n_beta = 0;
    int charge = 0;
    SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1;

    /// @brief Empty system (fields filled by the caller).
    NDDOSystem() = default;

    /// @brief Build an NDDO AO-indexing snapshot from a Frame.
    ///
    /// Defaults to AM1 when `method` is omitted. Throws if the method table
    /// lacks a parameter row for any atom, or if charge/multiplicity is
    /// inconsistent with the valence electron count.
    ///
    /// @param frame         CPU Frame (type column = Z).
    /// @param charge        Molecular charge.
    /// @param multiplicity  Spin multiplicity M = 2S + 1 (>= 1).
    /// @param method        AM1 or RM1 parameter table.
    NDDOSystem(const cpu::Frame& frame, int charge, int multiplicity,
               SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1);
};

/// @brief Parameter table lookup for the active NDDO method.
const AM1Parameter* nddo_parameter(SemiEmpiricalMethod method, int z) noexcept;

/// @brief True when the method table covers atomic number `z`.
bool nddo_has_parameter(SemiEmpiricalMethod method, int z) noexcept;

/// @brief Number of AM1 valence orbitals for atomic number `z`.
int am1_orbital_count(int z) noexcept;

/// @brief NDDO reference AM1 valence/core charge (`tore`) for atomic number `z`.
double am1_core_charge(int z) noexcept;

/// @brief Packed lower-triangle index for 0-based AO indices.
std::size_t packed_index(int i, int j) noexcept;

/// @brief NDDO reference `chrge`: sum density diagonal entries over each atom's AO range.
std::vector<double> atomic_populations_from_packed_density(
    const NDDOSystem& system, const std::vector<double>& packed_density);

/// @brief Net Mulliken-like atom charges: core charge minus electron population.
std::vector<double>
atomic_charges_from_packed_density(const NDDOSystem& system,
                                   const std::vector<double>& packed_density);

/// @brief NDDO reference initial diagonal density guess (`pdiag`) for AM1 RHF setup.
std::vector<double> initial_diagonal_density_guess(const NDDOSystem& system);

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
