/// @file cpu/semiempirical/am1_parameters.h
/// @brief AM1 parameter accessors sourced from AM1 reference.
///
/// The table in this module is intentionally a data layer, not an algorithm.
/// Values are ported from AM1 reference `src/models/parameters_for_AM1_C.F90`
/// and orbital/core metadata from `src/models/parameters_C.F90`.
/// `cpu::semiempirical::NDDOSolver` consumes them through the typed
/// `AM1Parameter` shape instead of NDDO reference's global arrays.
#pragma once

#include <array>

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief One element's AM1 MNDO-form parameters.
struct AM1Parameter {
    int atomic_number = 0;
    const char* symbol = "";

    double uss = 0.0;
    double upp = 0.0;
    double udd = 0.0;
    double betas = 0.0;
    double betap = 0.0;
    double betad = 0.0;
    double zs = 0.0;
    double zp = 0.0;
    double zd = 0.0;
    double alp = 0.0;
    double gss = 0.0;
    double gsp = 0.0;
    double gpp = 0.0;
    double gp2 = 0.0;
    double hsp = 0.0;
    double polvol = 0.0;
    double zsn = 0.0;
    double zpn = 0.0;
    double zdn = 0.0;
    double f0sd = 0.0;
    double g2sd = 0.0;
    int ios = 0;
    int iop = 0;
    int iod = 0;
    int n_orbitals = 0;
    double core_charge = 0.0;
    int principal_s = 0;
    int principal_p = 0;
    int principal_d = 0;
    double dd = 0.0;
    double qq = 0.0;
    double am = 0.0;
    double ad = 0.0;
    double aq = 0.0;
    double eisol = 0.0;
    std::array<double, 4> guess1{};
    std::array<double, 4> guess2{};
    std::array<double, 4> guess3{};
};

/// @brief Return AM1 parameters for atomic number `z`, or null when absent.
const AM1Parameter* am1_parameter(int z) noexcept;

/// @brief True when this build carries AM1 parameters for atomic number `z`.
bool am1_has_parameter(int z) noexcept;

/// @brief Experimental gas-phase atomic heat of formation [kcal/mol] at 298 K.
///
/// Used with `eisol` to convert the NDDO total energy into a MOPAC-style
/// heat of formation. Throws if `z` has AM1 electronic parameters but no
/// tabulated atomic heat (no silent fallback).
/// @param z Atomic number.
/// @return ΔHf(atom, gas, 298 K) [kcal/mol].
double am1_atomic_heat_kcal_mol(int z);

/// @brief eV → kcal/mol factor used by MOPAC/AM1 thermochemistry (23.061).
inline constexpr double kAm1EvToKcalMol = 23.061;

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
