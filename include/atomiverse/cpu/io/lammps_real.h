/// @file cpu/io/lammps_real.h
/// @brief LAMMPS `units real` → Atomiverse engineering unit converters.
///
/// Real units: energy kcal/mol, length Å, angle degree (coeffs).
/// Atomiverse: energy eV, length Å, angle rad for bonded angle/dihedral
/// equilibrium values.
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {

/// @brief kcal/mol → eV (matches tools/debug_amorphous pin).
inline constexpr double kKcalPerMolInEv = 1.0 / 23.060547830619026;

/// @brief Convert energy from kcal/mol to eV.
inline double kcal_per_mol_to_ev(double e_kcal) {
    return e_kcal * kKcalPerMolInEv;
}

/// @brief Convert degrees to radians.
inline double degree_to_rad(double deg) {
    return deg * ATV::units::kDegreeInRad;
}

/// @brief Single-term LAMMPS `dihedral_style fourier` → Charmm parameters.
///
/// Fourier: E = Σ_i K_i (1 + cos(n_i φ − d_i)) with d in degrees.
/// For m=1 this is identical to `DihedralCharmm` after converting K to eV
/// and d to radians.
struct FourierTermReal {
    double K_kcal = 0; ///< force constant [kcal/mol]
    int n = 1;         ///< multiplicity (≥1)
    double d_deg = 0;  ///< phase [degree]
};

struct CharmmDihedralEv {
    double K_ev = 0;     ///< [eV]
    int n = 1;           ///< multiplicity
    double delta_rad = 0; ///< phase [rad]
};

/// @brief Map one fourier term to Charmm constructor args.
/// @throws std::invalid_argument if n < 1.
inline CharmmDihedralEv fourier_term_to_charmm(const FourierTermReal& t) {
    if (t.n < 1)
        throw std::invalid_argument(
            "fourier_term_to_charmm: multiplicity n must be ≥ 1");
    return {kcal_per_mol_to_ev(t.K_kcal), t.n, degree_to_rad(t.d_deg)};
}

/// @brief Map a vector of single-term fourier types (one term per type).
/// @throws std::invalid_argument if any entry has n < 1.
/// @note Multi-term (m>1) Fourier per type is not supported here — call
///       once per term only after expanding types, or throw via
///       `reject_multi_term_fourier`.
inline void fourier_table_to_charmm(const std::vector<FourierTermReal>& in,
                                    std::vector<double>& K_ev,
                                    std::vector<int>& n,
                                    std::vector<double>& delta_rad) {
    K_ev.clear();
    n.clear();
    delta_rad.clear();
    K_ev.reserve(in.size());
    n.reserve(in.size());
    delta_rad.reserve(in.size());
    for (const auto& t : in) {
        auto c = fourier_term_to_charmm(t);
        K_ev.push_back(c.K_ev);
        n.push_back(c.n);
        delta_rad.push_back(c.delta_rad);
    }
}

/// @brief Reject multi-term Fourier (m>1) for this phase of the GAFF path.
inline void reject_multi_term_fourier(int m) {
    if (m > 1)
        throw std::invalid_argument(
            "multi-term dihedral_style fourier (m>1) is not supported; "
            "use single-term terms mapped to DihedralCharmm");
    if (m < 1)
        throw std::invalid_argument("fourier m must be ≥ 1");
}

} // namespace cpu
} // namespace ATV
