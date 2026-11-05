/// @file core/periodic_torsion.h
/// @brief The periodic 4-body sum V = Σ_m K_m (1 + cos(n_m φ − δ_m)) shared
///        by the CPU `PeriodicTorsionSet` and the CUDA torsion kernel: the
///        validated per-type term table and the per-tuple term sum.
///
/// Three LAMMPS styles are this one sum with different parameter shapes:
/// `dihedral_style charmm` (weight 0: one `{K, n, δ}`), `dihedral_style
/// fourier` (`m` × `{K, n, δ}`) and `improper_style cvff` (one
/// `{K, n, δ ∈ {0, π}}`).
#pragma once

#include <vector>

#if defined(__CUDACC__)
#define ATV_TORSION_HD __host__ __device__
#else
#define ATV_TORSION_HD
#endif

namespace ATV {

/// @brief One periodic term K (1 + cos(n φ − δ)).
struct TorsionTerm {
    double K = 0.0;     ///< Amplitude [eV].
    int n = 1;          ///< Multiplicity, 0 ≤ n ≤ 6 (n = 0 is a constant).
    double delta = 0.0; ///< Phase [rad].
};

/// LAMMPS cvff bound; fourier never above.
inline constexpr int kMaxTorsionMultiplicity = 6;

/// @brief Validated terms flattened as CSR over types: the terms of type t
///        are [offset[t], offset[t+1]).
struct TorsionTable {
    /// @param terms Terms per type (index = 0-based type id); a type with no
    ///        terms, or only K = 0 terms, contributes exactly nothing.
    /// @param style Name used in error messages.
    /// @throws std::invalid_argument on an empty table, a non-finite K or δ,
    ///         or a multiplicity outside [0, 6].
    TorsionTable(const std::vector<std::vector<TorsionTerm>>& terms,
                 const char* style);

    int n_types() const { return static_cast<int>(active.size()); }

    std::vector<int> offset;
    std::vector<double> K;
    std::vector<int> n;
    std::vector<double> cos_delta;
    std::vector<double> sin_delta;
    std::vector<char> active; ///< type has at least one K ≠ 0 term
};

/// @brief `dihedral_style charmm` (weight 0): one term per type, n ≥ 1.
/// @throws std::invalid_argument on size mismatch or n < 1.
std::vector<std::vector<TorsionTerm>>
charmm_torsion_terms(const std::vector<double>& K, const std::vector<int>& n,
                     const std::vector<double>& delta);

/// @brief `improper_style cvff`: V = K (1 + d cos(n φ)), d = ±1.
/// @throws std::invalid_argument on size mismatch or d ∉ {−1, +1}.
std::vector<std::vector<TorsionTerm>>
cvff_torsion_terms(const std::vector<double>& K, const std::vector<int>& d,
                   const std::vector<int>& n);

/// @brief cos(nφ), sin(nφ) from (cos φ, sin φ) by recurrence — no atan2.
ATV_TORSION_HD inline void multiple_angle(int n, double c, double s, double& cn,
                                          double& sn) {
    switch (n) {
    case 0:
        cn = 1.0;
        sn = 0.0;
        return;
    case 1:
        cn = c;
        sn = s;
        return;
    case 2:
        cn = c * c - s * s;
        sn = 2.0 * s * c;
        return;
    case 3:
        cn = (4.0 * c * c - 3.0) * c;
        sn = (3.0 - 4.0 * s * s) * s;
        return;
    default:
        cn = 1.0;
        sn = 0.0;
        for (int k = 0; k < n; ++k) {
            const double cn1 = cn * c - sn * s;
            sn = sn * c + cn * s;
            cn = cn1;
        }
    }
}

/// @brief Energy, dE/dφ and d²E/dφ² of the terms [first, last) at φ.
ATV_TORSION_HD inline void torsion_sum(int first, int last, const double* K,
                                       const int* n, const double* cos_delta,
                                       const double* sin_delta, double cos_phi,
                                       double sin_phi, double& energy,
                                       double& dE_dphi, double& d2E_dphi2) {
    energy = dE_dphi = d2E_dphi2 = 0.0;
    for (int k = first; k < last; ++k) {
        double cn, sn;
        multiple_angle(n[k], cos_phi, sin_phi, cn, sn);
        const double carg = cn * cos_delta[k] + sn * sin_delta[k];
        const double sarg = sn * cos_delta[k] - cn * sin_delta[k];
        const double m = static_cast<double>(n[k]);
        energy += K[k] * (1.0 + carg);
        dE_dphi -= K[k] * m * sarg;
        d2E_dphi2 -= K[k] * m * m * carg;
    }
}

} // namespace ATV
