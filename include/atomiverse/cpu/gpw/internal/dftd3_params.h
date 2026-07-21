/// @file cpu/gpw/internal/dftd3_params.h
/// @brief Grimme DFT-D3 reference parameters (zero-damping), matching CP2K's
///        `qs_dispersion_*` exactly. All data is in-tree static tables
///        (basis_data/dftd3_tables.h) — nothing is parsed at build, test or
///        run time.
///
/// Three tables feed the D3 model (Grimme, Antony, Ehrlich, Krieg,
/// J. Chem. Phys. 132, 154104 (2010)):
///   1. The tabulated, coordination-number-resolved reference C6 coefficients
///      for the supported element set (same encoding as
///      `qs_dispersion_d3::dftd3_c6_param`).
///   2. The element covalent radii `rcov` (Pyykkö & Atsumi, Chem. Eur. J. 15,
///      2009, 188; metals reduced 10%), pre-scaled by k2 = 4/3 and converted to
///      Bohr — exactly the `dispersion_env%rcov = k2*rcov*bohr` of CP2K.
///   3. The `r2r4` values (PBE0/def2-QZVP) post-processed to
///      `sqrt(0.5*r2r4_raw*sqrt(Z))` so that C8 = 3*C6*r2r4_A*r2r4_B, and the
///      packed lower-triangular `r0ab(za,zb)` cutoff-radius table [Bohr].
///
/// Element tables cover Z = 1..94 (`kMaxElem`); C6 records cover the supported
/// GPW element set. Physics lives in `dftd3.h`.
#pragma once

#include <array>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Largest atomic number supported by the D3 tables.
constexpr int kMaxElem = 94;

/// @brief Reference-state entry of the CN-resolved C6 table for one element pair.
struct D3C6Ref {
    double c6 = 0.0;   ///< Reference C6^AB [a.u.] for this (CN_A, CN_B) pair.
    double cn_a = 0.0; ///< Reference coordination number of A.
    double cn_b = 0.0; ///< Reference coordination number of B.
};

/// @brief Parsed DFT-D3 parameter set (everything geometry-independent).
///
/// `rcov`, `r2r4` and `r0ab` are indexed by atomic number Z (1-based; index 0
/// unused). `c6ab[za][zb]` is the list of reference states for that ordered
/// element pair (symmetric: both (za,zb) and (zb,za) are populated).
struct D3Params {
    /// @brief k2-folded covalent radii [Bohr], rcov[Z] = (4/3)*rcov_Pyykko*bohr.
    std::array<double, kMaxElem + 1> rcov{};
    /// @brief Processed r2r4 [a.u.], r2r4[Z] = sqrt(0.5*r2r4_raw*sqrt(Z)).
    std::array<double, kMaxElem + 1> r2r4{};
    /// @brief Cutoff radii r0ab[Za][Zb] [Bohr] (symmetric).
    std::array<std::array<double, kMaxElem + 1>, kMaxElem + 1> r0ab{};
    /// @brief CN-resolved reference C6 states per ordered element pair.
    std::array<std::array<std::vector<D3C6Ref>, kMaxElem + 1>, kMaxElem + 1>
        c6ab{};

    /// @brief Build the full parameter set from the in-tree tables.
    ///
    /// Radii [Bohr], C6 [Hartree*Bohr^6]. C6 records cover the supported GPW
    /// element set only; an element outside it yields no C6 record and the
    /// caller throws (no fallback value) — check `supports_element` first.
    D3Params();

    /// @brief Whether the in-tree C6 table covers atomic number `z`.
    /// @param z Atomic number.
    /// @return true iff `z` is one of the supported elements.
    static bool supports_element(int z);
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
