/// @file cpu/scf/internal/scf_types.h
/// @brief **PRIVATE** SpinResolution — (electron count, multiplicity) →
///        (n_alpha, n_beta). Results live on the concrete method (`energy()` / `determinant()`).
#pragma once

#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Occupation resolver: (electron count, spin multiplicity) →
///        (n_alpha, n_beta). Direct instantiation — no factory function.
///
/// Implements N_e = Σ_A Z_A − q (charge already folded into
/// `IntegralPack::n_electrons` by its Frame constructor), M = 2S + 1:
///   n_alpha = (N_e + M − 1) / 2,   n_beta = (N_e − M + 1) / 2.
///
/// Feed the resolved counts to a concrete open-shell method (`UHF`/`UKS`):
/// @code
///   SpinResolution occ(ints.n_electrons, /*multiplicity=*/2);
///   cpu::scf::UHF uhf;
///   uhf.n_alpha = occ.n_alpha;
///   uhf.n_beta  = occ.n_beta;
/// @endcode
///
/// Matches PySCF's convention contrast: PySCF's `mol.spin` is 2S =
/// n_alpha − n_beta, NOT the multiplicity M.
struct SpinResolution {
    int n_alpha; ///< Alpha electrons (dimensionless count).
    int n_beta;  ///< Beta electrons (dimensionless count).

    /// @brief Resolve and validate the occupation split.
    /// @param n_electrons Total electron count N_e (dimensionless, >= 0).
    /// @param multiplicity Spin multiplicity M = 2S + 1 (dimensionless, >= 1).
    /// @throws std::invalid_argument when n_electrons < 0, multiplicity < 1,
    ///         N_e and M − 1 differ in parity (n_alpha/n_beta would be
    ///         non-integer), or n_beta would be negative (M > N_e + 1).
    SpinResolution(int n_electrons, int multiplicity) {
        if (n_electrons < 0 || multiplicity < 1
            || (n_electrons + multiplicity - 1) % 2 != 0
            || n_electrons - multiplicity + 1 < 0) {
            throw std::invalid_argument(
                "SpinResolution: invalid electron count / multiplicity "
                "combination (n_electrons="
                + std::to_string(n_electrons)
                + ", multiplicity=" + std::to_string(multiplicity)
                + "): requires n_electrons >= 0, multiplicity >= 1, "
                  "matching parity (even N_e with odd M or odd N_e with "
                  "even M), and multiplicity <= n_electrons + 1");
        }
        n_alpha = (n_electrons + multiplicity - 1) / 2;
        n_beta = (n_electrons - multiplicity + 1) / 2;
    }
};

} // namespace scf
} // namespace cpu
} // namespace ATV
