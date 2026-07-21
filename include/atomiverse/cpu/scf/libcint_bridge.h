/// @file cpu/scf/libcint_bridge.h
/// @brief libcint argument bridge for the CPU SCF two-electron gradient path.
///
/// Mirrors the basis normalization and atm/bas/env layout used by PySCF's
/// `make_env` / `make_bas_env` helpers for point nuclei and contracted GTOs.
///
/// @note The current bridge expects exactly 3 primitives per shell, so it
///       only supports the STO-3G basis. `accumulate_two_electron_gradient`
///       guards on `nao` matching the SCF basis dimension and falls back to
///       the native Rys-quadrature derivative engine for any other basis.
#pragma once

#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace scf {

#ifdef ATV_HAS_LIBCINT

/// @brief Prepared libcint arguments for the current molecule.
struct LibcintContext {
    std::vector<int> atm;    ///< libcint atm table [natm * ATM_SLOTS].
    std::vector<int> bas;    ///< libcint bas table [nshells * BAS_SLOTS].
    std::vector<double> env; ///< libcint env array (Bohr / normalized coeffs).
    std::vector<int> shell_offsets; ///< AO offset per shell.
    int natm = 0;                   ///< Number of atoms.
    int nshells = 0;                ///< Number of libcint shells.
    int nao = 0;                    ///< Number of cartesian AOs (s/p only).

    /// @brief Empty context (fields filled by the caller or left default).
    LibcintContext() = default;

    /// @brief Build libcint atm/bas/env arrays for the molecule.
    ///
    /// Reads the generic `basis_data::sto3g()` table and emits one libcint bas
    /// entry per shell (s, then s, then p for 2nd-row atoms). Because the basis
    /// only uses l <= 1, cartesian and spherical AO dimensions match.
    ///
    /// @param frame CPU Frame with atom positions (Angstrom, converted to Bohr
    ///              internally) and nuclear charges on `kAtom_type`.
    ///
    /// @note Expects 3 primitives per shell (STO-3G). Callers must verify the
    ///       resulting `nao` against the SCF basis dimension before use.
    explicit LibcintContext(const Frame& frame);
};

#endif

} // namespace scf
} // namespace cpu
} // namespace ATV
