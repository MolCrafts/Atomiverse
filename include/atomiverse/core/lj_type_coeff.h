/// @file core/lj_type_coeff.h
/// @brief Backend-neutral type-pair coefficient rows for typed nonbonded LJ.
///
/// Both backends (`cpu::PairLJCut`, `cpu::PairLJCutCoulLong`,
/// `cuda::PairLJCut`, `cuda::PairLJCutCoulLong`) take the same rows and
/// complete them with one table builder (`core/lj_type_table.h`), so a
/// coefficient list means the same Hamiltonian on either backend.
#pragma once

#include <optional>

namespace ATV {

/// @brief One Lennard-Jones type-pair coefficient row in public units.
///
/// Mirrors LAMMPS `pair_coeff i j epsilon sigma [cut_lj]`. A row given for
/// `(i, i)` is a diagonal; a row for `(i, j)`, `i != j`, is an explicit
/// cross term that overrides mixing. Rows are symmetric in `(i, j)`.
struct LJTypeCoeff {
    int type_i;     ///< First atom type (Frame `kAtom_type` value).
    int type_j;     ///< Second atom type.
    double epsilon; ///< Well depth [eV].
    double sigma;   ///< Length scale [Å].
    /// Per-pair LJ cutoff [Å]. Unset on an explicit row means the style's
    /// global cutoff (LAMMPS `pair_coeff` with four arguments); a pair that
    /// has no row at all mixes its cutoff arithmetically from the two
    /// diagonal cutoffs (LAMMPS `init_one`).
    std::optional<double> cutoff = std::nullopt;
};

/// @brief One force-shifted r⁻⁴ attraction row in public units.
///
/// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
/// else 0, so U(r_c) = F(r_c) = 0. An ATV-native term (no LAMMPS style),
/// used for e.g. a Li⁺–ether-oxygen solvation well in coarse-grained
/// polymer electrolytes. Rows are symmetric in `(i, j)`; a pair without a
/// row has no r⁻⁴ term (no mixing).
struct R4TypeCoeff {
    int type_i;      ///< First atom type (Frame `kAtom_type` value).
    int type_j;      ///< Second atom type.
    double strength; ///< Well strength S [eV]; S > 0 attracts.
    double sigma;    ///< Length scale σ [Å].
    double cutoff;   ///< Cutoff r_c [Å] of this term.
};

} // namespace ATV
