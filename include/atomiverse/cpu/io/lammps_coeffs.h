/// @file cpu/io/lammps_coeffs.h
/// @brief Parser for LAMMPS `units real` coefficient include files (`.ff`).
///
/// Reads the style/coefficient command subset that a data-file companion
/// carries (`bond_coeff` / `angle_coeff` / `dihedral_coeff` / `pair_coeff`
/// plus their `*_style` declarations and `special_bonds`), converting every
/// value to Atomiverse engineering units (eV, Å, rad) at parse time.
///
/// **Data only** — the result is plain typed tables shaped for direct
/// construction of `BondHarmonic` / `AngleHarmonic` / `DihedralCharmm` /
/// `DihedralMultiHarmonic` / `PairLJCutCoulLong`. Composition stays the
/// caller's job.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/lj_type_coeff.h"

namespace ATV {
namespace cpu {

/// @brief Parsed LAMMPS-real coefficient tables in Atomiverse units.
///
/// Vectors are indexed by 0-based type (LAMMPS 1-based ids shifted down) and
/// sized to the largest type id seen. `dihedral_style hybrid charmm
/// multi/harmonic` is split into two aligned tables covering **all** dihedral
/// types: a type owned by the other style gets exact-zero coefficients
/// (K = 0 / Cn = {0}), which contributes zero energy and force, so both
/// components can be registered over the full topology.
struct LammpsCoeffs {
    std::string pair_style;      ///< e.g. "lj/cut/coul/long", "lj/cut"
    double pair_cut_lj = 0.0;    ///< global LJ cutoff [Å] (0 = not given)
    double pair_cut_coul = 0.0;  ///< Coulomb cutoff [Å] (0 = same as LJ)
    std::vector<LJTypeCoeff> lj; ///< pair_coeff rows, ε in eV, σ in Å

    std::vector<double> bond_K;  ///< [eV/Å²] (LAMMPS K, energy/length²)
    std::vector<double> bond_r0; ///< [Å]

    std::vector<double> angle_K;      ///< [eV/rad²]
    std::vector<double> angle_theta0; ///< [rad]

    bool has_charmm = false; ///< any charmm dihedral types present
    bool has_multi = false;  ///< any multi/harmonic dihedral types present
    std::vector<double> dihedral_charmm_K;     ///< [eV], 0 for non-charmm
    std::vector<int> dihedral_charmm_n;        ///< multiplicity, 1 placeholder
    std::vector<double> dihedral_charmm_delta; ///< [rad]
    std::vector<std::vector<double>> dihedral_multi_Cn; ///< [eV], {0} for
                                                        ///< non-multi types

    double special_lj[3] = {0.0, 0.0, 0.0};   ///< 1-2 / 1-3 / 1-4 LJ weights
    double special_coul[3] = {0.0, 0.0, 0.0}; ///< 1-2 / 1-3 / 1-4 Coulomb
    bool has_special = false; ///< a special_bonds command was present

    double kspace_accuracy = 0.0; ///< kspace_style pppm/pme accuracy (0=none)
    std::string kspace_style;     ///< e.g. "pppm" ("" = no kspace command)

    bool tail_correction = false; ///< pair_modify tail yes

    /// Commands seen but not interpreted (first token of each line).
    std::vector<std::string> skipped;
};

/// @brief Parse a LAMMPS `units real` coefficient/include file.
///
/// Recognised commands: `pair_style` (lj/cut, lj/cut/coul/long),
/// `pair_coeff i j ε σ [cut]`, `bond_style harmonic` + `bond_coeff`,
/// `angle_style harmonic` + `angle_coeff`, `dihedral_style`
/// (charmm, multi/harmonic, or `hybrid` of exactly those two) +
/// `dihedral_coeff`, `special_bonds` (amber, or `lj a b c coul d e f`),
/// `kspace_style` (pppm/ewald accuracy), `pair_modify` (`tail yes` only).
/// Unrecognised commands are collected into `skipped`, never silently
/// dropped mid-coefficient: malformed rows of a recognised command throw.
///
/// @param path file to parse (LAMMPS `units real`: kcal/mol, Å, degree)
/// @return typed tables in eV / Å / rad
/// @throws std::runtime_error on unreadable file or malformed rows
/// @throws std::invalid_argument on unsupported styles (e.g. a hybrid
///         member other than charmm / multi/harmonic, charmm weight ≠ 0)
LammpsCoeffs load_lammps_coeffs(const std::string& path);

} // namespace cpu
} // namespace ATV
