/// @file core/pppm_accuracy.h
/// @brief Accuracy → (α, mesh) for ik PPPM, shared by both backends.
///
/// Ported from LAMMPS `src/KSPACE/pppm.cpp` (`PPPM::PPPM` `acons` table,
/// `PPPM::init`, `set_grid_global`, `compute_df_kspace`,
/// `estimate_ik_error`, `adjust_gewald`, `newton_raphson_f`, `derivf`,
/// `final_accuracy`) and `src/kspace.cpp` (`two_charge`, `lamda2xT`,
/// `x2lamdaT`), checkout e8c4c278a8:
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.
///
/// The request mirrors `kspace_style pppm <accuracy>` with the
/// `kspace_modify gewald / mesh / order` overrides, for the ik
/// differentiation (`differentiation_flag = 0`, no `stagger`), a 3-d periodic
/// cell (`slab_volfactor = 1`) and point charges (`qdist = 0`).
/// `two_charge_force` and `q2 = Σq² · qqrd2e` come from `request.unit_system`
/// (LAMMPS `KSpace::two_charge` and `PPPM::qsum_qsq`). The default system is
/// lj, where both factors are 1. The mesh this returns is `PPPM::init`.
/// A later cell change refreshes the Green's function at those counts
/// (`PPPM::setup`) and does not re-run `set_grid_global`.
///
/// Deviations, each deliberate:
///  * The Atomiverse stencil wraps one period, so a mesh axis is never below
///    order + 1; a smaller LAMMPS mesh is raised to the smallest 2·3·5 size
///    ≥ order + 1 (the rule `KSpacePPPM` applies to a spacing-derived mesh)
///    before α is adjusted. LAMMPS runs such meshes.
///  * The final estimate has no `estimate_table_accuracy` term: Atomiverse
///    evaluates erfc analytically, i.e. LAMMPS `pair_modify table 0`.
///  * The cell is LAMMPS-triclinic only when a tilt is nonzero; LAMMPS also
///    applies the triclinic mesh scaling to a `triclinic` box with zero tilts.
///  * LAMMPS reduces the order when a stencil reaches beyond the nearest
///    neighbor processor; that depends on the domain decomposition and is not
///    modelled — the order is the requested one.
#pragma once

#include "atomiverse/core/unit_system.h"

namespace ATV::longrange {

/// LAMMPS `MAXORDER` for the accuracy estimate (`acons` rows 1 … 7).
inline constexpr int kPppmAccuracyMaxOrder = 7;

/// @brief One `kspace_style pppm <accuracy>` setup (LAMMPS `PPPM::init`).
struct PppmAccuracyRequest {
    /// Relative RMS force accuracy (LAMMPS `accuracy_relative`,
    /// dimensionless, in (0, 1]). The absolute target is this times
    /// `unit_system.two_charge_force()` (LAMMPS `KSpace::two_charge`).
    double accuracy = 0.0;
    /// Real-space Coulomb cutoff [Å] (LAMMPS `cut_coul`), > 0.
    double cutoff = 0.0;
    /// Charge-assignment order (count), in [3, 7] (`kspace_modify order`;
    /// LAMMPS allows 2, the Atomiverse solvers start at 3).
    int order = 6;
    /// Row-major cell H [Å], lattice vectors in columns (s = H⁻¹ r).
    double cell[9] = {};
    /// Number of atoms (count), > 0 (LAMMPS `atom->natoms`).
    long long natoms = 0;
    /// Σ q_i² over every atom [e²] (LAMMPS `qsqsum`).
    double qsqsum = 0.0;
    /// Fixed α [1/Å] (`kspace_modify gewald`); ≤ 0 derives it.
    double alpha = 0.0;
    /// Fixed mesh (`kspace_modify mesh`); all three > 0 fixes it, all zero
    /// derives it. Raised to 2·3·5 sizes as in LAMMPS.
    int mesh[3] = {0, 0, 0};
    /// LAMMPS `units` row for `two_charge` and `qqrd2e`. Default lj.
    units::UnitSystem unit_system = units::UnitSystem::lj();
};

/// @brief What LAMMPS `PPPM::init` settles on and prints.
struct PppmAccuracyResult {
    double alpha = 0.0;      ///< G vector α [1/Å] (`g_ewald`).
    int mesh[3] = {0, 0, 0}; ///< Grid points per axis (`n{x,y,z}_pppm`).
    int order = 0;           ///< Stencil order (count).
    /// Estimated absolute RMS force accuracy in the request's force unit
    /// (`final_accuracy`). lj: the force unit is ε/σ, and this equals the
    /// relative accuracy times 1.
    double accuracy_absolute = 0.0;
    /// `accuracy_absolute / two_charge_force` (dimensionless).
    double accuracy_relative = 0.0;
    double df_real = 0.0;   ///< Real-space RMS force error [request force].
    double df_kspace = 0.0; ///< k-space RMS force error [request force].
};

/// @brief LAMMPS `estimate_ik_error`: ik RMS force error of one axis.
/// @param h      Mesh spacing along the axis [Å] (or its lamda equivalent).
/// @param prd    Box extent along the axis [Å].
/// @param natoms Atom count (count); 0 returns 0.
/// @param q2     Σq² · qqrd2e, in the request's energy·length.
/// @param alpha  α [1/length].
/// @param order  Stencil order, in [1, 7].
/// @return RMS force error contribution in the request's force unit.
double pppm_estimate_ik_error(double h, double prd, long long natoms, double q2,
                              double alpha, int order);

/// @brief LAMMPS `PPPM::init` for ik PPPM: α, mesh and the final estimate.
/// @throws std::invalid_argument on a non-positive accuracy, cutoff, natoms
///         or cell volume, an order outside [3, 7], a mesh that is neither
///         all-zero nor all-positive, or Σq² = 0 without a fixed α (LAMMPS
///         "Must use 'kspace_modify gewald' for uncharged system").
/// @throws std::runtime_error when the Newton iteration on α does not
///         converge (LAMMPS "Could not compute g_ewald").
PppmAccuracyResult pppm_from_accuracy(const PppmAccuracyRequest& request);

} // namespace ATV::longrange
