/// @file cpu/scf/grid.h
/// @brief Molecular integration grid for DFT: Lebedev angular + radial + Becke partitioning.
///
/// Atom-centered quadrature grids for numerical XC integration.
///
/// References:
///   - Lebedev-Laikov: Doklady Math. 59, 477 (1999)
///   - Mura-Knowles radial: JCP 104, 9848 (1996)
///   - Becke partitioning: JCP 88, 2547 (1988)
#pragma once

#include "atomiverse/cpu/scf/native_integrals.h"
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Single quadrature grid point with position and weight.
struct GridPoint {
    double x, y, z; ///< Position in Bohr.
    double
        weight; ///< Quadrature weight (includes radial, angular, and Becke partition).
    int parent_atom; ///< Index of the atom this point is centered on (the
                     ///< atom whose atomic quadrature generated it). The
                     ///< point rides rigidly with this atom; needed by the
                     ///< XC grid-response nuclear gradient.
};

/// @brief A single radial quadrature point.
struct RadialPoint {
    double r; ///< Radius in Bohr.
    double w; ///< Quadrature weight, includes r^2 Jacobian (so integrals
              ///< of f(r) over 3-space are sum_i w_i * f(r_i) * 4*pi).
};

/// @brief Named DFT integration-grid quality level.
///
/// Selects the radial scheme, the angular Lebedev order set, and the
/// pruning policy for atom-centered DFT quadrature grids.
///   - `SG0`:        legacy uniform-order grid (`n_radial`, `n_angular`
///                   honoured verbatim, Treutler-Ahlrichs radial, no
///                   per-shell angular pruning). Kept bit-for-bit
///                   compatible with the pre-pruning default (50, 110).
///   - `SG1`:        Gill-Johnson-Pople SG-1 standard grid
///                   (Chem. Phys. Lett. 209, 506 (1993)) -- Euler-Maclaurin
///                   radial (50 points), 5-region Lebedev pruning with
///                   orders {6, 38, 86, 194, 86}, Becke size-adjusted
///                   atomic radii. Design accuracy ~0.2 kcal/mol.
///   - `MatchPySCF`: high-accuracy grid that reproduces PySCF's default
///                   `level=3` -- Treutler-Ahlrichs M4 radial with
///                   per-element (n_rad) counts and nwchem-style angular
///                   pruning bounded by the level-3 maximum order
///                   (590 for Z <= 10, 770 for Z > 10).
enum class GridLevel { SG0, SG1, MatchPySCF };

/// @brief Molecular integration grid: atom-centered DFT quadrature points.
struct MolecularGrid {
    std::vector<GridPoint> points; ///< All grid points.
    int n_radial = 0;              ///< Radial points per atom.
    int n_angular = 0;             ///< Lebedev angular points per atom.

    /// @brief Empty grid (no points); populated later by copy/move
    ///        assignment (e.g. a supplied fixed grid or a level-built one).
    MolecularGrid() = default;

    /// @brief Build atom-centered molecular grid at a named quality level.
    ///
    /// Level-aware grid builder: selects the radial scheme and the
    /// per-radial-shell angular Lebedev order from `level` (see
    /// ::GridLevel). For `GridLevel::SG0` uses the supplied `n_radial` /
    /// `n_angular`. For `SG1` / `MatchPySCF` the `n_radial` / `n_angular`
    /// arguments are ignored in favour of the level's own constants.
    ///
    /// @param atoms Array of atom descriptors (Z, position in Bohr).
    /// @param natm  Number of atoms.
    /// @param level Named grid quality level.
    /// @param n_radial Radial point count (only used for `GridLevel::SG0`).
    /// @param n_angular Angular point count (only used for `GridLevel::SG0`).
    MolecularGrid(const native::Atom* atoms, int natm, GridLevel level,
                  int n_radial = 50, int n_angular = 110);
};

/// @brief Lebedev angular order to use at a single radial shell.
///
/// Implements the per-shell angular pruning for ::GridLevel::SG1 (the
/// 5-region alpha-quadruple table of Gill-Johnson-Pople) and
/// ::GridLevel::MatchPySCF (nwchem-style pruning bounded by the per-element
/// level-3 maximum order). For ::GridLevel::SG0 it always returns the
/// uniform order requested by the caller.
///
/// @param level        Named grid quality level.
/// @param Z            Atomic number of the shell's parent atom.
/// @param r            Radius of the current radial shell in Bohr.
/// @param R            Atom-size-adjusted radius in Bohr (Becke convention).
/// @param radial_index Zero-based index of the radial shell.
/// @param n_radial     Total number of radial shells for this atom.
/// @param uniform_order Angular order to return for ::GridLevel::SG0.
/// @return Lebedev angular order; one of
///         {6, 38, 50, 86, 110, 194, 302, 590, 770}.
int lebedev_order_for_shell(GridLevel level, int Z, double r, double R,
                            int radial_index, int n_radial, int uniform_order);

/// @brief Euler-Maclaurin radial quadrature for the SG-1 parent grid.
///
/// Implements the Euler-Maclaurin radial mapping used by the SG-1 standard
/// grid (Gill, Johnson, Pople, Chem. Phys. Lett. 209, 506 (1993)):
///   r_i = R * i^2 / (n + 1 - i)^2,   i = 1..n
/// with the matching quadrature weight including the r^2 Jacobian, so the
/// 3-space integral of f(r) is sum_i w_i * f(r_i) after multiplication by
/// the angular weight times 4*pi.
///
/// @param n Number of radial points.
/// @param R Atom-size-adjusted radius in Bohr.
/// @return Vector of n radial points and weights.
std::vector<RadialPoint> radial_grid_euler_maclaurin(int n, double R);

/// @brief Test-only accessor for the internal Becke partition weights.
///
/// Exposes `becke_partition_weights` (otherwise file-static in grid.cpp)
/// so the partition-of-unity acceptance test can probe it directly. Not
/// intended for production callers -- the `MolecularGrid` constructor applies
/// the partition internally.
///
/// @param atoms   Array of atom descriptors (Z, position in Bohr).
/// @param natm    Number of atoms.
/// @param rx      Grid-point x-coordinate in Bohr.
/// @param ry      Grid-point y-coordinate in Bohr.
/// @param rz      Grid-point z-coordinate in Bohr.
/// @param weights Output buffer [natm]; receives normalized cell weights.
void becke_partition_weights_for_test(const native::Atom* atoms, int natm,
                                      double rx, double ry, double rz,
                                      double* weights);

/// @brief Nuclear-coordinate derivative of the Becke fuzzy-cell quadrature weight.
///
/// Verbatim port of PySCF's `grids_response_becke` / `gen_grid_partition`
/// (`pyscf/grad/rks.py`) and `get_smooth_function_derivative`, restricted to
/// the `original_becke` switching scheme (iterated `f(x) = (3x - x^3)/2`
/// applied three times). For one grid point `r_g` whose parent (owning) atom
/// is `parent`, it returns `dw_g/dR_A` -- the derivative of the *normalized*
/// Becke partition weight of `parent` with respect to every nuclear
/// coordinate `R_{A,d}`.
///
/// The returned quantity is the normalized cell-weight derivative WITHOUT the
/// radial/angular `vol` factor; the full grid-weight derivative is recovered
/// by `dW_g/dR_A = (W_g / w_parent) * dw_g/dR_A`, where `W_g` is the stored
/// grid weight and `w_parent` the normalized partition weight of `parent`
/// (the value `becke_partition_weights_for_test` returns for `parent`).
///
/// The self term `dw_A/dR_A` is obtained from translational invariance,
/// `dw_A/dR_A = -sum_{B != A} dw_A/dR_B`, exactly as PySCF does -- a grid
/// point sitting on an atom drives `mu -> 0` and a direct differentiation
/// there is numerically unstable. The result therefore satisfies the
/// rigid-shift identity `sum_A dw_g/dR_A = 0` per axis by construction.
///
/// The size-adjustment `a_ij` uses the same atomic-radii table and clamping
/// as the production `becke_partition_weights`, selected by `level`:
/// `GridLevel::SG0` / `SG1` use the original Becke-1988 heteronuclear
/// adjustment, `GridLevel::MatchPySCF` uses PySCF's treutler adjustment.
///
/// @param atoms   Array of atom descriptors (Z, position in Bohr).
/// @param natm    Number of atoms.
/// @param rx      Grid-point x-coordinate in Bohr.
/// @param ry      Grid-point y-coordinate in Bohr.
/// @param rz      Grid-point z-coordinate in Bohr.
/// @param parent  Index of the grid point's parent (owning) atom, in [0,natm).
/// @param level   Grid quality level; selects the size-adjustment scheme so
///                the derivative stays consistent with `becke_partition_weights`.
/// @param[out] dw_out Output buffer [natm*3]; `dw_out[A*3 + d]` receives
///                `d(w_parent)/dR_{A,d}` in units of 1/Bohr.
/// @return The normalized Becke partition weight of `parent` at this grid
///         point (dimensionless, in [0,1]) -- the conversion factor needed
///         to lift `dw_out` to the full grid-weight derivative
///         `dW_g/dR_A = (W_g / w_parent) * dw_out`.
double becke_weight_derivative(const native::Atom* atoms, int natm, double rx,
                               double ry, double rz, int parent,
                               GridLevel level, double* dw_out);

/// @brief Bragg-Slater atomic radius for element Z, in Bohr.
///
/// Throws std::invalid_argument for Z outside the supported range [1, 86].
/// Values are Slater 1964 (JCP 41, 3199) with noble-gas radii from
/// Clementi-Raimondi-Reinhardt 1967 (JCP 47, 1300) and At from Cordero 2008.
///
/// @param Z Atomic number (1 <= Z <= 86).
/// @return Bragg-Slater radius in Bohr.
double bragg_slater_radius(int Z);

/// @brief Mura-Knowles radial quadrature on (0, infinity).
///
/// Reference: M. E. Mura and P. J. Knowles, "Improved radial grids for
/// quadrature in molecular density-functional calculations", J. Chem.
/// Phys. 104, 9848 (1996), DOI 10.1063/1.471749.
///
/// Uses the log3 mapping r = -R * ln(1 - x^3) with x = (i + 0.5)/n
/// (uniform open Newton-Cotes) and weights w_i = R * 3 x_i^2 / (n (1 - x_i^3)) * r_i^2.
/// Matches PySCF's default radial grid scheme.
///
/// @param n Number of radial points.
/// @param R_bragg Element-specific Bragg-Slater radius in Bohr.
/// @return Vector of n radial points and weights.
std::vector<RadialPoint> radial_grid_mura_knowles(int n, double R_bragg);

} // namespace scf
} // namespace cpu
} // namespace ATV
