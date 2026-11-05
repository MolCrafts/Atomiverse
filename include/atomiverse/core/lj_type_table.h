/// @file core/lj_type_table.h
/// @brief One typed LJ (+ r⁻⁴) coefficient table for every backend.
///
/// Ported from LAMMPS `PairLJCut::init_one` (`src/pair_lj_cut.cpp`),
/// `PairLJCutCoulLong::init_one` (`src/KSPACE/pair_lj_cut_coul_long.cpp`)
/// and `Pair::mix_energy` / `Pair::mix_distance` (`src/pair.cpp`, mix
/// arithmetic), LAMMPS checkout `e8c4c278a8`:
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
/// ATV differences, each deliberate: a type with no diagonal row has no LJ
/// site (every pair it is in that has no explicit row gets ε = 0 and a zero
/// cutoff) where LAMMPS refuses to run; the r⁻⁴ attraction is an ATV-native
/// term carried in the same per-pair record.
#pragma once

#include <cmath>
#include <vector>

#include "atomiverse/core/lj_type_coeff.h"

namespace ATV {

/// @brief Resolved coefficients of one ordered type pair (i, j).
///
/// LJ follows LAMMPS: with r6inv = r⁻⁶,
/// F·r̂/r = r⁻² r6inv (lj1 r6inv − lj2) and
/// E = r6inv (lj3 r6inv − lj4) − offset, both for r² < cut_ljsq.
/// The r⁻⁴ term is evaluated by `r4_attraction` for r² < r4_cutsq.
struct LJPairCoeff {
    double epsilon = 0.0;   ///< Resolved well depth ε_ij [eV].
    double sigma = 0.0;     ///< Resolved length σ_ij [Å].
    double cut_lj = 0.0;    ///< Resolved LJ cutoff [Å]; 0 = no LJ site.
    double lj1 = 0.0;       ///< 48 ε σ¹² [eV·Å¹²].
    double lj2 = 0.0;       ///< 24 ε σ⁶ [eV·Å⁶].
    double lj3 = 0.0;       ///< 4 ε σ¹² [eV·Å¹²].
    double lj4 = 0.0;       ///< 4 ε σ⁶ [eV·Å⁶].
    double cut_ljsq = 0.0;  ///< cut_lj² [Å²].
    double offset = 0.0;    ///< LJ energy at cut_lj when shifted [eV].
    double r4_a = 0.0;      ///< S σ⁴ [eV·Å⁴]; 0 = no r⁻⁴ term.
    double r4_cutsq = 0.0;  ///< r_c² of the r⁻⁴ term [Å²]; 0 = none.
    double r4_rc5inv = 0.0; ///< r_c⁻⁵ [Å⁻⁵].
    double r4_e0 = 0.0;     ///< 5 S σ⁴ / r_c⁴ [eV].
};

/// @brief Energy and pair force of one r⁻⁴ interaction.
/// @tparam T Arithmetic type of the evaluating kernel.
template <typename T>
struct R4Term {
    T energy; ///< U(r) [eV].
    T fpair;  ///< −(dU/dr)/r [eV/Å²]: F_i = fpair · (x_i − x_j).
};

#if defined(__CUDACC__)
#define ATV_LJ_TABLE_HD __host__ __device__
#else
#define ATV_LJ_TABLE_HD
#endif

/// @brief Force-shifted r⁻⁴ attraction, one formula for both backends.
///
/// With A = S σ⁴:
/// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ]
///      = −A r⁻⁴ − 4 A r_c⁻⁵ r + 5 A r_c⁻⁴,
/// −dU/dr = −4 A (r⁻⁵ − r_c⁻⁵), so U(r_c) = 0 and F(r_c) = 0. The caller
/// applies the cutoff test r² < r_c².
///
/// @tparam T Arithmetic type of the evaluating kernel.
/// @param r2 Squared separation [Å²], > 0.
/// @param a S σ⁴ [eV·Å⁴] (`LJPairCoeff::r4_a`).
/// @param rc5inv r_c⁻⁵ [Å⁻⁵] (`LJPairCoeff::r4_rc5inv`).
/// @param e0 5 S σ⁴ / r_c⁴ [eV] (`LJPairCoeff::r4_e0`).
/// @return Energy [eV] and pair force −(dU/dr)/r [eV/Å²].
template <typename T>
ATV_LJ_TABLE_HD inline R4Term<T> r4_attraction(T r2, T a, T rc5inv, T e0) {
    const T r2inv = T(1) / r2;
#if defined(__CUDA_ARCH__)
    const T rinv = rsqrt(r2);
#else
    const T rinv = T(1) / std::sqrt(r2);
#endif
    const T r4inv = r2inv * r2inv;
    const T r = r2 * rinv;
    const T energy = -a * r4inv - T(4) * a * rc5inv * r + e0;
    const T fpair = -T(4) * a * (r4inv * r2inv - rc5inv * rinv);
    return R4Term<T>{energy, fpair};
}

#undef ATV_LJ_TABLE_HD

/// @brief LJ long-range tail prefactors (LAMMPS `pair_modify tail yes`).
///
/// With V the cell volume, the correction energy is `energy / V` and each
/// diagonal virial component gains `virial / V`, so the pressure gains
/// `virial / V²` (LAMMPS `ComputePressure::virial_compute`).
struct LJTail {
    double energy = 0.0; ///< Σ etail_ij [eV·Å³].
    double virial = 0.0; ///< Σ ptail_ij [eV·Å³].
};

/// @brief Tail of one ordered type pair, LAMMPS `PairLJCut::init_one`:
///        etail_ij = 8π N_i N_j ε σ⁶ (σ⁶ − 3 r_c⁶) / (9 r_c⁹),
///        ptail_ij = 16π N_i N_j ε σ⁶ (2σ⁶ − 3 r_c⁶) / (9 r_c⁹)
///        (homogeneous g(r) = 1 beyond each pair's own LJ cutoff).
/// @param epsilon ε_ij [eV].
/// @param sigma σ_ij [Å].
/// @param cut_lj The pair's LJ cutoff r_c [Å]; ≤ 0 = no LJ site (no tail).
/// @param n_i Global count of type i atoms [dimensionless].
/// @param n_j Global count of type j atoms [dimensionless].
/// @return The pair's prefactors; summed over ordered (i, j) they give the
///         LAMMPS `Pair::init` total (off-diagonal pairs counted twice).
LJTail lj_tail_pair(double epsilon, double sigma, double cut_lj, double n_i,
                    double n_j);

/// @brief Typed LJ coefficient table with LAMMPS `init_one` semantics.
///
/// Built from the rows a pair style was constructed with:
/// - an explicit (i, j) row uses its own ε, σ and cutoff (unset = the
///   global cutoff);
/// - a pair with no row mixes from the diagonals: ε geometric, σ and the
///   cutoff arithmetic (LAMMPS `pair_modify mix arithmetic`);
/// - a type with no diagonal row has no LJ site: every unlisted pair it is
///   in has ε = 0 and cutoff 0;
/// - lj1..lj4, cut_ljsq and (when shifted) offset as in LAMMPS.
/// r⁻⁴ rows land on their own pairs only. Later rows override earlier ones.
class LJTypeTable {
  public:
    /// @brief Validate and store the LJ rows.
    /// @param coeffs LJ rows; must not be empty.
    /// @param cutoff Global LJ cutoff [Å] (diagonal default), > 0.
    /// @throws std::invalid_argument on an empty list, a negative type,
    ///         ε < 0, σ ≤ 0 or a non-positive / non-finite cutoff.
    LJTypeTable(std::vector<LJTypeCoeff> coeffs, double cutoff);

    /// @brief Add or replace one LJ row (a later row overrides).
    /// @param row LJ row, validated as in the constructor.
    void set_coeff(const LJTypeCoeff& row);

    /// @brief Replace the r⁻⁴ rows.
    /// @param rows r⁻⁴ rows; empty removes the term.
    /// @throws std::invalid_argument on a negative type, a non-finite
    ///         strength, σ ≤ 0 or r_c ≤ 0.
    void set_r4(std::vector<R4TypeCoeff> rows);

    /// @brief Global LJ cutoff given at construction.
    /// @return Cutoff [Å].
    double cutoff() const { return cutoff_; }

    /// @brief Largest LJ or r⁻⁴ cutoff any pair can reach.
    ///
    /// Mixed cutoffs are arithmetic means of diagonal ones, so the maximum
    /// is attained on a row. Known before `build`, so a style can report its
    /// neighbor cutoff at construction.
    /// @return Cutoff [Å].
    double max_cutoff() const;

    /// @brief One past the largest type any row names.
    /// @return Type count [dimensionless].
    int named_types() const;

    /// @brief Whether any r⁻⁴ row is present.
    /// @return True when the r⁻⁴ term is active.
    bool has_r4() const { return !r4_.empty(); }

    /// @brief Resolve every ordered pair of `max(n_types, named_types())`
    ///        types (LAMMPS `init_one` over all i, j).
    /// @param n_types Minimum type count to cover (e.g. 1 + max Frame type).
    /// @param shift LAMMPS `pair_modify shift`: subtract the LJ energy at
    ///        each pair's own cutoff.
    void build(int n_types, bool shift);

    /// @brief Number of types the last `build` covered.
    /// @return Type count; 0 before `build`.
    int n_types() const { return n_types_; }

    /// @brief Resolved coefficients of (i, j) after `build`.
    /// @param i First type, 0 ≤ i < n_types().
    /// @param j Second type, 0 ≤ j < n_types().
    /// @return The pair record.
    const LJPairCoeff& operator()(int i, int j) const {
        return pairs_[static_cast<std::size_t>(i) * n_types_ + j];
    }

    /// @brief Row-major n_types() × n_types() pair records after `build`.
    /// @return The flattened table.
    const std::vector<LJPairCoeff>& pairs() const { return pairs_; }

    /// @brief Whether the built table is pure Lorentz–Berthelot on the
    ///        diagonals with every interacting pair at the global cutoff and
    ///        no r⁻⁴ term — the condition for per-atom (σ/2, 2√ε) kernels.
    /// @return True when such a per-atom form reproduces every pair.
    bool lorentz_berthelot_uniform() const;

    /// @brief LJ tail prefactors of the built table over global type counts
    ///        (LAMMPS `Pair::init` sum of `init_one` etail_ij / ptail_ij,
    ///        each pair at its own LJ cutoff; the r⁻⁴ term has no tail).
    /// @param counts Global atom count of each type, `counts.size()` ≥
    ///        n_types() [dimensionless]; types past the table have no LJ.
    /// @return Energy and virial prefactors [eV·Å³].
    LJTail tail(const std::vector<double>& counts) const;

  private:
    std::vector<LJTypeCoeff> rows_;
    std::vector<R4TypeCoeff> r4_;
    double cutoff_ = 0.0;
    int n_types_ = 0;
    std::vector<LJPairCoeff> pairs_;
};

} // namespace ATV
