/// @file cpu/potentials/pair_lj_cut.h
/// @brief LAMMPS-style PairLJCut — 12-6 with shifted potential (CPU).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/lj_type_coeff.h"
#include "atomiverse/cpu/potentials/pair.h"
#include "atomiverse/cpu/potentials/special_scale.h"

namespace ATV {
namespace cpu {

/// @brief Lennard-Jones 12-6 with shifted potential.
///
/// V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] − V_shift
/// F(r) = (24ε/r)·[2·(σ/r)¹² − (σ/r)⁶]·r̂
///
/// Optional special-bonds (LAMMPS convention): 1-2 / 1-3 / 1-4 weights
/// multiply pair energy and force. Scale 0 drops the pair; Amber/GAFF uses
/// `set_special_bonds(0.0, 0.0, 0.5)`. Lists come from `kAtom_nspecial` /
/// `kAtom_special` (via `Frame::build_special_list`). Default: no specials.
///
/// Thread-parallel with per-thread force reduction buffers
/// (Newton's 3rd law). xsimd ket-loop in the inner pair walk (Track B.2,
/// 1.21× over scalar on NEON64).
class PairLJCut : public Pair {
  public:
    /// @param epsilon Well depth [eV].
    /// @param sigma   Size parameter [Å].
    /// @param cutoff  Cutoff distance [Å].
    PairLJCut(double epsilon, double sigma, double cutoff);

    /// @brief Per-type constructor: one Lennard-Jones system, fully specified.
    ///
    /// Use this whenever the system has more than one element. The coefficient table is
    /// an argument, not something you attach afterwards, so the object is complete the
    /// moment it is constructed — symmetric with the single-type constructor above, and
    /// with no state in which a `PairLJCut` exists but cannot be used.
    ///
    /// It takes NO system-wide epsilon/sigma, because there is no such thing in a
    /// multi-element system. Supplying dummy values for the table to override would be
    /// a fallback value, and this engine does not have those.
    ///
    /// Entries you do NOT give are filled by **Lorentz-Berthelot** mixing from the
    /// diagonal (arithmetic mean of sigma, geometric mean of epsilon), so listing the
    /// like-pairs is usually enough. An explicit (i,j) entry overrides the mixed value.
    /// A type with **no diagonal to mix from has no LJ site**: every pair involving it
    /// gets epsilon = 0 and does not interact. That is deliberate — TIP3P's hydrogens
    /// have no LJ site, and falling back to some other sigma would silently reintroduce
    /// the bug this exists to fix.
    ///
    /// @code
    ///   // TIP3P: the LJ site is on oxygen only.
    ///   sim.set_evaluator(cpu::PairLJCut{{{8, 8, eps_OO, sigma_OO}}, cutoff});
    ///
    ///   // Cu-Ni alloy: give the like-pairs, let Lorentz-Berthelot mix the rest.
    ///   sim.set_evaluator(cpu::PairLJCut{{{29, 29, eps_Cu, sig_Cu},
    ///                               {28, 28, eps_Ni, sig_Ni}}, cutoff});
    /// @endcode
    ///
    /// **Performance.** The single-type constructor keeps the hoisted xsimd ket loop —
    /// the kernel benchmarked against LAMMPS. Per-type coefficients cannot hoist out of
    /// the inner loop, so this constructor selects a SCALAR path. Correct, not fast;
    /// vectorizing it needs a SIMD gather over the table, which nobody has needed yet.
    ///
    /// @param coeffs Type-pair coefficients; must not be empty.
    /// @param cutoff Interaction cutoff [Angstrom].
    /// @throws std::invalid_argument if `coeffs` is empty or a sigma/cutoff is <= 0.
    PairLJCut(std::vector<LJTypeCoeff> coeffs, double cutoff);

    /// @brief Change ONE entry of the coefficient table after construction.
    ///
    /// A mutator, not a builder step: the table already exists, this edits a cell of it.
    /// Adding a type pair the constructor did not mention is allowed (the table grows),
    /// which is what makes it useful for a parameter scan.
    ///
    /// @param type_i First atom type.
    /// @param type_j Second atom type.
    /// @param epsilon Well depth [eV].
    /// @param sigma Length scale [Angstrom].
    /// @throws std::logic_error if this PairLJCut is single-type (there is no table).
    PairLJCut& set_type_coeff(int type_i, int type_j, double epsilon,
                              double sigma) &;
    PairLJCut set_type_coeff(int type_i, int type_j, double epsilon,
                             double sigma) &&;

    /// @brief True when a per-type coefficient table is active (scalar path).
    bool is_multi_type() const { return n_types_ > 0; }

  private:
    /// @brief Complete the table by Lorentz-Berthelot mixing and precompute
    ///        sigma^6 + the per-pair energy shift. Called once from build().
    void finalize_type_table_();

  public:
    std::string name() const override { return "lj"; }

    /// @brief Enable/disable LAMMPS-style "special bonds" exclusions.
    /// See the class-level special-bonds description for full semantics.
    PairLJCut& set_special_bonds(double s12, double s13, double s14) &;
    PairLJCut set_special_bonds(double s12, double s13, double s14) &&;

    /// @brief Enable/disable per-pair virial (stress) accumulation.
    ///
    /// The pair virial (6 Voigt components) feeds the pressure / barostat
    /// path only; a plain NVE run never consumes it. Disabling it when no
    /// stress consumer is active mirrors LAMMPS's per-step `vflag` gating and
    /// skips the horizontal reduce per atom, the scalar-tail products and the
    /// flush to `kStress`. Forces and energy are unaffected — bit-identical
    /// either way. Default: enabled.
    ///
    /// The six **batch** products inside the xsimd ket loop are computed
    /// regardless: the branch that used to skip them made GCC 14.3 at
    /// `-O3 -march=native` drop most of the reduction (see the comment at that
    /// site in `pair_lj_cut.cpp`). A 60 %-short stress is not a trade worth
    /// six FMAs.
    PairLJCut& set_compute_stress(bool on) &;
    PairLJCut set_compute_stress(bool on) &&;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Resident heap bytes: per-thread force accumulators (OMP>1 only)
    /// plus the exclusion map + filtered neighbor list (special-bonds runs).
    /// Classified under the "Pair" category by the driver memory summary.
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    /// @brief Rebuild `filtered_{numneigh,firstneigh,neighbors}` from
    /// `pairs` by walking each owner i's neighbors, resolving partner
    /// tags, and dropping any pair (i,j) whose j_tag appears in i's
    /// excluded set. Called from `compute()` only when `pairs.version`
    /// has advanced past the last snapshot we filtered.
    void rebuild_filtered_list(const Frame& frame, const md::PairList& pairs);

    double epsilon_, sigma_;

    /// Per-type coefficient tables, flattened `n_types_ x n_types_`. Empty = the
    /// single-epsilon fast path. Filled lazily by `set_type_coeffs`, then completed by
    /// Lorentz-Berthelot mixing at `build()`.
    bool per_type_mode_ = false; ///< Constructed with the cutoff-only ctor.
    int n_types_ = 0;
    std::vector<double> eps_tab_;   ///< eps_ij [eV]
    std::vector<double> sigma_tab_; ///< sigma_ij [A]
    std::vector<double> sig6_tab_;  ///< sigma_ij^6 [A^6]
    std::vector<double>
        shift_tab_;             ///< energy shift at the cutoff, per pair [eV]
    std::vector<char> set_tab_; ///< 1 = explicitly set by the user
    double sigma6_, cutoff_sq_, v_shift_;

    // Special-bonds weights (LAMMPS special_bonds). Amber: (0, 0, 0.5).
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;

    // When false, the inner loop skips the 6-component pair virial and the
    // final stress write (LAMMPS-style vflag gating). Forces/energy are
    // unchanged. Default true to preserve behavior for pressure/NPT runs.
    bool compute_stress_ = true;

    /// True when any of s12/s13/s14 != 1 — activates filtered CSR + scales.
    bool any_special_ = false;
    /// True when any scale is in (0,1) — forces scalar path (scale × E,F).
    bool has_partial_special_ = false;
    md::SpecialScaleTable special_;

    // Per-thread force reduction buffers
    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;

    // Special-filtered neighbor CSR (+ per-pair scale parallel to neighbors).
    std::vector<int> filtered_neighbors_;
    std::vector<int> filtered_numneigh_;
    std::vector<int> filtered_firstneigh_;
    std::vector<double> filtered_scale_;
    std::vector<double> filtered_shift_x_;
    std::vector<double> filtered_shift_y_;
    std::vector<double> filtered_shift_z_;
    bool filtered_has_shifts_ = false;
    int filtered_inum_ = 0;
    unsigned long long filtered_version_ = 0;
};

} // namespace cpu
} // namespace ATV
