/// @file cpu/potentials/pair_lj_cut.h
/// @brief LAMMPS-style PairLJCut — 12-6 with shifted potential (CPU).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair.h"
#include "atomiverse/cpu/potentials/pair_partition.h"
#include "atomiverse/cpu/potentials/special_scale.h"

namespace ATV {
namespace cpu {

/// @brief Lennard-Jones 12-6, shifted at the cutoff by default.
///
/// V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] − V_shift   (V_shift = 0 with set_energy_shift(false))
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
    /// The table is completed with LAMMPS `PairLJCut::init_one` semantics
    /// (`core/lj_type_table.h`): a pair you do NOT give is mixed from the diagonal by
    /// **Lorentz-Berthelot** (geometric epsilon, arithmetic sigma) and gets the
    /// arithmetic mean of the two diagonal cutoffs; an explicit (i,j) row overrides
    /// the mix and uses its own `cutoff` (unset = `cutoff` below). A type with **no
    /// diagonal to mix from has no LJ site**: every pair involving it that has no
    /// explicit row gets epsilon = 0 and does not interact. That is deliberate —
    /// TIP3P's hydrogens have no LJ site, and falling back to some other sigma would
    /// silently reintroduce the bug this exists to fix. With the energy shift on (the
    /// default, `set_energy_shift`) every pair is shifted to zero at its own cutoff.
    /// The neighbor cutoff this style reports is the largest
    /// per-pair cutoff (LJ or r⁻⁴).
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
    /// @param cutoff Global LJ cutoff [Å] for rows that name none.
    /// @throws std::invalid_argument if `coeffs` is empty, a type is negative,
    ///         epsilon < 0, or a sigma/cutoff is <= 0.
    PairLJCut(std::vector<LJTypeCoeff> coeffs, double cutoff);

    /// @brief Change ONE entry of the coefficient table after construction.
    ///
    /// A mutator, not a builder step: the table already exists, this edits a cell of it.
    /// Adding a type pair the constructor did not mention is allowed (the table grows),
    /// which is what makes it useful for a parameter scan. A larger per-pair cutoff
    /// raises the reported neighbor cutoff.
    ///
    /// @param row Type pair, epsilon [eV], sigma [Å] and optional cutoff [Å].
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::logic_error if this PairLJCut is single-type (there is no table).
    PairLJCut& set_type_coeff(const LJTypeCoeff& row) &;
    PairLJCut set_type_coeff(const LJTypeCoeff& row) &&;

    /// @brief Add a force-shifted r⁻⁴ attraction on listed type pairs.
    ///
    /// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
    /// else 0, so U and F vanish at r_c (`ATV::r4_attraction`). Evaluated in
    /// the same per-pair table walk as LJ, scaled by the same special-bond
    /// weight, and part of energy and virial. Pairs without a row have no
    /// such term. An ATV-native term, not a LAMMPS style. Replaces any rows
    /// set before; the neighbor cutoff grows to cover the largest r_c.
    ///
    /// @param rows Rows {type_i, type_j, S [eV], σ [Å], r_c [Å]}.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::logic_error on a single-type PairLJCut (no type table).
    /// @throws std::invalid_argument on a negative type, non-finite S, or
    ///         σ / r_c <= 0.
    PairLJCut& set_r4_attraction(std::vector<R4TypeCoeff> rows) &;
    PairLJCut set_r4_attraction(std::vector<R4TypeCoeff> rows) &&;

    /// @brief True when a per-type coefficient table is active (scalar path).
    bool is_multi_type() const { return table_.has_value(); }

    std::string name() const override { return "lj"; }
    /// A pure function of the geometry (no trial-dependent state).
    bool trial_invariant() const override { return true; }

    /// @brief Enable/disable LAMMPS-style "special bonds" exclusions.
    /// See the class-level special-bonds description for full semantics.
    PairLJCut& set_special_bonds(double s12, double s13, double s14) &;
    PairLJCut set_special_bonds(double s12, double s13, double s14) &&;

    /// @brief LAMMPS `pair_modify tail yes` — analytic LJ tail energy and
    ///        pressure beyond each pair's own cutoff (homogeneous g(r) = 1,
    ///        `ATV::lj_tail_pair`; the r⁻⁴ term has no tail).
    ///
    /// Global per-type counts (every atom one type on the single-type form)
    /// are gathered on the first evaluation after `build()`; every step adds
    /// E = etail/V to the energy and ptail/V to each diagonal virial
    /// component at the current box volume. Independent of the energy
    /// shift, as in LAMMPS. Default off (same on `cuda::PairLJCut`).
    /// @param on Whether the tail is added.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    PairLJCut& set_tail_correction(bool on) &;
    PairLJCut set_tail_correction(bool on) &&;

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

    /// @brief LAMMPS `pair_modify shift`: subtract V(r_c) from every counted
    ///        pair (per-type form: at that pair's own cutoff) so the energy
    ///        is continuous at the cutoff. Read at `build()`. Forces are
    ///        unaffected. Default on (the historical behaviour of this style);
    ///        a force field that states no shift runs it off.
    PairLJCut& set_energy_shift(bool on) &;
    PairLJCut set_energy_shift(bool on) &&;

    /// @brief The per-type table reads the neighbour's force-field type
    /// (`k_type[j]`); the single-pair form reads no type at all.
    static constexpr std::array kTypedGhosts = {
        GhostField::read_stable(kAtom_type, "PairLJCut"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        if (table_)
            return kTypedGhosts;
        return {};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Resident heap bytes: per-chunk force accumulators (OMP>1 only)
    /// plus the exclusion map + filtered neighbor list (special-bonds runs).
    /// Classified under the "Pair" category by the driver memory summary.
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    /// @brief Report the largest per-pair cutoff as this style's cutoff.
    void refresh_cutoff_();

    /// @brief Rebuild `filtered_{numneigh,firstneigh,neighbors}` from
    /// `pairs` by walking each owner i's neighbors, resolving partner
    /// tags, and dropping any pair (i,j) whose j_tag appears in i's
    /// excluded set. Called from `compute()` only when `pairs.version`
    /// has advanced past the last snapshot we filtered.
    void rebuild_filtered_list(const Frame& frame, const md::PairList& pairs);

    double epsilon_, sigma_;

    /// Per-type coefficient table (LAMMPS `init_one`, `core/lj_type_table.h`).
    /// Empty = the single-epsilon fast path. Resolved at `build()` over every
    /// type the Frame carries.
    std::optional<LJTypeTable> table_;
    int n_types_ = 0; ///< Types the resolved table covers (0 before build).
    // Neighbor build whose row types were last checked against the table.
    const md::PairList* types_source_ = nullptr;
    unsigned long long types_version_ = 0;
    double sigma6_, cutoff_sq_, v_shift_;

    /// `pair_modify shift` — see set_energy_shift. Applied at build().
    bool energy_shift_ = true;

    // Special-bonds weights (LAMMPS special_bonds). Amber: (0, 0, 0.5).
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;

    bool tail_ = false;
    bool tail_ready_ = false;
    /// E_tail = etail_pref_/V; each diagonal virial += ptail_pref_/V.
    double etail_pref_ = 0.0, ptail_pref_ = 0.0;

    // When false, the inner loop skips the 6-component pair virial and the
    // final stress write (LAMMPS-style vflag gating). Forces/energy are
    // unchanged. Default true to preserve behavior for pressure/NPT runs.
    bool compute_stress_ = true;

    /// True when any of s12/s13/s14 != 1 — activates filtered CSR + scales.
    bool any_special_ = false;
    /// True when any scale is in (0,1) — forces scalar path (scale × E,F).
    bool has_partial_special_ = false;
    md::SpecialScaleTable special_;

    // Special-filtered neighbor CSR (+ per-pair scale parallel to neighbors).
    std::vector<int> filtered_neighbors_;
    std::vector<int> filtered_numneigh_;
    std::vector<int> filtered_firstneigh_;
    std::vector<double> filtered_scale_;
    int filtered_inum_ = 0;
    // Null source denotes an unbuilt cache; version zero is a valid list.
    const md::PairList* filtered_source_ = nullptr;
    unsigned long long filtered_version_ = 0;
    // Team / row balance / touched rows of the list `compute()` walks,
    // rebuilt when that list changes (null source = not built).
    md::PairRowPartition partition_;
    const md::PairList* part_source_ = nullptr;
    unsigned long long part_version_ = 0;
};

} // namespace cpu
} // namespace ATV
