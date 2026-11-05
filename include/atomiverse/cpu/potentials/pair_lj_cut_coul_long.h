/// @file cpu/potentials/pair_lj_cut_coul_long.h
/// @brief Fused LJ 12-6 + real-space coul/long — one neighbor walk (CPU).
///
/// Matches LAMMPS `pair_style lj/cut/coul/long`. Multi-type LJ completed with
/// LAMMPS `init_one` semantics (`core/lj_type_table.h`: Lorentz–Berthelot
/// fill, per-pair LJ cutoffs), LJ and Coulomb special scales, optional r⁻⁴
/// attraction. erfc(αr) is evaluated analytically (vectorized fdlibm erfc,
/// `cpu/potentials/ewald_erfc.h`, ~1 ulp) — LAMMPS `pair_modify table 0`
/// semantics without its 1.5e-7 polynomial, and the same function as
/// `cuda::PairLJCutCoulLong`; there is no tabulated mode. Reciprocal space
/// is a separate `KSpacePME` / `KSpacePPPM`.
#pragma once

#include <array>
#include <span>

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/cpu/component/long_range_protocol.h"
#include "atomiverse/cpu/potentials/pair.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace ATV {
namespace cpu {

/// @brief Fused `lj/cut/coul/long` with multi-type LJ and Amber specials.
class PairLJCutCoulLong : public Pair, public LongRangeParticipant {
  public:
    /// @brief The fused kernel reads the neighbour charge (`q_ptr[j]`) and
    /// force-field type (`type_ptr[j]`, the LJ table index).
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_charge, "PairLJCutCoulLong"),
        GhostField::read_stable(kAtom_type, "PairLJCutCoulLong"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    /// @brief Construct from LJ type rows (LAMMPS `pair_style
    ///        lj/cut/coul/long cutoff` + `pair_coeff` rows).
    ///
    /// Unlisted pairs mix from the diagonals (geometric ε, arithmetic σ and
    /// cutoff); a row's own `cutoff` sets that pair's LJ cutoff, unset rows
    /// use `cutoff`. Coulomb always uses `cutoff`. Defaults: Amber specials
    /// (`set_special_bonds_amber`), shifted LJ, no tail correction. The
    /// neighbor cutoff is max(`cutoff`, every LJ / r⁻⁴ cutoff).
    ///
    /// @param coeffs  Diagonal (and optional off-diagonal) LJ type pairs.
    /// @param alpha   Ewald α [1/Å] — must match the KSpace component.
    /// @param cutoff  Coulomb cutoff and global LJ cutoff [Å].
    /// @throws std::invalid_argument on an empty table, a bad row, or a
    ///         non-positive α / cutoff.
    PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs, double alpha,
                      double cutoff);

    std::string name() const override { return "lj/cut/coul/long"; }
    /// A pure function of the geometry (no trial-dependent state).
    bool trial_invariant() const override { return true; }

    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, alpha_, AccReal(cut_coul_),
                AccReal(0)};
    }

    /// @brief Amber `special_bonds amber`: LJ (0,0,0.5), coul (0,0,5/6).
    PairLJCutCoulLong& set_special_bonds_amber() &;
    PairLJCutCoulLong set_special_bonds_amber() &&;

    /// @brief Explicit LJ / Coulomb special scales (LAMMPS special_bonds).
    PairLJCutCoulLong& set_special_bonds(double lj12, double lj13, double lj14,
                                         double c12, double c13, double c14) &;
    PairLJCutCoulLong set_special_bonds(double lj12, double lj13, double lj14,
                                        double c12, double c13, double c14) &&;

    /// @brief Change ONE entry of the LJ coefficient table after
    ///        construction (same semantics as `PairLJCut::set_type_coeff`).
    ///
    /// Adds or overrides the (i, j) row (a later row wins); a type pair the
    /// constructor did not name extends the table. Takes effect at the next
    /// `build()`. A larger per-pair LJ cutoff raises the reported neighbor
    /// cutoff; the Coulomb cutoff is unchanged.
    /// @param row Type pair, ε [eV], σ [Å] and optional LJ cutoff [Å].
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::invalid_argument on a negative type, ε < 0, or a
    ///         σ / cutoff <= 0.
    PairLJCutCoulLong& set_type_coeff(const LJTypeCoeff& row) &;
    PairLJCutCoulLong set_type_coeff(const LJTypeCoeff& row) &&;

    /// @brief Add a force-shifted r⁻⁴ attraction on listed type pairs.
    ///
    /// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
    /// else 0 (`ATV::r4_attraction`); evaluated in the same table walk as
    /// LJ, scaled by the LJ special weight, part of energy and virial. An
    /// ATV-native term. Replaces earlier rows; the neighbor cutoff grows to
    /// cover the largest r_c.
    /// @param rows Rows {type_i, type_j, S [eV], σ [Å], r_c [Å]}.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::invalid_argument on a negative type, non-finite S, or
    ///         σ / r_c <= 0.
    PairLJCutCoulLong& set_r4_attraction(std::vector<R4TypeCoeff> rows) &;
    PairLJCutCoulLong set_r4_attraction(std::vector<R4TypeCoeff> rows) &&;

    PairLJCutCoulLong& set_compute_stress(bool on) &;
    PairLJCutCoulLong set_compute_stress(bool on) &&;

    /// @brief LAMMPS `pair_modify shift` — subtract each pair's V_LJ at its
    ///        own LJ cutoff so the LJ energy is zero there. Default on (same
    ///        on `cuda::PairLJCutCoulLong`). LAMMPS production decks with
    ///        `tail yes` and OpenMM comparisons run shift off.
    PairLJCutCoulLong& set_energy_shift(bool on) &;
    PairLJCutCoulLong set_energy_shift(bool on) &&;

    /// @brief LAMMPS `pair_modify tail yes` — analytic LJ tail energy and
    ///        pressure beyond each pair's own LJ cutoff (homogeneous
    ///        g(r) = 1, `ATV::lj_tail_pair`; the r⁻⁴ term has no tail).
    ///        Global per-type counts are gathered on the first evaluation
    ///        after `build()`; every step adds E = etail/V to the energy and
    ///        ptail/V to each diagonal virial component at the current box
    ///        volume. Default off (same on `cuda::PairLJCutCoulLong`).
    /// @param on Whether the tail is added.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    PairLJCutCoulLong& set_tail_correction(bool on) &;
    PairLJCutCoulLong set_tail_correction(bool on) &&;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    /// @brief Report max(Coulomb, LJ, r⁻⁴ cutoffs) as this style's cutoff.
    void refresh_cutoff_();
    void rebuild_filtered_list(const Frame& frame, const md::PairList& pairs);

    double alpha_ = 0.0;
    double cut_coul_ = 0.0;   ///< Coulomb cutoff [Å].
    double cut_coulsq_ = 0.0; ///< [Å²]
    double cut_sq_ = 0.0;     ///< Largest pair cutoff squared [Å²].

    /// LAMMPS `init_one` table (lj1..lj4, cut_ljsq, offset, r⁻⁴), resolved at
    /// build() over every type the Frame carries.
    LJTypeTable table_;
    int n_types_ = 0;
    // Neighbor build whose row types were last checked against the table.
    const md::PairList* types_source_ = nullptr;
    unsigned long long types_version_ = 0;

    double lj12_ = 1.0, lj13_ = 1.0, lj14_ = 1.0;
    double c12_ = 1.0, c13_ = 1.0, c14_ = 1.0;
    bool any_special_ = false;

    bool energy_shift_ = true;
    bool tail_ = false;
    bool tail_ready_ = false;
    /// E_tail = etail_pref_/V; P_tail = ptail_pref_/V² (see compute()).
    double etail_pref_ = 0.0, ptail_pref_ = 0.0;
    bool compute_stress_ = true;

    /// Owner row → (partner tag, tier) CSR from kAtom_special, rebuilt with
    /// the pair list (sort / migration reorder rows). Tier 1/2/3 = 1-2/1-3/
    /// 1-4, listed nearest tier first so the first match wins as in LAMMPS.
    std::vector<int> special_off_;
    std::vector<TagInt> special_tag_;
    std::vector<std::uint8_t> special_tier_;
    /// Weights by tier; index 0 is the ordinary pair (1.0). Kept in double:
    /// Amber's 5/6 is not a float.
    double tier_lj_[4] = {1.0, 1.0, 1.0, 1.0};
    double tier_coul_[4] = {1.0, 1.0, 1.0, 1.0};

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;

    /// Parallel to PairList::neighbors[] (no compact copy): the pair's tier,
    /// 0 for an ordinary pair. One byte per pair instead of two floats.
    std::vector<std::uint8_t> pair_tier_;
    // Null source denotes an unbuilt cache; version zero is a valid list.
    const md::PairList* scale_source_ = nullptr;
    unsigned long long scale_version_ = 0;
};

} // namespace cpu
} // namespace ATV
