/// @file cuda/potentials/pair_lj_cut_coul_long.cuh
/// @brief Fused typed LJ and Ewald real space with special-bond exceptions.
#pragma once

#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/cuda/component/long_range_protocol.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"
#include <memory>
#include <vector>

namespace ATV::cuda {

/// @brief Single-GPU typed LJ + screened Coulomb tile evaluator — the CUDA
///        twin of `cpu::PairLJCutCoulLong` (LAMMPS `lj/cut/coul/long`).
///
/// The LJ table is completed with LAMMPS `init_one` semantics
/// (`core/lj_type_table.h`), identically to the CPU style: unlisted pairs
/// mix from the diagonals (geometric ε, arithmetic σ and cutoff), a row's
/// own cutoff is that pair's LJ cutoff, a type with no diagonal has no LJ
/// site. Coulomb uses the global cutoff. When the resolved table is pure
/// Lorentz–Berthelot at one cutoff with no r⁻⁴ term, the tile kernel uses
/// per-atom (σ/2, 2√ε) parameters (the OpenMM-benchmarked path); otherwise
/// each pair reads its record from a device type table.
///
/// Special bonds: bond-graph distances 1, 2 and 3 carry LJ / Coulomb
/// weights. A pair whose nearest shell has weights ≠ (1, 1) is an
/// exception: it is masked out of the tiles and evaluated by a sparse pass
/// (minimum image) with the LAMMPS `PairLJCutCoulLong::compute` cutoffs, as
/// factor_lj · (LJ − offset) inside the pair's LJ cutoff, factor_lj · r⁻⁴
/// inside its r⁻⁴ cutoff, and the Ewald correction
/// qq (factor_coul − erf(αr)) / r inside the Coulomb cutoff only (a bonded
/// pair beyond it keeps its full reciprocal-space share, as on the CPU and
/// in LAMMPS). Reciprocal PME is a separate component.
class PairLJCutCoulLong : public Pair, public LongRangeParticipant {
  public:
    /// @brief Construct from LJ type rows.
    ///
    /// Defaults match `cpu::PairLJCutCoulLong`: Amber specials, shifted LJ.
    /// The neighbor cutoff is max(`cutoff`, every LJ / r⁻⁴ cutoff).
    ///
    /// @param coeffs Diagonal (and optional off-diagonal) LJ type pairs.
    /// @param alpha Ewald splitting parameter [1/Å], matching reciprocal space.
    /// @param cutoff Coulomb cutoff and global LJ cutoff [Å].
    /// @throws std::invalid_argument on an empty table, a bad row, or a
    ///         non-positive / non-finite α or cutoff.
    PairLJCutCoulLong(std::vector<LJTypeCoeff> coeffs, double alpha,
                      double cutoff);
    ~PairLJCutCoulLong() override;
    PairLJCutCoulLong(PairLJCutCoulLong&&) noexcept;
    PairLJCutCoulLong& operator=(PairLJCutCoulLong&&) noexcept;

    std::string name() const override { return "lj/cut/coul/long"; }
    bool graph_capturable() const override { return true; }
    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, AccReal(alpha_), AccReal(cut_coul_),
                0};
    }
    /// @brief Set Amber LJ (0,0,1/2) and Coulomb (0,0,5/6) exceptions.
    /// @return This evaluator.
    PairLJCutCoulLong& set_special_bonds_amber() &;
    /// @brief Set Amber exceptions on a temporary evaluator.
    /// @return The configured evaluator by value.
    PairLJCutCoulLong set_special_bonds_amber() &&;
    /// @brief LAMMPS `special_bonds lj a b c coul d e f`.
    /// @param lj12 LJ weight of 1-2 pairs [dimensionless].
    /// @param lj13 LJ weight of 1-3 pairs [dimensionless].
    /// @param lj14 LJ weight of 1-4 pairs [dimensionless].
    /// @param c12 Coulomb weight of 1-2 pairs [dimensionless].
    /// @param c13 Coulomb weight of 1-3 pairs [dimensionless].
    /// @param c14 Coulomb weight of 1-4 pairs [dimensionless].
    /// @return This evaluator (lvalue) or the moved evaluator (rvalue).
    PairLJCutCoulLong& set_special_bonds(double lj12, double lj13, double lj14,
                                         double c12, double c13, double c14) &;
    PairLJCutCoulLong set_special_bonds(double lj12, double lj13, double lj14,
                                        double c12, double c13, double c14) &&;
    /// @brief LAMMPS `pair_modify shift`: subtract each pair's LJ energy at
    ///        its own LJ cutoff. Default on, as on the CPU style. OpenMM
    ///        comparisons run it off.
    /// @param on Whether LJ energies are shifted.
    /// @return This evaluator (lvalue) or the moved evaluator (rvalue).
    PairLJCutCoulLong& set_energy_shift(bool on) &;
    PairLJCutCoulLong set_energy_shift(bool on) &&;
    /// @brief LAMMPS `pair_modify tail yes`, the CUDA twin of
    ///        `cpu::PairLJCutCoulLong::set_tail_correction`: per-type counts
    ///        at `build()`; each step whose energy or virial is requested
    ///        adds E = etail/V and ptail/V on each diagonal virial component
    ///        (each pair at its own LJ cutoff), with V read on the device
    ///        from the live cell. Default off.
    /// @param on Whether the tail is added.
    /// @return This evaluator (lvalue) or the moved evaluator (rvalue).
    PairLJCutCoulLong& set_tail_correction(bool on) &;
    PairLJCutCoulLong set_tail_correction(bool on) &&;
    /// @brief Change ONE entry of the LJ coefficient table after
    ///        construction (same semantics as
    ///        `cpu::PairLJCutCoulLong::set_type_coeff`): adds or overrides
    ///        the (i, j) row, extending the table; takes effect at the next
    ///        `build()`; a larger LJ cutoff raises the neighbor cutoff.
    /// @param row Type pair, ε [eV], σ [Å] and optional LJ cutoff [Å].
    /// @return This evaluator (lvalue) or the moved evaluator (rvalue).
    /// @throws std::invalid_argument on a negative type, ε < 0, or a
    ///         σ / cutoff <= 0.
    PairLJCutCoulLong& set_type_coeff(const LJTypeCoeff& row) &;
    PairLJCutCoulLong set_type_coeff(const LJTypeCoeff& row) &&;
    /// @brief Add a force-shifted r⁻⁴ attraction on listed type pairs.
    ///
    /// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
    /// else 0 (`ATV::r4_attraction`), in the same typed tile walk as LJ,
    /// scaled by the LJ special weight, part of energy and virial. Selects
    /// the typed tile kernel. Replaces earlier rows; the neighbor cutoff
    /// grows to cover the largest r_c.
    /// @param rows Rows {type_i, type_j, S [eV], σ [Å], r_c [Å]}.
    /// @return This evaluator (lvalue) or the moved evaluator (rvalue).
    /// @throws std::invalid_argument on a negative type, non-finite S, or
    ///         σ / r_c <= 0.
    PairLJCutCoulLong& set_r4_attraction(std::vector<R4TypeCoeff> rows) &;
    PairLJCutCoulLong set_r4_attraction(std::vector<R4TypeCoeff> rows) &&;
    /// @brief Packed I×J tiles of the latest neighbor rebuild (diagnostics).
    ///        Reads the device count; call outside a run (it synchronizes).
    /// @return Tile count [dimensionless]; 0 before the first force call.
    int packed_tiles() const;
    /// @brief Cluster-pair CSR entries with at least one interacting pair in
    ///        the upper triangle at the latest rebuild (diagnostics). Reads
    ///        the device count; call outside a run (it synchronizes).
    /// @return Entry count [dimensionless]; 0 before the first force call.
    int valid_tiles() const;
    /// @brief Whether the last build selected the typed (device table) tile
    ///        kernel rather than the per-atom Lorentz–Berthelot one
    ///        (diagnostics; kept after destroy).
    /// @return True for the typed path; false before the first build.
    bool typed_kernel() const;
    void build(const BuildContext& ctx) override;
    void invalidate_inputs(InputChange changes,
                           const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void refresh_cutoff_();

    LJTypeTable table_;
    double alpha_;
    double cut_coul_;
    double lj_[4] = {1.0, 1.0, 1.0, 1.0};   ///< Index = graph distance.
    double coul_[4] = {1.0, 1.0, 1.0, 1.0}; ///< Index = graph distance.
    bool shift_ = true;
    bool tail_ = false;
    LJTail tail_pref_{}; ///< Prefactors from the build-time type counts.
    bool typed_ = false; ///< Kernel selected by the last build.
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace ATV::cuda
