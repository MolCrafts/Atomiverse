/// @file cuda/potentials/pair_lj_cut.cuh
/// @brief LAMMPS-style PairLJCut — 12-6 with shifted potential (CUDA).
///
/// Was `ATV::cuda::LJ` before the LAMMPS-naming refactor.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/lj_type_table.h"
#include "atomiverse/cuda/potentials/pair.cuh"

namespace ATV {
namespace cuda {

/// @brief Lennard-Jones 12-6 pair potential evaluator (CUDA).
///
/// Must be explicitly bound to a force phase (cross-driver component).
/// Requires a populated PairList published on `cuda::Context::pair_list`
/// by a NeighborList component.
class PairLJCut : public Pair {
  public:
    /// @brief Construct with LJ parameters.
    /// @param epsilon Well depth [eV].
    /// @param sigma   Collision diameter [Å].
    /// @param cutoff  Interaction cutoff [Å].
    PairLJCut(double epsilon, double sigma, double cutoff);

    /// @brief Per-type constructor, the CUDA twin of
    ///        `cpu::PairLJCut(std::vector<LJTypeCoeff>, double)`.
    ///
    /// The table is completed with LAMMPS `PairLJCut::init_one` semantics
    /// (`core/lj_type_table.h`): unlisted pairs mix from the diagonals
    /// (geometric ε, arithmetic σ and cutoff), a row's own `cutoff` is that
    /// pair's cutoff (unset = `cutoff`), a type with no diagonal has no LJ
    /// site. Every pair is shifted at its own cutoff. Each pair reads its
    /// record from a read-only device table; the reported neighbor cutoff
    /// is the largest per-pair cutoff.
    ///
    /// @param coeffs Type-pair coefficients; must not be empty.
    /// @param cutoff Global LJ cutoff [Å] for rows that name none.
    /// @throws std::invalid_argument on an empty list, a negative type,
    ///         ε < 0, or a σ / cutoff <= 0.
    PairLJCut(std::vector<LJTypeCoeff> coeffs, double cutoff);

    /// @brief Change ONE entry of the coefficient table after construction,
    ///        the CUDA twin of `cpu::PairLJCut::set_type_coeff`.
    ///
    /// Adds or overrides the (i, j) row (a later row wins); a type pair the
    /// constructor did not name extends the table. Takes effect at the next
    /// `build()`; a larger per-pair cutoff raises the reported neighbor
    /// cutoff.
    /// @param row Type pair, ε [eV], σ [Å] and optional cutoff [Å].
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::logic_error on a single-type PairLJCut (no type table).
    /// @throws std::invalid_argument on a negative type, ε < 0, or a
    ///         σ / cutoff <= 0.
    PairLJCut& set_type_coeff(const LJTypeCoeff& row) &;
    PairLJCut set_type_coeff(const LJTypeCoeff& row) &&;

    /// @brief LAMMPS `pair_modify tail yes`, the CUDA twin of
    ///        `cpu::PairLJCut::set_tail_correction`.
    ///
    /// Per-type counts are taken at `build()` (every atom one type on the
    /// single-type form); each step whose energy or virial is requested adds
    /// E = etail/V and ptail/V on each diagonal virial component, with V
    /// read on the device from the live cell. Single GPU only. Default off.
    /// @param on Whether the tail is added.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    PairLJCut& set_tail_correction(bool on) &;
    PairLJCut set_tail_correction(bool on) &&;

    /// @brief Add a force-shifted r⁻⁴ attraction on listed type pairs.
    ///
    /// U(r) = −S [ (σ/r)⁴ − (σ/r_c)⁴ + 4 (σ/r_c)⁴ (r/r_c − 1) ] for r < r_c,
    /// else 0 (`ATV::r4_attraction`), in the same typed tile walk as LJ,
    /// dropped with LJ by a 0 special-bond scale, part of energy and
    /// virial. Replaces earlier rows; the neighbor cutoff grows to cover the
    /// largest r_c.
    /// @param rows Rows {type_i, type_j, S [eV], σ [Å], r_c [Å]}.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    /// @throws std::logic_error on a single-type PairLJCut (no type table).
    /// @throws std::invalid_argument on a negative type, non-finite S, or
    ///         σ / r_c <= 0.
    PairLJCut& set_r4_attraction(std::vector<R4TypeCoeff> rows) &;
    PairLJCut set_r4_attraction(std::vector<R4TypeCoeff> rows) &&;

    std::string name() const override { return "lj"; }
    bool graph_capturable() const override { return true; }

    /// @brief Enable LAMMPS-style `special_bonds` 1-2 / 1-3 / 1-4
    ///        pair exclusions.
    ///
    /// Mirrors LAMMPS `special_bonds lj/coul s12 s13 s14` semantics, with
    /// the same fractional weights as `cpu::PairLJCut` (e.g. Amber
    /// `(0, 0, 0.5)`). The nearest shell wins when one pair has two
    /// topological distances. Every pair whose shell weight is not 1 leaves
    /// the tiles (the neighbor list bakes it out of the masks); a weight
    /// other than 0 and 1 is then restored by a sparse exception pass
    /// (minimum image) as
    /// w · (LJ − offset) inside the pair's own cutoff plus w · r⁻⁴ inside
    /// its r⁻⁴ cutoff — the CPU `factor_lj` result. Fractional weights need
    /// a single GPU (`build()` throws `std::invalid_argument` on ghost rows).
    ///
    /// Partner shells come from the device special list
    /// (`Frame::build_special_list`), else from `frame.bonds / angles /
    /// dihedrals`, at `build()` time, and are uploaded as a sorted CSR.
    ///
    /// @param s12 Scale for 1-2 (bonded) pairs.
    /// @param s13 Scale for 1-3 (across-angle) pairs.
    /// @param s14 Scale for 1-4 (across-dihedral) pairs.
    /// @return `*this`, chainable.
    /// @throws std::invalid_argument on a non-finite weight.
    ///
    /// @note Two ref-qualified overloads (mirrors `cpu::PairLJCut`): the
    /// lvalue form returns `PairLJCut&` for `obj.set_special_bonds(…)`
    /// usage; the rvalue form returns `PairLJCut` BY VALUE so a chained
    /// `cuda::PairLJCut{…}.set_special_bonds(…)` expression yields an
    /// rvalue (not a dangling lvalue reference to a temporary). Passing
    /// the latter into `sim.set_evaluator(…)` moves the configured component
    /// into the driver's `unique_ptr`, instead of storing a dangling
    /// `&temporary` that would later trigger a "pure virtual method
    /// called" abort when the driver invokes `compute()` / `build()`.
    PairLJCut& set_special_bonds(double s12, double s13, double s14) &;
    PairLJCut set_special_bonds(double s12, double s13, double s14) &&;

    /// @brief Enable/disable per-pair virial (stress) accumulation.
    ///
    /// The pair virial (6 Voigt components, double accumulators) feeds the
    /// pressure / barostat path only; a plain NVE run never consumes it.
    /// Disabling it selects a kernel template that skips the 6 inner-loop
    /// virial products AND frees their registers (relieving the spill the
    /// `__launch_bounds__(256,4)` budget otherwise forces) and drops the
    /// per-block stress reduction. A veto on top of the per-step demand
    /// (LAMMPS `vflag`): the virial is computed only on steps whose
    /// consumers ask for it. Forces are unaffected. Default: enabled.
    PairLJCut& set_compute_stress(bool on) &;
    PairLJCut set_compute_stress(bool on) &&;

    /// @brief `cuda::Pair` hook — enables full (0,0,0) special-bonds
    ///        exclusions, equivalent to `set_special_bonds(0,0,0)`. Lets the
    ///        multi-GPU engine request exclusions through a `Pair*`.
    void enable_full_exclusions() override;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    double epsilon() const { return epsilon_; }
    double sigma() const { return sigma_; }
    double cutoff() const { return cutoff_; }

    /// @brief Cluster force writes owned atoms in one pass. Arbitrary
    ///        interior/boundary row subsets are not supported.
    bool supports_force_rows() const override { return false; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    /// Per-type table; empty = the single-ε path.
    std::optional<LJTypeTable> table_;
    /// Device pair records: the typed table, or on the single-type path a
    /// one-record table used only by the exception pass.
    void* d_table_ = nullptr;
    int n_types_ = 0;
    int* spt_ = nullptr; ///< Cluster-sorted atom types (typed path only).

    double epsilon_; ///< Well depth [eV].
    double sigma_;   ///< Collision diameter [Å].
    double v_shift_; ///< Potential shift at cutoff [eV].

    // Special-bonds configuration (host-side).
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;
    double s_excl_ = 1.0; ///< Single scale used by kernel when collapsed.
    bool has_exclusions_ = false;

    /// Fractional-weight exceptions (i < j, weight, table index), evaluated
    /// by the sparse pass after the tiles. Empty without fractional weights.
    void* d_exceptions_ = nullptr;
    int n_exceptions_ = 0;

    bool tail_ = false;
    LJTail tail_pref_{}; ///< Prefactors from the build-time type counts.

    /// When false, the force kernel skips the 6-component pair virial and
    /// the stress reduction (LAMMPS-style vflag gating). Forces/energy are
    /// unchanged. Default true to preserve behavior for pressure/NPT runs.
    bool compute_stress_ = true;

    // Device CSR — d_excl_offsets_[n_atoms+1], d_excl_partners_[n_excl_total_].
    // Both `nullptr` when has_exclusions_ is false (kernel fast path).
    int* d_excl_offsets_ = nullptr;
    int* d_excl_partners_ = nullptr;
    int n_excl_total_ = 0;

    // Cluster-sorted current-position scratch (gathered each compute() so the
    // force kernel reads neighbour positions coalesced). Sized to
    // padded Frame capacity in build(), with no allocation in compute().
    // Freed in destroy(). Runtime container — not part of the frame.
    force_t* spx_ = nullptr;
    force_t* spy_ = nullptr;
    force_t* spz_ = nullptr;
    int spos_cap_ = 0;
};

} // namespace cuda
} // namespace ATV
