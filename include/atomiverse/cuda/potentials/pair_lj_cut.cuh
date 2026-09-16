/// @file cuda/potentials/pair_lj_cut.cuh
/// @brief LAMMPS-style PairLJCut — 12-6 with shifted potential (CUDA).
///
/// Was `ATV::cuda::LJ` before the LAMMPS-naming refactor.
#pragma once

#include <string>

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

    std::string name() const override { return "lj"; }

    /// @brief Enable LAMMPS-style `special_bonds` 1-2 / 1-3 / 1-4
    ///        pair exclusions.
    ///
    /// Mirrors LAMMPS `special_bonds lj/coul s12 s13 s14` semantics.
    /// The CUDA arm supports two combinations in this revision (per
    /// `msu-pe-cuda-components-05-cuda-lj-exclusions` design):
    ///   - `(1.0, 1.0, 1.0)` — no exclusions, kernel takes fast path.
    ///   - All three equal (typically `(0.0, 0.0, 0.0)` for Dreiding-UA
    ///     polyethylene): every 1-2/1-3/1-4 pair is multiplied by the
    ///     shared scale `s`. `s == 0` ⇒ completely skipped.
    /// Other combinations throw `std::invalid_argument`. The supported
    /// pair (0,0,0) is what MSU PE protocol uses.
    ///
    /// Exclusion partner lists are enumerated from `frame.bonds /
    /// angles / dihedrals` at `build()` time, sorted+uniqued per atom,
    /// and uploaded to device as a CSR. The kernel does a per-pair
    /// binary search against the CSR.
    ///
    /// @param s12 Scale for 1-2 (bonded) pairs.
    /// @param s13 Scale for 1-3 (across-angle) pairs.
    /// @param s14 Scale for 1-4 (across-dihedral) pairs.
    /// @return `*this`, chainable.
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
    /// per-block stress reduction. Mirrors LAMMPS per-step `vflag` gating;
    /// forces and energy are unaffected. Default: enabled.
    PairLJCut& set_compute_stress(bool on) &;
    PairLJCut set_compute_stress(bool on) &&;

    /// @brief `Component` hook — the driver gates per-step virial on
    ///        whether any downstream component consumes stress. Equivalent to
    ///        `set_compute_stress(required)`; lets the MD driver disable the
    ///        virial path for plain NVE plans without name matching.
    void set_stress_required(bool required) override {
        compute_stress_ = required;
    }

    /// @brief `cuda::Pair` hook — enables full (0,0,0) special-bonds
    ///        exclusions, equivalent to `set_special_bonds(0,0,0)`. Lets the
    ///        multi-GPU engine request exclusions through a `Pair*`.
    void enable_full_exclusions() override;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    double epsilon() const { return epsilon_; }
    double sigma() const { return sigma_; }
    double cutoff() const { return cutoff_; }

    /// @brief `compute()` honors `Context::n_force_rows`/`force_rows` (the
    ///        multi-GPU owned-row restriction + interior/boundary split).
    bool supports_force_rows() const override { return true; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double epsilon_; ///< Well depth [eV].
    double sigma_;   ///< Collision diameter [Å].
    double v_shift_; ///< Potential shift at cutoff [eV].

    // Special-bonds configuration (host-side).
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;
    double s_excl_ = 1.0; ///< Single scale used by kernel when collapsed.
    bool has_exclusions_ = false;

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
    // num_clusters*kClusterSize; grown when that count increases. Freed in
    // destroy(). Runtime container — not part of the frame.
    real_t* spx_ = nullptr;
    real_t* spy_ = nullptr;
    real_t* spz_ = nullptr;
    int spos_cap_ = 0;

    // AoS [x,y,z,_] packed positions, re-packed each compute() from the frame's
    // current x/y/z. The per-atom Verlet force kernel gathers neighbour
    // positions as one packed float4 per neighbour (a single ≤16-byte
    // transaction) instead of three scattered SoA loads — ncu showed the SoA
    // gather L2-bandwidth-bound (23.8 sectors/req, 5.3 useful bytes/sector) at
    // large multi-GPU subdomain sizes. Sized to n_atoms; grown as needed; freed
    // in destroy(). Runtime container — not part of the frame.
    float4* pos4_ = nullptr;
    int pos4_cap_ = 0;
};

} // namespace cuda
} // namespace ATV
