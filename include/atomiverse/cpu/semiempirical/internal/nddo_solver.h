/// @file cpu/semiempirical/internal/nddo_solver.h
/// @brief Internal NDDO engine (not a public bag API).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_matrix.h"
// internal only

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOConfig {
    ATV::semiempirical::SemiEmpiricalMethod method =
        ATV::semiempirical::SemiEmpiricalMethod::AM1;
    int charge = 0;
    int multiplicity = 1;
    int max_iter = 200;
    double density_tol = 1.0e-8;
    double energy_tol = 1.0e-10;
};

struct NDDOScfState {
    std::vector<double> initial_diagonal_density;
    std::vector<double> initial_packed_density;
    std::vector<double> initial_spin_density;
    std::vector<double> packed_hcore;
    std::vector<double> initial_one_center_fock;
    double nuclear_repulsion_ev = 0.0;
};

/// @brief Inner solver for MNDO-form NDDO methods.
///
/// `build()` validates and sizes method data for the current atom set.
/// `run()` executes the NDDO reference-shaped AM1 RHF loop for currently ported s/p
/// systems and throws a typed runtime error for unsupported cases.
///
/// Nuclear forces use three-direction forward analytic differentiation of the
/// same scalar pair integrals used by the energy, contracted with converged
/// density. The NDDO basis is orthonormal, so there is no overlap Pulay term.
/// Stationarity removes the density response term. Pair work is quadratic in
/// atom count and temporary derivative matrices contain only two atoms.
///
/// Self-consistency is `NddoMethod` (`cpu::Driver<nddo::Phase>`). This
/// class is the step body: `scf_prepare`, then one call each of
/// `scf_diagonalize`, `scf_density`, `scf_fock`, and `scf_check` per
/// iteration. Forces feed Evaluate / MD / Minimize through `AM1` / `RM1`.
class NDDOSolver {
  public:
    explicit NDDOSolver(NDDOConfig config);

    void build(const cpu::Frame& reference);

    /// @brief Update numeric SCF controls without rebuilding method workspace.
    /// @param max_iter Positive SCF iteration budget.
    /// @param density_tol Positive finite packed-density residual tolerance [e].
    /// @param energy_tol Positive finite energy-change tolerance [eV].
    /// @throws std::invalid_argument if any supplied control is invalid.
    void set_scf_controls(int max_iter, double density_tol, double energy_tol);

    /// @brief Guess Fock and warm-start density for one SCF.
    /// @param frame Geometry [Å]. Not modified.
    /// Rebuilds when atom count or elements change; discards the old guess.
    /// @throws std::runtime_error if `build()` has not run or the system is open-shell.
    /// @throws std::invalid_argument for empty/non-finite/coincident geometry.
    void scf_prepare(const cpu::Frame& frame);
    /// @brief Diagonalize the current packed Fock matrix.
    void scf_diagonalize();
    /// @brief Build the new closed-shell density and its residual.
    void scf_density();
    /// @brief Rebuild the Fock matrix from the new density.
    void scf_fock();
    /// @brief Energy, charges, and the density / energy residuals.
    ///
    /// When both residuals are inside their tolerances, fills the heat of
    /// formation. Accepted warm-start density changes only in commit_density().
    /// Does not set `result().converged`: `NddoMethod` polls a Criterion on the
    /// published residuals. Throws for non-finite energy or missing parameters.
    void scf_check();

    /// @brief Accept current converged density as the next normal solve's guess.
    /// Called only after the complete method operation succeeds; initial-force
    /// preparation and rejected or exploratory trials must not call this.
    void commit_density();

    /// @brief Max absolute packed-density change of this iteration.
    /// @return |ΔP|, or +∞ before `scf_density`.
    const double& density_residual() const { return scf_delta_; }

    /// @brief Absolute total-energy change of this iteration [eV].
    /// @return |ΔE| [eV], or +∞ before `scf_check`.
    const double& energy_residual() const { return scf_energy_delta_; }

    /// @brief Mutable result of the SCF in progress or just finished.
    ATV::semiempirical::NDDOResult& result() { return result_; }

    /// @brief Analytic nuclear forces [eV/Å] at the last converged SCF density.
    /// @param frame Geometry matching that SCF. Frame outputs are not modified.
    /// @return Method result with equal-and-opposite pair forces accumulated.
    /// @throws std::runtime_error for missing convergence, wrong atom count,
    /// or a non-finite derivative.
    ATV::semiempirical::NDDOResult& compute_forces(cpu::Frame& frame);

    const ATV::semiempirical::NDDOResult& last_result() const {
        return result_;
    }
    const NDDOConfig& config() const { return config_; }
    const NDDOSystem& system() const { return system_; }
    const NDDOScfState& state() const { return state_; }
    bool built() const noexcept { return built_; }
    bool has_density() const noexcept { return has_density_; }

  protected:
    void validate_supported_atoms_(const cpu::Frame& frame) const;

  private:
    NDDOConfig config_;
    bool built_ = false;
    bool has_density_ = false;
    int n_atoms_ = 0;
    NDDOSystem system_;
    NDDOScfState state_;
    ATV::semiempirical::NDDOResult result_;
    // Allocated at build; force evaluation never constructs per-pair vectors.
    NDDOSystem force_pair_;
    std::vector<double> force_scratch_;
    std::vector<double> density_;
    std::vector<double> spin_density_;

    const StorageReal* scf_x_ = nullptr;
    const StorageReal* scf_y_ = nullptr;
    const StorageReal* scf_z_ = nullptr;
    std::vector<double> scf_hcore_;
    double scf_nuc_ev_ = 0.0;
    std::vector<double> scf_density_;
    std::vector<double> scf_spin_;
    std::vector<double> scf_fock_;
    std::vector<double> scf_coeff_;
    NDDODiagonalizer diagonalizer_;
    double scf_delta_ = 0.0;
    double scf_energy_delta_ = 0.0;
    double scf_prev_energy_ev_ = 0.0;
    int scf_iter_ = 0;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
