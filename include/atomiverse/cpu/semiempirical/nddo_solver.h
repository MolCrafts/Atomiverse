/// @file cpu/semiempirical/nddo_solver.h
/// @brief Low-level MNDO/NDDO-family semiempirical solver scaffold.
///
/// This is the semiempirical analogue of `cpu::scf::SCFSolver` and
/// `cpu::pw::PWSolver`: an inner numerical solver owned by an Evaluator, not an
/// outer `Driver<>` subclass.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOConfig {
    SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1;
    int charge = 0;
    int multiplicity = 1;
    int max_iter = 200;
    double density_tol = 1.0e-8;
    double energy_tol = 1.0e-10;
};

struct NDDOResult {
    SemiEmpiricalMethod method = SemiEmpiricalMethod::AM1;
    bool converged = false;
    int iterations = 0;
    /// @brief Total electronic + nuclear energy [Hartree].
    double electronic_energy_hartree = 0.0;
    /// @brief Total electronic + nuclear energy [eV] (same quantity as Hartree).
    double total_energy_ev = 0.0;
    double heat_of_formation_kcal_mol = 0.0;
    std::vector<double> mulliken_charges;
    /// @brief Nuclear Cartesian forces [eV/Å], layout (fx0,fy0,fz0, fx1, …).
    /// Empty until `compute_forces()` succeeds after a converged `run()`.
    std::vector<double> forces_ev_a;
    std::string reference;
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
/// Nuclear forces (`compute_forces`) are the SCF analytic gradient of the
/// NDDO energy: Hellmann–Feynman contraction of the fixed SCF density with
/// two-center integral / nuclear-repulsion derivatives (central difference of
/// the energy functional at fixed P). No new Driver — forces feed ordinary
/// Evaluate / MD / Minimize via `AM1Evaluator`.
class NDDOSolver {
  public:
    explicit NDDOSolver(NDDOConfig config);

    void build(const cpu::Frame& reference);
    NDDOResult& run(const cpu::Frame& frame);

    /// @brief Analytic nuclear forces [eV/Å] at the last SCF density.
    ///
    /// @param frame Geometry (must match last `run()` atom count).
    /// @param step_angstrom Central-difference step for integral derivatives [Å].
    /// @return Reference to `last_result()` with `forces_ev_a` filled (3 * nlocal).
    /// @pre `run()` converged on this geometry (or same nlocal after rebuild).
    NDDOResult& compute_forces(cpu::Frame& frame,
                               double step_angstrom = 1.0e-4);

    const NDDOResult& last_result() const { return result_; }
    const NDDOConfig& config() const { return config_; }
    const NDDOSystem& system() const { return system_; }
    const NDDOScfState& state() const { return state_; }
    bool built() const noexcept { return built_; }
    bool has_density() const noexcept { return has_density_; }

  protected:
    void validate_supported_atoms_(const cpu::Frame& frame) const;

  private:
    /// @brief Total energy [eV] of the NDDO functional at fixed packed density.
    double energy_at_fixed_density_ev_(const StorageReal* sp_x,
                                       const StorageReal* sp_y,
                                       const StorageReal* sp_z) const;

    NDDOConfig config_;
    bool built_ = false;
    bool has_density_ = false;
    int n_atoms_ = 0;
    NDDOSystem system_;
    NDDOScfState state_;
    NDDOResult result_;
    std::vector<double> density_;
    std::vector<double> spin_density_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
