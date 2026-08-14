/// @file cpu/semiempirical/internal/nddo_solver.h
/// @brief Internal NDDO engine (not a public bag API).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
// internal only

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
/// Nuclear forces (`compute_forces`) are a **central finite difference of the
/// NDDO energy functional at fixed density** \(P\): converge the SCF once,
/// then displace one Cartesian coordinate by \(\pm h\), rebuild the
/// geometry-dependent integrals and core-core repulsion at that displaced
/// geometry with \(P\) unchanged, and difference the two energies.
///
/// Freezing \(P\) costs nothing in exactness. At convergence the energy is
/// stationary with respect to density variations that preserve the electron
/// count, so the response term \((\partial E/\partial P)(dP/dR)\) vanishes;
/// and NDDO's zero-differential-overlap approximation makes the orbital basis
/// orthonormal (`NDDODiagonalizer` solves a *standard* eigenproblem, no
/// overlap matrix), so there is no basis-derivative — "Pulay" — term to
/// recover either. What remains approximate is the difference quotient: it
/// equals the exact NDDO gradient to \(O(h^{2})\) plus round-off.
///
/// No new Driver — forces feed ordinary Evaluate / MD / Minimize via
/// `AM1Evaluator`.
class NDDOSolver {
  public:
    explicit NDDOSolver(NDDOConfig config);

    void build(const cpu::Frame& reference);
    NDDOResult& run(const cpu::Frame& frame);

    /// @brief Nuclear forces \(F = -\partial E/\partial R\) [eV/Å] by central
    ///        finite difference of the energy at the last SCF density.
    ///
    /// Only \(3(N-1)\) differences are taken for \(N \ge 2\): the force on the
    /// last atom is set to minus the sum of all the others, which is exact for
    /// the isolated molecule NDDO models (a rigid translation cannot change
    /// \(E\)). The returned forces therefore sum to zero by construction —
    /// that is not an independent check of the gradient.
    ///
    /// @param frame Geometry (must match last `run()` atom count). Nothing on
    ///        it is modified: the positions are copied out and displaced in
    ///        local buffers, and the force columns are written by
    ///        `AM1::publish`, not here.
    /// @param step_angstrom Displacement \(h\) applied to one Cartesian
    ///        coordinate at a time [Å]. Truncation error is \(O(h^{2})\), so
    ///        shrinking it indefinitely trades truncation for cancellation
    ///        round-off.
    /// @return Reference to `last_result()` with `forces_ev_a` filled (3 * nlocal).
    /// @pre `run()` converged on this geometry (or same nlocal after rebuild).
    /// @throws std::runtime_error if `build()` never ran, if no converged
    ///         density is held, or if `frame.nlocal()` differs from the atom
    ///         count `build()` sized for.
    /// @throws std::invalid_argument if `step_angstrom` is not positive finite.
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
