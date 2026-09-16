/// @file cpu/semiempirical/am1_solver.h
/// @brief AM1 concrete method — public members + Frame build/run.
#pragma once

#include <memory>
#include <utility>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using AM1Result = NDDOResult;

/// @brief AM1 (*Austin Model 1*) — a concrete semiempirical method, same shape
///        as `cpu::scf::RHF`: parameters as members, `run(frame)` as the entry.
///
/// Semiempirical here means an NDDO (*neglect of diatomic differential
/// overlap*) Hamiltonian over a minimal valence orbital basis, with the
/// surviving integrals supplied by per-element fitted parameters instead of
/// being computed.
///
/// Owns an internal NDDO engine. Parameters are public members — there is no
/// public `NDDOConfig` / `NDDOSolver` bag. `AM1Evaluator` wraps this type.
/// Implements `cpu::QmEngine` for multiscale couplings (no embedding path).
class AM1 : public cpu::QmEngine {
  public:
    /// @brief Total molecular charge [e].
    int charge = 0;
    /// @brief Spin multiplicity 2S+1 (dimensionless).
    int multiplicity = 1;
    /// @brief SCF iteration cap (dimensionless).
    int max_iter = 200;
    /// @brief Density residual tolerance (dimensionless).
    double density_tol = 1.0e-8;
    /// @brief Energy residual tolerance [eV].
    double energy_tol = 1.0e-10;

    /// @brief Default parameters (charge 0, singlet, max_iter 200, …).
    AM1();
    /// @brief Construct with SCF parameters assigned immediately.
    /// @param charge Total molecular charge [e].
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param max_iter SCF iteration cap (dimensionless).
    /// @param density_tol Density residual tolerance (dimensionless).
    /// @param energy_tol Energy residual tolerance [eV].
    AM1(int charge, int multiplicity, int max_iter = 200,
        double density_tol = 1.0e-8, double energy_tol = 1.0e-10);
    ~AM1();
    AM1(AM1&&) noexcept;
    AM1& operator=(AM1&&) noexcept;
    /// @brief Copy parameters only (does not copy SCF engine state).
    AM1(const AM1& other);
    AM1& operator=(const AM1& other);

    /// @brief Size the NDDO engine for this atom set and snapshot the members
    ///        as its configuration.
    /// @param reference Frame whose atomic numbers / count fix the orbital
    ///        layout. Throws for an element AM1 has no parameters for.
    void build(const cpu::Frame& reference);

    /// @brief Run the SCF at the frame's geometry (warm-started from the last
    ///        converged density when there is one).
    /// @param frame Geometry [Å]; not modified.
    /// @return `last_result()`: energies [eV / Hartree / kcal·mol⁻¹] and
    ///         Mulliken charges [e].
    /// @throws std::runtime_error if the SCF does not converge in `max_iter`.
    AM1Result& run(const cpu::Frame& frame);

    /// @brief Nuclear forces \(F = -\partial E/\partial R\) [eV/Å] by central
    ///        finite difference of the energy at the converged density.
    ///
    /// See `NDDOSolver::compute_forces` for why holding the density fixed is
    /// exact here and what the difference step costs in accuracy.
    /// @param frame Geometry [Å]; not modified (displacements happen on local
    ///        copies of the coordinates).
    /// @param step_angstrom Finite-difference displacement \(h\) [Å];
    ///        truncation error \(O(h^{2})\).
    /// @return `last_result()` with `forces_ev_a` filled (3 · nlocal).
    /// @pre `run()` converged on this geometry.
    AM1Result& compute_forces(cpu::Frame& frame, double step_angstrom = 1.0e-4);

    /// @brief Results of the most recent `run()` / `compute_forces()`
    ///        (all-zero if neither has run).
    const AM1Result& last_result() const;
    /// @brief Has `build()` sized the engine for an atom set?
    bool built() const noexcept;
    /// @brief Is a converged density held (the warm start / force prerequisite)?
    bool has_density() const noexcept;
    /// @brief Orbital layout and per-atom parameters resolved by `build()`.
    const NDDOSystem& system() const;
    /// @brief SCF starting state captured at `build()` from the reference
    ///        geometry: initial density guess, core Hamiltonian, core-core
    ///        repulsion [eV] and the first Fock matrix. `run()` rebuilds these
    ///        from the frame it is given, so this is the *starting* state, not
    ///        the current one.
    const NDDOScfState& state() const;

    // ── QmEngine: the QM half of a QM/MM coupling ─────────────────────
    std::string qm_engine_name() const override { return "am1"; }
    void prepare(const cpu::BuildContext& qm_ctx) override;
    void evaluate(const cpu::LaunchContext& qm_ctx) override;
    // as_embedding() default nullptr — NDDO has no external-charge path yet.
    void reset_history() override;

    /// @brief Run the SCF and write every result channel onto the frame:
    ///        total energy [eV] to `kEnergy` and `kEnergyElectronic`, Mulliken
    ///        charges [e] to `kAtom_charge`, and — when asked — forces [eV/Å]
    ///        to `kAtom_f{x,y,z}`.
    ///
    /// Shared by both callers so the write-back contract exists once: the
    /// `QmEngine::evaluate` path (multiscale) and `AM1Evaluator::launch`.
    /// @param frame Target frame; geometry in, results out.
    /// @param forces_requested Also run `compute_forces()` and write the
    ///        gradient. False leaves the force columns untouched.
    /// @param accumulate true → `+=` energy/forces, the contract for a
    ///        component summed with other energy sources in a driver phase;
    ///        false → `=`. Charges are always last-writer-wins (a state
    ///        channel, not a sum).
    void publish(cpu::Frame& frame, bool forces_requested, bool accumulate);

  private:
    void sync_engine_();
    std::unique_ptr<NDDOSolver> engine_;
};

/// @brief Historical alias of the concrete AM1 method.
using AM1Solver = AM1;

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
