/// @file cpu/semiempirical/am1_solver.h
/// @brief AM1 concrete method — public members + Frame build/run.
#pragma once

#include <memory>
#include <utility>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include "atomiverse/cpu/semiempirical/nddo_method.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

using AM1Result = ATV::semiempirical::NDDOResult;

/// @brief AM1 (*Austin Model 1*) — a concrete semiempirical method, same shape
///        as `cpu::scf::RHF`: parameters as members, `run(frame)` as the entry.
///
/// Semiempirical here means an NDDO (*neglect of diatomic differential
/// overlap*) Hamiltonian over a minimal valence orbital basis, with the
/// surviving integrals supplied by per-element fitted parameters instead of
/// being computed.
///
/// Owns an internal NDDO engine. Parameters are public members — there is no
/// public `NDDOConfig` / `NDDOSolver` bag. The self-consistency loop is
/// `NddoMethod`; mount the method itself in an outer loop. Implements
/// `cpu::QmEngine` for multiscale couplings (no embedding path).
class AM1 : public NddoMethod, public cpu::QmEngine {
  public:
    /// @brief Total molecular charge [e].
    const int charge;
    /// @brief Spin multiplicity 2S+1 (dimensionless).
    const int multiplicity;
    /// @brief SCF iteration cap (dimensionless).
    int max_iter = 200;
    /// @brief Density residual tolerance (dimensionless).
    double density_tol = 1.0e-8;
    /// @brief Energy residual tolerance [eV].
    double energy_tol = 1.0e-10;

    /// @brief Construct with SCF parameters assigned immediately.
    /// @param charge Total molecular charge [e].
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param max_iter SCF iteration cap (dimensionless).
    /// @param density_tol Density residual tolerance (dimensionless).
    /// @param energy_tol Energy residual tolerance [eV].
    explicit AM1(int charge = 0, int multiplicity = 1, int max_iter = 200,
                 double density_tol = 1.0e-8, double energy_tol = 1.0e-10);
    ~AM1();
    AM1(AM1&&);
    AM1& operator=(AM1&&) = delete;
    /// @brief Copy parameters only (does not copy SCF engine state).
    AM1(const AM1& other);
    AM1& operator=(const AM1&) = delete;

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

    /// @brief Write total energy [eV] and Mulliken charges [e], preserving forces.
    /// @param frame Geometry [Å]; energy and charges out (overwritten).
    /// @return `last_result()`.
    /// @throws std::runtime_error if the SCF does not converge in `max_iter`.
    AM1Result& run(cpu::Frame& frame);

    /// @brief SCF + analytic nuclear forces \(F = -\partial E/\partial R\) [eV/Å]
    ///        contracted from pair-integral derivatives at converged density.
    ///
    /// See `NDDOSolver::compute_forces` for the stationary-density derivative. Writes
    /// `kAtom_fx/fy/fz` [eV/Å] (overwritten, not accumulated) and
    /// `frame.meta[kEnergy]` [eV].
    /// @param frame Geometry [Å]; forces and energy out.
    /// @return `last_result()` with `forces_ev_a` filled (3 · nlocal).
    /// @throws std::runtime_error if the SCF does not converge.
    AM1Result& forces(cpu::Frame& frame);

    /// @brief Results of the most recent `run()` / `forces()`
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
    /// @brief Display name in plan dumps and binding diagnostics.
    /// @return `"am1"`.
    std::string name() const { return "am1"; }
    std::string qm_engine_name() const override { return "am1"; }
    void prepare(const cpu::BuildContext& qm_ctx) override;
    void evaluate(const cpu::LaunchContext& qm_ctx) override;
    // as_embedding() default nullptr — NDDO has no external-charge path yet.
    void reset_history() override;

    /// @brief See `NddoMethod::publish` (energy [eV], charges [e], forces
    ///        [eV/Å]; accumulate or overwrite).
    /// @param frame Target frame; geometry in, results out.
    /// @param forces_requested Also compute analytic forces.
    /// @param accumulate true → `+=` energy/forces; false → `=`.
    void publish(cpu::Frame& frame, bool forces_requested,
                 bool accumulate) override;

  private:
    void sync_engine_();
    std::unique_ptr<NDDOSolver> engine_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
