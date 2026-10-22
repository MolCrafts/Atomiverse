/// @file cpu/semiempirical/nddo_method.h
/// @brief Shared NDDO self-consistency Driver for AM1 and RM1.
#pragma once

#include <memory>

#include "atomiverse/cpu/driver/driver.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

class NDDOSolver;
struct NDDOResult;
class NddoStep;
class NddoConverged;

/// @brief `cpu::Driver<nddo::Phase>` shared by `AM1` and `RM1`.
///
/// One iteration is `diagonalize → density → fock → check`. The guess Fock
/// is built in `NDDOSolver::scf_prepare` before the first iteration. Stages
/// are rewired on every `drive_` so a step lambda never outlives that call.
///
/// Mounted in an outer loop (`md.on(md::Phase::force).add(AM1{})`), the
/// method is a nested node: each launch runs the SCF on the parent Runtime
/// and accumulates energy [eV], Mulliken charges [e] and — when the phase
/// requests forces — finite-difference forces [eV/Å] onto the Frame.
class NddoMethod : public cpu::Driver<ATV::nddo::Phase> {
  public:
    bool requires_full_frame() const override { return true; }
    ~NddoMethod() override;

    /// @brief Energy source for an outer loop's validation.
    /// @return true.
    bool provides_energy() const { return true; }
    /// @brief Force source for an outer loop's validation.
    /// @return true.
    bool provides_force() const { return true; }

    /// @brief Run the SCF and write every result channel onto the frame:
    ///        total energy [eV] to `kEnergy` and `kEnergyElectronic`, Mulliken
    ///        charges [e] to `kAtom_charge`, and — when asked — forces [eV/Å]
    ///        to `kAtom_f{x,y,z}`.
    /// @param frame Target frame; geometry in, results out.
    /// @param forces_requested Also compute finite-difference forces.
    /// @param accumulate true → `+=` energy/forces (a co-author in a driver
    ///        phase); false → `=`. Charges are always last-writer-wins.
    virtual void publish(cpu::Frame& frame, bool forces_requested,
                         bool accumulate) = 0;

    /// @brief Nested mount: remember the parent Runtime the SCF runs on.
    /// @param rt Parent Runtime.
    void build_on(cpu::Runtime& rt) override;
    /// @brief One nested solve at the frame's geometry (molecule unwrapped
    ///        around atom 0 for the solve), accumulated onto the frame.
    /// @param ctx Parent Context (force request read from `ctx.exec`).
    /// @param frame Parent Frame.
    /// @param budget Ignored; the SCF runs to convergence or `max_iter`.
    /// @return Iterations of the solve.
    int run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                          int budget) override;
    /// @brief Detach from the parent Runtime.
    /// @param rt Parent Runtime.
    void destroy_on(cpu::Runtime& rt) override;

    NddoMethod(const NddoMethod&) = delete;
    NddoMethod& operator=(const NddoMethod&) = delete;
    NddoMethod(NddoMethod&&) = delete;
    NddoMethod& operator=(NddoMethod&&) = delete;

  protected:
    /// Defined out of line: the stage `unique_ptr`s are incomplete here.
    NddoMethod();

    /// @brief Run the NDDO self-consistency loop on an already-built solver.
    /// @param solver Inner NDDO engine. Must already be `build()`-ed.
    /// @param frame Geometry [Å]. Not modified.
    /// @return The solver result. Energies [eV] and Mulliken charges [e].
    /// @throws std::runtime_error if the SCF does not converge.
    NDDOResult& drive_(NDDOSolver& solver, const cpu::Frame& frame);

    /// @brief Write a finished SCF onto a Frame.
    ///
    /// Calls `drive_`, then optionally the fixed-density finite-difference
    /// forces. Charges always overwrite. Energy and forces assign or
    /// accumulate.
    /// @param solver Inner engine.
    /// @param force_step Finite-difference displacement [Å].
    /// @param frame Geometry in; energy [eV], charges [e], forces [eV/Å] out.
    /// @param forces_requested Also write `kAtom_f*`.
    /// @param accumulate True → `+=` energy and forces. False → `=`.
    void publish_(NDDOSolver& solver, double force_step, cpu::Frame& frame,
                  bool forces_requested, bool accumulate);

    /// @brief Drop staged components and the inner backend.
    void reset_built_driver_();

    void validate_components() const override {}

    /// Iterations executed by the most recent solve.
    int iterations_ = 0;

  private:
    /// Parent Runtime while nested (`build_on`); null for standalone entries.
    cpu::Runtime* host_runtime_ = nullptr;
    /// True during a nested solve: `drive_` steps on `host_runtime_`.
    bool nested_active_ = false;
    bool loop_converged_ = false;
    std::unique_ptr<NddoStep> diagonalize_;
    std::unique_ptr<NddoStep> density_;
    std::unique_ptr<NddoStep> fock_;
    std::unique_ptr<NddoStep> check_;
    std::unique_ptr<NddoConverged> converged_;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
