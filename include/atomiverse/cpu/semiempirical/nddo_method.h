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
class NddoMethod : public cpu::Driver<ATV::nddo::Phase> {
  public:
    bool requires_full_frame() const override { return true; }
    ~NddoMethod() override;

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

  private:
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
