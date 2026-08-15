/// @file cpu/pw/pw_evaluator.h
/// @brief PWEvaluator — plane-wave DFT energy (+ optional analytic forces).
///
/// Mirrors the Gaussian methods (`cpu::scf::RHF` / `RKS` / …): holds a `PW`
/// inner solver built once;
/// each `launch()` runs the plane-wave SCF and writes energy [eV]. With
/// `with_gradient(true)` (default **false**), ionic forces are filled via
/// `pw_forces_analytic` [eV/Å]. FD oracle remains `pw_forces_fd`.
#pragma once

#include <stdexcept>
#include <utility>

#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Plane-wave DFT total-energy evaluator component.
///
/// Kind `PW`. Writes `frame.meta.get(kEnergy)` [eV]; Ry/Hartree stay
/// on `last_result()`. Forces (when enabled) use analytic local HF + Ewald
/// (`pw_forces_analytic`).
class PWEvaluator : public cpu::PlaneWave {
  public:
    /// @brief Construct from a plane-wave configuration.
    /// @param config Cutoff, k-mesh, convergence, mixing parameters.
    PWEvaluator() = default;
    explicit PWEvaluator(PW method) : method_(std::move(method)) {}

    std::string name() const override { return "pw_evaluator"; }

    /// @brief Enable/disable ionic force evaluation (analytic).
    /// @param enabled When true, `launch()` also writes per-atom forces [eV/Å].
    PWEvaluator& with_gradient(bool enabled) & noexcept {
        compute_forces_ = enabled;
        return *this;
    }
    PWEvaluator&& with_gradient(bool enabled) && noexcept {
        compute_forces_ = enabled;
        return std::move(*this);
    }

    /// @brief True when forces will be written on launch.
    bool compute_forces() const noexcept { return compute_forces_; }

    /// @brief Enable/disable the cell stress write to `kStress`.
    ///
    /// `pw_stress` returns the intensive stress σ [eV/Å³]; the launch path
    /// multiplies it by the cell volume before accumulating, because `kStress`
    /// is the **extensive** Voigt virial [eV] for every producer
    /// (`core/frame_schema.h`). Default **false**.
    /// @param enabled When true, `launch()` also accumulates σ·V [eV].
    PWEvaluator& with_stress(bool enabled) & noexcept {
        compute_stress_ = enabled;
        return *this;
    }
    PWEvaluator&& with_stress(bool enabled) && noexcept {
        compute_stress_ = enabled;
        return std::move(*this);
    }
    bool compute_stress() const noexcept { return compute_stress_; }

    /// @brief Legacy FD step [Å] (only used if force path is switched to FD).
    /// @deprecated Prefer `pw_forces_fd` for oracle tests; evaluator uses analytic.
    PWEvaluator& set_fd_step(double h_ang) & {
        if (h_ang <= 0.0)
            throw std::invalid_argument("PWEvaluator: fd step must be > 0");
        fd_step_ = h_ang;
        return *this;
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Most recent plane-wave SCF result.
    const PWResult& last_result() const { return method_.last_result(); }

    /// @brief Owned concrete method (parameters as members).
    const PW& method() const { return method_; }
    PW& method() { return method_; }

  private:
    PW method_;
    bool compute_forces_ = false;
    bool compute_stress_ = false;
    double fd_step_ = 1e-3; // retained for API compatibility / future FD opt-in
};

} // namespace pw
} // namespace cpu
} // namespace ATV
