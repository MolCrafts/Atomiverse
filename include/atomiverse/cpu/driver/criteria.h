/// @file cpu/driver/criteria.h
/// @brief The built-in termination Criteria — the loop node's stock verdicts.
///
/// Each of these used to BE a driver category, then a non-component policy
/// object. They are now ~10 lines apiece of verdict-only `Criterion`, and they
/// compose: a
/// Minimize run that must also give up on numerical insanity is
/// `add_criterion(ForceConverged{}) + add_criterion(StopOnNaN{})`, which needed
/// a new class twice over before.
///
/// They live in `cpu/driver/` — not `cpu/component/` — because they are the
/// loop node's built-in termination defaults (`Minimize` attaches
/// `ForceConverged` as a build-time default in `install_defaults()`,
/// `PathDriver` attaches `AllReplicasConverged` in its constructor), and
/// because their whole point is cross-workflow
/// generality: `StopOnNaN` serves MD, Minimize, Path and MC alike. The
/// `Criterion` base (the component KIND) is `cpu/component/criterion.h`;
/// domain-specific criteria stay in their domain (SCF's `ScfConverged` is in
/// `cpu/scf/stages/`).
///
/// ## Writing your own
///
/// @code
///   class StopWhenTooHot : public cpu::Criterion {
///     public:
///       explicit StopWhenTooHot(double t_max) : t_max_(t_max) {}
///       std::string name() const override { return "stop_when_too_hot"; }
///       void launch(const cpu::LaunchContext&) override {} // verdict-only
///       Verdict verdict(const cpu::LoopState& s) override {
///           return (temperature(s.frame) > t_max_) ? Verdict::Stop
///                                                  : Verdict::Continue;
///       }
///     private:
///       double t_max_;
///   };
///
///   sim.add_criterion(StopWhenTooHot{2000.0});
/// @endcode
///
/// This is LAMMPS's `fix halt`, written from OUTSIDE the library.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/criterion.h"

namespace ATV {
namespace cpu {

/// @brief The five-criteria geometry-optimization convergence test.
///
/// ALL five must hold. It reads `ConvergenceMetrics`, which the Optimizer
/// component publishes — so the tolerances live here, with the *policy*, and
/// not (as they did) as five of the eight positional constructor arguments of
/// `LBFGS`, where changing a tolerance meant touching the algorithm and where
/// `CG` had to duplicate the identical test.
class ForceConverged : public Criterion {
  public:
    /// @brief Construct the convergence test from its five tolerances.
    /// @param ftol Max force [eV/Å].
    /// @param rms_ftol RMS force [eV/Å].
    /// @param max_dtol Max displacement [Å].
    /// @param rms_dtol RMS displacement [Å].
    /// @param etol Energy change [eV].
    explicit ForceConverged(double ftol = 0.05, double rms_ftol = 0.03,
                            double max_dtol = 0.002, double rms_dtol = 0.001,
                            double etol = 1e-5)
        : ftol_(ftol), rms_ftol_(rms_ftol), max_dtol_(max_dtol),
          rms_dtol_(rms_dtol), etol_(etol) {}

    /// @brief Identifier reported as the stop reason.
    /// @return `"force_converged"`.
    std::string name() const override { return "force_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Declare convergence when all five criteria hold.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` if every criterion holds, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const ConvergenceMetrics& m = s.metrics;
        // Judge the METRICS — never the iteration counter. Whether a displacement is
        // meaningful is a fact only the component knows (it has the previous geometry),
        // and it says so in `has_displacement`. A criterion that second-guesses this with
        // `iteration == 0` gets it wrong, and I know because I wrote that first: in the
        // minimizer phase order the optimizer has ALREADY taken a step by the time
        // `convergence_check` runs, so driver-iteration 0 is not the optimizer's first
        // step. Skipping the displacement test there declared an untouched lattice
        // converged, having moved nothing.
        const bool disp_ok = !m.has_displacement
                             || (m.max_displacement < max_dtol_
                                 && m.rms_displacement < rms_dtol_);
        const bool ok = m.max_force < ftol_ && m.rms_force < rms_ftol_
                        && disp_ok && m.energy_change < etol_;
        return ok ? Verdict::Converged : Verdict::Continue;
    }

  private:
    double ftol_, rms_ftol_, max_dtol_, rms_dtol_, etol_;
};

/// @brief Stop when isotropic pressure is within a tolerance of a target.
///
/// Reads Voigt stress from `frame.meta` (`kStress`) and the box volume.
/// Intended for `IsotropicCellRelax` variable-cell runs (0 K: kinetic stress
/// omitted). Units: eV/Å³.
class PressureConverged : public Criterion {
  public:
    /// @param P_target Target pressure [eV/Å³].
    /// @param ptol Absolute pressure tolerance [eV/Å³].
    explicit PressureConverged(double P_target = 0.0, double ptol = 1e-5)
        : P_target_(P_target), ptol_(ptol) {
        if (!(ptol > 0.0))
            throw std::invalid_argument("PressureConverged: ptol must be > 0");
    }

    std::string name() const override { return "pressure_converged"; }
    void launch(const LaunchContext&) override {}

    Verdict verdict(const LoopState& s) override {
        const double V = s.frame.box.volume();
        if (!(V > 0.0))
            return Verdict::Stop;
        const auto& st = s.frame.meta.get(kStress);
        const double P = (st[0] + st[1] + st[2]) / (3.0 * V);
        return (std::fabs(P - P_target_) < ptol_) ? Verdict::Converged
                                                  : Verdict::Continue;
    }

  private:
    double P_target_;
    double ptol_;
};

/// @brief Converged when every relaxed diagonal stress is at its target.
///
/// Reads Voigt `kStress` as **intensive stress [eV/Å³]** — the PW
/// (`with_stress`) / GPW (`compute_stress`) convention — and never divides by
/// volume. For classical extensive-virial writers use `PressureConverged`.
/// Per-axis maskable: pair the mask with `AnisotropicCellRelax::relax_*` so a
/// constrained axis's reaction stress cannot block convergence
/// (CP2K `CELL_OPT` `CONSTRAINT` twin).
class StressConverged : public Criterion {
  public:
    /// @param sigma_target Target diagonal stress [eV/Å³] (0 = relaxed cell).
    /// @param stol Absolute stress tolerance [eV/Å³].
    /// @param check_x Judge σ_xx.
    /// @param check_y Judge σ_yy.
    /// @param check_z Judge σ_zz.
    explicit StressConverged(double sigma_target = 0.0, double stol = 1e-4,
                             bool check_x = true, bool check_y = true,
                             bool check_z = true)
        : sigma_target_(sigma_target), stol_(stol),
          check_{check_x, check_y, check_z} {
        if (!(stol > 0.0))
            throw std::invalid_argument("StressConverged: stol must be > 0");
        if (!check_x && !check_y && !check_z)
            throw std::invalid_argument("StressConverged: every axis masked");
    }

    std::string name() const override { return "stress_converged"; }
    void launch(const LaunchContext&) override {}

    Verdict verdict(const LoopState& s) override {
        // kStress is the extensive Voigt virial [eV] (core/frame_schema.h), so
        // the intensive stress this criterion judges is st[d]/V. Reading st[d]
        // directly — as this did before the convention was fixed — compared an
        // eV number against an eV/Å³ tolerance, which for a classical evaluator
        // is off by the cell volume (~10³ Å³) and fired on the first iteration.
        const double V = s.frame.box.volume();
        if (!(V > 0.0))
            throw std::runtime_error("StressConverged: non-positive volume");
        const auto& st = s.frame.meta.get(kStress);
        double r = 0.0;
        for (std::size_t d = 0; d < 3; ++d)
            if (check_[d])
                r = std::max(r, std::fabs(st[d] / V - sigma_target_));
        return (r < stol_) ? Verdict::Converged : Verdict::Continue;
    }

  private:
    double sigma_target_;
    double stol_;
    bool check_[3];
};

/// @brief A coupled chain is converged when EVERY replica is — never when the first is.
///
/// The AND that `PathDriver` used to spell out inline. Reads
/// `ConvergenceMetrics::replicas_converged`, which the band optimizer publishes
/// as a fraction.
class AllReplicasConverged : public Criterion {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"all_replicas_converged"`.
    std::string name() const override { return "all_replicas_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Declare convergence when the whole band has converged.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` when `metrics.replicas_converged` reaches 1.0, else
    ///         `Continue`.
    Verdict verdict(const LoopState& s) override {
        return (s.metrics.replicas_converged >= 1.0) ? Verdict::Converged
                                                     : Verdict::Continue;
    }
};

/// @brief Abandon a run whose energy has gone non-finite.
///
/// `Stop`, never `Converged` — which is exactly why the severity order matters. A blown-up
/// system has zero force on every atom (they are all NaN, and NaN fails every `<`
/// comparison... but a *diverged* one can genuinely park at a force minimum of garbage).
/// Reporting success for it would be worse than crashing.
class StopOnNaN : public Criterion {
  public:
    /// @brief Identifier reported as the stop reason.
    /// @return `"stop_on_nan"`.
    std::string name() const override { return "stop_on_nan"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Abandon the run when the measured energy is not finite.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Stop` if `metrics.energy` [eV] is NaN or infinite, else `Continue`.
    Verdict verdict(const LoopState& s) override {
        const double e = s.metrics.energy;
        return std::isfinite(e) ? Verdict::Continue : Verdict::Stop;
    }
};

} // namespace cpu
} // namespace ATV
