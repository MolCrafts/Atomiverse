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

/// @brief Stop when the isotropic pressure is within a tolerance of a target.
///
/// The *virial* is the configurational stress tensor an evaluator accumulates
/// while it computes forces; the Frame carries it on `kStress` in **Voigt
/// order** — the six independent components of a symmetric 3×3 tensor listed
/// as (xx, yy, zz, xy, xz, yz). That key is **extensive**: it holds a
/// whole-cell energy \(W_{ab}\) in eV, not a stress, for every producer
/// (`core/frame_schema.h`). The pressure is therefore its trace over three
/// times the cell volume \(V\) [Å³],
///
/// \[ P = \frac{W_{xx}+W_{yy}+W_{zz}}{3V} \quad [\mathrm{eV/Å^3}], \]
///
/// and this criterion returns `Converged` once \(|P - P_{\mathrm{target}}|\)
/// drops below `ptol`. Sign convention is LAMMPS's, compression-positive;
/// 1 bar ≈ 6.2415e-7 eV/Å³.
///
/// The kinetic (ideal-gas) term is deliberately not included — it lives on
/// its own key `kStressKinetic` — which is what makes this the right test for
/// a 0 K `IsotropicCellRelax` / `JointIsotropicRelax` run, where the atoms
/// have no velocities to speak of.
///
/// `StressConverged` is the per-axis sibling: it reads the same extensive key
/// and divides by the same volume, but judges each diagonal component on its
/// own instead of their mean, and can mask axes out.
class PressureConverged : public Criterion {
  public:
    /// @brief Construct the isotropic-pressure test.
    /// @param P_target Target pressure [eV/Å³]; 0 is a relaxed cell, negative
    ///                 is a tensile setpoint.
    /// @param ptol Absolute pressure tolerance [eV/Å³]; must be > 0.
    /// @throws std::invalid_argument when `ptol <= 0` — a non-positive
    ///         tolerance can never be met, so the loop would never stop.
    explicit PressureConverged(double P_target = 0.0, double ptol = 1e-5)
        : P_target_(P_target), ptol_(ptol) {
        if (!(ptol > 0.0))
            throw std::invalid_argument("PressureConverged: ptol must be > 0");
    }

    /// @brief Identifier reported as the stop reason.
    /// @return `"pressure_converged"`.
    std::string name() const override { return "pressure_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Judge the isotropic virial pressure of the current Frame.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` when |P − P_target| < ptol [eV/Å³]; `Stop` when the
    ///         cell volume is not positive (a collapsed box has no pressure to
    ///         judge); otherwise `Continue`.
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
/// Same input as `PressureConverged` — the **extensive** Voigt virial
/// \(W_{ab}\) [eV] on `kStress`, for every producer (`core/frame_schema.h`) —
/// and the same divide by the cell volume \(V\) [Å³]. What differs is the
/// *shape* of the test: instead of the mean \(\mathrm{tr}(W)/(3V)\), this one
/// judges each diagonal component separately as the intensive stress
///
/// \[ \sigma_{dd} = W_{dd}/V \quad [\mathrm{eV/Å^3}],\qquad d \in \{x,y,z\}, \]
///
/// and returns `Converged` once \(\max_d |\sigma_{dd} -
/// \sigma_{\mathrm{target}}|\) over the **unmasked** axes falls below `stol`.
/// Producers that naturally compute an intensive stress (plane-wave
/// `PWEvaluator::with_stress`, GPW `GPWEvaluator::compute_stress`) multiply by
/// the volume at the Frame write, so they land on the same convention as the
/// classical pair/bonded virials and need no special handling here.
///
/// Per-axis maskable: pair the mask with `AnisotropicCellRelax::relax_*` so a
/// constrained axis's reaction stress — the stress the frozen axis pushes back
/// with, which no cell step is driving to the target — cannot block
/// convergence (CP2K `CELL_OPT` `CONSTRAINT` twin).
class StressConverged : public Criterion {
  public:
    /// @brief Construct the per-axis diagonal-stress test.
    /// @param sigma_target Target diagonal stress [eV/Å³] (0 = relaxed cell).
    /// @param stol Absolute stress tolerance [eV/Å³]; must be > 0.
    /// @param check_x Judge σ_xx.
    /// @param check_y Judge σ_yy.
    /// @param check_z Judge σ_zz.
    /// @throws std::invalid_argument when `stol <= 0`, or when all three axes
    ///         are masked — a criterion that judges nothing would declare
    ///         victory on iteration zero.
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

    /// @brief Identifier reported as the stop reason.
    /// @return `"stress_converged"`.
    std::string name() const override { return "stress_converged"; }

    /// @brief Verdict-only: nothing to measure, nothing to launch.
    void launch(const LaunchContext&) override {}

    /// @brief Judge every unmasked diagonal stress of the current Frame.
    /// @param s Iteration index, frame, context, metrics.
    /// @return `Converged` when the largest unmasked |σ_dd − sigma_target| is
    ///         below `stol` [eV/Å³], else `Continue`.
    /// @throws std::runtime_error when the cell volume is not positive.
    Verdict verdict(const LoopState& s) override {
        // kStress is the extensive Voigt virial [eV] (core/frame_schema.h), so
        // the intensive stress this criterion judges is st[d]/V. Comparing
        // st[d] itself against `stol` would put an eV number against an
        // eV/Å³ tolerance — off by the cell volume (~10³ Å³), which fires on
        // the first iteration and reports an unrelaxed cell as converged.
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
