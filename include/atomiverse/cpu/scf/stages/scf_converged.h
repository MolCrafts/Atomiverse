/// @file cpu/scf/stages/scf_converged.h
/// @brief cpu::scf::ScfConverged — the SCF `check` stage as a Criterion.
///
/// The whole-object incarnation of `internal::DualConvergenceCheck`: measure
/// and judge in ONE Component. Bound to `scf::Phase::check`, its `launch()`
/// MEASURES the energy change and the combined α/β DIIS-error Frobenius norm
/// (byte-for-byte the `DualConvergenceCheck::step` arithmetic) into its own
/// plainly-named members; its `verdict()` JUDGES those members against the two
/// tolerances and nothing else.
///
/// It never writes `ConvergenceMetrics` or the loop's status — it only RETURNS a
/// `Verdict`, which the driver pulls once per iteration and composes by
/// severity (Stop > Converged > Continue). A component still cannot declare its
/// own loop converged.
///
/// Flag-gated placement: the `Criterion` base (`cpu/component/criterion.h`) is
/// unconditional, but this concrete SCF criterion lives in the SCF domain dir
/// and is only compiled into `ATV_BUILD_HF` translation units.
#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <xtensor/reducers/xnorm.hpp>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief SCF convergence test — PySCF-aligned `|ΔE| ∧ ‖err‖`, as a Criterion.
///
/// Wires to the method driver's iteration state by reference: it READS the
/// running total energy and the two DIIS-error channels, and owns its rolling
/// `prev_energy_`. Restricted SCF judges the α channel alone; unrestricted
/// combines both channels' Frobenius norms.
class ScfConverged : public cpu::Criterion {
  public:
    /// @brief Construct the convergence criterion, wiring the state it reads.
    /// @param tol_energy Energy-change tolerance |ΔE| must fall below [Hartree].
    /// @param tol_grad DIIS-error tolerance ‖err‖_F must fall below
    ///        [dimensionless Frobenius norm].
    /// @param restricted True ⇒ judge the α channel only; false ⇒ combine α/β.
    /// @param e_total Running total electronic + nuclear energy [Hartree].
    /// @param err_alpha α-channel DIIS error matrix [dimensionless].
    /// @param err_beta β-channel DIIS error matrix [dimensionless]; ignored when
    ///        `restricted`.
    ScfConverged(double tol_energy, double tol_grad, bool restricted,
                 const double& e_total, const RealMatrix& err_alpha,
                 const RealMatrix& err_beta)
        : tol_energy_(tol_energy), tol_grad_(tol_grad), restricted_(restricted),
          e_total_(e_total), err_alpha_(err_alpha), err_beta_(err_beta) {}

    /// @brief Component identifier, reported as the stop reason.
    /// @return `"scf_converged"`.
    std::string name() const override { return "scf_converged"; }

    /// @brief Bind the measurement to the SCF `check` phase.
    /// @return Exactly one binding: `scf::Phase::check`, order 0.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::scf::Phase::check, 0)};
    }

    /// @brief MEASURE ΔE and the DIIS-error norm for the iteration just built.
    ///
    /// Self-guards on `scf::Phase::check`. The first launch after `reset()`
    /// reports ΔE = +∞ (no prior energy), mirroring
    /// `DualConvergenceCheck::step` (dual_convergence_check.h:57). The DIIS
    /// error reproduces `DualConvergenceCheck::step` exactly
    /// (dual_convergence_check.h:60-66): restricted uses ‖err_α‖_F; unrestricted
    /// combines √(‖err_α‖² + ‖err_β‖²). Writes only its own members — never
    /// `ConvergenceMetrics`.
    /// @param ctx Launch context; `ctx.at(scf::Phase::check)` is the self-guard.
    void launch(const LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::check))
            return;

        delta_energy_ = std::isnan(prev_energy_)
                            ? std::numeric_limits<double>::infinity()
                            : e_total_ - prev_energy_;

        const double ea = xt::norm_l2(err_alpha_)();
        if (restricted_) {
            diis_error_ = ea;
        } else {
            const double eb = xt::norm_l2(err_beta_)();
            diis_error_ = std::sqrt(ea * ea + eb * eb);
        }

        prev_energy_ = e_total_;
    }

    /// @brief JUDGE the iteration from this criterion's own measurements only.
    ///
    /// Reads solely `delta_energy_` and `diis_error_` — never `state.metrics`,
    /// never the iteration counter (iteration 0 Continues naturally because
    /// `delta_energy_` is +∞ there).
    /// @param state Loop state (unread — this criterion judges its own members).
    /// @return `Converged` iff `|ΔE| < tol_energy` and `‖err‖ < tol_grad`, else
    ///         `Continue`.
    ATV::Verdict verdict(const cpu::LoopState& state) override {
        (void)state;
        const bool converged =
            std::abs(delta_energy_) < tol_energy_ && diis_error_ < tol_grad_;
        return converged ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }

    /// @brief Energy change measured at the last `launch()`.
    /// @return |ΔE| signed value E_total − E_prev [Hartree] (+∞ before the first
    ///         post-reset measurement).
    double delta_energy() const { return delta_energy_; }

    /// @brief DIIS-error norm measured at the last `launch()`.
    /// @return ‖err‖_F [dimensionless Frobenius norm].
    double diis_error() const { return diis_error_; }

    /// @brief Reset the rolling energy state at the start of a new run.
    ///
    /// Restores the prev-energy sentinel so the next `launch()` reports ΔE = +∞.
    void reset() {
        prev_energy_ = std::numeric_limits<double>::quiet_NaN();
        delta_energy_ = std::numeric_limits<double>::infinity();
        diis_error_ = 0.0;
    }

    /// @brief Retarget the two tolerances for the next run.
    ///
    /// Parameters-are-members house style: the driver sets these so an ASPC
    /// corrector can run against a loosened threshold (and restores the full
    /// threshold afterwards) without a second criterion object. Judgement still
    /// reads only these two members. @see verdict
    /// @param tol_energy New |ΔE| tolerance [Hartree].
    /// @param tol_grad New DIIS-error tolerance [dimensionless Frobenius norm].
    void set_tolerances(double tol_energy, double tol_grad) {
        tol_energy_ = tol_energy;
        tol_grad_ = tol_grad;
    }

  private:
    double tol_energy_;           ///< Energy tolerance [Hartree].
    double tol_grad_;             ///< DIIS-error tolerance [Frobenius norm].
    bool restricted_;             ///< α-only (true) vs combined α/β (false).
    const double& e_total_;       ///< Running total energy [Hartree].
    const RealMatrix& err_alpha_; ///< α-channel DIIS error [dimensionless].
    const RealMatrix& err_beta_;  ///< β-channel DIIS error [dimensionless].

    double delta_energy_ =
        std::numeric_limits<double>::infinity(); ///< Last |ΔE| [Hartree].
    double diis_error_ = 0.0;                    ///< Last ‖err‖_F.
    double prev_energy_ =
        std::numeric_limits<double>::quiet_NaN(); ///< Prev-energy sentinel.
};

} // namespace scf
} // namespace cpu
} // namespace ATV
