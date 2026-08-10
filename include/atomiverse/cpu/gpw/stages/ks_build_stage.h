/// @file cpu/gpw/stages/ks_build_stage.h
/// @brief KsBuildStage — `gpw::Phase::ks_build`: H[P] + energy + DIIS error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief KS Hamiltonian build stage for the GPW Driver loop.
///
/// Invokes a path-local builder (Γ RKS collocate→Poisson→XC→integrate, …)
/// supplied by the method at wire time, then forms the DIIS commutator
/// residual. Sets `skip_update` when the shipping GPW gate is already met
/// (`|ΔE| < tol` ∧ `max|err| < err_tol`, iter > 0) so Mix/Diag/Density leave
/// the raw H for `store_converged_state` (break-before-mix semantics).
class KsBuildStage : public cpu::Component {
  public:
    /// @brief Builder signature: write H and e_total from the current density P.
    using Builder = std::function<void(const linalg::RealMatrix& P,
                                       linalg::RealMatrix& H, double& e_total)>;

    /// @param builder    Path-local KS build.
    /// @param P          Current density (read).
    /// @param S          Overlap (read).
    /// @param X          Löwdin orthogonalizer (read).
    /// @param H          KS matrix [Hartree] (write).
    /// @param err        DIIS error matrix (write).
    /// @param e_total    Total energy [Hartree] (write).
    /// @param err_norm   max|err| residual (write).
    /// @param prev_energy Previous iteration energy [Hartree] (read/write).
    /// @param tol_energy |ΔE| gate [Hartree].
    /// @param tol_err    max|err| gate (dimensionless).
    /// @param skip_update Set true when gate met (Mix/Diag/Density no-op).
    KsBuildStage(Builder builder, const linalg::RealMatrix& P,
                 const linalg::RealMatrix& S, const linalg::RealMatrix& X,
                 linalg::RealMatrix& H, linalg::RealMatrix& err, double& e_total,
                 double& err_norm, double& prev_energy, double tol_energy,
                 double tol_err, bool& skip_update)
        : builder_(std::move(builder)), P_(P), S_(S), X_(X), H_(H), err_(err),
          e_total_(e_total), err_norm_(err_norm), prev_energy_(prev_energy),
          tol_energy_(tol_energy), tol_err_(tol_err), skip_update_(skip_update) {
    }

    /// @brief Component name.
    /// @return `"gpw_ks_build_stage"`.
    std::string name() const override { return "gpw_ks_build_stage"; }

    /// @brief Default binding: `gpw::Phase::ks_build` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::ks_build, 0)};
    }

    /// @brief Build H[P], energy, DIIS residual; arm skip_update if converged.
    /// @param ctx Launch context; self-guards on `gpw::Phase::ks_build`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::ks_build))
            return;
        builder_(P_, H_, e_total_);
        err_ = internal::gpw_diis_error(H_, P_, S_, X_);
        err_norm_ = 0.0;
        for (const double v : err_)
            err_norm_ = std::max(err_norm_, std::abs(v));

        const bool no_stop = std::getenv("ATV_GPW_NOSTOP") != nullptr;
        const bool has_prev = !std::isnan(prev_energy_);
        const double dE = has_prev ? (e_total_ - prev_energy_)
                                   : std::numeric_limits<double>::infinity();
        skip_update_ = !no_stop && has_prev && std::abs(dE) < tol_energy_
                       && err_norm_ < tol_err_;
        prev_energy_ = e_total_;
    }

    /// @brief Retarget tolerances for this run.
    /// @param tol_energy New |ΔE| [Hartree].
    /// @param tol_err New max|err|.
    void set_tolerances(double tol_energy, double tol_err) {
        tol_energy_ = tol_energy;
        tol_err_ = tol_err;
    }

  private:
    Builder builder_;
    const linalg::RealMatrix& P_;
    const linalg::RealMatrix& S_;
    const linalg::RealMatrix& X_;
    linalg::RealMatrix& H_;
    linalg::RealMatrix& err_;
    double& e_total_;
    double& err_norm_;
    double& prev_energy_;
    double tol_energy_;
    double tol_err_;
    bool& skip_update_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
