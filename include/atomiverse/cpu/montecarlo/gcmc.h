/// @file cpu/montecarlo/gcmc.h
/// @brief grand-canonical Monte Carlo acceptance policy.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/outcome_policy.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief Grand-canonical Metropolis acceptance (Frenkel & Smit §5.6,
///        Algorithms 13–14), single species:
///
///   - displacement: `min(1, exp(-β ΔE))`
///   - insertion  (N → N+1): `min(1, V / (Λ³ (N+1)) · exp(β(μ − ΔE)))`
///   - deletion   (N → N−1): `min(1, Λ³ N / V · exp(-β(μ + ΔE)))`
///
/// with `Λ = h/√(2π m k_B T)` the thermal de Broglie wavelength
/// (`units::lambda_thermal`) and `ΔE = E_trial − E_accepted`. The trial verb
/// is read off `ctx.mc.proposal`; N and V off the trial frame.
class GcmcMetropolis : public OutcomePolicy {
  public:
    /// @brief Construct the μVT acceptance test.
    /// @param temperature Ensemble temperature in run units (must be > 0).
    /// @param mu Chemical potential μ in run energy units (total, including the ideal
    ///        part — the same convention as the NIST SRSW LJ μVT tables once
    ///        converted from reduced units).
    /// @param mass Mass of the (single) insertable species in run mass units
    ///        (must be > 0; sets Λ).
    /// @param rng The run's single RNG stream (shared with the Movers).
    ///        Must outlive this policy.
    /// @throws std::invalid_argument on a non-positive temperature or mass.
    GcmcMetropolis(AccReal temperature, AccReal mu, double mass,
                   std::mt19937_64& rng)
        : temperature_(temperature), mu_(mu), mass_(mass), rng_(rng) {
        if (!std::isfinite(temperature) || temperature <= AccReal(0))
            throw std::invalid_argument(
                "GcmcMetropolis: temperature must be finite and positive");
        if (!std::isfinite(mu))
            throw std::invalid_argument(
                "GcmcMetropolis: chemical potential must be finite");
        if (!std::isfinite(mass) || mass <= 0.0)
            throw std::invalid_argument(
                "GcmcMetropolis: mass must be finite and positive");
    }

    /// @brief Stable identifier, for diagnostics.
    const char* name() const override { return "gcmc_metropolis"; }

    /// @brief Seed the reference energy from the initial configuration.
    void on_start(const cpu::LoopState& s) override {
        e_cur_ = s.frame.meta.get(kEnergy);
        // Λ in the run's units (LAMMPS fix gcmc: hplanck, boltz, mvv2e).
        const double lam = units::lambda_thermal(
            mass_, static_cast<double>(temperature_), s.runtime.units);
        lambda3_ = lam * lam * lam;
    }

    /// @brief Judge the trial now in the frame, by its verb.
    /// @throws std::runtime_error on a verb this ensemble has no rule for
    ///         (a volume move belongs to NPT) or a missing proposal.
    Outcome after_stages(const cpu::LoopState& s) override {
        const mc::Transaction* txn = s.runtime.mc.proposal;
        if (txn == nullptr)
            throw std::runtime_error(
                "GcmcMetropolis: ctx.mc.proposal not published — must run "
                "under an MC preset");
        const AccReal e_trial = s.frame.meta.get(kEnergy);
        const double beta =
            1.0 / (s.runtime.units.boltz * static_cast<double>(temperature_));
        const double dE = static_cast<double>(e_trial - e_cur_);
        const double mu = static_cast<double>(mu_);
        const double v = s.frame.box.volume();

        if (txn->verb == Transaction::Verb::Insert
            || txn->verb == Transaction::Verb::Delete) {
            if (!txn->exchange_mass || *txn->exchange_mass != mass_)
                throw std::invalid_argument("GCMC mover and acceptance policy "
                                            "must use the same exchange mass");
            if (!std::isfinite(v) || v <= 0.0)
                throw std::invalid_argument(
                    "GCMC requires a positive finite box volume");
        }

        // Acceptance probability written as exp(-arg): arg = βΔE − ln(prefactor).
        double arg;
        switch (txn->verb) {
        case Transaction::Verb::None:
            // An empty proposal (e.g. a trial deletion from N = 0) is
            // "simply rejected" — no Boltzmann draw, nothing to undo.
            return Outcome::Rollback;
        case Transaction::Verb::Displace:
            arg = beta * dE;
            break;
        case Transaction::Verb::Insert: {
            // The trial frame already holds N+1 atoms.
            const double n_new = static_cast<double>(s.frame.nlocal());
            arg = beta * (dE - mu) - std::log(v / (lambda3_ * n_new));
            break;
        }
        case Transaction::Verb::Delete: {
            // The trial frame holds N−1 atoms; the rule's N is the pre-delete
            // count.
            const double n_old = static_cast<double>(s.frame.nlocal() + 1);
            arg = beta * (dE + mu) - std::log(lambda3_ * n_old / v);
            break;
        }
        default:
            throw std::runtime_error(
                "GcmcMetropolis: no muVT acceptance rule for this trial verb "
                "(a volume move needs the NPT policy)");
        }

        const double correction = txn->require_log_proposal_ratio();
        if (correction != -std::numeric_limits<double>::infinity()
            && accept_(arg - correction)) {
            e_cur_ = e_trial;
            return Outcome::Commit;
        }
        return Outcome::Rollback;
    }

    /// @brief Energy in run units of the current (last-accepted) configuration.
    AccReal current_energy() const { return e_cur_; }

    /// @brief Λ³ in run volume units at the ensemble temperature (for tests/diagnostics).
    double lambda_cubed() const { return lambda3_; }

  private:
    /// @brief Accept with probability `min(1, exp(-arg))`.
    bool accept_(double arg) {
        if (arg <= 0.0)
            return true;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng_) < std::exp(-arg);
    }

    AccReal temperature_;  ///< Ensemble temperature in run units.
    AccReal mu_;           ///< Chemical potential in run energy units.
    double mass_;          ///< Particle mass [style mass].
    double lambda3_ = 0;   ///< Λ³ [style volume] at `temperature_`, set by
                           ///< `on_start` in the run's units.
    std::mt19937_64& rng_; ///< The run's RNG stream, shared with the Movers.
    AccReal e_cur_ =
        0; ///< Energy of the last accepted state in run energy units.
};

} // namespace mc
} // namespace cpu
} // namespace ATV
