/// @file cpu/montecarlo/gcmc.h
/// @brief cpu::mc::GcmcMetropolis + cpu::mc::Gcmc — grand-canonical (μVT)
///        Monte Carlo: the acceptance policy and the preset that installs it.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/driver/outcome_policy.h"
#include "atomiverse/cpu/montecarlo/mc_base.h"
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
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param mu_ev Chemical potential μ [eV] (total, including the ideal
    ///        part — the same convention as the NIST SRSW LJ μVT tables once
    ///        converted from reduced units).
    /// @param mass_amu Mass of the (single) insertable species [amu]
    ///        (must be > 0; sets Λ).
    /// @param rng The run's single RNG stream (shared with the Movers).
    ///        Must outlive this policy.
    /// @throws std::invalid_argument on a non-positive temperature or mass.
    GcmcMetropolis(AccReal temperature_K, AccReal mu_ev, double mass_amu,
                   std::mt19937_64& rng)
        : temperature_(temperature_K), mu_(mu_ev), rng_(rng) {
        if (!(temperature_K > AccReal(0)))
            throw std::invalid_argument(
                "GcmcMetropolis: temperature must be > 0 K");
        if (!(mass_amu > 0.0))
            throw std::invalid_argument("GcmcMetropolis: mass must be > 0 amu");
        const double lam =
            units::lambda_thermal(mass_amu, static_cast<double>(temperature_K));
        lambda3_ = lam * lam * lam;
    }

    /// @brief Stable identifier, for diagnostics.
    const char* name() const override { return "gcmc_metropolis"; }

    /// @brief Seed the reference energy from the initial configuration.
    void on_start(const cpu::LoopState& s) override {
        e_cur_ = s.frame.meta.get(kEnergy);
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
            1.0 / (units::kB_ev_per_k * static_cast<double>(temperature_));
        const double dE = static_cast<double>(e_trial - e_cur_);
        const double mu = static_cast<double>(mu_);
        const double v = s.frame.box.volume();

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

        if (accept_(arg)) {
            e_cur_ = e_trial;
            return Outcome::Commit;
        }
        return Outcome::Rollback;
    }

    /// @brief Energy [eV] of the current (last-accepted) configuration.
    AccReal current_energy() const { return e_cur_; }

    /// @brief Λ³ [Å³] at the ensemble temperature (for tests/diagnostics).
    double lambda_cubed() const { return lambda3_; }

  private:
    /// @brief Accept with probability `min(1, exp(-arg))`.
    bool accept_(double arg) {
        if (arg <= 0.0)
            return true;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng_) < std::exp(-arg);
    }

    AccReal temperature_;  ///< Ensemble temperature [K].
    AccReal mu_;           ///< Chemical potential [eV].
    double lambda3_ = 0;   ///< Λ³ [Å³] at `temperature_`.
    std::mt19937_64& rng_; ///< The run's RNG stream, shared with the Movers.
    AccReal e_cur_ = 0;    ///< Energy of the last accepted state [eV].
};

/// @brief Grand-canonical (μVT) Monte Carlo — a preset over the shared MC
///        loop that attaches `GcmcMetropolis` in its constructor.
///
/// Pair it with a `MoveSelector` mixing displacements and insert/delete:
/// @code{.cpp}
///   cpu::mc::Gcmc mc{/*T=*/120.0, /*mu [eV]=*/-0.1, /*mass [amu]=*/39.948,
///                    /*seed=*/7u};
///   mc.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
///   mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
///   mc.set_mover(cpu::mc::MoveSelector{}
///                    .add(cpu::mc::TranslateMove{0.2}, 0.5)
///                    .add(cpu::mc::InsertDeleteMove{18, 39.948}, 0.5));
///   mc.build();
///   mc.run(frame, 200000);
///   double n_avg = ...; // observe frame.nlocal() along the chain
/// @endcode
class Gcmc : public McBase {
  public:
    /// @brief Construct the μVT sampler (single insertable species).
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param mu_ev Chemical potential μ [eV].
    /// @param mass_amu Mass of the insertable species [amu] (sets Λ).
    /// @param seed Seed for the single reproducible RNG stream.
    /// @throws std::invalid_argument on a non-positive temperature or mass.
    Gcmc(AccReal temperature_K, AccReal mu_ev, double mass_amu,
         std::uint64_t seed)
        : McBase(seed), policy_(this->set_outcome(GcmcMetropolis{
                            temperature_K, mu_ev, mass_amu, rng_})) {}

    /// @brief Energy [eV] of the current (last-accepted) configuration.
    AccReal current_energy() const { return policy_.current_energy(); }

  private:
    GcmcMetropolis& policy_; ///< The preset's own policy, for stat read-back.
};

} // namespace mc
} // namespace cpu
} // namespace ATV
