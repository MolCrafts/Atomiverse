/// @file cpu/montecarlo/npt_mc.h
/// @brief isothermal-isobaric Monte Carlo acceptance policy.
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

/// @brief NPT Metropolis acceptance (Frenkel & Smit §5.4): particle
///        displacements use the canonical rule; a volume move sampled
///        **uniformly in V** is accepted with
///        `min(1, exp(-β[ΔE + P·ΔV − N k_B T ln(V'/V)]))`.
///
/// Owns the energy AND volume of the last accepted configuration; reads the
/// trial verb off `ctx.mc.proposal` (an `OutcomePolicy` sees a `LoopState`,
/// and the Context slice is how it learns what kind of move it is judging
/// without ever touching the frame).
class NptMetropolis : public OutcomePolicy {
  public:
    /// @brief Construct the NPT acceptance test.
    /// @param temperature Ensemble temperature [temperature] (must be > 0).
    /// @param pressure External pressure in the run's pressure unit
    ///        (LAMMPS-standard sign: positive = compressive; the unit of the
    ///        MD barostats and of thermo `press`).
    /// @param rng The run's single RNG stream (shared with the Movers so one
    ///        seed reproduces one trajectory). Must outlive this policy.
    /// @throws std::invalid_argument if `temperature <= 0`.
    NptMetropolis(AccReal temperature, AccReal pressure, std::mt19937_64& rng)
        : temperature_(temperature), pressure_(pressure), rng_(rng) {
        if (!std::isfinite(temperature) || temperature <= AccReal(0))
            throw std::invalid_argument(
                "NptMetropolis: temperature must be finite and positive");
        if (!std::isfinite(pressure))
            throw std::invalid_argument(
                "NptMetropolis: pressure must be finite");
    }

    /// @brief Stable identifier, for diagnostics.
    const char* name() const override { return "npt_metropolis"; }

    /// @brief Seed the reference energy AND volume from the initial
    ///        configuration.
    void on_start(const cpu::LoopState& s) override {
        e_cur_ = s.frame.meta.get(kEnergy);
        v_cur_ = s.frame.box.volume();
    }

    /// @brief Judge the trial now in the frame, by its verb.
    /// @throws std::runtime_error on a verb this ensemble has no rule for
    ///         (insert/delete belong to μVT) or a missing proposal.
    Outcome after_stages(const cpu::LoopState& s) override {
        const mc::Transaction* txn = s.runtime.mc.proposal;
        if (txn == nullptr)
            throw std::runtime_error(
                "NptMetropolis: ctx.mc.proposal not published — must run "
                "under an MC preset");
        const AccReal e_trial = s.frame.meta.get(kEnergy);
        const double beta =
            1.0 / (s.runtime.units.boltz * static_cast<double>(temperature_));
        const double dE = static_cast<double>(e_trial - e_cur_);

        double arg; // exponent of the Boltzmann acceptance factor
        switch (txn->verb) {
        case Transaction::Verb::None:
            // An empty proposal (e.g. a volume draw clamped at V ≤ 0) is
            // rejected outright — no Boltzmann draw, nothing to undo.
            return Outcome::Rollback;
        case Transaction::Verb::Displace:
            arg = beta * dE;
            break;
        case Transaction::Verb::Box: {
            const double v_trial = s.frame.box.volume();
            const double n = static_cast<double>(s.frame.nlocal());
            // P·ΔV in energy: the pressure over the run's nktv2p.
            const double p_energy =
                static_cast<double>(pressure_) / s.runtime.units.nktv2p;
            arg = beta * (dE + p_energy * (v_trial - v_cur_))
                  - n * std::log(v_trial / v_cur_);
            break;
        }
        default:
            throw std::runtime_error(
                "NptMetropolis: no NPT acceptance rule for this trial verb "
                "(insert/delete need the muVT policy)");
        }

        const double correction = txn->require_log_proposal_ratio();
        if (correction != -std::numeric_limits<double>::infinity()
            && accept_(arg - correction)) {
            e_cur_ = e_trial;
            v_cur_ = s.frame.box.volume();
            return Outcome::Commit;
        }
        return Outcome::Rollback;
    }

    /// @brief Energy [eV] of the current (last-accepted) configuration.
    AccReal current_energy() const { return e_cur_; }

    /// @brief Volume [Å³] of the current (last-accepted) configuration.
    double current_volume() const { return v_cur_; }

  private:
    /// @brief Accept with probability `min(1, exp(-arg))`.
    bool accept_(double arg) {
        if (arg <= 0.0)
            return true;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng_) < std::exp(-arg);
    }

    AccReal temperature_;  ///< Ensemble temperature in run units.
    AccReal pressure_;     ///< External pressure [style pressure].
    std::mt19937_64& rng_; ///< The run's RNG stream, shared with the Movers.
    AccReal e_cur_ = 0;    ///< Energy of the last accepted state [eV].
    double v_cur_ = 0;     ///< Volume of the last accepted state [Å³].
};

} // namespace mc
} // namespace cpu
} // namespace ATV
