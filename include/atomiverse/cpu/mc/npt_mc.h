/// @file cpu/mc/npt_mc.h
/// @brief cpu::mc::NptMetropolis + cpu::mc::NptMc — isothermal–isobaric
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
#include "atomiverse/cpu/mc/mc_base.h"
#include "atomiverse/cpu/mc/transaction.h"

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
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param pressure_ev_A3 External pressure [eV/Å³] (LAMMPS-standard sign:
    ///        positive = compressive; same unit as the MD barostats).
    /// @param rng The run's single RNG stream (shared with the Movers so one
    ///        seed reproduces one trajectory). Must outlive this policy.
    /// @throws std::invalid_argument if `temperature_K <= 0`.
    NptMetropolis(AccReal temperature_K, AccReal pressure_ev_A3,
                  std::mt19937_64& rng)
        : temperature_(temperature_K), pressure_(pressure_ev_A3), rng_(rng) {
        if (!(temperature_K > AccReal(0)))
            throw std::invalid_argument(
                "NptMetropolis: temperature must be > 0 K");
    }

    /// @brief Stable identifier, for diagnostics.
    const char* name() const override { return "npt_metropolis"; }

    /// @brief Seed the reference energy AND volume from the initial
    ///        configuration.
    void on_start(const cpu::LoopState& s) override {
        e_cur_ = s.frame.meta.get(kEnergy);
        v_cur_ = s.frame.simbox.volume();
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
            1.0 / (units::kB_ev_per_k * static_cast<double>(temperature_));
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
            const double v_trial = s.frame.simbox.volume();
            const double n = static_cast<double>(s.frame.nlocal());
            arg =
                beta
                    * (dE + static_cast<double>(pressure_) * (v_trial - v_cur_))
                - n * std::log(v_trial / v_cur_);
            break;
        }
        default:
            throw std::runtime_error(
                "NptMetropolis: no NPT acceptance rule for this trial verb "
                "(insert/delete need the muVT policy)");
        }

        if (accept_(arg)) {
            e_cur_ = e_trial;
            v_cur_ = s.frame.simbox.volume();
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

    AccReal temperature_;  ///< Ensemble temperature [K].
    AccReal pressure_;     ///< External pressure [eV/Å³].
    std::mt19937_64& rng_; ///< The run's RNG stream, shared with the Movers.
    AccReal e_cur_ = 0;    ///< Energy of the last accepted state [eV].
    double v_cur_ = 0;     ///< Volume of the last accepted state [Å³].
};

/// @brief Isothermal–isobaric Monte Carlo — a preset over the shared MC loop
///        that attaches `NptMetropolis` in its constructor.
///
/// Pair it with a `MoveSelector` mixing `TranslateMove` and `VolumeMove`:
/// @code{.cpp}
///   cpu::mc::NptMc mc{300.0, /*P [eV/Å³]=*/1e-4, /*seed=*/7u};
///   mc.add_potential(cpu::PairLJCut{eps, sigma, cutoff});
///   mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
///   mc.set_mover(cpu::mc::MoveSelector{rig}
///                    .add(cpu::mc::TranslateMove{0.2}, 0.9)
///                    .add(cpu::mc::VolumeMove{50.0}, 0.1));
///   mc.build();
///   mc.run(frame, 100000);
/// @endcode
class NptMc : public McBase {
  public:
    /// @brief Construct the NPT sampler.
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param pressure_ev_A3 External pressure [eV/Å³].
    /// @param seed Seed for the single reproducible RNG stream.
    /// @throws std::invalid_argument if `temperature_K <= 0`.
    NptMc(AccReal temperature_K, AccReal pressure_ev_A3, std::uint64_t seed)
        : McBase(seed), policy_(this->set_outcome(NptMetropolis{
                            temperature_K, pressure_ev_A3, rng_})) {}

    /// @brief Energy [eV] of the current (last-accepted) configuration.
    AccReal current_energy() const { return policy_.current_energy(); }

    /// @brief Volume [Å³] of the current (last-accepted) configuration.
    double current_volume() const { return policy_.current_volume(); }

  private:
    NptMetropolis& policy_; ///< The preset's own policy, for stat read-back.
};

} // namespace mc
} // namespace cpu
} // namespace ATV
