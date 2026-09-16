/// @file cpu/montecarlo/mc.h
/// @brief cpu::mc::MC — concrete canonical (NVT) Metropolis Monte Carlo driver.
#pragma once

#include <cstdint>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/driver/outcome_policy.h"
#include "atomiverse/cpu/montecarlo/mc_base.h"

namespace ATV {
namespace cpu {
namespace mc {

/// @brief CPU canonical (NVT) Metropolis Monte Carlo — a **preset**, not a
///        category.
///
/// It runs a budget of trial moves, each proposed by a `Mover` (e.g.
/// `TranslateMove`), scored by any registered `Capability::Energy` evaluator
/// (e.g. `PairLJCut`), and kept or discarded by an `OutcomePolicy`. It reuses
/// the MD component set unchanged — energy evaluator, neighbor list, frame —
/// and mirrors the shape of `cpu::md::MD`.
///
/// ## Accept/reject is not a termination
///
/// The loop has two orthogonal policy slots, and only MC uses the second:
///
///   - `Criterion`     — should the loop stop?      (MC attaches none; the
///                                                   budget is it)
///   - `OutcomePolicy` — does this iteration count? (this preset attaches
///                                                   `Metropolis`, in its
///                                                   constructor)
///
/// Swapping in a grand-canonical or isobaric acceptance rule is a different
/// preset (`Gcmc`, `NptMc`) attaching a different policy — never a new driver
/// category. The loop itself lives on `McBase`. @see cpu/mc/mc_base.h,
/// cpu/driver/outcome_policy.h
///
/// Usage:
/// @code{.cpp}
///   cpu::mc::MC mc{/*T [K]=*/300.0, /*seed=*/12345u};
///   mc.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
///   mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, true});
///   mc.set_mover(cpu::mc::TranslateMove{/*delta [Å]=*/0.2});
///   mc.set_thermo(100);
///   mc.build();
///   mc.run(frame, 100000);            // 100k trial moves
///   double e = mc.current_energy();   // [eV]; mc.acceptance_ratio() in [0,1]
/// @endcode
///
/// @note For MC the neighbor list should rebuild whenever an accepted move
///       displaces an atom past the skin — construct it with `delay = 0`
///       (`NeighborList<>{cutoff, skin, 1, 0, true}`) so the displacement
///       check is never deferred.
class MC : public McBase {
  public:
    /// @brief Construct the NVT sampler.
    /// @param temperature_K Ensemble temperature [K] (must be > 0).
    /// @param seed Seed for the single reproducible RNG stream.
    /// @throws std::invalid_argument if `temperature_K <= 0`.
    MC(AccReal temperature_K, std::uint64_t seed)
        : McBase(seed),
          metropolis_(this->set_outcome(cpu::Metropolis{temperature_K, rng_})) {
    }

    /// @brief Energy [eV] of the current (last-accepted) configuration.
    AccReal current_energy() const { return metropolis_.current_energy(); }

    /// @brief Total trial moves attempted across all `run()` calls.
    long long attempts() const { return metropolis_.attempts(); }

    /// @brief Total trial moves accepted across all `run()` calls.
    long long accepts() const { return metropolis_.accepts(); }

    /// @brief Accepted fraction of attempted moves, in [0,1] (0 if none).
    double acceptance_ratio() const { return metropolis_.acceptance_ratio(); }

  private:
    /// The acceptance policy, owned by the base's outcome slot. This
    /// reference is how the preset reads back what its own policy decided
    /// (energy, acceptance ratio).
    cpu::Metropolis& metropolis_;
};

} // namespace mc
} // namespace cpu
} // namespace ATV
