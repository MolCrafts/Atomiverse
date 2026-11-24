/// @file cpu/montecarlo/kinetic.h
/// @brief Event components and solvers for continuous-time KMC.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/montecarlo/transaction.h"

namespace ATV::cpu::mc {

/// Physical time [ps]. A supplied value resumes the clock from a Frame.
inline constexpr MetaKey<double> kKineticTime{"kmc_time_ps", true};
/// State label for StateTransitions.
inline constexpr MetaKey<std::int64_t> kKineticState{"kmc_state", true};

inline constexpr MetaKey<std::int64_t> kKineticEventId{"kmc_event_id", true};
inline constexpr MetaKey<double> kKineticWaitingTime{"kmc_waiting_time_ps",
                                                     true};
inline constexpr MetaKey<MetaF64x3> kKineticDisplacement{"kmc_displacement",
                                                         true};
inline constexpr MetaKey<bool> kKineticHasDisplacement{"kmc_has_displacement",
                                                       true};

struct EventResult {
    /// Stable channel identity, independent of the current rate vector index.
    std::int64_t id;
    /// Unwrapped displacement of the transported particle, in Frame length units.
    /// An absent value means this event does not report a displacement.
    std::optional<MetaF64x3> displacement;
};

/// Defines the available transitions, their rates and their effect on state.
class Event {
  public:
    virtual ~Event() = default;

    /// Append rates [ps^-1] for the current state. The selected index refers
    /// to this enumeration. The driver clears the output before each call.
    /// Rates stay constant between events.
    virtual void rates(const cpu::Frame& frame, std::vector<double>& out) = 0;

    /// Apply the selected event with the Frame as the authoritative state.
    /// Record an atom row or the box before changing it so a throwing event
    /// can be rolled back. The driver already snapshots Frame metadata.
    virtual EventResult apply(cpu::Frame& frame, std::size_t index,
                              Transaction& transaction) = 0;
};

/// Sparse, time-homogeneous Markov chain on integer Frame state labels.
/// Each tuple is (source, destination, rate [ps^-1]). States without outgoing
/// transitions are absorbing. State changes are written to kmc_state.
/// Input row indices identify channels, including parallel transitions.
/// An optional state list declares isolated states.
class StateTransitions final : public Event {
  public:
    using Edge = std::tuple<std::int64_t, std::int64_t, double>;

    explicit StateTransitions(const std::vector<Edge>& edges,
                              const std::vector<std::int64_t>& states = {});
    void rates(const cpu::Frame& frame, std::vector<double>& out) override;
    EventResult apply(cpu::Frame& frame, std::size_t index,
                      Transaction& transaction) override;

  private:
    struct Channel {
        std::int64_t id;
        std::int64_t destination;
        double rate;
    };
    std::unordered_set<std::int64_t> states_;
    std::unordered_map<std::int64_t, std::vector<Channel>> outgoing_;
};

struct KineticDraw {
    std::size_t index;
    double waiting_time; ///< Time in ps before the selected event.
};

class KineticSolver {
  public:
    virtual ~KineticSolver() = default;

    /// Validate the full list of rates [ps^-1] and sample the next event
    /// and waiting time. This interface scans all rates on each call.
    /// A zero total rate is an absorbing state and consumes no RNG draws.
    std::optional<KineticDraw> draw(std::span<const double> rates,
                                    std::mt19937_64& rng);

  protected:
    /// Sample an event with probability rate / total_rate and an independent
    /// exponential waiting time with parameter total_rate.
    virtual KineticDraw draw_event(std::span<const double> rates,
                                   double total_rate, std::mt19937_64& rng) = 0;
};

/// Gillespie direct method with linear event selection.
class LinearKineticSolver final : public KineticSolver {
  protected:
    KineticDraw draw_event(std::span<const double> rates, double total_rate,
                           std::mt19937_64& rng) override;
};

} // namespace ATV::cpu::mc
