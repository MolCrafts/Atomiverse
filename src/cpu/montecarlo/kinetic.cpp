#include "atomiverse/cpu/montecarlo/kinetic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV::cpu::mc {

StateTransitions::StateTransitions(const std::vector<Edge>& edges,
                                   const std::vector<std::int64_t>& states)
    : states_(states.begin(), states.end()) {
    std::int64_t id = 0;
    for (const auto& [from, to, rate] : edges) {
        if (!std::isfinite(rate) || rate < 0.0 || from == to)
            throw std::invalid_argument(
                "StateTransitions requires distinct states and finite, "
                "nonnegative rates");
        states_.insert(from);
        states_.insert(to);
        outgoing_[from].push_back({id++, to, rate});
    }
}

void StateTransitions::rates(const cpu::Frame& frame,
                             std::vector<double>& out) {
    if (!frame.meta.contains(kKineticState.key))
        throw std::invalid_argument(
            "StateTransitions requires integer Frame.meta['kmc_state']");
    const auto state = frame.meta.get(kKineticState);
    if (!states_.contains(state))
        throw std::invalid_argument(
            "StateTransitions received an unknown kmc_state");
    const auto found = outgoing_.find(state);
    if (found == outgoing_.end())
        return;
    for (const auto& entry : found->second)
        out.push_back(entry.rate);
}

EventResult StateTransitions::apply(cpu::Frame& frame, std::size_t index,
                                    Transaction&) {
    const auto current = frame.meta.get(kKineticState);
    const auto& channel = outgoing_.at(current).at(index);
    frame.meta.set(kKineticState, channel.destination);
    return {channel.id, std::nullopt};
}

std::optional<KineticDraw> KineticSolver::draw(std::span<const double> rates,
                                               std::mt19937_64& rng) {
    double total = 0.0;
    for (const double rate : rates) {
        if (!std::isfinite(rate) || rate < 0.0)
            throw std::invalid_argument(
                "KMC event rates must be finite and nonnegative");
        total += rate;
    }
    if (!std::isfinite(total))
        throw std::overflow_error("KMC total event rate is not finite");
    if (total == 0.0)
        return std::nullopt;

    const auto selected = draw_event(rates, total, rng);
    if (selected.index >= rates.size() || rates[selected.index] == 0.0)
        throw std::runtime_error(
            "KMC solver selected an event with no positive rate");
    if (!std::isfinite(selected.waiting_time) || selected.waiting_time <= 0.0)
        throw std::runtime_error("KMC solver returned an invalid waiting time");
    return selected;
}

KineticDraw LinearKineticSolver::draw_event(std::span<const double> rates,
                                            double total_rate,
                                            std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target =
        std::min(uniform(rng) * total_rate, std::nextafter(total_rate, 0.0));
    double waiting_draw = uniform(rng);
    while (waiting_draw <= 0.0 || waiting_draw >= 1.0)
        waiting_draw = uniform(rng);

    double cumulative = 0.0;
    for (std::size_t index = 0; index < rates.size(); ++index) {
        cumulative += rates[index];
        if (target < cumulative)
            return {index, -std::log(waiting_draw) / total_rate};
    }
    throw std::logic_error("KMC event selection did not reach the total rate");
}

} // namespace ATV::cpu::mc
