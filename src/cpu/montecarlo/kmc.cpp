#include "atomiverse/cpu/montecarlo/kmc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ATV::cpu::mc {

KMC::KMC(std::uint64_t seed) : MC(seed, EmptyPlan{}) {}

KMC::~KMC() { this->flush_live_(); }

KMC& KMC::set_event(std::shared_ptr<Event> event) {
    this->require_mutable_();
    if (!event)
        throw std::invalid_argument(
            "KMC::set_event requires an event component");
    if (this->live_ || this->attached_runtime_)
        throw std::logic_error("Configure the KMC event before running");
    event_ = std::move(event);
    return *this;
}

KMC& KMC::set_solver(std::shared_ptr<KineticSolver> solver) {
    this->require_mutable_();
    if (!solver)
        throw std::invalid_argument("KMC::set_solver requires a solver");
    if (this->live_ || this->attached_runtime_)
        throw std::logic_error("Configure the KMC solver before running");
    solver_ = std::move(solver);
    return *this;
}

AccReal KMC::current_energy() const {
    throw std::logic_error("KMC does not evaluate energy");
}

void KMC::require_trial_statistics_() const {
    throw std::logic_error(
        "KMC counts executed events and physical time. Trial statistics are "
        "only available for equilibrium MC.");
}

void KMC::validate_components() const {
    if (!event_)
        throw std::runtime_error("KMC requires an event component (set_event)");
    if (!solver_)
        throw std::runtime_error("KMC requires a solver (set_solver)");
    if (this->outcome_)
        throw std::runtime_error(
            "KMC cannot use an equilibrium outcome policy");
    for (const auto& binding : this->mounted()) {
        if (binding.phase != ATV::slot_of(ATV::sampling::Phase::end_of_step))
            throw std::runtime_error("KMC only runs end-of-step components: "
                                     + binding.component->name());
    }
}

void KMC::prepare_sampling_(cpu::Runtime& runtime, cpu::Frame& frame) {
    if (runtime.context().mpi_size != 1)
        throw std::runtime_error("KMC requires a single MPI rank");
    constexpr MetaKey<double> previous_clock{"kmc_time", true};
    const bool has_clock = frame.meta.contains(kKineticTime.key);
    double time = 0.0;
    if (has_clock)
        time = frame.meta.get(kKineticTime);
    else if (frame.meta.contains(previous_clock.key))
        time = frame.meta.get(previous_clock);
    if (!std::isfinite(time) || time < 0.0)
        throw std::invalid_argument("KMC time must be finite and nonnegative");
    if (!has_clock)
        frame.meta.insert(kKineticTime, time);
}

bool KMC::advance_step_(cpu::Runtime& runtime, cpu::Frame& frame, BigInt) {
    cpu::Runtime::DispatchScope dispatch(&runtime);
    event_rates_.clear();
    event_->rates(frame, event_rates_);
    const auto selected = solver_->draw(event_rates_, rng_);
    if (!selected)
        return false;

    const double previous_time = frame.meta.get(kKineticTime);
    const double event_time = previous_time + selected->waiting_time;
    if (!std::isfinite(event_time) || event_time <= previous_time)
        throw std::overflow_error("KMC time cannot represent the next event");

    const auto metadata_before = frame.meta;
    proposal_.begin(Transaction::Verb::None, frame);
    try {
        const auto result = event_->apply(frame, selected->index, proposal_);
        if (result.displacement
            && !std::ranges::all_of(*result.displacement,
                                    [](double x) { return std::isfinite(x); }))
            throw std::invalid_argument(
                "KMC event displacement must be finite");
        frame.meta.upsert_serializable(kKineticEventId.key, result.id);
        frame.meta.upsert_serializable(kKineticWaitingTime.key,
                                       selected->waiting_time);
        frame.meta.upsert_serializable(kKineticHasDisplacement.key,
                                       result.displacement.has_value());
        frame.meta.upsert_serializable(
            kKineticDisplacement.key,
            result.displacement.value_or(MetaF64x3{}));
        frame.meta.set(kKineticTime, event_time);
    } catch (...) {
        proposal_.rollback(frame);
        frame.meta = metadata_before;
        throw;
    }
    return true;
}

} // namespace ATV::cpu::mc
