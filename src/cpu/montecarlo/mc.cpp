#include "atomiverse/cpu/montecarlo/mc.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/montecarlo/gcmc.h"
#include "atomiverse/cpu/montecarlo/npt_mc.h"

namespace ATV::cpu::mc {

using Phase = ATV::sampling::Phase;

MC::MC(std::uint64_t seed, EmptyPlan) : rng_(seed) {}

MC::MC(std::uint64_t seed) : MC(seed, EmptyPlan{}) {
    this->on(Phase::energy)
        .add(cpu::ZeroForces{}, -1000)
        .add(cpu::ZeroEnergy{}, -999);
}

MC::MC(AccReal temperature_K, std::uint64_t seed) : MC(seed) {
    set_nvt(temperature_K);
}

MC::~MC() { this->flush_live_(); }

MC& MC::set_nvt(AccReal temperature_K) {
    this->set_outcome(cpu::Metropolis{temperature_K, rng_});
    return *this;
}

MC& MC::set_npt(AccReal temperature_K, AccReal pressure) {
    this->set_outcome(NptMetropolis{temperature_K, pressure, rng_});
    return *this;
}

MC& MC::set_gcmc(AccReal temperature_K, AccReal mu, AccReal mass) {
    this->set_outcome(GcmcMetropolis{temperature_K, mu, mass, rng_});
    return *this;
}

BigInt MC::run(cpu::Frame& frame, BigInt budget) {
    if (budget < 0)
        throw std::invalid_argument("MC step budget must be nonnegative");
    if (this->plan().kind != PhaseKind::Sampling)
        throw std::logic_error("Build the MC driver before running it");
    if (this->begin_run_(frame)) {
        bootstrapped_ = false;
        changed_connectivity_ = false;
        current_energy_ = 0;
        current_volume_ = 0;
    }
    const BigInt completed = advance_(*this->backend_, frame, budget);
    this->backend_->drain_capture();
    return completed;
}

void MC::build_on(cpu::Runtime& runtime) {
    Driver<Phase>::build_on(runtime);
    bootstrapped_ = false;
}

int MC::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame, int budget) {
    if (!ctx.exec.runtime)
        throw std::runtime_error(
            "MC nested execution requires a parent Runtime");
    if (budget < 0)
        throw std::invalid_argument("MC step budget must be nonnegative");
    auto& runtime = *ctx.exec.runtime;
    struct Restore {
        cpu::Context& ctx;
        SamplingServices mc;
        LoopServices loop;
        cpu::Frame& frame;
        ResolutionWindow window;
        ~Restore() {
            ctx.mc = mc;
            ctx.loop = loop;
            frame.restore_resolution_window(window);
        }
    } restore{ctx, ctx.mc, ctx.loop, frame, frame.resolution_window()};
    frame.restore_resolution_window(ResolutionWindow{});
    ctx.loop.status = &this->mutable_outer_status();
    ctx.loop.metrics = &this->metrics();
    cpu::Runtime::ScopedPlan active(runtime, this->plan(), 0);
    return static_cast<int>(advance_(runtime, frame, budget));
}

void MC::install_defaults() {
    const auto scan = this->scan_pending_infra_();
    this->throw_if_missing_neighbor_list_(
        scan, "MC",
        "Add `mc.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, "
        "true})`.");
    // A geometry patch is indexed by row. Scoring must preserve row order
    // until the trial has been committed or rolled back.
    this->for_each_pending_mut([](cpu::Component& component) {
        if (cpu::NeighborProvider* provider = component.neighbor_provider())
            provider->forbid_row_reordering();
    });
}

void MC::validate_components() const {
    this->require_mover();
    const auto& bindings = this->mounted();
    const auto generators = std::count_if(
        bindings.begin(), bindings.end(), [](const auto& binding) {
            return binding.phase == ATV::slot_of(Phase::propose)
                   && dynamic_cast<const cpu::Mover*>(binding.component)
                          != nullptr;
        });
    if (generators != 1)
        throw std::runtime_error(
            "MC requires exactly one proposer; use MoveSelector to mix movers");
    for (const auto& binding : bindings) {
        if (binding.phase == ATV::slot_of(Phase::energy)
            && binding.component->frequency() > 1)
            throw std::runtime_error(
                "MC energy components must run at every trial: "
                + binding.component->name());
    }
    this->require_energy();
    if (!this->outcome_)
        throw std::runtime_error(
            "MC requires an equilibrium outcome policy (set_nvt, set_npt "
            "or set_gcmc)");
}

bool MC::phase_wants_forces(Phase phase) const {
    return phase != Phase::energy;
}

BigInt MC::advance_(cpu::Runtime& runtime, cpu::Frame& frame, BigInt budget) {
    auto& context = runtime.context();
    context.mc.proposal = &proposal_;
    context.mc.rng = &rng_;
    prepare_sampling_(runtime, frame);
    current_volume_ = frame.box.volume();

    BigInt completed = 0;
    while (completed < budget) {
        this->mutable_outer_status().reset();
        const BigInt step = this->step_counter_;
        if (!advance_step_(runtime, frame, step))
            break;
        this->next_step_();
        current_volume_ = frame.box.volume();
        runtime.step_stage(ATV::slot_of(Phase::end_of_step), step);
        ++completed;
        if (this->poll_criteria_(step, frame, context) != Verdict::Continue)
            break;
    }
    return completed;
}

void MC::prepare_sampling_(cpu::Runtime& runtime, cpu::Frame& frame) {
    if (bootstrapped_)
        return;
    score_(runtime, -1);
    this->start_outcome_(frame, runtime.context());
    current_energy_ = frame.meta.get(kEnergy);
    bootstrapped_ = true;
}

bool MC::advance_step_(cpu::Runtime& runtime, cpu::Frame& frame, BigInt step) {
    const auto metadata_before = frame.meta;
    proposal_.begin(Transaction::Verb::None, frame);
    Outcome outcome;
    try {
        {
            auto& context = runtime.context();
            auto& evaluate_trial = context.mc.evaluate_trial;
            struct RestoreEvaluator {
                const std::function<AccReal()>*& evaluate;
                const std::function<AccReal()>* saved;
                ~RestoreEvaluator() { evaluate = saved; }
            } restore{evaluate_trial, evaluate_trial};
            const std::function<AccReal()> score_candidate = [&] {
                auto& flag = context.exec.trial_evaluation;
                struct RestoreFlag {
                    bool& flag;
                    bool saved;
                    ~RestoreFlag() { flag = saved; }
                } restore_flag{flag, flag};
                flag = true;
                runtime.run_stage(Phase::energy, step);
                return frame.meta.get(kEnergy);
            };
            evaluate_trial = &score_candidate;
            runtime.step_stage(ATV::slot_of(Phase::propose), step);
        }
        score_(runtime, step);
        outcome = this->decide_outcome_(frame, step, runtime.context());
    } catch (...) {
        proposal_.rollback(frame);
        frame.meta = metadata_before;
        throw;
    }
    const auto verb = static_cast<std::size_t>(proposal_.verb);
    if (outcome == Outcome::Rollback) {
        proposal_.rollback(frame);
        frame.meta = metadata_before;
        // Evaluators can publish outputs outside the geometry patch.
        score_(runtime, step);
    } else {
        current_energy_ = frame.meta.get(kEnergy);
        changed_connectivity_ |= proposal_.connectivity_changed;
        ++accepts_[verb];
    }
    ++attempts_[verb];
    return true;
}

void MC::score_(cpu::Runtime& runtime, BigInt step) {
    auto& exec = runtime.context().exec;
    struct Restore {
        bool& flag;
        bool saved;
        ~Restore() { flag = saved; }
    } restore{exec.trial_evaluation, exec.trial_evaluation};
    exec.trial_evaluation = true;
    runtime.step_stage(ATV::slot_of(Phase::energy), step);
}

long long MC::attempts() const {
    require_trial_statistics_();
    return std::accumulate(attempts_.begin(), attempts_.end(), 0LL);
}

AccReal MC::current_energy() const { return current_energy_; }

long long MC::accepts() const {
    require_trial_statistics_();
    return std::accumulate(accepts_.begin(), accepts_.end(), 0LL);
}

long long MC::attempts(Transaction::Verb verb) const {
    require_trial_statistics_();
    return attempts_[static_cast<std::size_t>(verb)];
}

long long MC::accepts(Transaction::Verb verb) const {
    require_trial_statistics_();
    return accepts_[static_cast<std::size_t>(verb)];
}

double MC::acceptance_ratio() const {
    const auto total = attempts();
    return total > 0 ? static_cast<double>(accepts()) / total : 0.0;
}

double MC::acceptance_ratio(Transaction::Verb verb) const {
    const auto total = attempts(verb);
    return total > 0 ? static_cast<double>(accepts(verb)) / total : 0.0;
}

} // namespace ATV::cpu::mc
