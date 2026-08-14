/// @file cpu/task/task_materializer.cpp
/// @brief Typed TaskSpec registration replay onto MD/Evaluate/Minimize.

#include "atomiverse/cpu/task/task_materializer.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"

namespace ATV::cpu::task {
namespace {

void register_md_entry(md::MD& driver, const MDComponentEntry& entry) {
    std::visit(
        [&](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            if (entry.binding) {
                auto bind = [&](auto component) {
                    driver.on(entry.binding->phase)
                        .add(std::move(component), entry.binding->order);
                };
                if constexpr (std::is_same_v<P, PairLJCutParams>)
                    bind(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
                else if constexpr (std::is_same_v<P, NeighborListParams>)
                    bind(cpu::md::NeighborList<>{p.cutoff, p.skin});
                else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                    bind(cpu::BondHarmonic{p.K, p.r0});
                else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                    bind(cpu::AngleHarmonic{p.K, p.theta0});
                else if constexpr (std::is_same_v<P,
                                                  DihedralMultiHarmonicParams>)
                    bind(cpu::DihedralMultiHarmonic{p.Cn});
                else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                    bind(cpu::md::VelocityVerlet{p.dt});
                else if constexpr (std::is_same_v<P, InitVelocityParams>)
                    bind(cpu::md::InitVelocity{p.temperature, p.seed});
                else if constexpr (std::is_same_v<P, NoseHooverParams>)
                    bind(cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
                else if constexpr (std::is_same_v<P, FixLangevinParams>)
                    bind(cpu::md::FixLangevin{p.T_start, p.T_stop, p.damp, p.dt,
                                              p.seed});
                else if constexpr (std::is_same_v<P, FixCsvrParams>)
                    bind(cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
                else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                    bind(cpu::md::NoseHooverNPT{p.T_target, p.tau_T, p.P_target,
                                                p.tau_P, p.dt});
                else if constexpr (std::is_same_v<P, ThermoParams>)
                    bind(cpu::Thermo{p.freq});
                return;
            }
            // Categorical path — no public Driver::add
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                driver.set_evaluator(
                    cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                driver.set_neighbor(cpu::md::NeighborList<>{p.cutoff, p.skin});
            else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                driver.set_evaluator(cpu::BondHarmonic{p.K, p.r0});
            else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                driver.set_evaluator(cpu::AngleHarmonic{p.K, p.theta0});
            else if constexpr (std::is_same_v<P, DihedralMultiHarmonicParams>)
                driver.set_evaluator(cpu::DihedralMultiHarmonic{p.Cn});
            else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                driver.set_integrator(cpu::md::VelocityVerlet{p.dt});
            else if constexpr (std::is_same_v<P, InitVelocityParams>)
                driver.set_observer(
                    cpu::md::InitVelocity{p.temperature, p.seed});
            else if constexpr (std::is_same_v<P, NoseHooverParams>)
                driver.set_thermostat(
                    cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
            else if constexpr (std::is_same_v<P, FixLangevinParams>)
                driver.set_thermostat(cpu::md::FixLangevin{
                    p.T_start, p.T_stop, p.damp, p.dt, p.seed});
            else if constexpr (std::is_same_v<P, FixCsvrParams>)
                driver.set_thermostat(
                    cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
            else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                driver.set_integrator(cpu::md::NoseHooverNPT{
                    p.T_target, p.tau_T, p.P_target, p.tau_P, p.dt});
            else if constexpr (std::is_same_v<P, ThermoParams>)
                driver.set_thermo(p.freq);
        },
        entry.params);
}

void register_minimize_entry(minimizer::Minimize& driver,
                             const MinimizeComponentEntry& entry) {
    std::visit(
        [&](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            if (entry.binding) {
                auto bind = [&](auto component) {
                    driver.on(entry.binding->phase)
                        .add(std::move(component), entry.binding->order);
                };
                if constexpr (std::is_same_v<P, PairLJCutParams>)
                    bind(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
                else if constexpr (std::is_same_v<P, NeighborListParams>)
                    bind(cpu::md::NeighborList<>{p.cutoff, p.skin});
                else if constexpr (std::is_same_v<P, LBFGSParams>)
                    bind(cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls});
                return;
            }
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                driver.set_evaluator(
                    cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                driver.set_neighbor(cpu::md::NeighborList<>{p.cutoff, p.skin});
            else if constexpr (std::is_same_v<P, LBFGSParams>)
                driver.set_optimizer(
                    cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls});
        },
        entry.params);
}

} // namespace

TaskMaterializer::TaskMaterializer(const TaskSpec& spec) : kind_(spec.kind()) {
    if (kind_ == TaskKind::MD) {
        const auto& task = spec.md();
        budget_ = static_cast<int>(task.n_steps);
        md_.emplace();
        for (const auto& entry : task.entries)
            register_md_entry(*md_, entry);
        md_->build();
        return;
    }

    if (kind_ == TaskKind::SinglePoint) {
        const auto& p = spec.single_point().evaluator;
        eval_.emplace();
        eval_->set_serial_on_replica();
        eval_->on(ATV::evaluate::Phase::evaluate)
            .add(semiempirical::AM1Evaluator{p.charge, p.multiplicity,
                                             p.max_iter, p.density_tol,
                                             p.energy_tol});
        eval_->build();
        return;
    }

    const auto& task = spec.minimize();
    budget_ = static_cast<int>(task.max_iter);
    minimize_.emplace();
    const auto& c = task.convergence;
    minimize_->add_criterion(cpu::ForceConverged{c.ftol, c.rms_ftol, c.max_dtol,
                                                 c.rms_dtol, c.etol});
    for (const auto& entry : task.entries)
        register_minimize_entry(*minimize_, entry);
    minimize_->build();
}

TaskResult TaskMaterializer::run(cpu::Frame& frame) {
    completed_dd_.reset();
    if (kind_ == TaskKind::MD) {
        const int iterations = md_->run(frame, budget_);
        return TaskResult{iterations, md_->converged(), md_->stop_reason()};
    }
    if (kind_ == TaskKind::SinglePoint) {
        eval_->run(frame);
        return TaskResult{1, false, {}};
    }

    auto completed = minimize_->run_detached(frame, budget_);
    const int iterations = completed.iterations;
    completed_dd_ = std::move(completed.decomposition);
    return TaskResult{iterations, minimize_->converged(),
                      minimize_->stop_reason()};
}

std::unique_ptr<md::mpi::DomainDecomposition> TaskMaterializer::release_dd() {
    if (kind_ == TaskKind::MD)
        return md_->release_dd();
    if (kind_ == TaskKind::Minimize)
        return std::move(completed_dd_);
    return nullptr;
}

const Plan<cpu::Component>& TaskMaterializer::plan() const {
    if (kind_ == TaskKind::MD)
        return md_->plan();
    if (kind_ == TaskKind::SinglePoint)
        return eval_->plan();
    return minimize_->plan();
}

} // namespace ATV::cpu::task
