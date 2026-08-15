/// @file cpu/task/task_spec.cpp
/// @brief Task-specific invariant validation for TaskSpec v5.

#include "atomiverse/cpu/task/task_spec.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ATV::cpu::task {
namespace {

void require_budget(const char* task, const char* field, std::int64_t value) {
    if (value < 1)
        throw std::invalid_argument(std::string("task_spec: ") + task + " "
                                    + field + " must be >= 1, got "
                                    + std::to_string(value));
    if (value > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("task_spec: ") + task + " "
                                    + field + " " + std::to_string(value)
                                    + " exceeds the int budget limit");
}

template <typename Phase>
void require_phase(const char* task, Phase phase) {
    const int slot = ATV::slot_of(phase);
    if (slot < 0 || slot >= ATV::slot_count<Phase>())
        throw std::invalid_argument(
            std::string("task_spec: ") + task + " binding phase slot "
            + std::to_string(slot) + " is outside its typed phase domain");
}

void require_positive_finite(const char* task, const char* field,
                             double value) {
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(std::string("task_spec: ") + task + " "
                                    + field + " must be positive and finite");
}

} // namespace

TaskSpec::TaskSpec(MDTask task) : payload_(std::move(task)) {
    const auto& md_task = md();
    require_budget("MD", "n_steps", md_task.n_steps);
    for (const auto& entry : md_task.entries)
        if (entry.binding)
            require_phase("MD", entry.binding->phase);
}

TaskSpec::TaskSpec(SinglePointTask task) : payload_(std::move(task)) {
    const auto& config = single_point().evaluator;
    if (config.max_iter < 1)
        throw std::invalid_argument(
            "task_spec: SinglePoint max_iter must be >= 1");
    if (config.multiplicity < 1)
        throw std::invalid_argument(
            "task_spec: SinglePoint multiplicity must be >= 1");
    require_positive_finite("SinglePoint", "density_tol", config.density_tol);
    require_positive_finite("SinglePoint", "energy_tol", config.energy_tol);
}

TaskSpec::TaskSpec(MinimizeTask task) : payload_(std::move(task)) {
    const auto& min_task = minimize();
    require_budget("Minimize", "max_iter", min_task.max_iter);
    for (const auto& entry : min_task.entries) {
        if (entry.binding)
            require_phase("Minimize", entry.binding->phase);
        if (const auto* lbfgs = std::get_if<LBFGSParams>(&entry.params)) {
            if (lbfgs->memory < 1)
                throw std::invalid_argument(
                    "task_spec: Minimize LBFGS memory must be >= 1");
            require_positive_finite("Minimize", "LBFGS alpha0", lbfgs->alpha0);
            if (lbfgs->max_ls < 1)
                throw std::invalid_argument(
                    "task_spec: Minimize LBFGS max_ls must be >= 1");
        }
    }

    // The criterion is a variant since v5, so the tolerance checks have to
    // reach INTO the alternative. Each arm mirrors exactly what the C++
    // criterion's own constructor refuses (`cpu/driver/criteria.h`), so a
    // payload can never smuggle past the wire only to throw at materialization.
    std::visit(
        [](const auto& c) {
            using C = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<C, ForceConvergedParams>) {
                require_positive_finite("Minimize", "ForceConverged ftol",
                                        c.ftol);
                require_positive_finite("Minimize", "ForceConverged rms_ftol",
                                        c.rms_ftol);
                require_positive_finite("Minimize", "ForceConverged max_dtol",
                                        c.max_dtol);
                require_positive_finite("Minimize", "ForceConverged rms_dtol",
                                        c.rms_dtol);
                require_positive_finite("Minimize", "ForceConverged etol",
                                        c.etol);
            } else if constexpr (std::is_same_v<C, PressureConvergedParams>) {
                // Only the tolerance: `PressureConverged` throws on
                // `ptol <= 0` and on nothing else, so a negative target
                // (a tensile setpoint) stays a legal payload.
                require_positive_finite("Minimize", "PressureConverged ptol",
                                        c.ptol);
            } else {
                require_positive_finite("Minimize", "StressConverged stol",
                                        c.stol);
                if (!c.check_x && !c.check_y && !c.check_z)
                    throw std::invalid_argument(
                        "task_spec: Minimize StressConverged has every axis "
                        "masked — it would judge nothing and declare victory");
            }
        },
        min_task.convergence);
}

TaskKind TaskSpec::kind() const noexcept {
    if (std::holds_alternative<MDTask>(payload_))
        return TaskKind::MD;
    if (std::holds_alternative<SinglePointTask>(payload_))
        return TaskKind::SinglePoint;
    return TaskKind::Minimize;
}

} // namespace ATV::cpu::task
