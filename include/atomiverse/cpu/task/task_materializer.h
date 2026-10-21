/// @file cpu/task/task_materializer.h
/// @brief TaskSpec typed alternative -> one run-ready CPU driver.
#pragma once

#include <memory>
#include <optional>

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"

#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#endif
#ifdef ATV_HAS_EVALUATE
#include "atomiverse/cpu/evaluate/evaluate.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/minimize.h"
#endif
#include "atomiverse/cpu/task/task_spec.h"

namespace ATV::cpu::task {

class TaskMaterializer {
  public:
    /// @throws std::runtime_error if the task kind's domain (MD, MINIMIZER,
    ///         SEMIEMPIRICAL) was compiled out.
    explicit TaskMaterializer(const TaskSpec& spec);

    /// Execute through one task-kind-agnostic entry. Frame remains the sole
    /// state currency; the return value contains loop control metadata only.
    TaskResult run(cpu::Frame& frame);

    TaskKind kind() const noexcept { return kind_; }

    /// Release the distributed owner map for MD/Minimize result gathering.
    /// SinglePoint uses full replicas and returns nullptr.
    std::unique_ptr<md::mpi::DomainDecomposition> release_dd();

    /// Read-only compiled Plan for structural equivalence tests.
    const Plan<cpu::Component>& plan() const;

  private:
    TaskKind kind_;
#ifdef ATV_HAS_MD
    std::optional<md::MD> md_;
#endif
#ifdef ATV_HAS_EVALUATE
    std::optional<Evaluate> eval_;
#endif
#ifdef ATV_HAS_MINIMIZER
    std::optional<minimizer::Minimize> minimize_;
#endif
    std::unique_ptr<md::mpi::DomainDecomposition> completed_dd_;
    int budget_ = 1;
};

} // namespace ATV::cpu::task
