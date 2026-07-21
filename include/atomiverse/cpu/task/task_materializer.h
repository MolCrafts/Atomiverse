/// @file cpu/task/task_materializer.h
/// @brief TaskSpec typed alternative -> one run-ready CPU driver.
#pragma once

#include <memory>
#include <optional>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/cpu/task/task_spec.h"

namespace ATV::cpu::task {

class TaskMaterializer {
  public:
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
    std::optional<md::MD> md_;
    std::optional<Evaluate> eval_;
    std::optional<minimizer::Minimize> minimize_;
    std::unique_ptr<md::mpi::DomainDecomposition> completed_dd_;
    int budget_ = 1;
};

} // namespace ATV::cpu::task
