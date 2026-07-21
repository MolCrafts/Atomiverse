/// @file cuda/potentials/pair.cuh
/// @brief LAMMPS-style abstract base for non-bonded pair styles (CUDA).
///
/// Mirror of `cpu::Pair`. Independent hierarchy — no shared ancestor
/// between CPU and CUDA potential layers; only the names match.
#pragma once
#include "atomiverse/facade/component_role.h"

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/component/neighbor_protocol.cuh"

namespace ATV {
namespace cuda {

/// @brief Abstract base for non-bonded pair styles (CUDA backend).
class Pair : public Component, public NeighborConsumer {
  public:
    Role role() const override { return 0; }
    ~Pair() override = default;

    /* role via Potential/kind base */
    /// @brief Pair styles depend on a neighbor list — used by
    ///        `cuda::md::MD::install_infra_` to detect a missing
    ///        `cuda::NeighborList` registration without `dynamic_cast`.
    const NeighborConsumer* as_neighbor_consumer() const final { return this; }

    /// @brief The neighbor list must cover the pair interaction cutoff —
    ///        the CUDA side now reports it through the protocol like the
    ///        CPU `PairEvaluator` (closes the historical hint gap).
    /// @return `pair_cutoff()` [Å].
    double neighbor_cutoff() const final { return pair_cutoff(); }

    /// @brief Framework dispatch — `final`, calls `compute()`.
    void launch(const LaunchContext& ctx) final { compute(ctx); }

    /// @brief Per-type pair cutoff [Å]. Reports the value stored at
    /// construction; derived classes can override to compute from
    /// per-type tables.
    double pair_cutoff() const { return cutoff_; }

    /// @brief Engine-facing hook: enable full 1-2/1-3/1-4 special-bonds
    ///        exclusions (s12=s13=s14=0). Default no-op so the force engine
    ///        (`cuda::md::MultiGpu`) can request exclusions on any `Pair`
    ///        without knowing its concrete type. A distinct name from the
    ///        derived ref-qualified `set_special_bonds(…)` setters (which a
    ///        non-ref-qualified base virtual could not override).
    virtual void enable_full_exclusions() {}

    /// @brief Whether `compute()` honors `Context::n_force_rows` /
    ///        `force_rows` (the multi-GPU owned-row restriction and
    ///        interior/boundary split). The engine consults this before
    ///        issuing split (two-launch) pair calls — a style that ignores
    ///        the row fields would otherwise compute every row twice.
    ///        Default false: the engine then issues ONE full launch (the
    ///        historical behavior). Styles that implement the row fields
    ///        (PairLJCut) override to true.
    virtual bool supports_force_rows() const { return false; }

  protected:
    explicit Pair(double cutoff) : cutoff_(cutoff) {}

    /// @brief Hot-path override — derived class launches its CUDA kernel.
    virtual void compute(const LaunchContext& ctx) = 0;

    double cutoff_ = 0.0;
};

} // namespace cuda
} // namespace ATV
