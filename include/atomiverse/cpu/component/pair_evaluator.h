/// @file cpu/component/pair_evaluator.h
/// @brief PairPotential — Potential + NeighborConsumer (depth ≤ 2).
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {

/// @brief Non-bonded pair Potential (PairLJCut → PairPotential → Component).
class PairPotential : public Potential, public NeighborConsumer {
  public:
    const NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }

    AccReal pair_cutoff() const { return cutoff_; }
    AccReal neighbor_cutoff() const override { return pair_cutoff(); }

    void launch(const LaunchContext& ctx) final { compute(ctx); }

  protected:
    explicit PairPotential(AccReal cutoff) : cutoff_(cutoff) {}

    virtual void compute(const LaunchContext& ctx) = 0;

    AccReal cutoff_ = 0.0;
};

using PairEvaluator = PairPotential;
using Pair = PairPotential;

} // namespace cpu
} // namespace ATV
