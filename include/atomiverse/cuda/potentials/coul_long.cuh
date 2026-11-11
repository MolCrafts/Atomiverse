/// @file cuda/potentials/coul_long.cuh
/// @brief Real-space screened Coulomb (LAMMPS `pair_style coul/long`) — CUDA.
///
/// Short-range half of an Ewald/PME split: `erfc(αr)/r` on the cluster-tile
/// engine shared with `PairLJCut` and `PairLJCutCoulLong`. Must share α with
/// `cuda::KSpaceEwald` / `cuda::KSpacePME`.
#pragma once

#include <memory>
#include <string>

#include "atomiverse/core/precision.h"
#include "atomiverse/cuda/component/long_range_protocol.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"

namespace ATV {
namespace cuda {

/// @brief Real-space screened Coulomb force/energy (Ewald/PME real part).
class CoulLong : public Pair, public LongRangeParticipant {
  public:
    /// @param alpha  Ewald splitting parameter α [1/Å]. Must match reciprocal.
    /// @param cutoff Real-space cutoff [Å] (≤ the neighbor-list cutoff).
    CoulLong(double alpha, double cutoff);
    ~CoulLong() override;
    CoulLong(CoulLong&&) noexcept;
    CoulLong& operator=(CoulLong&&) noexcept;

    std::string name() const override { return "coul/long"; }
    bool graph_capturable() const override { return true; }
    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, AccReal(alpha_),
                AccReal(pair_cutoff()), AccReal(0)};
    }

    bool supports_force_rows() const override {
        // Cluster force currently covers all cluster slots; multi-GPU row
        // split is phase C of cuda-openmm-neighbor-force.
        return false;
    }

    void build(const BuildContext& ctx) override;
    void invalidate_inputs(InputChange changes,
                           const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double alpha_;

    /// Cluster-tile engine (packed tiles, slot charges, partials), sized in
    /// build(); compute() only launches.
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV
