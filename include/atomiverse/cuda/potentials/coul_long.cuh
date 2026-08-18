/// @file cuda/potentials/coul_long.cuh
/// @brief Real-space screened Coulomb (LAMMPS `pair_style coul/long`) — CUDA.
///
/// Short-range half of an Ewald/PME split: `erfc(αr)/r` on the **OpenMM
/// cluster** neighbor CSR (`cl_*`), matching `PairLJCut`. Must share α with
/// `cuda::KSpaceEwald` / `cuda::KSpacePME`.
#pragma once

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

    std::string name() const override { return "coul/long"; }
    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, AccReal(alpha_),
                AccReal(pair_cutoff()), AccReal(0)};
    }

    void set_stress_required(bool required) override {
        compute_stress_ = required;
    }

    bool supports_force_rows() const override {
        // Cluster force currently covers all cluster slots; multi-GPU row
        // split is phase C of cuda-openmm-neighbor-force.
        return false;
    }

    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double alpha_;
    bool compute_stress_ = true;

    // Cluster-sorted scratch (positions + charge), sized to num_clusters*8.
    real_t* spx_ = nullptr;
    real_t* spy_ = nullptr;
    real_t* spz_ = nullptr;
    real_t* spq_ = nullptr;
    int spos_cap_ = 0;

    // Legacy pos4_ field kept unused until destroy cleans any prior alloc.
    float4* pos4_ = nullptr;
    int pos4_cap_ = 0;
};

} // namespace cuda
} // namespace ATV
