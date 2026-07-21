/// @file cuda/potentials/coul_long.cuh
/// @brief Real-space screened Coulomb (LAMMPS `pair_style coul/long`) — CUDA.
///
/// Short-range half of an Ewald/PME split: `erfc(αr)/r` over the cluster
/// neighbor list, mirroring `cpu::CoulLong`. Must share α with the
/// reciprocal-space `cuda::KSpaceEwald` / `cuda::KSpacePME`. A cross-driver
/// `Pair` evaluator — bind to the force phase alongside an LJ pair style.
#pragma once

#include <string>

#include "atomiverse/cuda/potentials/pair.cuh"

namespace ATV {
namespace cuda {

/// @brief Real-space screened Coulomb force/energy (Ewald/PME real part).
class CoulLong : public Pair {
  public:
    /// @param alpha  Ewald splitting parameter α [1/Å]. Must match the
    ///               reciprocal-space component.
    /// @param cutoff Real-space cutoff [Å] (≤ the neighbor-list cutoff).
    CoulLong(double alpha, double cutoff);

    std::string name() const override { return "coul/long"; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double alpha_;
};

} // namespace cuda
} // namespace ATV
