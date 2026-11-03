/// @file cpu/potentials/angle_cosine.h
/// @brief LAMMPS-style AngleCosine — V = K(1 + cos θ) (CPU).
///
/// Port of LAMMPS `src/MOLECULE/angle_cosine.cpp` (Sandia, GPL; checkout
/// SHA e8c4c278a8). The per-tuple law is `core/angle_cosine.h`, shared with
/// `cuda::AngleCosine`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/angle.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

/// @brief Cosine angle style \(V = K(1+\cos\theta)\), per angle type.
///
/// The energy is minimal (zero) at \(\theta = \pi\), so this is the bending
/// stiffness of a semiflexible Kremer–Grest chain: there is no equilibrium
/// angle parameter. Reads the frame's angle topology; `K[type]` indexes the
/// angle type. Writes `kEnergy`, `kEnergyAngle`, forces and (when stress is
/// requested) the angle virial \(d_1\otimes F_1 + d_2\otimes F_3\) to
/// `kStress`.
class AngleCosine : public Angle {
  public:
    /// @param K Angle stiffness per angle type [eV]. Non-empty, finite.
    /// @throws std::invalid_argument on an empty or non-finite K.
    explicit AngleCosine(std::vector<double> K);

    std::string name() const override { return "cosine_angle"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    /// @throws std::runtime_error on an angle type outside `[0, K.size())`.
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    /// @brief Resolved-tuple cache; rebuilt on migration / halo full rebuild.
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
