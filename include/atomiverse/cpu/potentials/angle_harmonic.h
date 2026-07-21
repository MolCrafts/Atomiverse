/// @file cpu/potentials/angle_harmonic.h
/// @brief LAMMPS-style AngleHarmonic — V = K(θ − θ₀)² (CPU).
///
/// Was `ATV::HarmonicAngle` before the LAMMPS-naming refactor.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/angle.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

class AngleHarmonic : public Angle {
  public:
    /// @param K Force constants per angle type [eV/rad²].
    /// @param theta0 Equilibrium angles per angle type [radians].
    AngleHarmonic(std::vector<double> K, std::vector<double> theta0);

    std::string name() const override { return "harmonic_angle"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, theta0_;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    /// @brief Resolved-tuple cache; rebuilt on migration / halo full rebuild.
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
