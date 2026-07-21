/// @file cpu/potentials/improper_harmonic.h
/// @brief LAMMPS-style ImproperHarmonic — V = K(χ − χ₀)² (CPU).
///
/// χ is the improper torsion angle (same plane-normal geometry as a proper
/// dihedral). Topology lives in the improper per-atom columns; energy goes to
/// `kEnergy` / `kEnergyImproper`. Reuses `dihedral_geom` with DihedralCharmm /
/// multi-harmonic.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
#include "atomiverse/cpu/potentials/improper.h"

namespace ATV {
namespace cpu {

/// @brief Harmonic improper torsion.
///
/// Per type: V = K (χ − χ₀)² with K [eV/rad²], χ₀ [rad]. Matches LAMMPS
/// `improper_style harmonic` (angle between the two planes of the 4-tuple).
class ImproperHarmonic : public Improper {
  public:
    /// @param K     Force constants per improper type [eV/rad²].
    /// @param chi0  Equilibrium improper angles per type [radians].
    ImproperHarmonic(std::vector<double> K, std::vector<double> chi0);

    std::string name() const override { return "harmonic_improper"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_;
    std::vector<double> chi0_;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
