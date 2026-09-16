/// @file cpu/potentials/bond_fene.h
/// @brief FENE bond — Kremer–Grest U = −½ K R₀² ln(1 − (r/R₀)²) (CPU).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/bond.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

/// @brief Finite extensible nonlinear elastic bond (per bond type).
///
/// \(U=-\frac12 K R_0^2\ln(1-(r/R_0)^2)\) for \(r<R_0\); throws if \(r\ge R_0\).
/// \(K\) [eV/Å²], \(R_0\) [Å].
class BondFene : public Bond {
  public:
    /// @param K Spring constants per bond type [eV/Å²].
    /// @param R0 Maximum extensions per bond type [Å].
    BondFene(std::vector<double> K, std::vector<double> R0);

    std::string name() const override { return "fene_bond"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, R0_;
    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
