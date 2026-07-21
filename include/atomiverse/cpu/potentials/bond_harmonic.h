/// @file cpu/potentials/bond_harmonic.h
/// @brief LAMMPS-style BondHarmonic — V = K(r − r₀)² (CPU).
///
/// Was `ATV::HarmonicBond` before the LAMMPS-naming refactor.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/bond.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

/// @brief Harmonic bond stretching: V = K(r − r₀)².
///
/// Thread-parallel with per-thread force reduction buffers
/// (Newton's 3rd law). Multi-rank ownership filter is owned by the
/// `cpu::Bond` base — derived class only computes per-bond math.
class BondHarmonic : public Bond {
  public:
    /// @param K Force constants per bond type [eV/Å²].
    /// @param r0 Equilibrium distances per bond type [Å].
    BondHarmonic(std::vector<double> K, std::vector<double> r0);

    std::string name() const override { return "harmonic_bond"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_, r0_;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    /// @brief Resolved-tuple cache; rebuilt on migration / halo full rebuild.
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
