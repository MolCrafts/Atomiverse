/// @file cpu/potentials/dihedral_multi_harmonic.h
/// @brief LAMMPS-style DihedralMultiHarmonic — V = Σₙ Cₙ cosⁿ(φ) (CPU).
///
/// Was `ATV::MultiHarmonicDihedral` before the LAMMPS-naming refactor.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/dihedral.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

class DihedralMultiHarmonic : public Dihedral {
  public:
    /// @param Cn Coefficients per dihedral type. Cn[type] = {C0, C1, C2, ...}.
    explicit DihedralMultiHarmonic(std::vector<std::vector<double>> Cn);

    std::string name() const override { return "multi_harmonic_dihedral"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<std::vector<double>> Cn_;
    /// Per-type "any nonzero coefficient" flag — an all-zero type (the other
    /// member of a hybrid dihedral split owns it) is skipped in the hot loop.
    std::vector<char> active_;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    /// @brief Resolved-tuple cache; rebuilt on migration / halo full rebuild.
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
