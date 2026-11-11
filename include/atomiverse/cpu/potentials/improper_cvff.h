/// @file cpu/potentials/improper_cvff.h
/// @brief LAMMPS `improper_style cvff` — V = K (1 + d cos(n φ)), d = ±1.
///
/// The Amber/GAFF improper: a periodic torsion over the improper 4-tuple in
/// the file's atom order (Amber puts the central atom third). Equivalent to a
/// one-term periodic torsion with δ = 0 (d = +1) or δ = π (d = −1). Topology
/// is the per-atom improper columns; energy goes to `kEnergy` /
/// `kEnergyImproper`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/improper.h"
#include "atomiverse/cpu/potentials/periodic_torsion.h"

namespace ATV {
namespace cpu {

/// @brief CVFF/Amber improper torsion.
class ImproperCvff : public Improper {
  public:
    /// @param K Amplitude per improper type [eV].
    /// @param d Sign per type, +1 or −1.
    /// @param n Multiplicity per type, 0 ≤ n ≤ 6.
    /// @throws std::invalid_argument on size mismatch, an empty table, d ∉
    ///         {−1, +1} or n outside [0, 6].
    ImproperCvff(std::vector<double> K, std::vector<int> d, std::vector<int> n);

    std::string name() const override { return "cvff_improper"; }

    /// @brief Skip the improper virial when no barostat needs stress.
    /// @param on Accumulate the virial when stress is requested.
    /// @return This component.
    ImproperCvff& set_compute_stress(bool on) & {
        set_.set_compute_stress(on);
        return *this;
    }
    /// @copydoc set_compute_stress(bool) &
    ImproperCvff set_compute_stress(bool on) && {
        set_.set_compute_stress(on);
        return std::move(*this);
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    md::PeriodicTorsionSet set_;
};

} // namespace cpu
} // namespace ATV
