/// @file cpu/potentials/dihedral_fourier.h
/// @brief LAMMPS `dihedral_style fourier` — V = Σ_m K_m (1 + cos(n_m φ − δ_m)).
///
/// The Amber/GAFF proper torsion: one dihedral carries every periodic term of
/// its type on one row (`dihedral_coeff t m K1 n1 d1 K2 n2 d2 …`), so the
/// geometry is computed once per torsion. Topology is the per-atom dihedral
/// columns; energy goes to `kEnergy` / `kEnergyDihedral`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/dihedral.h"
#include "atomiverse/cpu/potentials/periodic_torsion.h"

namespace ATV {
namespace cpu {

/// @brief Multi-term periodic dihedral (Amber proper torsion).
class DihedralFourier : public Dihedral {
  public:
    /// @param terms Terms per dihedral type: K [eV], n ∈ [0, 6], δ [rad].
    /// @throws std::invalid_argument on an empty table, non-finite values or
    ///         a multiplicity outside [0, 6].
    explicit DihedralFourier(std::vector<std::vector<TorsionTerm>> terms);

    std::string name() const override { return "fourier_dihedral"; }

    /// @brief Skip the dihedral virial when no barostat needs stress.
    /// @param on Accumulate the virial when stress is requested.
    /// @return This component.
    DihedralFourier& set_compute_stress(bool on) & {
        set_.set_compute_stress(on);
        return *this;
    }
    /// @copydoc set_compute_stress(bool) &
    DihedralFourier set_compute_stress(bool on) && {
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
