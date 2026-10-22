/// @file cpu/potentials/dihedral_charmm.h
/// @brief LAMMPS-style CHARMM periodic dihedral — V = K(1 + cos(nφ − δ)).
///
/// Functional form matches LAMMPS `dihedral_style charmm` **without** the
/// optional 1–4 LJ/Coulomb weight factors (those stay on Pair styles via
/// `special_bonds`). Topology is the existing per-atom dihedral columns;
/// energy accumulates into `kEnergy` / `kEnergyDihedral`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/dihedral.h"
#include "atomiverse/cpu/potentials/periodic_torsion.h"

namespace ATV {
namespace cpu {

/// @brief Periodic Fourier torsion (CHARMM / AMBER-style).
///
/// Per type: V = K (1 + cos(n φ − δ)) with K [eV], n ≥ 1 (multiplicity),
/// δ [rad] (phase). The loop is the shared `md::PeriodicTorsionSet`
/// (also behind DihedralFourier and ImproperCvff).
class DihedralCharmm : public Dihedral {
  public:
    /// @param K      Force constants per dihedral type [eV].
    /// @param n      Multiplicities per type (must be ≥ 1).
    /// @param delta  Phase shifts per type [radians].
    /// @throws std::invalid_argument on size mismatch, K finite check, or n < 1.
    DihedralCharmm(std::vector<double> K, std::vector<int> n,
                   std::vector<double> delta);

    std::string name() const override { return "charmm_dihedral"; }

    /// @brief Skip pair-virial from dihedrals when no barostat needs stress
    ///        (~extra 3 outer products per torsion). Default true.
    DihedralCharmm& set_compute_stress(bool on) & {
        set_.set_compute_stress(on);
        return *this;
    }
    DihedralCharmm set_compute_stress(bool on) && {
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
