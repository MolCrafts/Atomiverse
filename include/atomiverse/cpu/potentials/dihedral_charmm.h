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

#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/dihedral.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

/// @brief Periodic Fourier torsion (CHARMM / AMBER-style).
///
/// Per type: V = K (1 + cos(n φ − δ)) with K [eV], n ≥ 1 (multiplicity),
/// δ [rad] (phase). Reuses BondedTupleCache / BondedForceScratch /
/// dihedral_geom with DihedralMultiHarmonic.
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
        compute_stress_ = on;
        return *this;
    }
    DihedralCharmm set_compute_stress(bool on) && {
        compute_stress_ = on;
        return std::move(*this);
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<double> K_;
    std::vector<int> n_;
    std::vector<double> delta_;
    /// Precomputed cos(δ), sin(δ) — avoids per-tuple sin/cos of phase.
    std::vector<double> cos_delta_;
    std::vector<double> sin_delta_;
    bool compute_stress_ = true;

    md::BondedForceScratch scratch_;
    md::DropTally drops_;
    md::BondedTupleCache cache_;
};

} // namespace cpu
} // namespace ATV
