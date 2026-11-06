/// @file cuda/potentials/periodic_torsion.cuh
/// @brief CUDA periodic 4-body styles on one table-driven kernel:
///        V = Σ_m K_m (1 + cos(n_m φ − δ_m)) over the dihedral or improper
///        list, the device twin of `cpu::md::PeriodicTorsionSet`.
///
/// | Style | Terms per type | Topology |
/// |---|---|---|
/// | `DihedralCharmm` (LAMMPS `dihedral_style charmm`, weight 0) | one | dihedrals |
/// | `DihedralFourier` (LAMMPS `dihedral_style fourier`) | m | dihedrals |
/// | `ImproperCvff` (LAMMPS `improper_style cvff`) | one, δ ∈ {0, π} | impropers |
///
/// The term table, the multiple-angle sum and the geometry/force map are the
/// shared `core/periodic_torsion.h` and `core/dihedral_geom.h` (FP64).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/periodic_torsion.h"
#include "atomiverse/cuda/component/potential.cuh"

namespace ATV {
namespace cuda {

/// @brief Device term table + the launch over one 4-body topology list.
class PeriodicTorsionSet {
  public:
    /// @brief Which topology list the set walks (owner = atom j).
    enum class Topology { Dihedral, Improper };

    /// @throws std::invalid_argument as `TorsionTable`.
    PeriodicTorsionSet(std::vector<std::vector<TorsionTerm>> terms,
                       Topology topology, const char* style);
    ~PeriodicTorsionSet();
    PeriodicTorsionSet(PeriodicTorsionSet&&) noexcept;
    PeriodicTorsionSet& operator=(PeriodicTorsionSet&&) noexcept;

    /// @brief Upload the table; check every tuple type has coefficients.
    void build(const BuildContext& ctx);
    /// @brief Forces, energy (dihedral or improper term) and virial.
    void compute(const LaunchContext& ctx);
    void destroy();

  private:
    TorsionTable table_;
    Topology topology_;
    const char* style_;
    struct State;
    std::unique_ptr<State> state_;
};

/// @brief LAMMPS `dihedral_style charmm` (weight 0): V = K (1 + cos(nφ − δ)).
class DihedralCharmm : public Potential {
  public:
    /// @param K Force constants per type [eV]; @param n multiplicities
    ///        (≥ 1); @param delta phases [rad].
    DihedralCharmm(std::vector<double> K, std::vector<int> n,
                   std::vector<double> delta);
    std::string name() const override { return "charmm_dihedral"; }
    Role role() const override { return 0; }
    bool graph_capturable() const override { return true; }
    void build(const BuildContext& ctx) override { set_.build(ctx); }
    void launch(const LaunchContext& ctx) override { set_.compute(ctx); }
    void destroy(const DestroyContext&) override { set_.destroy(); }

  private:
    PeriodicTorsionSet set_;
};

/// @brief LAMMPS `dihedral_style fourier`: Σ_m K_m (1 + cos(n_m φ − δ_m)).
class DihedralFourier : public Potential {
  public:
    explicit DihedralFourier(std::vector<std::vector<TorsionTerm>> terms);
    std::string name() const override { return "fourier_dihedral"; }
    Role role() const override { return 0; }
    bool graph_capturable() const override { return true; }
    void build(const BuildContext& ctx) override { set_.build(ctx); }
    void launch(const LaunchContext& ctx) override { set_.compute(ctx); }
    void destroy(const DestroyContext&) override { set_.destroy(); }

  private:
    PeriodicTorsionSet set_;
};

/// @brief LAMMPS `improper_style cvff`: V = K (1 + d cos(n φ)), d = ±1.
class ImproperCvff : public Potential {
  public:
    ImproperCvff(std::vector<double> K, std::vector<int> d, std::vector<int> n);
    std::string name() const override { return "cvff_improper"; }
    Role role() const override { return 0; }
    bool graph_capturable() const override { return true; }
    void build(const BuildContext& ctx) override { set_.build(ctx); }
    void launch(const LaunchContext& ctx) override { set_.compute(ctx); }
    void destroy(const DestroyContext&) override { set_.destroy(); }

  private:
    PeriodicTorsionSet set_;
};

} // namespace cuda
} // namespace ATV
