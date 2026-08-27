/// @file cpu/potentials/coul_long.h
/// @brief Real-space screened Coulomb (LAMMPS `pair_style coul/long`) — CPU.
///
/// The short-range half of an Ewald/PME split. Evaluates
///
///   E_real = units::coulomb · Σ_{i<j} q_i q_j · erfc(α r_ij) / r_ij
///
/// over the neighbor list, where α is the Ewald splitting parameter shared
/// with the reciprocal-space `KSpaceEwald` / `KSpacePME` component.
///
/// Optional special-bonds scales apply to this real-space loop only
/// (Amber/GAFF: `set_special_bonds(0.0, 0.0, 0.5)`). Reciprocal + self terms
/// are unchanged (LAMMPS convention).
#pragma once

#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/cpu/component/long_range_protocol.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/potentials/special_scale.h"

namespace ATV {
namespace cpu {

/// @brief Real-space screened Coulomb force/energy (Ewald/PME real part).
class CoulLong : public cpu::Potential,
                 public cpu::NeighborConsumer,
                 public cpu::LongRangeParticipant {
  public:
    /// @brief Real-space Coulomb reads the neighbour charge.
    /// @param out Schema being accumulated at build().
    void declare_fields(Schema& out) const override {
        out.require(kAtom_charge, "CoulLong");
    }

    /// @param alpha  Ewald splitting parameter α [1/Å]. Must match the
    ///               reciprocal-space component.
    /// @param cutoff Real-space cutoff [Å]. The neighbor list must be built
    ///               with a cutoff ≥ this value.
    CoulLong(double alpha, double cutoff);

    std::string name() const override { return "coul/long"; }
    const NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }
    const LongRangeParticipant* as_long_range_participant() const override {
        return this;
    }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::RealSpace, alpha_, cutoff_, AccReal(0)};
    }

    AccReal neighbor_cutoff() const override { return cutoff_; }
    AccReal pair_cutoff() const { return cutoff_; }

    /// @brief Real-space special-bonds weights (LAMMPS special_bonds).
    /// @param s12 1-2 scale (0 = skip pair).
    /// @param s13 1-3 scale.
    /// @param s14 1-4 scale (Amber/GAFF: 0.5).
    CoulLong& set_special_bonds(double s12, double s13, double s14) &;
    CoulLong set_special_bonds(double s12, double s13, double s14) &&;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  private:
    double alpha_;
    double cutoff_, cutoff_sq_;
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;
    bool any_special_ = false;
    md::SpecialScaleTable special_;
    unsigned long long special_version_ = 0;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;
};

} // namespace cpu
} // namespace ATV
