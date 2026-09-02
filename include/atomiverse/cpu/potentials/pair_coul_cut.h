/// @file cpu/potentials/pair_coul_cut.h
/// @brief Real-space Coulomb with hard cutoff (LAMMPS `pair_style coul/cut`) — CPU.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/role.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/special_scale.h"

namespace ATV {
namespace cpu {

/// @brief Truncated Coulomb pair style (no Ewald/PME partner required).
///
/// Optional special-bonds (LAMMPS convention): 1-2 / 1-3 / 1-4 weights
/// multiply pair energy and force. Scale 0 drops the pair; Amber/GAFF uses
/// `set_special_bonds(0.0, 0.0, 0.5)`.
class PairCoulCut : public cpu::PairPotential {
  public:
    /// @param cutoff Real-space cutoff [Å].
    explicit PairCoulCut(double cutoff);

    /// @brief The cut Coulomb kernel reads the NEIGHBOUR charge, and a
    /// neighbour is a ghost across every cut.
    ///
    /// `stable`, matching `CoulLong` / `PairLJCutCoulLong`: a fixed-charge
    /// model never changes it, and a fluctuating-charge owner (QEq,
    /// constant-potential electrodes) refreshes ghosts itself through
    /// `ChargePacker` on the self-initiated `ForwardPacker` path.
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_charge, "PairCoulCut"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    std::string name() const override { return "coul/cut"; }

    /// @brief Enable LAMMPS-style special-bonds scales for Coulomb.
    /// @param s12 Weight for 1-2 pairs (0 = exclude; (0,1] multiplies E,F).
    /// @param s13 Weight for 1-3 pairs.
    /// @param s14 Weight for 1-4 pairs (Amber/GAFF: 0.5).
    PairCoulCut& set_special_bonds(double s12, double s13, double s14) &;
    PairCoulCut set_special_bonds(double s12, double s13, double s14) &&;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void rebuild_filtered_list(const Frame& frame, const md::PairList& pairs);

    double cutoff_sq_;
    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;
    bool any_special_ = false;
    md::SpecialScaleTable special_;

    std::vector<int> filtered_neighbors_;
    std::vector<int> filtered_numneigh_;
    std::vector<int> filtered_firstneigh_;
    std::vector<double> filtered_scale_;
    std::vector<double> filtered_shift_x_;
    std::vector<double> filtered_shift_y_;
    std::vector<double> filtered_shift_z_;
    bool filtered_has_shifts_ = false;
    int filtered_inum_ = 0;
    unsigned long long filtered_version_ = 0;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;
};

} // namespace cpu
} // namespace ATV
