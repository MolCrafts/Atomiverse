/// @file cpu/potentials/pair_lj_cut_tip4p_cut.h
/// @brief LAMMPS-style `lj/cut/tip4p/cut` — LJ on O + Coulomb on TIP4P M-site (CPU).
///
/// Four-site water: the oxygen charge is evaluated at a massless M site along
/// the HOH bisector (distance `qdist` from O). Forces on M are redistributed
/// to O, H1, H2 (Feenstra, J. Comput. Chem. 20, 786 (1999)) so total force
/// and torque on the molecule are preserved.
///
/// H atoms of oxygen with tag `t` must be tags `t+1` and `t+2` (LAMMPS tip4p
/// convention). Units: energy eV, length Å, charge e.
#pragma once

#include <array>
#include <cmath>
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

/// @brief Cutoff LJ + cut Coulomb with TIP4P virtual M-site on oxygen.
///
/// Parameters are constructor arguments (no public Config bag).
///
/// @code
///   // Classic Jorgensen TIP4P (1983/85), types 0=O, 1=H
///   const double theta = 104.52 * ATV::units::kDegreeInRad;
///   cpu::PairLJCutTip4pCut tip4p(
///       /*type_O=*/0, /*type_H=*/1,
///       /*qdist=*/0.15, /*r_oh=*/0.9572, /*theta=*/theta,
///       /*eps_OO=*/0.1550 / 23.060547830619026, /*sigma_OO=*/3.15365,
///       /*cut_lj=*/10.0, /*cut_coul=*/10.0);
///   tip4p.set_special_bonds(0.0, 0.0, 0.0);
/// @endcode
class PairLJCutTip4pCut : public cpu::PairPotential {
  public:
    /// @param type_O Oxygen type id (`kAtom_type`).
    /// @param type_H Hydrogen type id.
    /// @param qdist Distance O→M along the bisector [Å].
    /// @param r_oh Equilibrium O–H bond length [Å] (sets α).
    /// @param theta_hoh Equilibrium H–O–H angle [rad] (sets α).
    /// @param epsilon_OO LJ well depth for O–O [eV].
    /// @param sigma_OO LJ size for O–O [Å].
    /// @param cut_lj LJ cutoff [Å].
    /// @param cut_coul Coulomb cutoff [Å]; if ≤ 0, equals `cut_lj`.
    /// @throws std::invalid_argument on non-positive cutoffs / distances or
    ///         if α cannot be formed (θ ≥ π or r_oh ≤ 0).
    PairLJCutTip4pCut(int type_O, int type_H, double qdist, double r_oh,
                      double theta_hoh, double epsilon_OO, double sigma_OO,
                      double cut_lj, double cut_coul = -1.0);

    /// @brief The Coulomb half of this style reads the NEIGHBOUR charge, and a
    /// neighbour is a ghost across every cut.
    ///
    /// `stable`, matching `CoulLong` / `PairLJCutCoulLong`: a fixed-charge
    /// model never changes it, and a fluctuating-charge owner (QEq,
    /// constant-potential electrodes) refreshes ghosts itself through
    /// `ChargePacker` on the self-initiated `ForwardPacker` path.
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_charge, "PairLJCutTip4pCut"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    std::string name() const override { return "lj/cut/tip4p/cut"; }

    /// @brief Neighbor list must cover LJ and M-site Coulomb reach.
    AccReal neighbor_cutoff() const override {
        return static_cast<AccReal>(neigh_cut_);
    }

    /// @brief LAMMPS-style special-bonds scales (applied to LJ and Coulomb).
    /// @param s12 Weight for 1-2 pairs (0 = exclude).
    /// @param s13 Weight for 1-3 pairs.
    /// @param s14 Weight for 1-4 pairs.
    PairLJCutTip4pCut& set_special_bonds(double s12, double s13, double s14) &;
    PairLJCutTip4pCut set_special_bonds(double s12, double s13, double s14) &&;

    /// @brief α = qdist / (cos(θ/2) * r_oh) — dimensionless M-site scale.
    double alpha() const { return alpha_; }

    /// @brief O→M distance parameter [Å].
    double qdist() const { return qdist_; }

    /// @brief Place M for one water from O and two H Cartesian positions [Å].
    ///        Displacement H−O is *not* min-imaged here (caller supplies images).
    void compute_newsite(double xO, double yO, double zO, double xH1,
                         double yH1, double zH1, double xH2, double yH2,
                         double zH2, double& xM, double& yM, double& zM) const;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    void ensure_sites_(const Frame& f);
    void rebuild_filtered_list_(const Frame& frame, const md::PairList& pairs);

    int type_O_ = 0;
    int type_H_ = 0;
    double qdist_ = 0.0;
    double alpha_ = 0.0;
    double epsilon_ = 0.0;
    double sigma_ = 0.0;
    double sigma6_ = 0.0;
    double v_shift_ = 0.0;
    double cut_lj_ = 0.0;
    double cut_lj_sq_ = 0.0;
    double cut_coul_ = 0.0;
    double cut_coul_sq_ = 0.0;
    double cut_coul_plus_sq_ = 0.0;
    double neigh_cut_ = 0.0;

    double s12_ = 1.0, s13_ = 1.0, s14_ = 1.0;
    bool any_special_ = false;
    md::SpecialScaleTable special_;

    // Cached H indices + M-site positions (size = n_total).
    std::vector<int> h1_;
    std::vector<int> h2_;
    std::vector<char> is_O_;
    std::vector<double> mx_, my_, mz_;
    int n_sites_ = 0;

    std::vector<int> filtered_neighbors_;
    std::vector<int> filtered_numneigh_;
    std::vector<int> filtered_firstneigh_;
    std::vector<double> filtered_scale_;
    int filtered_inum_ = 0;
    unsigned long long filtered_version_ = 0;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;
};

} // namespace cpu
} // namespace ATV
