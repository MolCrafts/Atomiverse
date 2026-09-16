/// @file cpu/potentials/pair_coul_soft.h
/// @brief Beutler/GROMACS one-leg soft-core Coulomb (CPU).
///
/// Appearance form (create Coulomb as λ → 1):
///   r_λ = (α σ_C⁶ (1−λ)^p + r⁶)^{1/6}
///   V   = λ k_C q_i q_j (1/r_λ − 1/r_c)     for r < r_c
///   F   = λ k_C q_i q_j (r⁴ / r_λ⁷) · r_vec   (fpair form)
///
/// Cutoff is on the raw distance r (PairCoulCut convention). Shift so V(r_c)=0.
/// This is **not** a silent LAMMPS `coul/cut/soft` alias.
///
/// Refs: Beutler et al., Chem. Phys. Lett. 222, 529 (1994);
///       GROMACS free-energy soft-core Coulomb.
#pragma once

#include <array>
#include <span>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Soft-core Coulomb pair Potential (CPU).
///
/// Construct with (cutoff, σ_C); optional soft-core α and p (defaults 0.5, 1).
/// Mutate public `lambda` ∈ [0,1] between Evaluate/MD runs for one-leg FEP.
class PairCoulSoft : public Pair {
  public:
    /// @brief Alchemical coupling (dimensionless). Default 1 = hard cut Coulomb.
    double lambda = 1.0;

    /// @param cutoff  Interaction cutoff on raw r [Å].
    /// @param sigma_c Soft-core length parameter [Å] (> 0).
    /// @param alpha   Soft-core strength (dimensionless, default 0.5).
    /// @param p       Soft-core λ power (dimensionless, default 1).
    /// @throws std::invalid_argument if any length/α/p is non-positive.
    PairCoulSoft(double cutoff, double sigma_c, double alpha = 0.5,
                 double p = 1.0);

    /// @brief The softened Coulomb kernel reads the NEIGHBOUR charge, and a
    /// neighbour is a ghost across every cut.
    ///
    /// `stable`, matching `CoulLong` / `PairLJCutCoulLong`: a fixed-charge
    /// model never changes it, and a fluctuating-charge owner (QEq,
    /// constant-potential electrodes) refreshes ghosts itself through
    /// `ChargePacker` on the self-initiated `ForwardPacker` path.
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_charge, "PairCoulSoft"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    std::string name() const override { return "coul/soft"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Analytic ∂U/∂λ [eV] for the last compute() (single config).
    /// @return Sum of per-pair ∂V/∂λ at the last launch geometry.
    double dU_dlambda() const { return dU_dlambda_; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double sigma_c_ = 1.0;
    double alpha_ = 0.5;
    double p_ = 1.0;
    double sigma6_ = 0;
    double cutoff_sq_ = 0;
    double inv_rc_ = 0;
    double dU_dlambda_ = 0;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;
};

} // namespace cpu
} // namespace ATV
