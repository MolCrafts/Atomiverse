/// @file cpu/potentials/pair_lj_soft.h
/// @brief Beutler/GROMACS one-leg soft-core LJ 12-6 (CPU).
///
/// Appearance form (create LJ site as λ → 1):
///   r_λ = (α σ⁶ (1−λ)^p + r⁶)^{1/6}
///   V   = λ [ V_LJ(r_λ) − V_shift ]
///   F_r = λ F_LJ(r_λ) (r / r_λ)^5
///
/// Cutoff is on the raw distance r (PairLJCut convention). V_shift is the
/// hard LJ value at r_c so λ=1 matches shifted PairLJCut bit-for-bit.
///
/// This is **not** LAMMPS `lj/cut/soft` algebra (different soft-core map).
///
/// Refs: Beutler et al., Chem. Phys. Lett. 222, 529 (1994);
///       GROMACS free-energy interactions (one-leg soft-core).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Soft-core Lennard-Jones 12-6 pair Potential (single-type, CPU).
///
/// Construct with (ε, σ, cutoff); optional soft-core α and p (defaults 0.5, 1).
/// Mutate public `lambda` ∈ [0,1] between Evaluate/MD runs for one-leg FEP.
class PairLJSoft : public Pair {
  public:
    /// @brief Alchemical coupling (dimensionless). Default 1 = hard LJ limit.
    double lambda = 1.0;

    /// @param epsilon Well depth [eV].
    /// @param sigma   Length scale [Å].
    /// @param cutoff  Interaction cutoff on raw r [Å].
    /// @param alpha   Soft-core strength (dimensionless, default 0.5).
    /// @param p       Soft-core λ power (dimensionless, default 1).
    /// @throws std::invalid_argument if any length/energy/α/p is non-positive.
    PairLJSoft(double epsilon, double sigma, double cutoff, double alpha = 0.5,
               double p = 1.0);

    std::string name() const override { return "lj/soft"; }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Analytic ∂U/∂λ [eV] from the last compute() (fixed geometry).
    double dU_dlambda() const { return dU_dlambda_; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double epsilon_ = 0;
    double sigma_ = 0;
    double alpha_ = 0.5;
    double p_ = 1.0;
    double sigma6_ = 0;
    double cutoff_sq_ = 0;
    double v_shift_ = 0; ///< Hard LJ at raw rc [eV].
    double dU_dlambda_ = 0;

    int n_threads_ = 1;
    std::vector<std::vector<double>> fx_thr_, fy_thr_, fz_thr_;
};

} // namespace cpu
} // namespace ATV
