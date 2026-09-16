/// @file cuda/scf/scf_evaluator.cuh
/// @brief cuda::scf::SCF — **synthetic-path only** Evaluator (tests).
///
/// **User path for molecular RHF:** `cuda::scf::RHF` (`rhf.cuh`) — members +
/// Frame. This class is retained solely for the synthetic
/// `H(P)=H0+alpha*P` unit-test path. It does **not** accept integral packs
/// or other user bags.
///
/// @code{.cpp}
///   // Preferred (production):
///   cuda::scf::RHF rhf;
///   rhf.basis = cpu::scf::BasisSet::STO3G;
///   rhf.run(frame);
///
///   // Synthetic (tests only):
///   cuda::Evaluate eval;
///   eval.on(evaluate::Phase::evaluate)
///       .add(cuda::scf::SCF{n_occ, alpha, tol, max_iter});
/// @endcode
#pragma once

#include "atomiverse/cuda/component/hf.cuh"

#include <memory>
#include <string>

#include "atomiverse/core/role.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/scf/internal/synthetic_scf_cycle.cuh"
#include "atomiverse/cuda/scf/scf_workspace.cuh"

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Synthetic-path CUDA SCF component (not molecular RHF).
class SCF : public cuda::Component, public cuda::HF {
  public:
    /// @brief Synthetic evaluator `H(P) = H0 + alpha*P`.
    /// @param n_occ Occupied orbitals [dimensionless].
    /// @param alpha Density coupling [dimensionless].
    /// @param tol Frobenius residual tol [dimensionless].
    /// @param max_iter Max SCF iterations [dimensionless].
    SCF(int n_occ, double alpha, double tol, int max_iter);

    std::string name() const override { return "scf_evaluator"; }
    Role role() const override { return 0; }

    /// @brief Resolves the Component / HF dual-inheritance probe.
    bool provides_energy() const override { return true; }
    /// @brief Resolves the Component / HF dual-inheritance probe.
    bool provides_force() const override { return true; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Converged density / MO workspace (observers after build).
    /// @return Device-resident SCF workspace.
    const SCFWorkspace& workspace() const { return cycle_->workspace(); }

  private:
    std::unique_ptr<internal::SyntheticScfCycle> cycle_;
};

} // namespace scf
} // namespace cuda
} // namespace ATV
