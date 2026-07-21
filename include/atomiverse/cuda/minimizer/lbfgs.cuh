/// @file cuda/minimizer/lbfgs.cuh
/// @brief cuda::minimizer::LBFGS — L-BFGS Optimizer component.
///
/// CUDA mirror of `cpu::minimizer::LBFGS`: two-loop recursion with
/// optional Armijo backtracking line search, hot-swappable inside
/// `cuda::minimizer::Minimize` next to `CG` / `SteepestDescent`.
///
/// Execution model: the heavy vector work (history update, two-loop
/// recursion, step application) is GPU-pure — every intermediate dot
/// product lands in a device scalar that the next kernel reads by
/// pointer, the same device-scalar chaining as
/// `cuda::md::NoseHoover`. Two host touch points per iteration remain,
/// both **host-dependent compute** by necessity:
///   - `search_direction` reads back the 8-byte curvature product
///     `y·s` — the accept/reject decision gates the host-side circular
///     history bookkeeping that determines how many two-loop kernels
///     are launched.
///   - `convergence_check` runs the shared `ConvCheck` bridge (the
///     metrics must reach the host, where the driver polls its
///     handlers). Line-search scalars (applied α, g·d) ride this sync as
///     batched pinned D2H copies — no extra synchronize.
///
/// **Tolerances are not here.** They belong to the termination policy —
/// `cuda::ForceConverged`, which `cuda::minimizer::Minimize` attaches by default and
/// which the user replaces with `min.add_handler(cuda::ForceConverged{ftol, …})`. This
/// component measures (all five metrics: max/RMS force, max/RMS displacement, energy
/// change) and publishes them; it does not decide. Hence five fewer constructor
/// arguments, and no `LBFGS::converged()` — ask the loop (`min.converged()`).
///
/// At step 0: steepest descent (no history). From step 1: L-BFGS
/// direction. Step length is capped so no atom moves more than 0.1
/// Angstrom per iteration, as on the CPU.
///
/// References:
///   Nocedal, Math. Comp. 35, 773 (1980) — L-BFGS algorithm
///   Nocedal & Wright, Numerical Optimization, Ch. 7
///
/// @see cuda::minimizer::Minimize for the driver this component plugs into
/// @see cpu::minimizer::LBFGS for the CPU mirror
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/minimizer/conv_check.cuh"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief L-BFGS Optimizer component (two-loop recursion on device).
class LBFGS : public Component {
  public:
    /// @brief Construct L-BFGS step.
    /// @param memory    Number of correction pairs to store (default 5).
    /// @param alpha0    Initial step size [Angstrom^2/eV] (default 0.01).
    /// @param max_ls    Maximum backtracking attempts per step (default 10).
    explicit LBFGS(int memory = 5, double alpha0 = 0.01, int max_ls = 10)
        : memory_(memory), alpha0_(alpha0), max_ls_(max_ls) {}

    /// @brief Enable the backtracking line search (Armijo condition).
    ///
    /// Same contract as `cpu::minimizer::LBFGS::line_search`: a driver
    /// step whose trial fails `E_{k+1} − E_k <= c1 · α_k · g_k·d_k` is
    /// rolled back to `x_k`, α is scaled by @p contract, and the trial
    /// is re-taken along the same direction; history is only updated on
    /// accepted trials. Must be called before `opt.build()`.
    /// @param c1         Armijo constant in (0, 1) (default 1e-4).
    /// @param contract   Backtracking contraction factor (default 0.5).
    /// @param min_alpha  Minimum α before accepting the trial even on
    ///                   Armijo failure (default 1e-6).
    /// @return Reference to self for chaining.
    LBFGS& line_search(double c1 = 1e-4, double contract = 0.5,
                       double min_alpha = 1e-6) {
        ls_enabled_ = true;
        ls_c1_ = c1;
        ls_contract_ = contract;
        ls_min_alpha_ = min_alpha;
        return *this;
    }

    /// @brief Component name.
    /// @return "lbfgs".
    std::string name() const override { return "lbfgs"; }
    Role role() const override { return 0; }

    /// @brief Structural role of this component.
    /// @return ComponentRole::Optimizer.
    /* role via Optimizer base */

    /// @brief Default bindings: search_direction + convergence_check.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    /// @brief Allocate + zero history and scratch device buffers.
    void build(const BuildContext& ctx) override;

    /// @brief Dispatch the bound phase (search_direction / convergence_check).
    void launch(const LaunchContext& ctx) override;

    /// @brief Release all device + pinned buffers.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Max force magnitude at the latest convergence check.
    double max_force() const { return conv_.max_force(); }

    /// @brief Per-step optimization trace.
    const std::vector<OptRecord>& trace() const { return trace_; }

  private:
    int memory_;    ///< Correction-pair capacity.
    double alpha0_; ///< Initial step size [Angstrom^2/eV].
    int max_ls_;    ///< Backtracking budget (parity with the CPU signature).

    int n_ = 0;  ///< Atom count.
    int n3_ = 0; ///< 3 * n_.

    // Planar [x:0..n, y:n..2n, z:2n..3n] device buffers, work_t precision
    // (s = x − x_prev and y = g − g_prev are differences of nearly-equal
    // values; real_t storage would destroy the curvature information).
    work_t* d_x_ = nullptr;     ///< Current packed positions (3N).
    work_t* d_g_ = nullptr;     ///< Current packed gradient −F (3N).
    work_t* d_xprev_ = nullptr; ///< Positions at the last accepted step (3N).
    work_t* d_gprev_ = nullptr; ///< Gradient at the last accepted step (3N).
    work_t* d_s_ = nullptr;     ///< Position-difference ring (memory × 3N).
    work_t* d_y_ = nullptr;     ///< Gradient-difference ring (memory × 3N).
    work_t* d_q_ = nullptr;     ///< Two-loop scratch q → r (3N).
    work_t* d_dir_ = nullptr; ///< Search direction (3N, persists for rollback).
    work_t* d_alpha_ = nullptr; ///< Two-loop a_k scalars (memory).
    work_t* d_dot_ = nullptr;   ///< Device {dot, dir_max} scratch scalars.
    work_t* d_ys_ = nullptr;    ///< Device curvature product y·s.
    work_t* d_ls_ = nullptr; ///< Device {applied α, g·d} line-search scalars.
    work_t* h_ys_ = nullptr; ///< Pinned mirror of d_ys_.
    work_t* h_ls_ = nullptr; ///< Pinned mirror of d_ls_.

    // Host-side circular-buffer bookkeeping (gates kernel-launch counts).
    std::vector<double> rho_; ///< 1 / (y·s) per stored pair.
    int n_stored_ = 0;
    int oldest_ = 0;
    int step_ = 0;
    bool has_prev_ = false;

    // Line search (optional; off by default, as on the CPU).
    bool ls_enabled_ = false;
    double ls_c1_ = 1e-4;
    double ls_contract_ = 0.5;
    double ls_min_alpha_ = 1e-6;
    double ls_alpha_ = 0;         ///< α of the most recent trial (host copy).
    double ls_gd_ = 0;            ///< g·d at the trial origin (host copy).
    double ls_origin_energy_ = 0; ///< Energy at the trial origin [eV].
    bool ls_had_trial_ = false;
    bool ls_fetch_pending_ = false; ///< D2H of {α, g·d} queued for next check.

    ConvCheck conv_;
    std::vector<OptRecord> trace_;
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
