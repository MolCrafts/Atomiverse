/// @file cuda/minimize/lbfgs.cuh
/// @brief cuda::minimizer::LBFGS — L-BFGS Optimizer component.
///
/// CUDA mirror of `cpu::minimizer::LBFGS`: two-loop recursion with
/// optional Armijo backtracking line search, hot-swappable inside
/// `cuda::minimizer::Minimize` next to `CG` / `SteepestDescent`.
///
/// Execution model: the search-direction iteration (history update,
/// two-loop recursion, step application) is device-resident and replayed
/// as one captured graph. The curvature test and the pair ring live on the
/// device: a candidate pair is staged in a spare buffer and swapped in only
/// when y·s > 1e-10, so a rejected pair never overwrites the oldest stored
/// one. The recursion launches a fixed `2 memory + 1` vector passes (each
/// update fused with the next inner product, y·y cached per pair; passes
/// beyond the stored count exit at once), so the launch sequence does not
/// depend on device state. Host touch points per iteration:
///   - `convergence_check` runs the shared `ConvCheck` bridge (the
///     metrics must reach the host, where the driver polls its
///     handlers).
///   - The optional Armijo line search decides on the device: the
///     direction graph opens it (origin slope and energy), and each bounded
///     trial — one force evaluation, then the device verdict and the next
///     trial's move, replayed as a graph — ends in one stream sync after
///     which the host reads the verdict from a mapped host word (the CUDA
///     RHF convergence-flag pattern). No scalar is copied back; all
///     geometry and force arrays stay on device.
///
/// **Tolerances are not here.** They belong to the termination policy —
/// `cuda::ForceConverged`, which `cuda::minimizer::Minimize` attaches by default and
/// which the user replaces with `min.add_criterion(cuda::ForceConverged{ftol, …})`. This
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

#include "atomiverse/cuda/component/optimizer.cuh"
#include "atomiverse/cuda/device/graph_ring.cuh"

#include <string>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/minimize/conv_check.cuh"

namespace ATV {
namespace cuda {
namespace minimizer {

/// @brief L-BFGS Optimizer component (two-loop recursion on device).
class LBFGS : public cuda::Optimizer {
  public:
    /// @brief Construct L-BFGS step.
    /// @param memory    Number of correction pairs to store (default 5).
    /// @param alpha0    Initial step size [Angstrom^2/eV] (default 0.01).
    /// @param max_ls    Maximum backtracking attempts per step (default 10).
    explicit LBFGS(int memory = 5, double alpha0 = 0.01, int max_ls = 10)
        : memory_(memory), alpha0_(alpha0), max_ls_(max_ls) {}

    /// @brief Enable the backtracking line search (Armijo condition).
    ///
    /// A driver step whose trial fails `E_{k+1} − E_k <= c1 · α_k · g_k·d_k` is
    /// rolled back to `x_k`, α is scaled by @p contract, and the trial
    /// is re-taken along the same direction; history is only updated on
    /// accepted trials. Must be called before `opt.build()`.
    /// @param c1         Armijo constant in (0, 1) (default 1e-4).
    /// @param contract   Backtracking contraction factor (default 0.5).
    /// @param min_alpha  Smallest trial alpha; failure at the floor or budget
    ///                   restores the origin and throws (default 1e-6).
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

    /// @brief Default bindings: direction, bounded line search, measurement.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::minimizer::Phase::search_direction, 0),
                ATV::make_binding(ATV::minimizer::Phase::line_search, 0),
                ATV::make_binding(ATV::minimizer::Phase::convergence_check, 0)};
    }

    /// @brief Allocate + zero history and scratch device buffers.
    void build(const BuildContext& ctx) override;

    /// @brief Dispatch direction, finite Armijo search, or convergence measurement.
    void launch(const LaunchContext& ctx) override;

    /// @brief Release all device + pinned buffers.
    void destroy(const DestroyContext& ctx) override;

    /// @brief Max force magnitude at the latest convergence check.
    double max_force() const { return conv_.max_force(); }

    /// @brief Per-step optimization trace.
    const std::vector<ATV::minimizer::OptRecord>& trace() const {
        return trace_;
    }

  private:
    int memory_;    ///< Correction-pair capacity.
    double alpha0_; ///< Initial step size [Angstrom^2/eV].
    int max_ls_;    ///< Backtracking budget (parity with the CPU signature).

    int n_ = 0;  ///< Atom count.
    int n3_ = 0; ///< 3 * n_.

    // Planar [x:0..n, y:n..2n, z:2n..3n] device buffers, mixed_t precision
    // (s = x − x_prev and y = g − g_prev are differences of nearly-equal
    // values; force_t storage would destroy the curvature information).
    // Positions are the reconstructed (primary + correction) coordinates, so
    // a rollback from x_prev restores both stored columns exactly. Every
    // reduction target (dot products, max, y·s, g·d) is acc_t.
    mixed_t* d_x_ = nullptr;     ///< Current packed positions (3N).
    mixed_t* d_g_ = nullptr;     ///< Current packed gradient −F (3N).
    mixed_t* d_xprev_ = nullptr; ///< Positions at the last accepted step (3N).
    mixed_t* d_gprev_ = nullptr; ///< Gradient at the last accepted step (3N).
    mixed_t* d_s_ = nullptr;     ///< Position-difference ring (memory × 3N).
    mixed_t* d_y_ = nullptr;     ///< Gradient-difference ring (memory × 3N).
    mixed_t* d_q_ = nullptr;     ///< Two-loop scratch q → r (3N).
    mixed_t* d_dir_ =
        nullptr; ///< Search direction (3N, persists for rollback).
    mixed_t* d_alpha_ = nullptr; ///< Two-loop a_k scalars (memory).
    ImageInt* d_image_prev_ = nullptr;
    acc_t* d_dot_ = nullptr; ///< Per-iteration reductions (y·s, y·y, max|d|,
                             ///< one inner product per two-loop pass).
    int n_dots_ = 0;
    /// Device pair ring: {n_stored, oldest, spare} then slot[memory]. The
    /// curvature test and the ring bookkeeping run on the device.
    int* d_ring_state_ = nullptr;
    acc_t* d_rho_ = nullptr; ///< 1/(y·s) then y·y per pair buffer.
    acc_t* d_ls_ =
        nullptr; ///< Device {trial α, g·d, origin energy} line-search scalars.
    void* d_line_state_ = nullptr; ///< Device line-search state.
    int* h_verdict_ = nullptr;     ///< Mapped host word: the device verdict.
    int* d_verdict_ = nullptr;     ///< Device alias of `h_verdict_`.
    /// The search-direction iteration, captured once without and once with
    /// a candidate pair.
    GraphRing graphs_[2];
    /// One line-search verdict (judge + next trial's move).
    GraphRing trial_graph_;

    BigInt step_ = 0;
    bool has_prev_ = false;

    // Optional finite Armijo search; disabled by default.
    bool ls_enabled_ = false;
    double ls_c1_ = 1e-4;
    double ls_contract_ = 0.5;
    double ls_min_alpha_ = 1e-6;

    ConvCheck conv_;
    std::vector<ATV::minimizer::OptRecord> trace_;
};

} // namespace minimizer
} // namespace cuda
} // namespace ATV
