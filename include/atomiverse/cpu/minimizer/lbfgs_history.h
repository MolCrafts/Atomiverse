/// @file cpu/minimizer/lbfgs_history.h
/// @brief LbfgsHistory — the curvature history and the two-loop recursion, on their own.
///
/// The L-BFGS *algorithm* — store `m` recent `(s, y)` curvature pairs, recover a descent
/// direction `-Hg` from them via Nocedal's two-loop recursion — has nothing to do with
/// atoms. It operates on a vector. Until now it was nonetheless welded to one: the
/// recursion sat inline inside `LBFGS::launch`, reading positions and forces straight out
/// of `ctx.frame`, so the dimension was `owned_count(ctx) * 3` and there was no seam to
/// inject any other vector through.
///
/// That was correct while `Minimize` was the only caller. A band optimizer is the
/// **second** caller: its vector is the concatenation of every image in the chain, and
/// its dimension is `n_images * 3N`. Per the house rule, extraction happens when the
/// second caller arrives — not before, and not in anticipation.
///
/// ## The reduction is a constructor argument, not an ambient fact
///
/// Every inner product in the recursion is a *global* one under domain decomposition: a
/// rank holds only its own rows, so `s·q` is a partial sum that must be all-reduced
/// before it can scale anything, or the ranks compute different directions and the band
/// tears. The old code reached for `global_sum(ctx, ...)` — an MPI call keyed on the
/// LaunchContext — from inside the recursion, which is precisely what made it
/// unextractable. Here the reduction is a `Reducer` handed in at construction: the MD
/// minimizer passes its MPI all-reduce, and a band optimizer whose vector is already
/// rank-complete passes the identity.
#pragma once

#include <functional>
#include <vector>

namespace ATV {
namespace cpu {
namespace minimizer {

/// @brief Curvature history + two-loop recursion over a plain vector of length `dim`.
class LbfgsHistory {
  public:
    /// @brief Sums a scalar across whatever the caller considers "everyone".
    ///
    /// Under MPI domain decomposition this is an `MPI_Allreduce` over the spatial
    /// communicator. In a serial run, or for a vector that is already complete on this
    /// rank, it is the identity. It is not defaulted: a silent identity is exactly the
    /// bug that produces per-rank-divergent search directions, and it must be an
    /// explicit choice.
    using Reducer = std::function<double(double)>;

    /// @param memory Number of curvature pairs to retain (`m`).
    /// @param max_dim Capacity: the longest vector this history will ever see.
    /// @param sum Cross-rank scalar reduction. @see Reducer
    ///
    /// @note Capacity and working length are separate on purpose. Under domain
    ///       decomposition the working length is `owned_rows * 3`, and that CHANGES as
    ///       atoms migrate between ranks — so every method below takes the current `dim`
    ///       while the buffers stay allocated at `max_dim`. Sizing the buffers to the
    ///       working length instead would mean reallocating inside the step loop.
    LbfgsHistory(int memory, int max_dim, Reducer sum);

    /// @brief Grow the capacity (a rank that just received an influx of migrated atoms).
    void grow(int max_dim);

    /// @brief Discard the history and the previous point.
    ///
    /// Mandatory after atom migration: every stored `s` and `y` is indexed by local row,
    /// and migration reshuffles what row means. Curvature pairs from before the shuffle
    /// describe a different vector space. Keeping them does not throw — it quietly
    /// produces a garbage direction.
    void reset();

    /// @brief Fold the step just taken into the history.
    ///
    /// `s = x - x_prev`, `y = g - g_prev`, both against the state given to the last
    /// `save()`. A no-op on the first call (there is no previous state to difference).
    ///
    /// The pair is stored only if it passes the curvature condition
    /// `yᵀs > eps·‖s‖·‖y‖`, which is what keeps the implicit inverse Hessian positive
    /// definite. A strong-Wolfe line search guarantees it (N&W §3.1); Armijo does not,
    /// and a search that bailed out certainly does not — so it is *checked*, not assumed.
    void push(const double* x, const double* g, int dim);

    /// @brief Two-loop recursion: `dir = -H g`.
    /// @param g Gradient at the current point (length `dim`).
    /// @param dir_out Descent direction, overwritten (length `dim`).
    /// @param dim Current working length.
    void direction(const double* g, double* dir_out, int dim);

    /// @brief Record the current point as the base for the next `push()`.
    void save(const double* x, const double* g, int dim);

    /// @brief The last saved point — the base point a line search steps out from.
    const std::vector<double>& x_prev() const { return x_prev_; }
    const std::vector<double>& g_prev() const { return g_prev_; }
    bool has_prev() const { return has_prev_; }

    /// @brief How many curvature pairs are currently stored (`<= memory`).
    int stored() const { return n_stored_; }

  private:
    int memory_;
    int max_dim_;
    Reducer sum_;

    std::vector<std::vector<double>> s_history_; ///< Position differences.
    std::vector<std::vector<double>> y_history_; ///< Gradient differences.
    std::vector<double> rho_history_;            ///< 1 / (yᵀs).
    int n_stored_ = 0;
    int oldest_ = 0;

    std::vector<double> x_prev_;
    std::vector<double> g_prev_;
    bool has_prev_ = false;

    // Recursion workspaces — allocated once in the constructor. No allocation on the
    // hot path, same contract as a component's build()/launch() split.
    std::vector<double> q_ws_;
    std::vector<double> r_ws_;
    std::vector<double> alpha_ws_;
};

} // namespace minimizer
} // namespace cpu
} // namespace ATV
