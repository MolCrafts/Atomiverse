/// @file cpu/path/neb.h
/// @brief Nudged elastic band: the coupling component and the band optimizer.
///
/// The directory is `cpu/path/`, not `cpu/neb/` — a domain directory is named for its
/// domain, never for one algorithm in it (`minimizer/` holds LBFGS; `mc/` holds
/// TranslateMove). Calling it `neb/` would leave the next path method — a string method,
/// a dimer — with nowhere to live but a directory named after its rival.
///
/// Both classes here are **components**. Nothing about NEB is in `PathDriver`, which
/// knows only "advance N coupled replicas until all converge". The tangent estimator,
/// the spring, and the climbing-image flip are things you *plug in* — so that PIMD, when
/// it arrives, plugs its ring-polymer spring into the very same `couple` slot without
/// the driver learning a thing about beads.
///
/// References:
///   G. Henkelman, B. P. Uberuaga, H. Jónsson, "A climbing image nudged elastic band
///   method for finding saddle points and minimum energy paths", J. Chem. Phys. 113,
///   9901 (2000).
///   G. Henkelman, H. Jónsson, "Improved tangent estimate in the nudged elastic band
///   method for finding minimum energy paths and saddle points", J. Chem. Phys. 113,
///   9978 (2000).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/minimizer/lbfgs_history.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief The NEB force: project out, spring back, and let one image climb.
///
/// Reads every replica's *true* force (whatever evaluator ran at the `force` phase) and
/// **overwrites** it with the NEB force. That is the whole trick of the method, and it
/// is worth being blunt about what it means: after this component runs, the forces in
/// the frames are no longer the gradient of any energy. They are a construction designed
/// so that its zero is a minimum energy path.
///
/// For interior image `i`, with unit tangent `τ`:
/// @code
///   F = F_true - (F_true·τ)τ        (perpendicular part of the true force)
///     + k(|R_{i+1}-R_i| - |R_i-R_{i-1}|) τ    (spring, parallel only)
/// @endcode
/// Mixing the two projections up is the classic NEB failure: keep the *parallel* true
/// force and the band slides down into the minima; keep the *perpendicular* spring force
/// and the band corner-cuts around the saddle.
///
/// The **climbing image** — the highest-energy one — instead gets
/// `F = F_true - 2(F_true·τ)τ`: its parallel force is *inverted*, so it walks uphill
/// along the path while still relaxing perpendicular to it, and it feels no spring at
/// all. That is what drives it onto the saddle. A plain band does not sit on the saddle;
/// it straddles it, and the barrier you read off is an interpolation error.
///
/// The two **endpoints are frozen** (force zeroed): they are the reactant and product,
/// given, not searched for.
class NebCoupling : public cpu::PathCoupling {
  public:
    /// @param spring_k Spring constant along the band [eV/Å²].
    /// @param climbing Enable the climbing image (CI-NEB). Off = plain NEB, which
    ///        converges to a band *near* the saddle but never onto it.
    /// @param climb_after Iterations of plain NEB before the climb switches on. Climbing
    ///        from a raw linear interpolation is unstable — the highest image of a bad
    ///        initial band is often nowhere near the saddle, and inverting its force
    ///        sends it uphill in the wrong direction entirely. Standard practice is to
    ///        relax first, then climb.
    explicit NebCoupling(double spring_k, bool climbing = true,
                         int climb_after = 5)
        : k_(spring_k), climbing_(climbing), climb_after_(climb_after) {}

    std::string name() const override { return "neb_coupling"; }
    /* role via Potential/kind base */

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::path::Phase::couple, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Index of the climbing image in the most recent iteration (-1 if not
    ///        climbing yet). The saddle, once converged.
    int climbing_image() const { return climb_idx_; }

  private:
    double k_;
    bool climbing_;
    int climb_after_;
    int climb_idx_ = -1;

    std::vector<double> tangent_; ///< Scratch, sized in build().
};

/// @brief L-BFGS over the whole band at once.
///
/// The band's coordinate vector is the concatenation of every image, so the optimizer's
/// dimension is `n_images × 3N` and it sees the chain as one system. That is the second
/// caller that justified extracting `LbfgsHistory` from `cpu::minimizer::LBFGS`, where
/// the two-loop recursion had been welded to a single Frame.
///
/// Convergence is **per image** (`ReplicaSet::set_converged`), which the driver ANDs. It
/// is judged on the forces this iteration evaluated — i.e. on the geometry *before* this
/// iteration's step — so a converged band always takes one final, tiny step it did not
/// strictly need. That is the honest cost of deciding convergence in the phase named for
/// it rather than in the middle of the one that moves the atoms.
class BandLBFGS : public cpu::Optimizer {
  public:
    /// @param memory Curvature pairs to retain.
    /// @param max_step Cap on any single coordinate's displacement [Å]. NEB forces are
    ///        not the gradient of anything, so a line search has no energy to search on;
    ///        a trust-radius cap is the standard substitute.
    /// @param ftol Convergence: max |NEB force| on any interior image [eV/Å].
    BandLBFGS(int memory = 8, double max_step = 0.05, double ftol = 0.05)
        : memory_(memory), max_step_(max_step), ftol_(ftol) {}

    std::string name() const override { return "band_lbfgs"; }
    /* role via Optimizer base */

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::path::Phase::search_direction, 0),
                ATV::make_binding(ATV::path::Phase::convergence_check, 0)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Largest NEB-force magnitude over the interior images [eV/Å].
    double max_force() const { return max_force_; }

  private:
    int memory_;
    double max_step_;
    double ftol_;

    std::unique_ptr<cpu::minimizer::LbfgsHistory> hist_;
    std::vector<double> x_;   ///< Band coordinates, concatenated.
    std::vector<double> g_;   ///< Band gradient = -(NEB force).
    std::vector<double> dir_; ///< Search direction.
    std::vector<double>
        image_max_force_; ///< Per-image max |F| from this iteration.
    double max_force_ = 0.0;
};

} // namespace path
} // namespace cpu
} // namespace ATV
