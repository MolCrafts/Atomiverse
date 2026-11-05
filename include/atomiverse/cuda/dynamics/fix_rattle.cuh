/// @file cuda/dynamics/fix_rattle.cuh
/// @brief SHAKE/RATTLE distance constraints on the GPU, one thread per
///        constraint cluster; the device twin of `cpu::md::FixRattle`.
#pragma once

#include "atomiverse/cuda/component/constraint.cuh"
#include "atomiverse/cuda/component/constraint_protocol.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {
namespace md {

/// @brief Iterative holonomic distance constraints (Andersen's RATTLE with
///        SHAKE's position stage), clustered over shared atoms.
///
/// The constraint table comes from the frame's bonded topology, exactly as
/// `cpu::md::FixRattle` builds it: every bond whose type was passed to
/// `constrain_bond()`, every angle whose type was passed to
/// `constrain_angle()` (its 1–3 distance), in the CPU table order (per owner
/// atom: its bonds, then the angles it centres). Constraints that share an
/// atom form one cluster (an X–H₃ group is three bonds, plus three 1–3
/// distances when its H–X–H angles are rigid); one thread solves one cluster
/// by Gauss–Seidel sweeps in table order, so every cluster takes the CPU's
/// iterations and arrives at the CPU's answer.
///
/// Bindings and stages are the CPU twin's: `initial_integrate` @ -500
/// snapshots the reference bond vectors, `post_integrate` @ 0 corrects
/// positions along them (and the half-step velocities by Δx/dt),
/// `final_integrate` @ +200 removes the bond-direction relative velocities.
/// A cluster that does not converge within `max_iter` sweeps raises
/// `ForceFault::kConstraint` (the CPU twin throws): the run freezes at that
/// step and its retrieve throws. Graph-capturable.
class FixRattle : public cuda::Constraint,
                  public cuda::ConstraintProvider,
                  public cuda::TimestepConsumer {
  public:
    /// @brief Smallest relative tolerance the integration precision can
    ///        meet: 64 ulps of `mixed_t`.
    static constexpr double precision_tolerance_floor() noexcept {
        return 64.0
               * static_cast<double>(std::numeric_limits<mixed_t>::epsilon());
    }

    /// @brief 1e-10, raised to the precision floor when that is coarser.
    static constexpr double default_tolerance() noexcept {
        return precision_tolerance_floor() > 1.0e-10
                   ? precision_tolerance_floor()
                   : 1.0e-10;
    }

    /// @param dt Timestep (the position stage's velocity correction Δx/dt).
    /// @param tolerance Relative tolerance on |r² − d²|/d² and |r·v|/d².
    /// @param max_iter Sweeps per stage before the cluster faults.
    explicit FixRattle(double dt, double tolerance = default_tolerance(),
                       int max_iter = 500);
    ~FixRattle() override;
    FixRattle(FixRattle&&) noexcept;
    FixRattle& operator=(FixRattle&&) noexcept;

    /// @brief Constrain every bond of `bond_type` to `length` [Å].
    void constrain_bond(int bond_type, double length);
    /// @brief Constrain the 1–3 distance of every angle of `angle_type`.
    void constrain_angle(int angle_type, double distance_13);

    std::string name() const override { return "rattle"; }
    bool graph_capturable() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, -500),
                ATV::make_binding(ATV::md::Phase::post_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, +200)};
    }

    const cuda::ConstraintProvider* as_constraint() const override {
        return this;
    }
    BigInt n_constraints(const cpu::Frame& frame) const override;

    cuda::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return false; }
    void set_timestep(double dt) override;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

    /// @brief Number of clusters solved per stage (after `build()`).
    int n_clusters() const;

  private:
    struct Rule {
        int type = 0;
        double distance = 0.0;
    };
    double dt_;
    double tol_;
    int max_iter_;
    std::vector<Rule> bond_rules_;
    std::vector<Rule> angle_rules_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
