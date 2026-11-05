/// @file cuda/dynamics/fix_settle.cuh
/// @brief Analytic SETTLE for rigid 3-site water on the GPU (Miyamoto &
///        Kollman, J. Comput. Chem. 13, 952 (1992)); the device twin of
///        `cpu::md::FixSettle`, step for step.
#pragma once

#include "atomiverse/cuda/component/constraint.cuh"
#include "atomiverse/cuda/component/constraint_protocol.cuh"
#include "atomiverse/cuda/component/timestep_protocol.cuh"

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"

namespace ATV {
namespace cuda {
namespace md {

/// @brief SETTLE: analytic rigid-water constraint, one thread per water.
///
/// Same algorithm, bindings and failure semantics as `cpu::md::FixSettle`:
/// - `initial_integrate` @ -500 snapshots each water's constraint-satisfying
///   reference triangle (unwrapped about the oxygen) before any integrator
///   moves it;
/// - `post_integrate` @ 0 rotates the reference triangle onto the
///   unconstrained one's centre of mass (closed-form φ, ψ, θ);
/// - `final_integrate` @ +200 solves the three bond-direction impulses of
///   the velocity constraint exactly (one 3×3 system per water).
///
/// Where the CPU twin throws (oxygen out of plane, degenerate reference,
/// singular velocity system, a velocity residual not below 1e-8 of the
/// relative speed) the kernel raises `ForceFault::kConstraint` on the frame:
/// every state writer stops at that step and the run's retrieve throws. All
/// launches are graph-capturable.
///
/// Waters are named by stable tags. Single-GPU rows never move during a run,
/// so the rows are resolved once at `build()` from the device tag column.
class FixSettle : public cuda::Constraint,
                  public cuda::ConstraintProvider,
                  public cuda::TimestepConsumer {
  public:
    /// @param dt Timestep (accepted for the shared constructor shape; the
    ///        SETTLE stages do not use it).
    /// @param d_oh O–H distance [Å] (TIP3P default 0.9572).
    /// @param d_hh H–H distance [Å] (TIP3P default 1.5139).
    FixSettle(double dt, double d_oh = 0.9572, double d_hh = 1.5139);
    ~FixSettle() override;
    FixSettle(FixSettle&&) noexcept;
    FixSettle& operator=(FixSettle&&) noexcept;

    /// @brief Register one water by stable tags.
    void add_water(TagInt tag_O, TagInt tag_H1, TagInt tag_H2);

    std::string name() const override { return "settle"; }
    bool graph_capturable() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, -500),
                ATV::make_binding(ATV::md::Phase::post_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, +200)};
    }

    const cuda::ConstraintProvider* as_constraint() const override {
        return this;
    }
    BigInt n_constraints(const cpu::Frame& /*frame*/) const override {
        return 3 * static_cast<BigInt>(waters_.size());
    }

    cuda::TimestepConsumer* as_timestep_consumer() override { return this; }
    bool timestep_required() const override { return false; }
    void set_timestep(double dt) override;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

  private:
    struct Water {
        TagInt o, h1, h2;
    };
    double dt_;
    double d_oh_, d_hh_;
    std::vector<Water> waters_;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace md
} // namespace cuda
} // namespace ATV
