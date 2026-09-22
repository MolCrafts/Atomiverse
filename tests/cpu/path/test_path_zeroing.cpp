/// @file tests/cpu/path/test_path_zeroing.cpp
/// @brief Unit: `cpu::PathDriver` owns the per-replica-pass zeroing of the
///        force / energy accumulator channels.
///
/// Mirrors `include/atomiverse/cpu/path/path_driver.h` (`ATV::cpu::PathDriver`).
/// Spec: `.claude/specs/esmm-contract-01-zeroing` — ac-002.
///
/// `PathDriver::run()` replays `path::Phase::force` once per replica, per band
/// iteration, against that replica's own persistent Frame. Nothing in the loop
/// resets `kAtom_f*` between iterations: `NebCoupling` zeroes only the frozen
/// endpoints, and the NEB regression works today only because its local
/// Müller-Brown stub OVERWRITES (`f[0] = ...`) instead of accumulating.
///
/// With an accumulating (`+=`) source — which is what every in-tree evaluator
/// is — a band held at fixed geometry must still see the SAME force on every
/// iteration. Today iteration 2 sees iteration 1's force plus its own, i.e. the
/// force grows linearly with the iteration count.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/role.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/path_driver.h"
#include "atomiverse/cpu/replica/replica_expansion.h"

namespace {

constexpr double kSpring = 1.5;  ///< Spring constant [eV/Angstrom^2].
constexpr double kCentre = -1.0; ///< Harmonic centre x0 [Angstrom]; off the
                                 ///< band so every image feels a nonzero force.
constexpr int kImages = 3;       ///< Band size (the expansion's minimum).
constexpr int kIterations = 2;   ///< Band iterations to run.

/// @brief Image x-coordinates produced by `LinearInterpolation` for a band from
///        x = 0 to x = 0.6 with three images [Angstrom].
constexpr double kImageX[kImages] = {0.0, 0.3, 0.6};

/// @brief 1-D harmonic stub written the way every real evaluator writes:
///        `+=` into the accumulator channels.
///
/// `F_x = -k (x - x0)`, no y/z contribution.
class AccumulatingHarmonicX : public ATV::cpu::Potential {
  public:
    /// @brief Construct with a spring constant and centre.
    /// @param k Spring constant [eV/Angstrom^2].
    /// @param x0 Harmonic centre [Angstrom].
    AccumulatingHarmonicX(double k, double x0) : k_(k), x0_(x0) {}

    std::string name() const override { return "accumulating_harmonic_x"; }

    void launch(const ATV::cpu::LaunchContext& ctx) override {
        auto& f = ctx.frame;
        const int n = f.nlocal();
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            const double dx = f.span(ATV::kAtom_x)[i] - x0_;
            e += 0.5 * k_ * dx * dx;
            f.span(ATV::kAtom_fx)[i] += -k_ * dx;
        }
        f.meta.get(ATV::kEnergy) += e;
    }

  private:
    double k_;  ///< Spring constant [eV/Angstrom^2].
    double x0_; ///< Harmonic centre [Angstrom].
};

/// @brief Records `fx` of atom 0 at the tail of every per-replica force pass.
///
/// One entry per `(iteration, replica)` pass, in execution order — the band's
/// force history as the driver actually produced it.
class ForceProbe : public ATV::cpu::Component {
  public:
    std::vector<double> fx; ///< Recorded `kAtom_fx[0]` [eV/Angstrom].

    std::string name() const override { return "force_probe"; }
    ATV::Role role() const override {
        return static_cast<ATV::Role>(ATV::RoleFlag::FrameImmutable);
    }
    void launch(const ATV::cpu::LaunchContext& ctx) override {
        fx.push_back(ctx.frame.span(ATV::kAtom_fx)[0]);
    }
};

/// @brief Optimizer that moves nothing.
///
/// The band geometry must be bit-identical between the two iterations, so that
/// any change in the measured force is the accumulator, not the physics.
class FrozenOptimizer : public ATV::cpu::Optimizer {
  public:
    std::string name() const override { return "frozen_optimizer"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// @brief One atom on the x-axis.
/// @param x Position along x [Angstrom].
/// @return A single-atom Frame in a box large enough never to bind.
ATV::cpu::Frame particle_at(double x) {
    ATV::cpu::Frame f(1);
    f.span(ATV::kAtom_x)[0] = x;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 1.0;
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_tag)[0] = 1;
    f.box = ATV::cpu::Box::cubic(50.0);
    return f;
}

} // namespace

/// ac-002 — two band iterations at frozen geometry must report identical
/// replica forces. Today the second iteration reports double the first.
TEST(PathZeroing, SecondBandIterationRepeatsTheFirstForceAtFixedGeometry) {
    ATV::cpu::Frame initial = particle_at(kImageX[0]);
    ATV::cpu::Frame product = particle_at(kImageX[kImages - 1]);
    ATV::cpu::LinearInterpolation expand{product, kImages};

    ForceProbe probe;
    ATV::cpu::PathDriver band;
    band.on(ATV::path::Phase::force)
        .add(AccumulatingHarmonicX{kSpring, kCentre}, /*order=*/0);
    band.on(ATV::path::Phase::force).add_ref(probe, /*order=*/1000);
    band.on(ATV::path::Phase::search_direction)
        .add(FrozenOptimizer{}, /*order=*/0);
    band.set_threads(1);
    band.enable_timing(false);
    band.build();

    const int iters = band.run(initial, expand, kIterations);

    ASSERT_EQ(iters, kIterations)
        << "no criterion should have stopped an unconverged band";
    ASSERT_EQ(probe.fx.size(), static_cast<std::size_t>(kIterations * kImages));

    // The geometry never moved, so iteration 1's force IS iteration 0's force.
    for (int i = 0; i < kImages; ++i) {
        SCOPED_TRACE("replica " + std::to_string(i));
        EXPECT_NEAR(probe.fx[kImages + i], probe.fx[i], 1e-12)
            << "iteration 1 fx = " << probe.fx[kImages + i]
            << " vs iteration 0 fx = " << probe.fx[i]
            << "; a ratio of 2 means PathDriver never zeroed kAtom_f* before "
               "the per-replica force pass.";
    }

    // ... and that shared value is the single-pass analytic force, not a sum.
    for (int i = 0; i < kImages; ++i) {
        SCOPED_TRACE("replica " + std::to_string(i));
        const double f_ref = -kSpring * (kImageX[i] - kCentre);
        EXPECT_NEAR(probe.fx[i], f_ref, 1e-12);
        EXPECT_NEAR(probe.fx[kImages + i], f_ref, 1e-12);
    }
}
