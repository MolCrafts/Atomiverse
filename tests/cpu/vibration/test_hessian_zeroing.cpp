/// @file tests/cpu/vibration/test_hessian_zeroing.cpp
/// @brief Unit: `cpu::vib::Hessian` owns the per-displacement zeroing of the
///        force / energy accumulator channels.
///
/// Mirrors `src/cpu/vibration/hessian.cpp` (`ATV::cpu::vib::Hessian`).
/// Spec: `.claude/specs/esmm-contract-01-zeroing` — ac-001.
///
/// `Hessian::run()` replays `vib::Phase::force` `6N` times against ONE Frame.
/// Force sources in this tree ACCUMULATE (`f[i] += ...`, `kEnergy += ...`), so
/// unless the hosting driver zeroes `kAtom_f*` before each displacement pass the
/// evaluator writes into a running sum. Central-differencing a running sum of a
/// linear restoring force yields exactly HALF the true curvature — the "k or
/// k/2 depending on FD assembly convention" hedge fossilized in
/// `regressions/architecture/harmonic_driver_matrix` is that bug, not a
/// convention.
///
/// `vib::Phonon` already binds `ZeroForces` @ -1000 / `ZeroEnergy` @ -999 on its
/// force phase; `vib::Hessian` binds neither. These tests pin the fix on BOTH
/// construction paths (the full chain and the `WithoutAccumulator` ctor).

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vibration/hessian.h"
#include "atomiverse/cpu/vibration/hessian_accumulator.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"

#include "cpu/driver_test_helpers.h"

namespace {

/// @brief Force spring constant used by the harmonic stub [eV/Angstrom^2].
///
/// Deliberately not 1.0: `k/2` (the accumulating artifact) and `k` are then
/// distinct from each other AND from any accidental unit value.
constexpr double kSpring = 2.5;

/// @brief Isotropic harmonic stub centred on the origin, written the way every
///        real evaluator writes: `+=` into the accumulator channels.
///
/// `U = 1/2 k |r|^2`, `F = -k r`. A sole source, so with correct per-pass
/// zeroing the assembled Cartesian Hessian is exactly `k * I` (the force is
/// linear, so the central difference carries no truncation error).
class AccumulatingHarmonic : public ATV::cpu::Potential {
  public:
    /// @brief Construct with a spring constant.
    /// @param k Spring constant [eV/Angstrom^2].
    explicit AccumulatingHarmonic(double k) : k_(k) {}

    std::string name() const override { return "accumulating_harmonic"; }

    void launch(const ATV::cpu::LaunchContext& ctx) override {
        auto& f = ctx.frame;
        const int n = f.nlocal();
        double e = 0.0;
        for (int i = 0; i < n; ++i) {
            const double dx = f.span(ATV::kAtom_x)[i];
            const double dy = f.span(ATV::kAtom_y)[i];
            const double dz = f.span(ATV::kAtom_z)[i];
            e += 0.5 * k_ * (dx * dx + dy * dy + dz * dz);
            f.span(ATV::kAtom_fx)[i] += -k_ * dx;
            f.span(ATV::kAtom_fy)[i] += -k_ * dy;
            f.span(ATV::kAtom_fz)[i] += -k_ * dz;
        }
        f.meta.get(ATV::kEnergy) += e;
    }

  private:
    double k_; ///< Spring constant [eV/Angstrom^2].
};

/// @brief One atom sitting exactly at the harmonic minimum (the origin).
/// @return A single-atom Frame in a box large enough never to bind.
ATV::cpu::Frame atom_at_origin() {
    ATV::cpu::Frame f(1);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 12.0;
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_tag)[0] = 1;
    f.box = ATV::cpu::Box::cubic(50.0);
    return f;
}

/// @brief Component-name sequence of one compiled Plan phase slot.
/// @param plan Compiled plan to read.
/// @param phase Slot index (the phase enum's integer value).
/// @return Names in execution order.
std::vector<std::string> slot_names(const ATV::Plan<ATV::cpu::Component>& plan,
                                    std::size_t phase) {
    std::vector<std::string> names;
    for (const auto& op : plan.slots[phase].ops)
        names.push_back(op.component->name());
    return names;
}

} // namespace

/// ac-001 — the behaviour. An accumulating sole source under `vib::Hessian`
/// must produce the analytic curvature `k`, not the running-sum artifact `k/2`.
TEST(HessianZeroing, AccumulatingSourceYieldsTheAnalyticCurvature) {
    ATV::cpu::Frame frame = atom_at_origin();

    const std::string tmp = ::testing::TempDir();
    ATV::cpu::vib::Hessian hess(
        /*temperature=*/298.15, /*pressure=*/101325.0,
        /*multiplicity=*/1, /*symmetry_number=*/1,
        /*frequency_tsv=*/tmp + "atv_test_hessian_zeroing_freq.tsv",
        /*mode_xyz_prefix=*/tmp + "atv_test_hessian_zeroing_mode");
    hess.on(ATV::vib::Phase::force).add(AccumulatingHarmonic{kSpring});
    hess.set_threads(1);
    hess.enable_timing(false);
    hess.build();

    const ATV::cpu::vib::HessianResult res = hess.run(frame);

    ASSERT_EQ(res.n_atoms, 1);
    ASSERT_EQ(static_cast<int>(res.hessian.shape(0)), 3);
    ASSERT_EQ(static_cast<int>(res.hessian.shape(1)), 3);

    // Relative 1e-8 on the diagonal: a linear force differences exactly, so the
    // only error left is floating-point.
    const double tol = 1e-8 * kSpring;
    for (int c = 0; c < 3; ++c) {
        SCOPED_TRACE("Cartesian coordinate " + std::to_string(c));
        EXPECT_NEAR(res.hessian(c, c), kSpring, tol)
            << "H(" << c << "," << c << ") = " << res.hessian(c, c)
            << "; the analytic curvature is k = " << kSpring
            << ". Half of k means the driver let the accumulating evaluator "
               "sum forces across the 6N displacement passes.";
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j)
                EXPECT_NEAR(res.hessian(i, j), 0.0, tol)
                    << "off-diagonal H(" << i << "," << j << ")";
}

/// ac-001 — the plan shape, on BOTH construction paths. Zeroing is a driver
/// binding, so it must be there however the driver was built; the test-only
/// `WithoutAccumulator` ctor is a second wiring site that can silently drift.
TEST(HessianZeroing, BothConstructorsBindZeroForcesAndZeroEnergyOnForcePhase) {
    const std::vector<std::string> expected{"zero_forces", "zero_energy",
                                            "stub_force"};
    const auto force_slot = static_cast<std::size_t>(ATV::vib::Phase::force);

    { // full chain (the production ctor)
        ATV::cpu::vib::Hessian hess;
        atv_test::StubForceEvaluator force;
        hess.on(ATV::vib::Phase::force).add_ref(force);
        hess.build();
        EXPECT_EQ(slot_names(hess.plan(), force_slot), expected)
            << "Hessian() must bind ZeroForces @ -1000 and ZeroEnergy @ -999 "
               "before the user force source";
    }

    { // WithoutAccumulator ctor — a caller-supplied accumulator gets it past
        // validate_components() so the compiled Plan is observable.
        ATV::cpu::vib::HessianWorkspace ws;
        ATV::cpu::vib::Hessian hess{
            ATV::cpu::vib::Hessian::WithoutAccumulator{}};
        hess.on(ATV::vib::Phase::accumulate)
            .add(ATV::cpu::vib::HessianAccumulator{&ws}, 0);
        atv_test::StubForceEvaluator force;
        hess.on(ATV::vib::Phase::force).add_ref(force);
        hess.build();
        EXPECT_EQ(slot_names(hess.plan(), force_slot), expected)
            << "Hessian(WithoutAccumulator) must carry the same Zero bindings "
               "as the full chain";
    }
}
