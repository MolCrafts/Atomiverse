/// @file tests/cpu/path/test_replica.cpp
/// @brief Unit: ReplicaExpansion strategies + ReplicaSet's collective termination.
///
/// The two things probe #3 had to get right before NEB was worth writing:
///   1. replicas come from a STRATEGY applied to one Frame — never from the user handing
///      over N frames, which would fight the facade (`run()` takes one Frame) and would
///      not match how any of these methods actually work (nobody gives REMD sixteen
///      structures; they give it one, plus a temperature ladder);
///   2. a chain is converged only when EVERY replica is.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/replica/replica_expansion.h"
#include "atomiverse/cpu/replica/replica_set.h"

using namespace ATV;

namespace {

cpu::Frame two_atoms(double x0, double x1) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = x0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = x1;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_mass)[0] = f.span(ATV::kAtom_mass)[1] = 1.0;
    f.span(ATV::kAtom_type)[0] = f.span(ATV::kAtom_type)[1] = 1;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

} // namespace

/// Linear interpolation puts image k at a fraction k/(N-1) of the way to the product.
TEST(ReplicaExpansion, LinearInterpolationPlacesImagesAlongTheChord) {
    cpu::Frame initial = two_atoms(0.0, 1.0);
    cpu::Frame final_state = two_atoms(0.0, 5.0); // atom 1 travels 4 A

    cpu::LinearInterpolation expand{final_state, 5};
    const auto reps = expand.expand(initial);

    ASSERT_EQ(reps.size(), 5u);
    for (int k = 0; k < 5; ++k) {
        const double t = double(k) / 4.0;
        EXPECT_NEAR(reps[k].frame.span(ATV::kAtom_x)[1], 1.0 + t * 4.0, 1e-12);
        EXPECT_NEAR(reps[k].frame.span(ATV::kAtom_x)[0], 0.0, 1e-12)
            << "the fixed atom moved";
        // The replica's parameter IS its position along the path.
        EXPECT_NEAR(reps[k].parameter, t, 1e-12);
    }
    // The endpoints are exactly the endpoints — not an interpolation of them.
    EXPECT_NEAR(reps[0].frame.span(ATV::kAtom_x)[1], 1.0, 1e-14);
    EXPECT_NEAR(reps[4].frame.span(ATV::kAtom_x)[1], 5.0, 1e-14);
}

/// A band of two images has nothing between the endpoints to relax.
TEST(ReplicaExpansion, LinearInterpolationRefusesFewerThanThreeImages) {
    cpu::Frame f = two_atoms(0.0, 1.0);
    EXPECT_THROW(cpu::LinearInterpolation(f, 2), std::invalid_argument);
}

/// A band between different systems is not a band.
TEST(ReplicaExpansion, LinearInterpolationRefusesAnAtomCountMismatch) {
    cpu::Frame initial = two_atoms(0.0, 1.0);
    cpu::Frame final_state(3);
    final_state.box = ATV::cpu::Box::cubic(20.0);
    cpu::LinearInterpolation expand{final_state, 4};
    EXPECT_THROW(expand.expand(initial), std::invalid_argument);
}

/// IdentityReplication is the PIMD/REMD shape: N copies, each with its own parameter.
/// It ships with no in-tree user — it is the evidence that ReplicaExpansion is a real
/// strategy and not an NEB constructor in disguise.
TEST(ReplicaExpansion, IdentityReplicationCopiesAndTags) {
    cpu::Frame initial = two_atoms(0.0, 1.0);
    cpu::IdentityReplication expand{
        {300.0, 350.0, 400.0, 450.0}}; // a temperature ladder

    const auto reps = expand.expand(initial);
    ASSERT_EQ(reps.size(), 4u);
    EXPECT_EQ(expand.n_replicas(), 4);
    for (const auto& r : reps) {
        EXPECT_NEAR(r.frame.span(ATV::kAtom_x)[1], 1.0, 1e-14)
            << "identity replication changed the "
               "geometry";
    }
    EXPECT_NEAR(reps[0].parameter, 300.0, 1e-12);
    EXPECT_NEAR(reps[3].parameter, 450.0, 1e-12);
}

/// The parameter list's length IS the replica count — you cannot ask for 8 replicas and
/// hand over 7 temperatures, because there is no separate count to disagree with.
TEST(ReplicaExpansion, IdentityReplicationRefusesFewerThanTwo) {
    EXPECT_THROW(cpu::IdentityReplication(std::vector<double>{300.0}),
                 std::invalid_argument);
}

/// Frame in, Frame out: collapse selects the highest-energy image (the saddle).
TEST(ReplicaExpansion, LinearInterpolationCollapsesToTheHighestEnergyImage) {
    cpu::Frame initial = two_atoms(0.0, 1.0);
    cpu::Frame final_state = two_atoms(0.0, 5.0);
    cpu::LinearInterpolation expand{final_state, 5};

    auto reps = expand.expand(initial);
    reps[0].frame.meta.get(ATV::kEnergy) = -1.0;
    reps[1].frame.meta.get(ATV::kEnergy) = 0.5;
    reps[2].frame.meta.get(ATV::kEnergy) = 2.5; // the barrier top
    reps[3].frame.meta.get(ATV::kEnergy) = 0.7;
    reps[4].frame.meta.get(ATV::kEnergy) = -1.2;

    cpu::Frame out = two_atoms(0.0, 0.0);
    expand.collapse(reps, out);

    EXPECT_NEAR(out.span(ATV::kAtom_x)[1], reps[2].frame.span(ATV::kAtom_x)[1],
                1e-14);
    EXPECT_NEAR(out.meta.get(ATV::kEnergy), 2.5, 1e-14);
}

/// THE termination rule of a coupled band: AND, not OR.
///
/// The band publishes `converged_fraction()` onto ConvergenceMetrics and
/// `cpu::AllReplicasConverged` applies the AND — the band is converged at exactly 1.0.
/// If a component could write convergence straight onto a single shared status, the
/// first image to settle would have stopped the entire chain, leaving the rest of the
/// band wherever it happened to be.
TEST(ReplicaSetTest, OneUnconvergedImageKeepsTheChainRunning) {
    cpu::Frame initial = two_atoms(0.0, 1.0);
    cpu::Frame final_state = two_atoms(0.0, 5.0);
    cpu::LinearInterpolation expand{final_state, 5};
    cpu::ReplicaSet band{initial, expand};

    ASSERT_EQ(band.size(), 5);
    EXPECT_LT(band.converged_fraction(), 1.0)
        << "a fresh band cannot already be converged";

    for (int i = 0; i < 5; ++i)
        band.set_converged(i, true);
    EXPECT_GE(band.converged_fraction(), 1.0);

    // Image 3 backslides — the chain is NOT converged, however happy image 0 is.
    band.set_converged(3, false);
    EXPECT_LT(band.converged_fraction(), 1.0);

    band.set_converged(3, true);
    EXPECT_GE(band.converged_fraction(), 1.0);
}
