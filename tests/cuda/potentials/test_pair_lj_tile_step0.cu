/// @file tests/cuda/potentials/test_pair_lj_tile_step0.cu
/// @brief Step-0 LJ energy on OpenMM tile force path (cuda-openmm-md-03).

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

/// Two Ar-like LJ atoms: r = 2^(1/6) σ ⇒ attractive well minimum for unshifted LJ.
/// With epsilon=1, sigma=1: E_pair = -1 at r_min (before shift).
/// PairLJCut applies energy shift so U(rc)=0; with rc large enough shift ≈ 0.
double lj_pair_shifted(double r, double eps, double sig, double rc) {
    auto ulj = [&](double rr) {
        double sr = sig / rr;
        double sr6 = sr * sr * sr * sr * sr * sr;
        double sr12 = sr6 * sr6;
        return 4.0 * eps * (sr12 - sr6);
    };
    return ulj(r) - ulj(rc);
}

} // namespace

TEST(PairLjTileStep0, TwoAtomEnergyMatchesOracle) {
    const double eps = 1.0;
    const double sig = 1.0;
    const double rc = 3.0;
    // r_min for LJ 12-6
    const double r = std::pow(2.0, 1.0 / 6.0);
    const double e_ref = lj_pair_shifted(r, eps, sig, rc);

    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = r;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_mass)[0] = 1.0;
    frame.span(kAtom_mass)[1] = 1.0;
    frame.span(kAtom_type)[0] = 0;
    frame.span(kAtom_type)[1] = 0;
    frame.box = cpu::Box::cubic(20.0);

    cuda::md::MD sim;
    // dt = 0: one frozen step is a force/energy pass. `MD` requires an
    // Integrator, so the pass declares the one that does nothing rather
    // than leaving the slot empty.
    sim.set_integrator(cuda::md::VelocityVerlet{0.0});
    sim.set_neighbor(cuda::NeighborList{rc, 0.5});
    sim.set_evaluator(cuda::PairLJCut{eps, sig, rc});
    sim.build();
    sim.run(frame, 1);

    double e = static_cast<double>(frame.meta.get(kEnergy));
    ASSERT_TRUE(std::isfinite(e));
    // float work path + FULL 0.5 energy; tol loose for single+double mix
    EXPECT_NEAR(e, e_ref, 1e-3) << "e=" << e << " e_ref=" << e_ref;
}
