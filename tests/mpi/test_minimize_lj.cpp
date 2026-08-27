/// @file tests/mpi/test_minimize_lj.cpp
/// @brief Multi-rank geometry minimization: perturbed FCC LJ lattice relaxes
///        back to the perfect-lattice energy under domain decomposition.
///
/// Setup mirrors tests/mpi/test_lj_nve.cpp:
///   - FCC Argon-like system, 500 atoms in a 26.5 Å cubic box
///     (slab 13.25 Å at 8 ranks > 2 × 6.6 Å halo cutoff), σ=3.4 Å,
///     ε=0.0104 eV, r_cut=6.0 Å.
///   - Deterministic per-atom perturbation (global-index sinusoid, ±0.08 Å)
///     so the initial geometry — and hence the optimization problem — is
///     identical at every rank count.
///
/// Checks, per optimizer (LBFGS, CG):
///   - The optimizer converges within the iteration budget (driver early
///     stop, rank-uniform verdict).
///   - The minimized energy matches the perfect-lattice reference energy
///     computed through the same Minimize pipeline (so any halo / reduce /
///     owned-range error in the minimizer stack shows up as a mismatch).
///   - Energy strictly decreases from the perturbed starting point.
///
/// `Minimize::install_defaults` is exercised end-to-end: `md::mpi::Reduce`
/// on the force slot, halo / exchange on `ctx.md.comm`.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [Å]
constexpr double kCut = 6.0;    // [Å]
constexpr int kCells = 5;       // 5×5×5 FCC cells → 500 atoms
constexpr double kALat = 5.3;   // [Å]

cpu::Frame make_fcc_argon(double perturb_amp) {
    const int n = kCells * kCells * kCells * 4;
    cpu::Frame frame(n);
    const double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };
    // Quarter-lattice shift keeps FCC sites off the slab boundaries (see
    // test_lj_nve.cpp).
    const double shift = 0.25 * kALat;
    int idx = 0;
    for (int ix = 0; ix < kCells; ++ix)
        for (int iy = 0; iy < kCells; ++iy)
            for (int iz = 0; iz < kCells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    // Deterministic, rank-count-independent perturbation.
                    const double px = perturb_amp * std::sin(1.7 * idx + 0.3);
                    const double py = perturb_amp * std::cos(2.3 * idx + 1.1);
                    const double pz = perturb_amp * std::sin(3.1 * idx + 2.9);
                    frame.span(ATV::kAtom_x)[idx] =
                        (static_cast<double>(ix) + basis[b][0]) * kALat + shift
                        + px;
                    frame.span(ATV::kAtom_y)[idx] =
                        (static_cast<double>(iy) + basis[b][1]) * kALat + shift
                        + py;
                    frame.span(ATV::kAtom_z)[idx] =
                        (static_cast<double>(iz) + basis[b][2]) * kALat + shift
                        + pz;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.box = ATV::cpu::Box::cubic(kCells * kALat);
    return frame;
}

/// Perfect-lattice reference energy through the same Minimize pipeline
/// (zero initial forces → zero step at iteration 0 → force pass → reduced
/// global energy). One iteration is enough; the perfect lattice is already
/// at the minimum.
double reference_energy() {
    cpu::Frame frame = make_fcc_argon(/*perturb_amp=*/0.0);
    cpu::minimizer::Minimize opt;
    opt.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    opt.set_neighbor(cpu::md::NeighborList<>{kCut, 0.3});
    opt.set_optimizer(cpu::minimizer::LBFGS{});
    opt.build();
    opt.run(frame, 1);
    return frame.meta.get(ATV::kEnergy);
}

} // namespace

TEST(MinimizeLj, LbfgsRelaxesPerturbedLatticeAnyRankCount) {
    const double e_ref = reference_energy();
    ASSERT_LT(e_ref, 0.0); // bound LJ crystal

    cpu::Frame frame = make_fcc_argon(/*perturb_amp=*/0.08);

    cpu::minimizer::Minimize opt;
    opt.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    opt.set_neighbor(cpu::md::NeighborList<>{kCut, 0.3});
    cpu::minimizer::LBFGS lbfgs{}; // lvalue: post-run trace stays readable
    opt.set_optimizer_ref(lbfgs);
    opt.build();

    // Generous budget: the force criteria are met within ~50 iterations,
    // but the |dE| < 1e-5 eV criterion has a slow geometric tail (~0.98
    // per-step decay with the fixed-alpha step cap), crossing at ~400.
    const int budget = 800;
    const int iters = opt.run(frame, budget);

    ASSERT_FALSE(lbfgs.trace().empty());
    const double e_start = lbfgs.trace().front().energy;
    const double e_min = frame.meta.get(ATV::kEnergy);

    EXPECT_LT(iters, budget)
        << "LBFGS failed to converge in " << budget
        << " iterations (max_force=" << lbfgs.max_force() << ")";
    // The LOOP knows it converged, not the optimizer: LBFGS publishes metrics, a
    // ForceConverged handler judges them. `LBFGS::converged()` is gone.
    EXPECT_TRUE(opt.converged());
    EXPECT_LT(e_min, e_start) << "energy did not decrease";
    EXPECT_LT(std::abs(e_min - e_ref) / std::abs(e_ref), 1e-3)
        << "minimized energy " << e_min
        << " does not match perfect-lattice reference " << e_ref;
}

TEST(MinimizeLj, CgRelaxesPerturbedLatticeAnyRankCount) {
    const double e_ref = reference_energy();
    cpu::Frame frame = make_fcc_argon(/*perturb_amp=*/0.05);

    cpu::minimizer::Minimize opt;
    opt.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    opt.set_neighbor(cpu::md::NeighborList<>{kCut, 0.3});
    cpu::minimizer::CG cg{};
    opt.set_optimizer_ref(cg);
    opt.build();

    const int budget = 1000;
    const int iters = opt.run(frame, budget);

    ASSERT_FALSE(cg.trace().empty());
    const double e_start = cg.trace().front().energy;
    const double e_min = frame.meta.get(ATV::kEnergy);

    EXPECT_LT(iters, budget)
        << "CG failed to converge in " << budget
        << " iterations (max_force=" << cg.max_force() << ")";
    EXPECT_TRUE(opt.converged());
    EXPECT_LT(e_min, e_start) << "energy did not decrease";
    EXPECT_LT(std::abs(e_min - e_ref) / std::abs(e_ref), 1e-3)
        << "minimized energy " << e_min
        << " does not match perfect-lattice reference " << e_ref;
}

TEST(MinimizeLj, FireRelaxesPerturbedLatticeAnyRankCount) {
    const double e_ref = reference_energy();
    cpu::Frame frame = make_fcc_argon(/*perturb_amp=*/0.05);

    cpu::minimizer::Minimize opt;
    opt.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    opt.set_neighbor(cpu::md::NeighborList<>{kCut, 0.3});
    cpu::minimizer::FIRE fire{};
    // Soft LJ crystal needs a larger pseudo-time scale than ASE metal defaults.
    fire.dt0 = 1.0;
    fire.dt_max = 20.0;
    opt.set_optimizer_ref(fire);
    opt.build();

    const int budget = 2000;
    const int iters = opt.run(frame, budget);

    ASSERT_FALSE(fire.trace().empty());
    const double e_start = fire.trace().front().energy;
    const double e_min = frame.meta.get(ATV::kEnergy);

    EXPECT_LT(iters, budget)
        << "FIRE failed to converge in " << budget
        << " iterations (max_force=" << fire.max_force() << ")";
    EXPECT_TRUE(opt.converged());
    EXPECT_LT(e_min, e_start) << "energy did not decrease";
    EXPECT_LT(std::abs(e_min - e_ref) / std::abs(e_ref), 1e-3)
        << "minimized energy " << e_min
        << " does not match perfect-lattice reference " << e_ref;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
