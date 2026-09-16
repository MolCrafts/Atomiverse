/// @file tests/mpi/test_eam_multirank.cpp
/// @brief Multi-rank EAM: the domain-decomposed forces must equal a brute-force
///        O(N^2) serial reference.
///
/// This is the test that closes the probe-1 comm wall. EAM's second pass needs
/// F'(rho_j) for a GHOST j, and rho_j is a sum over *j's* neighbours — so a ghost
/// shell of one cutoff leaves every ghost's density silently truncated. The component
/// declares a ghost-of-ghost halo (`halo_cutoff() = 2 * cutoff`) so it can complete
/// rho on ghosts locally, which buys multi-rank EAM without inventing a mid-force
/// communication point the engine does not have.
///
/// The reference is deliberately brute force: it uses no neighbor list, no halo, no
/// decomposition. If the ghost-of-ghost reasoning is wrong, the density on
/// near-boundary atoms is too small, their embedding derivative is too large in
/// magnitude, and the forces differ — there is nowhere for the error to hide.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/eam.h"

using namespace ATV;

namespace {

constexpr double kForceTolerance = 1.0e-9;

constexpr double kCutoff = 3.5;
constexpr double kA0 = 3.61; // fcc Cu

/// Distorted fcc copper — built identically on every rank.
cpu::Frame copper(int n_cell) {
    const int n = 4 * n_cell * n_cell * n_cell;
    cpu::Frame f(n);
    const double b[4][3] = {
        {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    int idx = 0;
    for (int ix = 0; ix < n_cell; ++ix)
        for (int iy = 0; iy < n_cell; ++iy)
            for (int iz = 0; iz < n_cell; ++iz)
                for (int k = 0; k < 4; ++k) {
                    const double j = 0.10 * std::sin(double(13 * idx + 3));
                    f.span(ATV::kAtom_x)[idx] =
                        (double(ix) + b[k][0]) * kA0 + j;
                    f.span(ATV::kAtom_y)[idx] =
                        (double(iy) + b[k][1]) * kA0 - 0.7 * j;
                    f.span(ATV::kAtom_z)[idx] =
                        (double(iz) + b[k][2]) * kA0 + 0.3 * j;
                    f.span(ATV::kAtom_mass)[idx] = 63.546;
                    f.span(ATV::kAtom_type)[idx] = 29;
                    ++idx;
                }
    f.box = ATV::cpu::Box::cubic(kA0 * n_cell);
    return f;
}

} // namespace

/// Decomposed EAM forces == brute-force serial EAM forces.
TEST(EAMMultiRank, ForcesMatchBruteForce) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The box must be big enough that the GHOST-OF-GHOST halo (2 * cutoff, plus the
    // engine's 10% pad = 7.7 A) still fits inside a subdomain. cutoff 3.5 A captures
    // Cu's first two neighbour shells (2.55, 3.61 A); n_cell 6 -> box 21.7 A.
    const int n_cell = 6; // 864 atoms
    cpu::Frame frame = copper(n_cell);
    const int n = frame.nlocal();

    // The MD driver, NOT Evaluate: only Dynamics installs the decomposition and the
    // halo, so only it actually exercises ghosts. dt = 0 keeps the geometry fixed, so
    // the forces are those of the frame we hand the brute-force reference.
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, 0.3, 1, 0, true});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.on(md::Phase::force)
        .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"},
                      kCutoff});
    sim.enable_timing(false);
    sim.build();
    sim.run(frame, 1);

    auto dd = sim.release_dd(); // public handle to the decomposition
    ASSERT_NE(dd, nullptr);
    const auto g = cpu::md::mpi::gather_to_root(frame, *dd);

    if (rank != 0) {
        SUCCEED();
        return;
    }
    ASSERT_EQ(static_cast<int>(g.fx.size()), n);

    // ── Brute-force reference: no neighbor list, no halo, no decomposition ──
    const cpu::EAMPotential pot{molrs::Element::Cu, "sutton-chen"};
    const double phi_shift = pot.phi(kCutoff);
    const cpu::Frame ref = copper(n_cell); // the same geometry, unpartitioned

    std::vector<double> rho(n, 0.0), dF(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double r_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double dx = ref.span(ATV::kAtom_x)[i] - ref.span(ATV::kAtom_x)[j];
            double dy = ref.span(ATV::kAtom_y)[i] - ref.span(ATV::kAtom_y)[j];
            double dz = ref.span(ATV::kAtom_z)[i] - ref.span(ATV::kAtom_z)[j];
            ref.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= kCutoff)
                continue;
            r_sum += pot.f(r);
        }
        rho[i] = r_sum;
        dF[i] = pot.dF(r_sum);
    }

    std::vector<double> fx(n, 0.0), fy(n, 0.0), fz(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double dx = ref.span(ATV::kAtom_x)[i] - ref.span(ATV::kAtom_x)[j];
            double dy = ref.span(ATV::kAtom_y)[i] - ref.span(ATV::kAtom_y)[j];
            double dz = ref.span(ATV::kAtom_z)[i] - ref.span(ATV::kAtom_z)[j];
            ref.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= kCutoff)
                continue;
            const double dE_dr = pot.dphi(r) + (dF[i] + dF[j]) * pot.df(r);
            const double fpair = -dE_dr / r;
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
        }
    }
    (void)phi_shift; // the shift is a constant: it moves E, never a force.

    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(g.fx[i] - fx[i]));
        worst = std::max(worst, std::fabs(g.fy[i] - fy[i]));
        worst = std::max(worst, std::fabs(g.fz[i] - fz[i]));
    }
    EXPECT_LT(worst, kForceTolerance)
        << "on " << size
        << " ranks, the decomposed EAM forces differ from the "
           "brute-force reference by "
        << worst << " eV/A — the ghost densities are truncated";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
