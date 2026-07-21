/// @file tests/mpi/test_sw_multirank.cpp
/// @brief Multi-rank Stillinger-Weber: decomposed 3-body forces vs a brute-force
///        O(N^3) serial reference.
///
/// SW's multi-rank story is DIFFERENT from EAM's, and the difference is the claim under
/// test here. EAM needs a ghost-of-ghost halo (2 * cutoff) because a ghost's embedding
/// density is a sum over that ghost's OWN neighbours. SW needs no such thing: every
/// triplet is enumerated at its VERTEX, a triplet whose vertex is a ghost belongs to
/// the rank that owns that vertex, and one cutoff of ghosts supplies every j and k an
/// owned vertex can reach. Forces landing on ghost j/k ride home on the existing
/// reverse halo.
///
/// That reasoning was asserted in code and in a commit message before it was ever run.
/// This test runs it.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/gather.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/sw.h"

using namespace ATV;

namespace {

constexpr double kForceTolerance = 1.0e-9;

constexpr double kA0 = 5.431; // diamond Si

/// Distorted diamond silicon — built identically on every rank.
cpu::Frame silicon(int n_cell) {
    const double b[8][3] = {
        {0, 0, 0},     {0.25, 0.25, 0.25}, {0.5, 0.5, 0}, {0.75, 0.75, 0.25},
        {0.5, 0, 0.5}, {0.75, 0.25, 0.75}, {0, 0.5, 0.5}, {0.25, 0.75, 0.75}};
    const int n = 8 * n_cell * n_cell * n_cell;
    cpu::Frame f(n);
    int idx = 0;
    for (int ix = 0; ix < n_cell; ++ix)
        for (int iy = 0; iy < n_cell; ++iy)
            for (int iz = 0; iz < n_cell; ++iz)
                for (int k = 0; k < 8; ++k) {
                    const double j = 0.10 * std::sin(double(11 * idx + 5));
                    f.span(ATV::kAtom_x)[idx] =
                        (double(ix) + b[k][0]) * kA0 + j;
                    f.span(ATV::kAtom_y)[idx] =
                        (double(iy) + b[k][1]) * kA0 - 0.6 * j;
                    f.span(ATV::kAtom_z)[idx] =
                        (double(iz) + b[k][2]) * kA0 + 0.4 * j;
                    f.span(ATV::kAtom_mass)[idx] = 28.0855;
                    f.span(ATV::kAtom_type)[idx] = 14;
                    ++idx;
                }
    f.simbox = ATV::cpu::Box::cubic(kA0 * n_cell);
    return f;
}

} // namespace

TEST(SWMultiRank, ForcesMatchBruteForce) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const cpu::SWPotential pot{molrs::Element::Si, "stillinger-weber"};
    const double rc = pot.cutoff(); // 3.771 A
    const int n_cell = 3;           // 216 atoms, box 16.3 A
    cpu::Frame frame = silicon(n_cell);
    const int n = frame.nlocal();

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{rc, 0.3, 1, 0, true});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0}); // dt = 0: geometry frozen
    sim.on(md::Phase::force).add(cpu::StillingerWeber{pot});
    sim.enable_timing(false);
    sim.build();
    sim.run(frame, 1);

    auto dd = sim.release_dd();
    ASSERT_NE(dd, nullptr);
    const auto g = cpu::md::mpi::gather_to_root(frame, *dd);

    if (rank != 0) {
        SUCCEED();
        return;
    }
    ASSERT_EQ(static_cast<int>(g.fx.size()), n);

    // ── Brute force: no neighbor list, no halo, no decomposition ────────
    const cpu::Frame ref = silicon(n_cell);
    const double rc2 = rc * rc;
    const double cos0 = pot.cos_theta0();
    const double le = pot.lambda_eps();
    std::vector<double> fx(n, 0.0), fy(n, 0.0), fz(n, 0.0);

    const auto delta = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx = ref.span(ATV::kAtom_x)[i] - ref.span(ATV::kAtom_x)[j];
        dy = ref.span(ATV::kAtom_y)[i] - ref.span(ATV::kAtom_y)[j];
        dz = ref.span(ATV::kAtom_z)[i] - ref.span(ATV::kAtom_z)[j];
        ref.simbox.min_image(dx, dy, dz);
    };

    for (int i = 0; i < n; ++i) {
        // two-body
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            double dx, dy, dz;
            delta(i, j, dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2)
                continue;
            const double r = std::sqrt(r2);
            const double fpair = -pot.dphi2(r) / r;
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
        }
        // three-body, vertex at i
        for (int j = 0; j < n; ++j) {
            if (j == i)
                continue;
            double ax, ay, az;
            delta(i, j, ax, ay, az);
            const double r1sq = ax * ax + ay * ay + az * az;
            if (r1sq >= rc2)
                continue;
            const double r1 = std::sqrt(r1sq);
            for (int k = j + 1; k < n; ++k) {
                if (k == i)
                    continue;
                double bx, by, bz;
                delta(i, k, bx, by, bz);
                const double r2sq = bx * bx + by * by + bz * bz;
                if (r2sq >= rc2)
                    continue;
                const double r2 = std::sqrt(r2sq);

                const double g1 = pot.g(r1), g2 = pot.g(r2);
                const double dg1 = pot.dg(r1), dg2 = pot.dg(r2);
                const double cs = (ax * bx + ay * by + az * bz) / (r1 * r2);
                const double dl = cs - cos0;

                const double dcos = 2.0 * le * dl * g1 * g2;
                const double dE_dr1 = le * dl * dl * dg1 * g2;
                const double dE_dr2 = le * dl * dl * g1 * dg2;
                const double inv = 1.0 / (r1 * r2);

                const double fjx =
                    dE_dr1 * (ax / r1) + dcos * (bx * inv - cs * ax / r1sq);
                const double fjy =
                    dE_dr1 * (ay / r1) + dcos * (by * inv - cs * ay / r1sq);
                const double fjz =
                    dE_dr1 * (az / r1) + dcos * (bz * inv - cs * az / r1sq);
                const double fkx =
                    dE_dr2 * (bx / r2) + dcos * (ax * inv - cs * bx / r2sq);
                const double fky =
                    dE_dr2 * (by / r2) + dcos * (ay * inv - cs * by / r2sq);
                const double fkz =
                    dE_dr2 * (bz / r2) + dcos * (az * inv - cs * bz / r2sq);

                fx[j] += fjx;
                fy[j] += fjy;
                fz[j] += fjz;
                fx[k] += fkx;
                fy[k] += fky;
                fz[k] += fkz;
                fx[i] -= fjx + fkx;
                fy[i] -= fjy + fky;
                fz[i] -= fjz + fkz;
            }
        }
    }

    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        worst = std::max(worst, std::fabs(g.fx[i] - fx[i]));
        worst = std::max(worst, std::fabs(g.fy[i] - fy[i]));
        worst = std::max(worst, std::fabs(g.fz[i] - fz[i]));
    }
    EXPECT_LT(worst, kForceTolerance)
        << "on " << size
        << " ranks, the decomposed SW forces differ from brute force by "
        << worst
        << " eV/A — a triplet is being lost or double-counted at a rank "
           "boundary";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
