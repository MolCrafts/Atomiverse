/// @file tests/mpi/test_sw_multirank.cpp
/// @brief Multi-rank Stillinger-Weber: decomposed 3-body forces vs a brute-force
///        O(N^3) serial reference.
///
/// Every SW triplet is enumerated at its owned vertex, with ghost j/k
/// forces returned through the reverse halo. The mixed case adds LJ through
/// the half list derived from SW's full list and compares the summed energy
/// and forces against brute force. Kernels are invoked directly, without MD.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/pbc.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
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
    f.box = ATV::cpu::Box::cubic(kA0 * n_cell);
    return f;
}

} // namespace

class SWMultiRank : public ::testing::TestWithParam<bool> {};

TEST_P(SWMultiRank, ForcesAndEnergyMatchBruteForce) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const cpu::SWPotential pot{molrs::Element::Si, "stillinger-weber"};
    const double rc = pot.cutoff(); // 3.771 A
    const int n_cell = 3;           // 216 atoms, box 16.3 A
    cpu::Frame frame = silicon(n_cell);
    const int n = frame.nlocal();

    const bool mixed_lj = GetParam();
    constexpr double epsilon = 0.1, sigma = 2.0;
    cpu::Context context;
    context.mpi_rank = rank;
    context.mpi_size = size;
    cpu::communication::DomainRuntime domain;
    domain.allocate(context);
    cpu::md::wrap_owned(frame, n);
    domain.bind_frame(frame, context);
    cpu::md::NeighborList<> neighbors{rc, 0.3, 1, 0, true};
    neighbors.configure_lists(mixed_lj, true);
    domain.comm()->set_ghost_cutoff(neighbors.neighbor_reach());
    const cpu::LaunchContext launch{context, frame, 0, 0, true, true};
    domain.comm()->setup(launch);
    domain.comm()->borders(launch);
    neighbors.build({context, frame});
    neighbors.launch(launch);
    cpu::StillingerWeber sw{pot};
    sw.build({context, frame});
    sw.launch(launch);
    if (mixed_lj) {
        cpu::PairLJCut lj{epsilon, sigma, rc};
        lj.build({context, frame});
        lj.launch(launch);
        lj.destroy({context, frame});
    }
    domain.comm()->reverse_comm(launch);
    const auto g = cpu::md::mpi::gather_to_root(frame, *domain.dd());
    const double local_energy = frame.meta.get(kEnergy);
    double total_energy = 0.0;
    MPI_Allreduce(&local_energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    double total_stress[6] = {};
    MPI_Allreduce(frame.meta.get(kStress).data(), total_stress, 6, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    sw.destroy({context, frame});
    neighbors.destroy({context, frame});
    domain.release_comm(context);

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
    double energy = 0.0;
    double stress[6] = {};
    const double cutoff_s6 = std::pow(sigma / rc, 6);
    const double lj_shift = 4.0 * epsilon * (cutoff_s6 * cutoff_s6 - cutoff_s6);

    const auto delta = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx = ref.span(ATV::kAtom_x)[i] - ref.span(ATV::kAtom_x)[j];
        dy = ref.span(ATV::kAtom_y)[i] - ref.span(ATV::kAtom_y)[j];
        dz = ref.span(ATV::kAtom_z)[i] - ref.span(ATV::kAtom_z)[j];
        ref.box.min_image(dx, dy, dz);
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
            double fpair = -pot.dphi2(r) / r;
            energy += 0.5 * pot.phi2(r);
            if (mixed_lj) {
                const double s6 = std::pow(sigma / r, 6);
                energy += 0.5 * (4.0 * epsilon * (s6 * s6 - s6) - lj_shift);
                fpair += 24.0 * epsilon * (2.0 * s6 * s6 - s6) / r2;
            }
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
            stress[0] += 0.5 * dx * fpair * dx;
            stress[1] += 0.5 * dy * fpair * dy;
            stress[2] += 0.5 * dz * fpair * dz;
            stress[3] += 0.5 * dx * fpair * dy;
            stress[4] += 0.5 * dx * fpair * dz;
            stress[5] += 0.5 * dy * fpair * dz;
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
                energy += le * dl * dl * g1 * g2;

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
                stress[0] -= ax * fjx + bx * fkx;
                stress[1] -= ay * fjy + by * fky;
                stress[2] -= az * fjz + bz * fkz;
                stress[3] -= ax * fjy + bx * fky;
                stress[4] -= ax * fjz + bx * fkz;
                stress[5] -= ay * fjz + by * fkz;
            }
        }
    }

    EXPECT_NEAR(total_energy, energy, 1.0e-9);
    for (int d = 0; d < 6; ++d)
        EXPECT_NEAR(total_stress[d], stress[d], 1.0e-8)
            << "virial component " << d;
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

TEST_P(SWMultiRank, SkinImageBeyondHalfBoxDoesNotBecomeAnotherInteraction) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4)
        GTEST_SKIP()
            << "the skin-image case requires a four-way x decomposition";

    const char* previous = std::getenv("ATV_FORCE_DIMS");
    const bool had_previous = previous != nullptr;
    const std::string previous_value = previous ? previous : "";
    ::setenv("ATV_FORCE_DIMS", "4,1,1", 1);
    cpu::Frame frame(2);
    frame.box = cpu::Box::orthorhombic(7.6, 20.0, 20.0);
    frame.span(kAtom_x)[0] = 0.2;
    frame.span(kAtom_x)[1] = 3.8;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = 10.0;
        frame.span(kAtom_mass)[i] = 28.0855;
    }
    const cpu::SWPotential pot{molrs::Element::Si, "stillinger-weber"};
    const double rc = pot.cutoff();
    cpu::Context context;
    context.mpi_rank = rank;
    context.mpi_size = size;
    cpu::communication::DomainRuntime domain;
    domain.allocate(context);
    domain.bind_frame(frame, context);
    if (had_previous)
        ::setenv("ATV_FORCE_DIMS", previous_value.c_str(), 1);
    else
        ::unsetenv("ATV_FORCE_DIMS");
    cpu::md::NeighborList<> neighbors{rc, 0.3, 1, 0, true};
    neighbors.configure_lists(GetParam(), true);
    domain.comm()->set_ghost_cutoff(neighbors.neighbor_reach());
    const cpu::LaunchContext launch{context, frame, 0, 0, true, true};
    domain.comm()->setup(launch);
    domain.comm()->borders(launch);
    neighbors.build({context, frame});
    neighbors.launch(launch);
    const auto& full = *context.md.full_pair_list;
    ASSERT_GT(frame.nghost(), 0);
    int far_local = 0, far_total = 0;
    for (int ii = 0; ii < full.inum; ++ii) {
        const int i = full.ilist[ii];
        for (int k = 0; k < full.numneigh[i]; ++k) {
            const int j = full.neighbors[full.firstneigh[i] + k];
            const double distance =
                std::abs(frame.span(kAtom_x)[i] - frame.span(kAtom_x)[j]);
            if (distance > rc && distance < rc + 0.3)
                ++far_local;
        }
    }
    MPI_Allreduce(&far_local, &far_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_GE(far_total, 2)
        << "the list must actually retain the far skin images";
    cpu::StillingerWeber sw{pot};
    sw.build({context, frame});
    sw.launch(launch);
    if (GetParam()) {
        cpu::PairLJCut lj{0.1, 2.0, rc};
        lj.build({context, frame});
        lj.launch(launch);
        lj.destroy({context, frame});
    }
    domain.comm()->reverse_comm(launch);
    const auto gathered = cpu::md::mpi::gather_to_root(frame, *domain.dd());
    double energy = 0.0, stress[6] = {};
    const double local_energy = frame.meta.get(kEnergy);
    MPI_Allreduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(frame.meta.get(kStress).data(), stress, 6, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    sw.destroy({context, frame});
    neighbors.destroy({context, frame});
    domain.release_comm(context);
    if (rank != 0)
        return;

    // Exactly one near neighbor per vertex: no three-body term is possible.
    constexpr double r = 3.6;
    double expected_energy = pot.phi2(r), derivative = pot.dphi2(r);
    if (GetParam()) {
        const double s6 = std::pow(2.0 / r, 6), c6 = std::pow(2.0 / rc, 6);
        expected_energy += 0.4 * (s6 * s6 - s6 - c6 * c6 + c6);
        derivative -= 2.4 * (2 * s6 * s6 - s6) / r;
    }
    EXPECT_NEAR(energy, expected_energy, 1e-12);
    ASSERT_EQ(gathered.fx.size(), 2u);
    EXPECT_NEAR(gathered.fx[0], derivative, 1e-12);
    EXPECT_NEAR(gathered.fx[1], -derivative, 1e-12);
    EXPECT_NEAR(stress[0], -r * derivative, 1e-12);
    for (int d = 1; d < 6; ++d)
        EXPECT_NEAR(stress[d], 0.0, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(NeighborCounting, SWMultiRank, ::testing::Bool());

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
