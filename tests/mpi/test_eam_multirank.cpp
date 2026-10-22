/// @file tests/mpi/test_eam_multirank.cpp
/// @brief Multi-rank EAM: the domain-decomposed forces must equal a brute-force
///        O(N^2) serial reference.
///
/// EAM consumes a Newton-on half list. Ghost density contributions return to
/// owners before embedding derivatives travel forward; one cutoff of halo is
/// sufficient. Small and larger boxes cover forwarding and two-face images.
/// The kernel is invoked directly: no MD trajectory or integrator is involved.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain_runtime.h"
#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/pbc.h"
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
class EAMMultiRank : public ::testing::TestWithParam<int> {};

TEST_P(EAMMultiRank, ForcesEnergyAndVirialMatchBruteForce) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_cell = GetParam();
    cpu::Frame frame = copper(n_cell);
    const int n = frame.nlocal();
    cpu::Context context;
    context.mpi_rank = rank;
    context.mpi_size = size;
    cpu::communication::DomainRuntime domain;
    domain.allocate(context);
    cpu::md::wrap_owned(frame, n);
    domain.bind_frame(frame, context);
    cpu::EAM eam{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, kCutoff};
    cpu::md::NeighborList<> neighbors{kCutoff, 0.3, 1, 0, true};
    EXPECT_EQ(eam.neighbor_list_mode(), cpu::NeighborListMode::Half);
    EXPECT_DOUBLE_EQ(eam.halo_cutoff(), kCutoff);
    EXPECT_FALSE(eam.requires_ghost_neighbor_rows());
    domain.comm()->set_ghost_cutoff(neighbors.neighbor_reach());
    const cpu::LaunchContext launch{context, frame, 0, 0};
    domain.comm()->setup(launch);
    domain.comm()->borders(launch);
    neighbors.build({context, frame});
    neighbors.launch(launch);
    eam.build({context, frame});
    eam.launch(launch);
    domain.comm()->reverse_comm(launch);
    const auto g = cpu::md::mpi::gather_to_root(frame, *domain.dd());
    double local[7] = {frame.meta.get(kEnergy)};
    for (int k = 0; k < 6; ++k)
        local[k + 1] = frame.meta.get(kStress)[k];
    double total[7]{};
    MPI_Allreduce(local, total, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Re-evaluating energy must reset density scratch, preserve forces/stress,
    // and retain the same result after both intermediate communication passes.
    frame.meta.get(kEnergy) = 0.0;
    frame.meta.get(kEnergyPair) = 0.0;
    eam.launch({context, frame, 0, 1, false, false});
    EXPECT_NEAR(frame.meta.get(kEnergy), local[0], 1e-10);
    for (int k = 0; k < 6; ++k)
        EXPECT_DOUBLE_EQ(frame.meta.get(kStress)[k], local[k + 1]);
    const auto repeated = cpu::md::mpi::gather_to_root(frame, *domain.dd());
    neighbors.destroy({context, frame});
    domain.release_comm(context);

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

    double energy = 0.0;
    double virial[6]{};
    for (int i = 0; i < n; ++i)
        energy += pot.F(rho[i]);
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
            energy += 0.5 * (pot.phi(r) - phi_shift);
            virial[0] += 0.5 * dx * fpair * dx;
            virial[1] += 0.5 * dy * fpair * dy;
            virial[2] += 0.5 * dz * fpair * dz;
            virial[3] += 0.5 * dx * fpair * dy;
            virial[4] += 0.5 * dx * fpair * dz;
            virial[5] += 0.5 * dy * fpair * dz;
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
        }
    }
    EXPECT_NEAR(total[0], energy, 1e-9);
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(total[k + 1], virial[k], 1e-9);
    EXPECT_EQ(g.fx, repeated.fx);
    EXPECT_EQ(g.fy, repeated.fy);
    EXPECT_EQ(g.fz, repeated.fz);

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
        << worst
        << " eV/A — density/derivative communication or half-list ownership is "
           "wrong";
}

INSTANTIATE_TEST_SUITE_P(PeriodicBricks, EAMMultiRank, ::testing::Values(3, 6));
