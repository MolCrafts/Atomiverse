/// @file tests/mpi/test_pme_nacl_parity.cpp
/// @brief cpu-breadth-03 ac-007: charged NaCl CoulLong+PME (+ short NPT)
///        agrees on 1 and 2 ranks after global reduce / tag-ordered gather.
///
/// Replicated-grid PME: every rank holds the full FFT mesh (Allreduce of the
/// charge grid), then interpolates forces on owned atoms only. Reciprocal
/// energy/stress are rank-0 only and then reduced. This test proves:
///   * static energy/stress match the Madelung literature value on n=1 and n=2,
///   * all ranks see the same reduced energy/stress/volume after one force step,
///   * tag-gathered forces vanish by cubic symmetry and agree across ranks,
///   * a short NoseHooverNPT run keeps H finite, rank-identical, and the cell
///     moving — without double-counting owned contributions.
///
/// Soft LJ core is used only on the NPT path (bare Coulomb collapses under
/// P_target=0). Static Madelung path is pure electrostatic.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/gather.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kD = 2.82;
// n=6 → L=16.92 Å. Under 2 ranks the thinnest subdomain is L/2 ≈ 8.46 Å,
// which must exceed the neighbor/halo shell (cutoff+skin ≈ 5.9 Å). n=4
// (L=11.28 → half 5.64) is too thin and HaloSend throws.
constexpr int kSide = 6;
constexpr int kN = kSide * kSide * kSide;
constexpr double kMadelung = 1.747564594633182;
constexpr double kCutoff = 1.9 * kD;
constexpr double kAlpha = 3.5 / kCutoff;
constexpr double kSpacing = 0.5;
constexpr int kOrder = 6;

// Reduction-order / FP budget: static electrostatics is tighter; NPT after
// many reductions is looser but still catches double-counting.
constexpr double kStaticEnergyRel = 5e-4; // matches unit Madelung gate
constexpr double kStaticStressAbs = 0.5;  // eV on ~95 eV diagonals
constexpr double kRankAgreeRel = 1e-12;
constexpr double kNptHAgreeRel = 1e-9;

cpu::Frame make_nacl(double scale) {
    cpu::Frame frame(kN);
    int idx = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k) {
                frame.span(ATV::kAtom_x)[idx] = (i * kD) * scale;
                frame.span(ATV::kAtom_y)[idx] = (j * kD) * scale;
                frame.span(ATV::kAtom_z)[idx] = (k * kD) * scale;
                frame.span(ATV::kAtom_charge)[idx] =
                    ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.span(ATV::kAtom_mass)[idx] =
                    ((i + j + k) % 2 == 0) ? 22.9898 : 35.453;
                frame.span(ATV::kAtom_type)[idx] =
                    ((i + j + k) % 2 == 0) ? 0 : 1;
                frame.span(ATV::kAtom_tag)[idx] = idx + 1;
                frame.span(ATV::kAtom_vx)[idx] = 0.0;
                frame.span(ATV::kAtom_vy)[idx] = 0.0;
                frame.span(ATV::kAtom_vz)[idx] = 0.0;
                ++idx;
            }
    frame.simbox = cpu::Box::cubic(kSide * kD * scale);
    return frame;
}

void build_static_electrostatics(cpu::md::MD& sim) {
    // dt = 0 freezes positions; one step evaluates forces + Reduce.
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, /*skin=*/0.3});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kCutoff});
    sim.add(cpu::KSpacePME{kAlpha, kSpacing, kOrder});
}

} // namespace

// ── Static: Madelung + rank-identical reduced state + gathered forces ──

TEST(PmeNaclParity, StaticEnergyStressForceOn1And2Ranks) {
    int world_size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world_size != 1 && world_size != 2)
        GTEST_SKIP() << "1- or 2-rank specific; got world_size=" << world_size;

    auto frame = make_nacl(1.0);
    cpu::md::MD sim;
    build_static_electrostatics(sim);
    sim.build();
    sim.run(frame, 1);

    const double e_ref = -0.5 * kN * kMadelung * units::coulomb / kD;
    const double E = frame.meta.get(ATV::kEnergy);
    const double rel = std::fabs(E / e_ref - 1.0);
    EXPECT_LT(rel, kStaticEnergyRel)
        << "rank " << rank << " world=" << world_size << " Madelung rel=" << rel
        << " E=" << E << " ref=" << e_ref;

    // Every rank must see the same reduced energy (Reduce all-reduces).
    double E_min = E, E_max = E;
    MPI_Allreduce(MPI_IN_PLACE, &E_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &E_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NEAR(E_min, E_max, kRankAgreeRel * std::max(1.0, std::fabs(E)))
        << "ranks disagree on reduced energy";

    // Stress diagonals equal by cubic symmetry; rank-identical after reduce.
    double sxx = frame.meta.get(ATV::kStress)[0];
    double syy = frame.meta.get(ATV::kStress)[1];
    double szz = frame.meta.get(ATV::kStress)[2];
    EXPECT_NEAR(sxx, syy, kStaticStressAbs);
    EXPECT_NEAR(syy, szz, kStaticStressAbs);
    for (int c = 3; c < 6; ++c)
        EXPECT_NEAR(frame.meta.get(ATV::kStress)[c], 0.0, 1e-4)
            << "shear Voigt " << c;

    double sxx_min = sxx, sxx_max = sxx;
    MPI_Allreduce(MPI_IN_PLACE, &sxx_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sxx_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_NEAR(sxx_min, sxx_max, kStaticStressAbs * 1e-6)
        << "ranks disagree on reduced stress_xx";

    // Cell must be identical on every rank (NPT/deform path will write it).
    double V = frame.simbox.volume();
    double V_min = V, V_max = V;
    MPI_Allreduce(MPI_IN_PLACE, &V_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &V_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NEAR(V_min, V_max, 1e-12 * V);

    // Tag-ordered gather: forces vanish by rock-salt symmetry; n_total == N
    // proves each atom is owned exactly once under the decomposition.
    auto dd = sim.release_dd();
    ASSERT_NE(dd, nullptr);
    auto gathered = cpu::md::mpi::gather_to_root(frame, *dd);
    if (rank == 0) {
        ASSERT_EQ(gathered.n_total, kN);
        double fmax = 0.0;
        double fsum[3] = {0, 0, 0};
        for (int i = 0; i < gathered.n_total; ++i) {
            fsum[0] += gathered.fx[i];
            fsum[1] += gathered.fy[i];
            fsum[2] += gathered.fz[i];
            fmax =
                std::max(fmax, std::max(std::fabs(gathered.fx[i]),
                                        std::max(std::fabs(gathered.fy[i]),
                                                 std::fabs(gathered.fz[i]))));
        }
        // Rock-salt forces vanish by symmetry; residual is PME grid noise
        // (larger under 2-rank real-space reduction order than serial).
        EXPECT_NEAR(fsum[0], 0.0, 1e-4);
        EXPECT_NEAR(fsum[1], 0.0, 1e-4);
        EXPECT_NEAR(fsum[2], 0.0, 1e-4);
        EXPECT_NEAR(fmax, 0.0, 1e-3) << "gathered max |F| not near zero";
    }
}

// ── Short NPT: conserved H rank-identical, cell mutates, no freeze ─────

TEST(PmeNaclParity, ShortNptConservedAndCellOn1And2Ranks) {
    int world_size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (world_size != 1 && world_size != 2)
        GTEST_SKIP() << "1- or 2-rank specific; got world_size=" << world_size;

    // Mildly expanded start so NPT contracts under cohesion + P=0. Keep the
    // run short so the shrinking cell never undercuts the halo shell.
    auto frame = make_nacl(1.02);
    const double V0 = frame.simbox.volume();

    const double dt = 0.5;
    const double T = 300.0;
    constexpr double lj_eps = 0.05;
    constexpr double lj_sigma = 1.5;
    constexpr double lj_cut = 2.5 * lj_sigma;
    const double neigh_cut = std::max(kCutoff, lj_cut);

    cpu::md::MD sim;
    sim.add(cpu::md::InitVelocity{T, /*seed=*/42u});
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = T,
                                               .tau_T = 50.0,
                                               .P_target = 0.0,
                                               .tau_P = 500.0,
                                               .dt = dt}});
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sigma, lj_cut});
    sim.set_neighbor(cpu::md::NeighborList<>{neigh_cut, /*skin=*/0.5});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kCutoff});
    sim.add(cpu::KSpacePME{kAlpha, kSpacing, kOrder});
    sim.build();

    sim.run(frame, 1);
    const double H0 = frame.meta.get(ATV::kConserved);
    ASSERT_TRUE(std::isfinite(H0)) << "rank " << rank << " H0 non-finite";

    // Ranks must share the same extended Hamiltonian after reduce.
    double H0_min = H0, H0_max = H0;
    MPI_Allreduce(MPI_IN_PLACE, &H0_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &H0_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_NEAR(H0_min, H0_max, kNptHAgreeRel * std::max(1.0, std::fabs(H0)))
        << "ranks disagree on H after first NPT step";

    constexpr int n_steps = 40;
    sim.run(frame, n_steps);
    const double H1 = frame.meta.get(ATV::kConserved);
    const double V1 = frame.simbox.volume();
    ASSERT_TRUE(std::isfinite(H1)) << "rank " << rank << " H1 non-finite";
    ASSERT_TRUE(std::isfinite(frame.meta.get(ATV::kEnergy)));
    ASSERT_TRUE(std::isfinite(frame.meta.get(ATV::kKinetic)));

    double H1_min = H1, H1_max = H1;
    MPI_Allreduce(MPI_IN_PLACE, &H1_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &H1_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_NEAR(H1_min, H1_max, kNptHAgreeRel * std::max(1.0, std::fabs(H1)))
        << "ranks disagree on H after NPT";

    double V1_min = V1, V1_max = V1;
    MPI_Allreduce(MPI_IN_PLACE, &V1_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &V1_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_NEAR(V1_min, V1_max, 1e-12 * std::max(1.0, V1))
        << "ranks disagree on cell volume after NPT";

    // Barostat moved the cell; soft core prevents free collapse.
    EXPECT_LT(V1, V0) << "expanded cohesive start should contract";
    EXPECT_GT(V1, 0.2 * V0) << "cell collapsed";
    EXPECT_GT(frame.meta.get(ATV::kKinetic), 0.1) << "kinetic freeze";

    // ac-007 is rank parity, not a tight serial conservation gate (that lives
    // in reg_npt_pme_nacl). Allow a short-run MTK envelope; ranks already
    // agreed on H0/H1 above.
    const double H_drift = std::fabs(H1 - H0) / std::max(1.0, std::fabs(H0));
    EXPECT_LT(H_drift, 0.15)
        << "rank " << rank << " world=" << world_size << " NPT H drift "
        << H_drift << " (secular runaway?)";

    // Gathered owner count must be N (each atom owned exactly once).
    auto dd = sim.release_dd();
    ASSERT_NE(dd, nullptr);
    auto gathered = cpu::md::mpi::gather_to_root(frame, *dd);
    if (rank == 0)
        EXPECT_EQ(gathered.n_total, kN)
            << "owned atoms not a partition of N after NPT";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
