/// @file tests/mpi/test_lj_energy_bruteforce.cpp
/// @brief Decisive pair-set guard: distributed LJ energy == brute force.
///
/// The other multi-rank LJ tests check energy *conservation* (drift over a
/// run). A conserving integrator can still be systematically wrong: drop one
/// far pair on every rank and the trajectory still conserves its (incorrect)
/// energy. This test removes that blind spot — it pins the *absolute* total
/// LJ potential energy of a fixed configuration against an independent
/// O(N²) minimum-image brute force computed from a pristine copy of the
/// global frame.
///
/// Because the same configuration is checked at both 1 rank and 8 ranks
/// against the same oracle, it is simultaneously the "1-rank vs 8-rank
/// energy comparison" the cpu-md-parallel-perf acceptance contract (ac-003)
/// requires, and the equivalence guard for ac-001 (multi-rank half-stencil
/// must yield the identical pair set) and ac-002 (multi-rank per-pair-shift
/// LJ kernel must equal the min-image kernel).
///
/// The box / cutoff are sized so each 2×2×2 rank slab has ≥3 cells per axis,
/// which is the precondition that activates the half-stencil + per-cell-shift
/// neighbor build under MPI (otherwise the cell list falls back to the
/// full-stencil path and the half-stencil code would never run here).

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEnergyRelativeTolerance = 1.0e-9;

// FCC lattice, quarter-cell shifted off the slab boundaries so no site lands
// exactly on a partition cut (mirrors test_lj_nve's make_fcc_argon rationale).
cpu::Frame make_fcc_argon(int n_cells, double a_lat) {
    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    const double shift = 0.25 * a_lat;
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] =
                        (static_cast<double>(ix) + basis[b][0]) * a_lat + shift;
                    frame.span(ATV::kAtom_y)[idx] =
                        (static_cast<double>(iy) + basis[b][1]) * a_lat + shift;
                    frame.span(ATV::kAtom_z)[idx] =
                        (static_cast<double>(iz) + basis[b][2]) * a_lat + shift;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    frame.span(ATV::kAtom_vx)[idx] = 0.0;
                    frame.span(ATV::kAtom_vy)[idx] = 0.0;
                    frame.span(ATV::kAtom_vz)[idx] = 0.0;
                    ++idx;
                }
    frame.box = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

// Shifted LJ 12-6 pair energy, replicating PairLJCut exactly (4ε(s12-s6)
// minus the value at the cutoff). Returns 0 outside the cutoff.
double lj_pair_energy(double r2, double eps, double sigma, double cutoff) {
    const double cutoff_sq = cutoff * cutoff;
    if (r2 >= cutoff_sq)
        return 0.0;
    const double sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
    const double rc2inv = 1.0 / cutoff_sq;
    const double rc6inv = rc2inv * rc2inv * rc2inv * sigma6;
    const double v_shift = 4.0 * eps * (rc6inv * rc6inv - rc6inv);
    const double r2inv = 1.0 / r2;
    const double r6inv = r2inv * r2inv * r2inv * sigma6;
    return 4.0 * eps * (r6inv * r6inv - r6inv) - v_shift;
}

double brute_force_lj_energy(const cpu::Frame& frame, double eps, double sigma,
                             double cutoff) {
    const int n = frame.nlocal();
    const double lx = frame.box.cell()(0, 0), ly = frame.box.cell()(1, 1),
                 lz = frame.box.cell()(2, 2);
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx =
                frame.span(ATV::kAtom_x)[i] - frame.span(ATV::kAtom_x)[j];
            double dy =
                frame.span(ATV::kAtom_y)[i] - frame.span(ATV::kAtom_y)[j];
            double dz =
                frame.span(ATV::kAtom_z)[i] - frame.span(ATV::kAtom_z)[j];
            // Minimum image (multi-wrap safe).
            while (dx > 0.5 * lx)
                dx -= lx;
            while (dx < -0.5 * lx)
                dx += lx;
            while (dy > 0.5 * ly)
                dy -= ly;
            while (dy < -0.5 * ly)
                dy += ly;
            while (dz > 0.5 * lz)
                dz -= lz;
            while (dz < -0.5 * lz)
                dz += lz;
            e +=
                lj_pair_energy(dx * dx + dy * dy + dz * dz, eps, sigma, cutoff);
        }
    }
    return e;
}

} // namespace

// ── Absolute LJ energy matches brute force at 1 and 8 ranks ─────────────

TEST(LjEnergyBruteForce, GlobalEnergyMatchesBruteForce) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1 && world_size != 8) {
        GTEST_SKIP() << "1- or 8-rank specific; got world_size=" << world_size;
    }

    // Box = 8 * 5.3 = 42.4 Å. At 8 ranks (2×2×2) each slab is 21.2 Å; with
    // r_cut+skin ≈ 5.2 Å that is ~6 cells/axis ≥ 3 → half-stencil active.
    // FCC NN = 5.3/√2 ≈ 3.75 Å < r_cut = 4.0 Å so every atom has LJ neighbors.
    const double eps = 0.0104, sigma = 3.4, cutoff = 4.0;
    auto frame = make_fcc_argon(8, 5.3);

    // Pristine copy of the *global* configuration for the oracle. Built
    // before the pipeline partitions/migrates the live frame.
    const cpu::Frame ref = frame;
    const double e_bf = brute_force_lj_energy(ref, eps, sigma, cutoff);
    ASSERT_GT(std::abs(e_bf), 1e-9) << "degenerate oracle energy";

    cpu::md::MD sim;
    sim.set_integrator(
        cpu::md::VelocityVerlet{0.0}); // dt = 0 → positions frozen
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff});
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{eps, sigma, cutoff});
    sim.build();

    sim.run(frame, 1); // one frozen step → Reduce fills global energy

    // frame.meta.get(ATV::kEnergy) is the cross-rank reduced LJ potential energy.
    const double e_dist = frame.meta.get(ATV::kEnergy);
    const double rel = std::abs(e_dist - e_bf) / std::abs(e_bf);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "rank " << my_rank << " world=" << world_size
        << ": distributed LJ energy " << e_dist
        << " disagrees with brute force " << e_bf << " (rel=" << rel
        << ") — multi-rank neighbor pair set or per-pair-shift kernel is wrong";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
