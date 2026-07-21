/// @file tests/mpi/test_halo_dual_image_ghosts.cpp
/// @brief Pin the dims==2 dual-image ghost guard (silent LJ double-count).
///
/// With 2 ranks along an axis the +d and -d neighbour are the SAME rank;
/// when 2*halo_cutoff exceeds the subdomain span, every overlap-band atom
/// ships through BOTH faces and arrives as two periodic images of one tag.
/// The full-stencil neighbor fallback then min-images the decomposed axis,
/// folds both images onto one separation, and the same physical pair is
/// counted TWICE — observed as a silent ~16% LJ energy error on a
/// 3^3-lattice 15.78 A box at cutoff 6 (2026-07-15). The fix: the
/// full-stencil neighbor paths min-image ONLY non-extended axes (ghosts
/// sit at their true periodic image; the far duplicate is rejected by its
/// real distance). This test pins the fixed behavior in BOTH geometries
/// against an O(N^2) minimum-image brute force.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <stdexcept>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [A]
constexpr double kEnergyRelativeTolerance = 1.0e-9;

/// Simple-cubic n^3 lattice, spacing a0, cubic box n*a0.
cpu::Frame make_sc_lattice(int n_side, double a0) {
    const int n = n_side * n_side * n_side;
    cpu::Frame frame(n);
    int idx = 0;
    for (int i = 0; i < n_side; ++i)
        for (int j = 0; j < n_side; ++j)
            for (int k = 0; k < n_side; ++k) {
                frame.span(ATV::kAtom_x)[idx] = (i + 0.5) * a0;
                frame.span(ATV::kAtom_y)[idx] = (j + 0.5) * a0;
                frame.span(ATV::kAtom_z)[idx] = (k + 0.5) * a0;
                frame.span(ATV::kAtom_mass)[idx] = 39.948;
                frame.span(ATV::kAtom_tag)[idx] = idx;
                ++idx;
            }
    frame.simbox = ATV::cpu::Box::cubic(n_side * a0);
    return frame;
}

/// Shifted LJ pair energy, replicating PairLJCut (energy shifted to 0 at rc).
double lj_pair_energy(double r2, double cutoff) {
    const double rc2 = cutoff * cutoff;
    if (r2 >= rc2)
        return 0.0;
    auto e = [](double rr2) {
        const double s2 = (kSigma * kSigma) / rr2;
        const double s6 = s2 * s2 * s2;
        return 4.0 * kEps * (s6 * s6 - s6);
    };
    return e(r2) - e(rc2);
}

/// O(N^2) minimum-image total LJ energy oracle.
double brute_force_energy(const cpu::Frame& frame, double cutoff) {
    const int n = frame.nlocal();
    const double L = frame.simbox.length(0);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx =
                frame.span(ATV::kAtom_x)[i] - frame.span(ATV::kAtom_x)[j];
            double dy =
                frame.span(ATV::kAtom_y)[i] - frame.span(ATV::kAtom_y)[j];
            double dz =
                frame.span(ATV::kAtom_z)[i] - frame.span(ATV::kAtom_z)[j];
            dx -= L * std::round(dx / L);
            dy -= L * std::round(dy / L);
            dz -= L * std::round(dz / L);
            total += lj_pair_energy(dx * dx + dy * dy + dz * dz, cutoff);
        }
    return total;
}

/// Build the standard LJ MD pipeline and run one frozen step.
void run_one_frozen_step(cpu::Frame& frame, double cutoff) {
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.0}); // dt = 0: positions frozen
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{cutoff}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, cutoff});
    sim.build();
    sim.run(frame, 1);
}

} // namespace

// The formerly-defective geometry: 3^3 lattice, box 15.78 A, cutoff 6 ->
// auto halo 6.6 A; at 2 ranks the split-axis span is 7.89 A < 2*6.6, so
// every rank-partner atom arrives as TWO periodic images. Before the fix
// this silently computed E 16% too negative; now it must match the oracle.
TEST(HaloDualImageGhosts, OverlappingFaceBandsMatchBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1 && world != 2)
        GTEST_SKIP() << "dual-image shape covers 1 and 2 ranks";

    cpu::Frame frame = make_sc_lattice(3, 5.26);
    const cpu::Frame ref = frame;
    const double e_bf = brute_force_energy(ref, 6.0);
    ASSERT_GT(std::abs(e_bf), 1e-9);

    run_one_frozen_step(frame, 6.0);

    const double rel =
        std::abs(frame.meta.get(ATV::kEnergy) - e_bf) / std::abs(e_bf);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "distributed " << frame.meta.get(ATV::kEnergy) << " vs brute force "
        << e_bf;
}

// Companion: the SAME shape with 2*halo < span (6^3 lattice, box 31.56 A,
// 2-rank span 15.78 A > 2*6.6) must still run and match the brute-force
// oracle — pinning that the guard fires on exactly the self-overlap
// predicate and nothing wider.
TEST(HaloDualImageGhosts, NonOverlappingGeometryStillMatchesBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1 && world != 2)
        GTEST_SKIP() << "reference shape covers 1 and 2 ranks";

    cpu::Frame frame = make_sc_lattice(6, 5.26);
    const cpu::Frame ref = frame; // pristine copy for the oracle
    const double e_bf = brute_force_energy(ref, 6.0);
    ASSERT_GT(std::abs(e_bf), 1e-9);

    run_one_frozen_step(frame, 6.0);

    const double rel =
        std::abs(frame.meta.get(ATV::kEnergy) - e_bf) / std::abs(e_bf);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "distributed " << frame.meta.get(ATV::kEnergy) << " vs brute force "
        << e_bf;
}

// ── Custom main that initializes MPI before/after GoogleTest ────────────
//
// See test_topology.cpp for the OpenMPI 5.0.x macOS teardown rationale.

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
