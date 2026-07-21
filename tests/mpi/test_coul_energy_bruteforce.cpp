/// @file tests/mpi/test_coul_energy_bruteforce.cpp
/// @brief Decisive multi-rank guard for the real-space Coulomb (CoulLong):
///        distributed energy == O(N²) brute force, at 1 and 8 ranks.
///
/// This is the proof for force-infra-02: ghost charges are now carried in the
/// `build_halo` border message (x,y,z,tag,q) and CoulLong reads them directly
/// from `frame.span(ATV::kAtom_charge)` — the old `MPI_Allgatherv` tag→charge table is
/// gone. If ghost charge were missing or wrong, a neighbor `j` that is a ghost
/// (its owner on another rank) would contribute the wrong erfc term and the
/// 8-rank energy would diverge from the 1-rank / brute-force value.
///
/// Same configuration is checked at 1 and 8 ranks against the same erfc oracle,
/// so it doubles as the "1-rank vs 8-rank" CoulLong parity (ac-005).

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

constexpr double kEnergyRelativeTolerance = 1.0e-9;

// FCC lattice (quarter-cell shifted off the slab cuts) with net-neutral
// alternating ±q charges so every atom carries a nonzero charge and ghost
// charges genuinely matter for cross-rank neighbors.
cpu::Frame make_charged_fcc(int n_cells, double a_lat, double q_mag) {
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
                    frame.span(ATV::kAtom_charge)[idx] =
                        (idx % 2 == 0) ? q_mag : -q_mag;
                    frame.span(ATV::kAtom_vx)[idx] = 0.0;
                    frame.span(ATV::kAtom_vy)[idx] = 0.0;
                    frame.span(ATV::kAtom_vz)[idx] = 0.0;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

// Real-space screened Coulomb pair energy, replicating CoulLong exactly:
// E = units::coulomb · q_i q_j · erfc(α r) / r for r < cutoff.
double brute_force_coul_energy(const cpu::Frame& frame, double alpha,
                               double cutoff) {
    const int n = frame.nlocal();
    const double lx = frame.simbox.cell()(0, 0), ly = frame.simbox.cell()(1, 1),
                 lz = frame.simbox.cell()(2, 2);
    const double rc_sq = cutoff * cutoff;
    double e = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx =
                frame.span(ATV::kAtom_x)[i] - frame.span(ATV::kAtom_x)[j];
            double dy =
                frame.span(ATV::kAtom_y)[i] - frame.span(ATV::kAtom_y)[j];
            double dz =
                frame.span(ATV::kAtom_z)[i] - frame.span(ATV::kAtom_z)[j];
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
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc_sq || r2 == 0.0)
                continue;
            const double r = std::sqrt(r2);
            e += units::coulomb * frame.span(ATV::kAtom_charge)[i]
                 * frame.span(ATV::kAtom_charge)[j] * std::erfc(alpha * r) / r;
        }
    }
    return e;
}

} // namespace

// ── Real-space Coulomb energy matches brute force at 1 and 8 ranks ───────────

TEST(CoulEnergyBruteForce, GlobalEnergyMatchesBruteForce) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1 && world_size != 8)
        GTEST_SKIP() << "1- or 8-rank specific; got world_size=" << world_size;

    const double alpha = 0.30, cutoff = 4.0, q_mag = 1.0;
    auto frame = make_charged_fcc(8, 5.3, q_mag);

    const cpu::Frame ref = frame; // pristine global oracle source
    const double e_bf = brute_force_coul_energy(ref, alpha, cutoff);
    ASSERT_GT(std::abs(e_bf), 1e-9) << "degenerate oracle energy";

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.0}); // dt = 0 → positions frozen
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{cutoff}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::CoulLong{alpha, cutoff});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{cutoff}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.build();

    sim.run(frame, 1); // one frozen step → Reduce fills global energy

    const double e_dist = frame.meta.get(ATV::kEnergy);
    const double rel = std::abs(e_dist - e_bf) / std::abs(e_bf);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "rank " << my_rank << " world=" << world_size
        << ": distributed real-space Coulomb energy " << e_dist
        << " disagrees with brute force " << e_bf << " (rel=" << rel
        << ") — ghost charge (border message) is missing or wrong";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
