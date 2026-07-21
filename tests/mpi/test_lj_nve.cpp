/// @file tests/mpi/test_lj_nve.cpp
/// @brief Phase 2.7 exit-gate: multi-rank LJ NVE energy conservation.
///
/// End-to-end check that the halo-enabled pipeline produces physically
/// consistent forces. Setup:
///   - FCC Argon-like system, 500 atoms in a 26.5 Å cubic box
///     (slab 13.25 Å at 8 ranks), σ=3.4 Å, ε=0.0104 eV, r_cut=6.0 Å.
///   - The slab is intentionally wider than 2·r_cut so no owned atom is
///     shipped to the same MPI neighbour via two opposing faces (which
///     would create duplicate ghost copies and double-count forces under
///     the Phase-2 face-only halo).
///   - Small initial velocities (~Maxwell-Boltzmann @ 10 K) so atoms do
///     not migrate across slab boundaries within the run (Phase 2 has no
///     atom migration; see spec §A.4).
///   - 20 steps of NVE integration with `dt = 1 fs`.
///   - Each step: compute total energy = `Σ local frame.meta.get(ATV::kEnergy) +
///     Σ local kinetic` reduced across ranks via `MPI_Allreduce`. In NVE
///     the total energy drift must stay below 1e-2 relative.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/driver/backend.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

cpu::Frame make_fcc_argon(int n_cells, double a_lat) {
    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };
    // Shift the lattice off the slab boundaries by a quarter lattice
    // constant so no FCC site lands exactly on `sublo[d]`/`subhi[d]` --
    // otherwise FP rounding plus tiny-velocity drift can push a boundary
    // atom across the slab edge in the first integration step, and
    // Phase 2 (no migration) would then drop it from its rank's owner
    // list when `Driver::run()` is invoked a second time.
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
                    // Tiny structured velocity pattern: deterministic, sum-to-zero.
                    const double s = 1e-4 * ((idx % 2 == 0) ? 1.0 : -1.0);
                    frame.span(ATV::kAtom_vx)[idx] = s;
                    frame.span(ATV::kAtom_vy)[idx] = s * 0.5;
                    frame.span(ATV::kAtom_vz)[idx] = s * 0.25;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

double total_energy_global(const cpu::Frame& frame) {
    // Reduce (Phase 0.4) runs at end_of_step @ -50 and writes the
    // globally-reduced sum back into frame.meta.get(ATV::kEnergy) / frame.meta.get(ATV::kKinetic), so
    // every rank sees the same global value here. Pre-Phase-0.4 callers
    // had to MPI_Allreduce manually -- doing so now would double-reduce.
    return frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
}

} // namespace

// ── NVE energy conservation on 1 rank (baseline, no halo activity) ──────

TEST(LjNve, EnergyConservationSingleRank) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1) {
        GTEST_SKIP() << "single-rank baseline; skipped under multi-rank";
    }

    // Box = 4 * 5.3 = 21.2 Å; FCC NN = 5.3/√2 ≈ 3.75 Å, close to σ=3.4 Å.
    // LJ is gentle here; NVE drift over short runs should be small.
    auto frame =
        make_fcc_argon(5, 5.3); // 500 atoms, box 26.5 Å, slab 13.25 Å @ 8 ranks

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1.0}); // dt = 1 fs
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{6.0}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{0.0104, 3.4, 6.0});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{6.0}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.build();

    sim.run(frame, 1);
    const double e0 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    ASSERT_GT(std::abs(e0), 1e-12) << "initial energy should be nonzero";

    sim.run(frame, 20);
    const double e1 =
        frame.meta.get(ATV::kEnergy) + frame.meta.get(ATV::kKinetic);
    const double drift = std::abs(e1 - e0) / std::abs(e0);
    EXPECT_LT(drift, 1e-2) << "single-rank NVE energy drift " << drift
                           << " exceeds 1e-2 over 20 steps (E0=" << e0
                           << ", E1=" << e1 << ")";
}

// ── NVE energy conservation under multi-rank halo exchange ───────────

TEST(LjNve, EnergyConservationMultiRank) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size <= 1) {
        GTEST_SKIP() << "multi-rank test; skipped on single-rank";
    }

    // Same setup as 1-rank: 500 atoms, box 26.5 Å, slab 13.25 Å @ 2x2x2.
    // Cutoff 6.0 < slab/2 = 6.625, so each owner is shipped to a given
    // neighbour via at most one face -- no duplicate ghost copies.
    auto frame = make_fcc_argon(5, 5.3);

    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{1.0});
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{6.0}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{0.0104, 3.4, 6.0});
    sim.on(ATV::md::Phase::pre_force).add(cpu::comm::HaloSend{6.0}, -20);
    sim.on(ATV::md::Phase::post_force).add(cpu::comm::HaloRecv{}, -1000);
    sim.build();

    sim.run(frame, 1);
    const double e0 = total_energy_global(frame);
    ASSERT_GT(std::abs(e0), 1e-12);

    sim.run(frame, 20);
    const double e1 = total_energy_global(frame);
    const double drift = std::abs(e1 - e0) / std::abs(e0);
    EXPECT_LT(drift, 1e-2) << "multi-rank NVE energy drift " << drift
                           << " exceeds 1e-2 over 20 steps (E0=" << e0
                           << ", E1=" << e1 << ")";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
