/// @file tests/cuda/potentials/test_pair_lj_special_bonds.cu
/// @brief `cuda::PairLJCut::set_special_bonds` — LAMMPS-style 1-2 / 1-3
///        / 1-4 exclusion CSR + per-pair binary-search masking.
///
/// Covers `msu-pe-cuda-components-05-cuda-lj-exclusions`:
/// - public API exists and returns `*this` (ac-001).
/// - rejects mixed-scaling combinations with `std::invalid_argument` (ac-002).
/// - 4-atom chain with `(0,0,0)` exclusion yields zero LJ energy (ac-003).
/// - the no-exclusion fast path (no `set_special_bonds` call) keeps existing
///   LJ-fluid behaviour bit-identical — covered by the existing test_cuda_md
///   NVE regression suite.

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/driver/evaluate.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

/// Build a 4-atom collinear chain frame, sigma-spaced (so LJ generates
/// well-defined non-zero energy when exclusions are off). Bonds: 0-1,
/// 1-2, 2-3; angles: 0-1-2, 1-2-3; dihedral: 0-1-2-3.
cpu::Frame make_chain4() {
    cpu::Frame frame(4);
    for (int i = 0; i < 4; ++i) {
        frame.atoms.x[i] = double(i) * 1.0;
        frame.atoms.y[i] = 0.0;
        frame.atoms.z[i] = 0.0;
        frame.atoms.vx[i] = frame.atoms.vy[i] = frame.atoms.vz[i] = 0.0;
        frame.atoms.fx[i] = frame.atoms.fy[i] = frame.atoms.fz[i] = 0.0;
        frame.atoms.mass[i] = 1.0;
        frame.atoms.type[i] = 0;
        frame.atoms.tag[i] = i;
    }
    frame.box = Mat3::cubic(20.0);
    // Per-atom topology + 1-2/1-3/1-4 special-neighbor exclusion list.
    // build_special_list() does the BFS over the per-atom bond graph that
    // populates nspecial/special — the source of the LJ exclusion CSR.
    frame.atoms.map_clear();
    frame.atoms.map_set(0, frame.atoms.nlocal);
    frame.atoms.add_bond(0, 1, 0); // bonds 0-1, 1-2, 2-3
    frame.atoms.add_bond(1, 2, 0);
    frame.atoms.add_bond(2, 3, 0);
    frame.atoms.add_angle(0, 1, 2, 0); // angles 0-1-2, 1-2-3
    frame.atoms.add_angle(1, 2, 3, 0);
    frame.atoms.add_dihedral(0, 1, 2, 3, 0); // dihedral 0-1-2-3
    frame.atoms.build_special_list();
    return frame;
}

} // namespace

TEST(CudaPairLJSpecialBonds, ApiAcceptsAllEqualOrIdentity) {
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(1.0, 1.0, 1.0));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 0.0, 0.0));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.5, 0.5, 0.5));
}

TEST(CudaPairLJSpecialBonds, ApiRejectsMixedScaling) {
    EXPECT_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 1.0, 0.0),
        std::invalid_argument);
    EXPECT_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 0.0, 0.5),
        std::invalid_argument);
}

// Re-enabled after the LJ kernel got an `(i >= 0 && j >= 0)` ghost-pair
// guard. The cluster-based neighbor list export pads each cluster to
// kClusterSize atoms, and the padded slots leak into the flat PairList
// with j = -1 and garbage `dx/dy/dz` that can coincidentally give a
// finite pdist (here r = 1.0 for atom 1's padding slot). Those ghost
// pairs bypass the exclusion CSR mask (-1 is in no partner list) and
// erroneously contributed ~-0.40 eV to the total LJ energy. The new
// guard rejects pairs with negative atom indices outright; valid pairs
// still flow through the existing `r < cutoff && r >= kMinPairDist`
// filter as before.
TEST(CudaPairLJSpecialBonds, FourAtomChainAllExcludedZeroEnergy) {
    auto frame = make_chain4();
    const double cutoff = 3.0;
    // Use cuda::md::MD instead of cuda::Evaluate so we run through the
    // canonical force-phase wiring the MSU PE benchmark uses.
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.add_potential(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(frame, 1);
    // 4 atoms in a 4-chain: every pair (0-1, 0-2, 0-3, 1-2, 1-3, 2-3) is
    // either 1-2 (bonds), 1-3 (angle endpoints) or 1-4 (dihedral endpoints).
    // With (0,0,0) the LJ kernel must skip every pair → energy_pair == 0.
    EXPECT_NEAR(static_cast<double>(frame.scalars.energy), 0.0, 1e-9);
}
