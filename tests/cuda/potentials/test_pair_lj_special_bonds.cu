/// @file tests/cuda/potentials/test_pair_lj_special_bonds.cu
/// @brief `cuda::PairLJCut::set_special_bonds` — LAMMPS-style 1-2 / 1-3
///        / 1-4 exclusion CSR + per-pair binary-search masking.
///
/// Covers `msu-pe-cuda-components-05-cuda-lj-exclusions`:
/// - public API exists and returns `*this` (ac-001).
/// - binary scales, including mixed shells such as (1, 0, 0) (ac-002).
/// - fractional scales (LAMMPS factor_lj, e.g. Amber 0/0/0.5) are applied by
///   the exception pass; a non-finite scale is rejected.
/// - 4-atom chain with `(0,0,0)` exclusion yields zero LJ energy (ac-003).
/// - `(1,0,0)` keeps the three bond pairs and drops 1-3 / 1-4 (ac-004).
/// - a dihedral-free chain still drops the bond-graph 1-4 (PEGM has angles
///   and no dihedrals; the CSR comes from build_special_list, not from a
///   dihedral endpoint).
/// - the no-exclusion fast path (no `set_special_bonds` call) keeps existing
///   LJ-fluid behaviour bit-identical — covered by the existing test_cuda_md
///   NVE regression suite.

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

/// Build a 4-atom collinear chain frame, sigma-spaced (so LJ generates
/// well-defined non-zero energy when exclusions are off). Bonds: 0-1,
/// 1-2, 2-3; angles: 0-1-2, 1-2-3; dihedral: 0-1-2-3.
cpu::Frame make_chain4() {
    cpu::Frame frame(4);
    {
        auto x = frame.span(kAtom_x);
        auto y = frame.span(kAtom_y);
        auto z = frame.span(kAtom_z);
        auto vx = frame.span(kAtom_vx);
        auto vy = frame.span(kAtom_vy);
        auto vz = frame.span(kAtom_vz);
        auto fx = frame.span(kAtom_fx);
        auto fy = frame.span(kAtom_fy);
        auto fz = frame.span(kAtom_fz);
        auto mass = frame.span(kAtom_mass);
        auto type = frame.span(kAtom_type);
        auto tag = frame.span(kAtom_tag);
        for (int i = 0; i < 4; ++i) {
            x[i] = double(i) * 1.0;
            y[i] = 0.0;
            z[i] = 0.0;
            vx[i] = vy[i] = vz[i] = 0.0;
            fx[i] = fy[i] = fz[i] = 0.0;
            mass[i] = 1.0;
            type[i] = 0;
            tag[i] = i;
        }
    }
    frame.box = cpu::Box::cubic(20.0);
    // Per-atom topology + 1-2/1-3/1-4 special-neighbor exclusion list.
    // build_special_list() does the BFS over the per-atom bond graph that
    // populates nspecial/special — the source of the LJ exclusion CSR.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_bond(0, 1, 0); // bonds 0-1, 1-2, 2-3
    frame.add_bond(1, 2, 0);
    frame.add_bond(2, 3, 0);
    frame.add_angle(0, 1, 2, 0); // angles 0-1-2, 1-2-3
    frame.add_angle(1, 2, 3, 0);
    frame.add_dihedral(0, 1, 2, 3, 0); // dihedral 0-1-2-3
    frame.build_special_list();
    return frame;
}

/// Square 4-bead chain with no dihedral. Bonds 0-1, 1-2, 2-3 are each
/// length 1. Atom 0 and atom 3 are also 1 apart, but that contact is a
/// bond-graph 1-4 (path 0-1-2-3), not a bond, angle endpoint, or dihedral.
cpu::Frame make_folded_chain4() {
    cpu::Frame frame(4);
    {
        auto x = frame.span(kAtom_x);
        auto y = frame.span(kAtom_y);
        auto z = frame.span(kAtom_z);
        auto vx = frame.span(kAtom_vx);
        auto vy = frame.span(kAtom_vy);
        auto vz = frame.span(kAtom_vz);
        auto fx = frame.span(kAtom_fx);
        auto fy = frame.span(kAtom_fy);
        auto fz = frame.span(kAtom_fz);
        auto mass = frame.span(kAtom_mass);
        auto type = frame.span(kAtom_type);
        auto tag = frame.span(kAtom_tag);
        const double xy[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int i = 0; i < 4; ++i) {
            x[i] = xy[i][0];
            y[i] = xy[i][1];
            z[i] = 0.0;
            vx[i] = vy[i] = vz[i] = 0.0;
            fx[i] = fy[i] = fz[i] = 0.0;
            mass[i] = 1.0;
            type[i] = 0;
            tag[i] = i;
        }
    }
    frame.box = cpu::Box::cubic(20.0);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    frame.add_bond(0, 1, 0);
    frame.add_bond(1, 2, 0);
    frame.add_bond(2, 3, 0);
    frame.add_angle(0, 1, 2, 0);
    frame.add_angle(1, 2, 3, 0);
    frame.build_special_list();
    return frame;
}

} // namespace

TEST(CudaPairLJSpecialBonds, ApiAcceptsBinaryScales) {
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(1.0, 1.0, 1.0));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 0.0, 0.0));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(1.0, 0.0, 0.0));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 1.0, 0.0));
}

TEST(CudaPairLJSpecialBonds, ApiAcceptsFractionalRejectsNonFinite) {
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.5, 0.5, 0.5));
    EXPECT_NO_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 0.0, 0.5));
    EXPECT_THROW(
        cuda::PairLJCut(1.0, 1.0, 3.0).set_special_bonds(0.0, 0.0, NAN),
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
    sim.set_evaluator(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(frame, 1);
    // 4 atoms in a 4-chain: every pair (0-1, 0-2, 0-3, 1-2, 1-3, 2-3) is
    // either 1-2 (bonds), 1-3 (angle endpoints) or 1-4 (dihedral endpoints).
    // With (0,0,0) the LJ kernel must skip every pair → energy_pair == 0.
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), 0.0, 1e-9);
}

TEST(CudaPairLJSpecialBonds, FourAtomChainKeepsBondPairsOnly) {
    auto frame = make_chain4();
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.set_evaluator(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(1.0, 0.0, 0.0));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    // Zero steps: the initial force pass reports the energy of this
    // geometry. One step would move the strained bonds (|F| = 24 eV/A) and
    // shift the energy by ~6e-6 eV.
    sim.run(frame, 0);
    // Three bonds at r = 1. 1-3 (r = 2) and 1-4 (r = 3) are scale 0.
    auto lj = [](double r) {
        const double sr6 = std::pow(1.0 / r, 6.0);
        return 4.0 * (sr6 * sr6 - sr6);
    };
    const double expected = 3.0 * (lj(1.0) - lj(cutoff));
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), expected, 1e-8);
}

// PEGM: bonds + angles, no dihedrals. The 0–3 contact is inside the cutoff
// and is a 1-4 only because build_special_list walked the bond graph.
// Baking exclusions from dihedral endpoints leaves this pair at full LJ.
TEST(CudaPairLJSpecialBonds, FoldedChainWithoutDihedralDrops14) {
    auto frame = make_folded_chain4();
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.set_evaluator(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(1.0, 0.0, 0.0));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(frame, 0); // energy of this geometry, as above
    // Three bonds at r = 1. The 1-4 (atom 0, atom 3) is also at r = 1 and
    // must not contribute. 1-3 pairs sit at r = sqrt(2) and are scale 0.
    auto lj = [](double r) {
        const double sr6 = std::pow(1.0 / r, 6.0);
        return 4.0 * (sr6 * sr6 - sr6);
    };
    const double expected = 3.0 * (lj(1.0) - lj(cutoff));
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), expected, 1e-8);
}

TEST(CudaPairLJSpecialBonds, FoldedChainWithoutDihedralAllExcluded) {
    auto frame = make_folded_chain4();
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.set_evaluator(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(0.0, 0.0, 0.0));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(frame, 1);
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), 0.0, 1e-9);
}

// Amber (0, 0, 0.5) on the folded chain: bonds and 1-3 pairs drop, the
// bond-graph 1-4 contact (atoms 0 and 3, r = 1) counts at half weight
// through the exception pass.
TEST(CudaPairLJSpecialBonds, FoldedChainAmberHalvesThe14) {
    auto frame = make_folded_chain4();
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.set_evaluator(
        cuda::PairLJCut{1.0, 1.0, cutoff}.set_special_bonds(0.0, 0.0, 0.5));
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.build();
    sim.run(frame, 0); // energy of this geometry
    auto lj = [](double r) {
        const double sr6 = std::pow(1.0 / r, 6.0);
        return 4.0 * (sr6 * sr6 - sr6);
    };
    const double expected = 0.5 * (lj(1.0) - lj(cutoff));
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), expected, 1e-8);
}
