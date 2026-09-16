/// @file tests/cuda/md/test_multi_gpu.cu
/// @brief cuda::md::MultiGpu — partition a global frame across GPUs + gather
///        back; atom conservation and position round-trip.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/communication/multi_gpu.cuh"
#include "atomiverse/cuda/longrange/kspace.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"

using namespace ATV;

namespace {

// ── Stub force families for the factory-dispatch test (force-infra-01 P1.5) ──
// Minimal cuda::Pair / cuda::KSpace subclasses that do no physics, only count
// how many times the engine dispatches their compute() — proving MultiGpu drives
// arbitrary force families polymorphically through the factory seam.
struct StubLaunchCounts {
    static int pair;
    static int kspace;
    static void reset() {
        pair = 0;
        kspace = 0;
    }
};
int StubLaunchCounts::pair = 0;
int StubLaunchCounts::kspace = 0;

class StubPair : public cuda::Pair {
  public:
    StubPair() : cuda::Pair(1.0) {}
    std::string name() const override { return "stub-pair"; }

  protected:
    void compute(const cuda::LaunchContext&) override {
        ++StubLaunchCounts::pair;
    }
};

class StubKSpace : public cuda::KSpace {
  public:
    StubKSpace() = default;
    std::string name() const override { return "stub-kspace"; }
    cuda::LongRangeContract long_range_contract() const override {
        return {cuda::LongRangeRole::ReciprocalSpace, AccReal(0.3), AccReal(0),
                AccReal(0)};
    }

  protected:
    void compute(const cuda::LaunchContext&) override {
        ++StubLaunchCounts::kspace;
    }
};

// 4x4x4 grid of atoms in a cubic box, deterministic tags 0..63.
cpu::Frame make_grid(double L) {
    const int s = 4, n = s * s * s;
    cpu::Frame f(n);
    f.box.set_cell(Mat3::cubic(L));
    double sp = L / s;
    int idx = 0;
    for (int ix = 0; ix < s; ++ix)
        for (int iy = 0; iy < s; ++iy)
            for (int iz = 0; iz < s; ++iz) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * sp;
                f.span(kAtom_y)[idx] = (iy + 0.5) * sp;
                f.span(kAtom_z)[idx] = (iz + 0.5) * sp;
                f.span(kAtom_vx)[idx] = 0.01 * idx;
                f.span(kAtom_mass)[idx] = 1.0;
                f.span(kAtom_type)[idx] = 0;
                f.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    return f;
}

// 4x4x4 grid + bonds & angles along each x-row (x-adjacent atoms). Bonds run
// along x, so any x-direction domain split makes some of them cross-GPU — the
// case the re-materialization tag→local resolve must handle.
cpu::Frame make_bonded_grid(double L) {
    cpu::Frame f = make_grid(L);
    const int s = 4;
    // add_bond/add_angle resolve tags through the tag→local map; build it first.
    f.rebuild_tag_index(0, f.nlocal());
    auto tag_of = [&](int ix, int iy, int iz) {
        return (ix * s + iy) * s + iz;
    };
    for (int iy = 0; iy < s; ++iy)
        for (int iz = 0; iz < s; ++iz) {
            for (int ix = 0; ix < s - 1; ++ix)
                f.add_bond(tag_of(ix, iy, iz), tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 1; ++ix)
                f.add_angle(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                            tag_of(ix + 1, iy, iz), 0);
            for (int ix = 1; ix < s - 2; ++ix)
                f.add_dihedral(tag_of(ix - 1, iy, iz), tag_of(ix, iy, iz),
                               tag_of(ix + 1, iy, iz), tag_of(ix + 2, iy, iz),
                               0);
        }
    f.build_special_list();
    return f;
}
} // namespace

TEST(MultiGpu, PartitionConservesAtoms) {
    auto g = make_grid(12.0);
    cuda::md::MultiGpu mg;
    mg.partition(g);
    int total = 0;
    for (int s = 0; s < mg.count(); ++s)
        total += mg.n_local(s);
    EXPECT_EQ(total, g.nlocal()); // every atom owned by exactly one GPU
}

// multigpu-bonded P2.4/P2.5: after partition + halo, the per-subdomain flat
// bond lists re-materialized from per-atom topology (partner tags resolved to
// local/ghost rows) must, mapped back to global tags, exactly reproduce the
// single global bond set — each bond owned by exactly one GPU, none dropped.
TEST(MultiGpu, PartitionRematerializesBondedTopology) {
    cuda::md::MultiGpu mg;
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs for cross-boundary bonds";
    auto g = make_bonded_grid(12.0); // x-bond length = spacing 3.0
    mg.partition(g);
    mg.build_halo(3.5); // ghost shell covers the 3.0 cross-split bond
    mg.rebuild_bonded_tuples();

    // Reference: every bond once, from the global owner-filtered per-atom store.
    std::set<std::pair<long long, long long>> ref;
    for (int i = 0; i < g.nlocal(); ++i)
        for (int m = 0; m < g.span(kAtom_num_bond)[i]; ++m) {
            long long a = g.span(kAtom_tag)[i],
                      b = g.span(kAtom_bond_atom)(i, m);
            ref.insert({std::min(a, b), std::max(a, b)});
        }
    ASSERT_FALSE(ref.empty());

    // Union over GPUs of re-materialized bonds, local rows mapped back to tags.
    std::set<std::pair<long long, long long>> got;
    for (int s = 0; s < mg.count(); ++s) {
        const auto& F = mg.frame(s);
        const int nb = F.n_bonds();
        const int nall = mg.n_local(s) + mg.n_ghost(s);
        if (nb == 0)
            continue;
        std::vector<int> ai(nb), aj(nb);
        std::vector<long long> tags(nall);
        // UVA (cudaMemcpyDefault) infers the owning device — no set_device.
        cudaMemcpy(ai.data(), F.bonds.atomi, nb * sizeof(int),
                   cudaMemcpyDefault);
        cudaMemcpy(aj.data(), F.bonds.atomj, nb * sizeof(int),
                   cudaMemcpyDefault);
        cudaMemcpy(tags.data(), F.atoms.tag, nall * sizeof(long long),
                   cudaMemcpyDefault);
        for (int k = 0; k < nb; ++k) {
            long long a = tags[ai[k]], b = tags[aj[k]];
            got.insert({std::min(a, b), std::max(a, b)});
        }
    }
    // Diagnostics: report set sizes + a few mismatches before the hard check.
    std::cerr << "[diag] ref=" << ref.size() << " got=" << got.size() << "\n";
    int shown = 0;
    for (const auto& p : ref)
        if (!got.count(p) && shown++ < 8)
            std::cerr << "[diag] in ref, MISSING from got: (" << p.first << ","
                      << p.second << ")\n";
    shown = 0;
    for (const auto& p : got)
        if (!ref.count(p) && shown++ < 8)
            std::cerr << "[diag] in got, EXTRA vs ref: (" << p.first << ","
                      << p.second << ")\n";
    EXPECT_EQ(got, ref); // no dropped cross-GPU bonds, no duplicates
}

// multigpu-bonded P2.3: per-atom bonded topology must travel with an atom when
// migrate() re-homes it to another GPU. Drive migrations with a rigid x-drift
// (every atom the same velocity → all bond lengths preserved, so partners stay
// inside the halo) under negligible LJ (tiny eps) so motion is ~ballistic;
// reneigh_every=1 migrates + re-materializes every step. The re-materialized
// bond set (by global tags) must remain the full global set — any atom whose
// topology was lost in migration would drop its bonds.
TEST(MultiGpu, MigratePreservesBondedTopology) {
    cuda::md::MultiGpu mg;
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs to migrate between subdomains";
    auto g = make_bonded_grid(12.0);
    for (int i = 0; i < g.nlocal(); ++i) {
        g.span(kAtom_vx)[i] = 2.0; // rigid translation along x
        g.span(kAtom_vy)[i] = 0.0;
        g.span(kAtom_vz)[i] = 0.0;
    }
    std::set<std::pair<long long, long long>> ref;
    for (int i = 0; i < g.nlocal(); ++i)
        for (int m = 0; m < g.span(kAtom_num_bond)[i]; ++m) {
            long long a = g.span(kAtom_tag)[i],
                      b = g.span(kAtom_bond_atom)(i, m);
            ref.insert({std::min(a, b), std::max(a, b)});
        }
    ASSERT_FALSE(ref.empty());

    mg.partition(g);
    mg.setup_force_engine(3.5, 1e-12,
                          1.0); // halo 3.5 covers the 3.0 bond; eps≈0
    // ~4 A of rigid drift over 20 steps → atoms cross subdomain boundaries;
    // migrate (carrying topology) runs every step (reneigh_every=1).
    mg.run_nve_nlist(20, 0.1, 3.5, 1e-12, 1.0, 0.5, 1);

    // Validate TRANSPORT fidelity: reconstruct the bond set straight from each
    // owned atom's migrated per-atom SoA (num_bond + bond_atom GLOBAL TAGS) — no
    // tag→local resolve, so this is independent of the halo/PBC (the resolve
    // path is covered by the static Partition test). If migrate lost or
    // corrupted any atom's topology, a bond would be missing or wrong here.
    std::set<std::pair<long long, long long>> got;
    for (int s = 0; s < mg.count(); ++s) {
        const int nl = mg.n_local(s);
        if (nl == 0)
            continue;
        const auto& F = mg.frame(s);
        const int MB = F.atoms.bond_per_atom; // runtime stride (per frame)
        std::vector<int> nbv(nl);
        std::vector<long long> ba(static_cast<std::size_t>(nl) * MB), tags(nl);
        cudaMemcpy(nbv.data(), F.atoms.num_bond, nl * sizeof(int),
                   cudaMemcpyDefault);
        cudaMemcpy(ba.data(), F.atoms.bond_atom,
                   static_cast<std::size_t>(nl) * MB * sizeof(long long),
                   cudaMemcpyDefault);
        cudaMemcpy(tags.data(), F.atoms.tag, nl * sizeof(long long),
                   cudaMemcpyDefault);
        for (int k = 0; k < nl; ++k)
            for (int m = 0; m < nbv[k]; ++m) {
                long long a = tags[k],
                          b = ba[static_cast<std::size_t>(k) * MB + m];
                got.insert({std::min(a, b), std::max(a, b)});
            }
    }
    std::cerr << "[diag] migrate: ref=" << ref.size() << " got=" << got.size()
              << "\n";
    EXPECT_EQ(got, ref); // per-atom bonded topology preserved across migration
}

// multigpu-bonded-03/04: cross-GPU BondHarmonic + AngleHarmonic +
// DihedralMultiHarmonic. The total bonded force on each owned atom (and on ghost
// partners, then reverse_comm'd home) plus the total bonded energy must match a
// single-GPU run. Cross-GPU bonds/angles/dihedrals exercise the ghost-deposit +
// reverse_comm path for all three tuple types.
TEST(MultiGpu, BondForceParityVsSingleGpu) {
    cuda::md::MultiGpu mg; // all visible GPUs
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs for cross-boundary bonds";
    auto g = make_bonded_grid(12.0);             // x-bond length 3.0
    const std::vector<double> K{100.0}, r0{2.0}; // stretched (force nonzero)
    const std::vector<double> aK{50.0}, aT{109.5};
    const std::vector<std::array<double, 5>> dC{{1.0, 0.5, -0.3, 0.2, 0.1}};

    auto run = [&](cuda::md::MultiGpu& m, cpu::Frame& out, double& e) {
        m.partition(g);
        m.setup_force_engine(3.5, 0.0,
                             1.0); // ctx box/stream; LJ unused (eps 0)
        m.setup_bonded(K, r0, aK, aT, dC);
        m.build_halo(3.5);         // ghosts cover the 3.0 bond
        m.rebuild_bonded_tuples(); // resolve partners to local/ghost rows
        m.compute_bonded_forces(); // zero → bond (owned+ghost) → reverse_comm
        m.gather_forces(out);      // owned bond forces by tag (out pre-sized)
        e = m.bonded_energy();
    };

    cpu::Frame fmulti(g.nlocal()), fref(g.nlocal());
    double emulti = 0, eref = 0;
    run(mg, fmulti, emulti);
    {
        cuda::md::MultiGpu mg1(1); // single-GPU reference (no ghosts/reverse)
        run(mg1, fref, eref);
    }

    std::cerr << "[diag] bond E: multi=" << emulti << " ref=" << eref << "\n";
    EXPECT_NEAR(emulti, eref, 1e-3 * std::abs(eref) + 1e-3);
    for (int i = 0; i < g.nlocal(); ++i) {
        EXPECT_NEAR(fmulti.span(kAtom_fx)[i], fref.span(kAtom_fx)[i], 0.2)
            << "fx atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fy)[i], fref.span(kAtom_fy)[i], 0.2)
            << "fy atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fz)[i], fref.span(kAtom_fz)[i], 0.2)
            << "fz atom " << i;
    }
}

// multigpu-bonded-04 P4.3/P4.4: full PE = LJ (special-bonds excluded) + bonded.
// The combined per-atom force and total energy on 2/4 GPUs must match a 1-GPU
// run — exercises cross-GPU special-bonds exclusion (1-2 partner is a ghost on
// the owner) together with the cross-GPU bonded forces.
TEST(MultiGpu, FullPotentialParityVsSingleGpu) {
    cuda::md::MultiGpu mg;
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs for cross-boundary topology";
    auto g = make_bonded_grid(12.0);
    const std::vector<double> K{100.0}, r0{2.0};
    const std::vector<double> aK{50.0}, aT{109.5};
    const std::vector<std::array<double, 5>> dC{{1.0, 0.5, -0.3, 0.2, 0.1}};

    auto run = [&](cuda::md::MultiGpu& m, cpu::Frame& out, double& e) {
        m.partition(g);
        m.setup_force_engine(5.0, 0.1,
                             2.0); // real LJ; halo 5.0+skin covers bonds
        m.setup_bonded(K, r0, aK, aT, dC);
        m.enable_special_bonds();
        m.reneighbor();     // halo + re-materialize + exclusion CSR + nlist
        m.compute_forces(); // LJ (excluded) + bonded, single reverse_comm
        m.gather_forces(out);
        e = m.bonded_energy(); // total PE here
    };

    cpu::Frame fmulti(g.nlocal()), fref(g.nlocal());
    double emulti = 0, eref = 0;
    run(mg, fmulti, emulti);
    {
        cuda::md::MultiGpu mg1(1);
        run(mg1, fref, eref);
    }

    std::cerr << "[diag] full PE: multi=" << emulti << " ref=" << eref << "\n";
    EXPECT_NEAR(emulti, eref, 1e-3 * std::abs(eref) + 1e-3);
    for (int i = 0; i < g.nlocal(); ++i) {
        EXPECT_NEAR(fmulti.span(kAtom_fx)[i], fref.span(kAtom_fx)[i], 0.5)
            << "fx atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fy)[i], fref.span(kAtom_fy)[i], 0.5)
            << "fy atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fz)[i], fref.span(kAtom_fz)[i], 0.5)
            << "fz atom " << i;
    }
}

// Owned-row pair gating: ghost rows must carry ONLY bonded deposits when
// reverse_comm ships them home. With zero-strength bonds the total energy is
// pure LJ and every cross-boundary pair is live, exposing the historical
// double-count (ghost-row LJ forces scatter-added onto owners ≈ 2x boundary
// force; cross-boundary pair energy counted on both GPUs). The bonded-grid
// FullPotentialParity test cannot see this — its bond energy dominates the
// 1e-3 relative band — so this one pins the LJ-only parity TIGHT.
TEST(MultiGpu, LJOwnedRowParityVsSingleGpu) {
    cuda::md::MultiGpu mg;
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs for cross-boundary pairs";
    auto g = make_bonded_grid(12.0);
    const std::vector<double> K{0.0}, r0{2.0}; // zero bonds ⇒ energy is pure LJ

    auto run = [&](cuda::md::MultiGpu& m, cpu::Frame& out, double& e) {
        m.partition(g);
        m.setup_force_engine(5.0, 0.1,
                             2.0); // spacing-3.0 grid: many live pairs
        m.setup_bonded(K, r0);     // no special bonds: LJ unexcluded
        m.reneighbor();
        m.compute_forces(); // LJ owned-rows + (zero) bonded + reverse_comm
        m.gather_forces(out);
        e = m.bonded_energy(); // total PE — here pure LJ
    };

    cpu::Frame fmulti(g.nlocal()), fref(g.nlocal());
    double emulti = 0, eref = 0;
    run(mg, fmulti, emulti);
    {
        cuda::md::MultiGpu mg1(1); // no ghosts → trusted reference
        run(mg1, fref, eref);
    }

    std::cerr << "[diag] pure-LJ PE: multi=" << emulti << " ref=" << eref
              << "\n";
    ASSERT_NE(eref, 0.0);
    EXPECT_NEAR(emulti, eref, 1e-5 * std::abs(eref));
    for (int i = 0; i < g.nlocal(); ++i) {
        EXPECT_NEAR(fmulti.span(kAtom_fx)[i], fref.span(kAtom_fx)[i], 2e-3)
            << "fx atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fy)[i], fref.span(kAtom_fy)[i], 2e-3)
            << "fy atom " << i;
        EXPECT_NEAR(fmulti.span(kAtom_fz)[i], fref.span(kAtom_fz)[i], 2e-3)
            << "fz atom " << i;
    }
}

// force-infra-01 P1.5: the engine composes force families by abstract base via
// caller-injected factories. Inject a stub cuda::Pair and a stub cuda::KSpace
// and assert compute_forces() dispatches each once per subdomain — proving
// MultiGpu is potential-agnostic (no hardcoded PairLJCut, and a KSpace slot).
TEST(MultiGpu, ForceFamilyFactoryDispatch) {
    cuda::md::MultiGpu mg; // works on any GPU count (single-GPU included)
    auto g = make_bonded_grid(12.0);
    const std::vector<double> K{100.0}, r0{2.0};

    mg.partition(g);
    // Custom pair factory (stub) instead of the default PairLJCut.
    mg.setup_force_engine(3.5, 0.1, 2.0, -1.0,
                          [] { return std::make_unique<StubPair>(); });
    // Optional reciprocal family: a stub KSpace per subdomain.
    mg.set_kspace([] { return std::make_unique<StubKSpace>(); });
    mg.setup_bonded(K,
                    r0); // real harmonic bond (exercises the rest of the path)
    mg.reneighbor();

    StubLaunchCounts::reset();
    mg.compute_forces();

    // One stub-pair and one stub-kspace launch per subdomain.
    EXPECT_EQ(StubLaunchCounts::pair, mg.count());
    EXPECT_EQ(StubLaunchCounts::kspace, mg.count());
}

// multigpu-bonded-05 P5.4: per-step hot-path timing (forward_comm + the combined
// LJ+bonded compute_forces). Disabled by default; run with
// --gtest_also_run_disabled_tests --gtest_filter=*HotPathTiming*. Reports µs/step
// so the P2P-batching before/after can be compared across engine versions.
TEST(MultiGpu, DISABLED_HotPathTiming) {
    cuda::md::MultiGpu mg;
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >=2 GPUs";
    auto g = make_bonded_grid(12.0);
    const std::vector<double> K{100.0}, r0{2.0}, aK{50.0}, aT{109.5};
    const std::vector<std::array<double, 5>> dC{{1.0, 0.5, -0.3, 0.2, 0.1}};
    mg.partition(g);
    mg.setup_force_engine(5.0, 0.1, 2.0);
    mg.setup_bonded(K, r0, aK, aT, dC);
    mg.enable_special_bonds();
    mg.reneighbor();
    const int M = 2000;
    for (int i = 0; i < 100; ++i) { // warmup
        mg.forward_comm();
        mg.compute_forces();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        mg.forward_comm();   // per-step position halo (forward, packed)
        mg.compute_forces(); // LJ + bonded + reverse halo (packed)
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / M;
    std::cerr << "[timing] forward_comm+compute_forces: " << us << " us/step ("
              << mg.count() << " GPUs)\n";
}

TEST(MultiGpu, PartitionGatherRoundTrip) {
    auto g = make_grid(12.0);
    cuda::md::MultiGpu mg;
    mg.partition(g);

    cpu::Frame out(g.nlocal());
    for (int i = 0; i < g.nlocal(); ++i)
        out.span(kAtom_tag)[i] = i;
    mg.gather(out);

    // cuda::Frame stores per-atom data as real_t (float), so the
    // partition->materialize->gather round trip is exact only to float
    // precision (e.g. vx = 0.01*idx is not representable in float).
    for (int i = 0; i < g.nlocal(); ++i) {
        EXPECT_NEAR(out.span(kAtom_x)[i], g.span(kAtom_x)[i], 1e-5)
            << "atom " << i;
        EXPECT_NEAR(out.span(kAtom_y)[i], g.span(kAtom_y)[i], 1e-5);
        EXPECT_NEAR(out.span(kAtom_z)[i], g.span(kAtom_z)[i], 1e-5);
        EXPECT_NEAR(out.span(kAtom_vx)[i], g.span(kAtom_vx)[i], 1e-5);
    }
}

TEST(MultiGpu, HaloExchangeXSplit) {
    // 2-GPU x-split (dims {2,1,1}); a single forward halo pass is complete.
    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";

    auto g = make_grid(12.0); // grid coords 1.5/4.5/7.5/10.5
    mg.partition(g);          // creates the SpatialGrid
    ASSERT_EQ(mg.domain().dims(),
              (std::array<int, 3>{2, 1, 1})); // boundary at x = 6
    mg.build_halo(3.0);                       // cutoff 3.0

    // Each slab owns 32 atoms; boundary atoms within 3.0 of x=0/6/12 are sent
    // both ways, so each side receives the other's near-boundary atoms.
    EXPECT_GT(mg.n_ghost(0), 0);
    EXPECT_EQ(mg.n_ghost(0), mg.n_ghost(1)); // symmetric exchange

    // Ghosts on slab 0 must be slab-1 atoms (x in [6,12)); on slab 1, x in [0,6).
    for (int slot = 0; slot < 2; ++slot) {
        const auto& f = mg.frame(slot);
        int ng = mg.n_ghost(slot);
        std::vector<float> gx(ng);
        cudaSetDevice(mg.devices().device_id(slot));
        cudaMemcpy(gx.data(), f.atoms.x + mg.n_local(slot), ng * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int k = 0; k < ng; ++k) {
            if (slot == 0)
                EXPECT_GE(gx[k], 6.0f) << "slab0 ghost should be slab1 atom";
            else
                EXPECT_LT(gx[k], 6.0f) << "slab1 ghost should be slab0 atom";
        }
    }
}

namespace {
// Shift the x-coordinate of every owned atom by `delta` (test helper for
// forward_comm: emulates one integration step moving the owners).
__global__ void shift_owned_x(ATV::cuda::real_t* x, int n_local,
                              ATV::cuda::real_t delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_local)
        x[i] += delta;
}
} // namespace

TEST(MultiGpu, ForwardCommRefreshesGhostPositions) {
    // build_halo establishes the ghost SET + a gather plan; forward_comm then
    // refreshes only ghost POSITIONS from the current owners. Moving every
    // owned atom by a fixed delta in x must move every ghost by the same delta
    // (a ghost is a copy of some owner), with y/z and the ghost count unchanged.
    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    auto g = make_grid(12.0);
    mg.partition(g);
    mg.build_halo(3.0);

    auto read_ghosts = [&](int slot, std::vector<float>& gx,
                           std::vector<float>& gy, std::vector<float>& gz) {
        int ng = mg.n_ghost(slot);
        gx.resize(ng);
        gy.resize(ng);
        gz.resize(ng);
        const auto& f = mg.frame(slot);
        int off = mg.n_local(slot);
        cudaSetDevice(mg.devices().device_id(slot));
        cudaMemcpy(gx.data(), f.atoms.x + off, ng * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gy.data(), f.atoms.y + off, ng * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gz.data(), f.atoms.z + off, ng * sizeof(float),
                   cudaMemcpyDeviceToHost);
    };

    std::vector<std::vector<float>> bx(2), by(2), bz(2);
    for (int s = 0; s < 2; ++s)
        read_ghosts(s, bx[s], by[s], bz[s]);

    const float delta = 0.123f;
    for (int s = 0; s < 2; ++s) {
        int nl = mg.n_local(s);
        cudaSetDevice(mg.devices().device_id(s));
        int block = 128, grid = (nl + block - 1) / block;
        shift_owned_x<<<grid, block>>>(mg.frame(s).atoms.x, nl, delta);
    }
    mg.forward_comm();

    for (int s = 0; s < 2; ++s) {
        std::vector<float> ax, ay, az;
        read_ghosts(s, ax, ay, az);
        ASSERT_EQ(ax.size(), bx[s].size())
            << "ghost count changed on slot " << s;
        for (size_t k = 0; k < ax.size(); ++k) {
            EXPECT_NEAR(ax[k], bx[s][k] + delta, 1e-5)
                << "slot " << s << " k " << k;
            EXPECT_NEAR(ay[k], by[s][k], 1e-5);
            EXPECT_NEAR(az[k], bz[s][k], 1e-5);
        }
    }
}

TEST(MultiGpu, LjForceParityVsSingleGpu) {
    // The crux of multi-GPU correctness: an owned atom must feel the SAME LJ
    // force whether the system runs on one GPU or is split across GPUs with a
    // halo. Same brute-force kernel both ways; only the atom set in each owned
    // atom's loop differs (single: all; multi: owned+ghost). With halo cutoff
    // == LJ cutoff, the ghost set captures every cross-boundary neighbor, so
    // the per-atom force sums are identical (up to float summation order).
    auto g = make_grid(12.0); // spacing 3.0
    const int n = g.nlocal();
    // Break the perfect-lattice symmetry (else neighbor forces cancel to ~0
    // and the parity check is vacuous). Deterministic per-atom jitter.
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
    }
    const double cut = 3.5, eps = 1.0, sig = 2.5;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1); // whole system on one GPU
        mg1.partition(g);
        mg1.build_halo(cut); // no ghosts (self-neighbors skipped)
        mg1.compute_lj_bruteforce(cut, eps, sig);
        mg1.gather_forces(ref);
    }

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g);
    mg.build_halo(cut);
    mg.compute_lj_bruteforce(cut, eps, sig);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather_forces(mf);

    double maxabs = 0;
    for (int i = 0; i < n; ++i)
        maxabs = std::max(maxabs, std::abs(ref.span(kAtom_fx)[i]));
    EXPECT_GT(maxabs, 1e-3) << "reference forces should be nonzero";
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_fx)[i], ref.span(kAtom_fx)[i], 1e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fy)[i], ref.span(kAtom_fy)[i], 1e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fz)[i], ref.span(kAtom_fz)[i], 1e-3)
            << "atom " << i;
    }
}

namespace {
// s x s x s grid in a cubic box of edge L, deterministic tags.
cpu::Frame make_grid_n(int s, double L) {
    const int n = s * s * s;
    cpu::Frame f(n);
    f.box.set_cell(Mat3::cubic(L));
    double sp = L / s;
    int idx = 0;
    for (int ix = 0; ix < s; ++ix)
        for (int iy = 0; iy < s; ++iy)
            for (int iz = 0; iz < s; ++iz) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * sp;
                f.span(kAtom_y)[idx] = (iy + 0.5) * sp;
                f.span(kAtom_z)[idx] = (iz + 0.5) * sp;
                f.span(kAtom_mass)[idx] = 1.0;
                f.span(kAtom_type)[idx] = 0;
                f.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    return f;
}
} // namespace

TEST(MultiGpu, LjForceParityDims2Overlap) {
    // Regression for the dims==2 double-send bug: along an axis split into
    // exactly 2 subdomains, both faces point to the SAME neighbor (periodic
    // self-pair). An atom in the overlap band (within cutoff of BOTH shared
    // boundaries) used to be sent TWICE, and the min-image force kernel then
    // counted it twice -> doubled force. Here spacing 2.0 with cutoff 4.5 puts
    // many atoms in the overlap band AND makes them genuine cross-boundary
    // neighbors, so the bug would break parity. The dedup (mode=2 single send)
    // restores it.
    auto g =
        make_grid_n(6, 16.0); // box/2 = 8; slab width 8, overlap band [3.5,4.5]
    const int n = g.nlocal(); // spacing 2.667 > sigma -> gentle forces
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.3 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.3 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.3 * std::sin(1.9 * i + 0.3);
    }
    const double cut = 4.5, eps = 1.0, sig = 2.5;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 4.0); // generous ghost headroom for the dense band
        mg1.build_halo(cut);
        mg1.compute_lj_bruteforce(cut, eps, sig);
        mg1.gather_forces(ref);
    }

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g, 4.0);
    ASSERT_EQ(mg.domain().dims(), (std::array<int, 3>{2, 1, 1}));
    mg.build_halo(cut);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.compute_lj_bruteforce(cut, eps, sig);
    mg.gather_forces(mf);

    double maxabs = 0;
    for (int i = 0; i < n; ++i)
        maxabs = std::max(maxabs, std::abs(ref.span(kAtom_fx)[i]));
    EXPECT_GT(maxabs, 1e-3) << "reference forces should be nonzero";
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_fx)[i], ref.span(kAtom_fx)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fy)[i], ref.span(kAtom_fy)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fz)[i], ref.span(kAtom_fz)[i], 2e-3)
            << "atom " << i;
    }
}

TEST(MultiGpu, LjForceParityNlistVsSingleGpu) {
    // Stage C: the production force path uses the optimized atom-centric CLUSTER
    // neighbor-list kernel per subdomain instead of brute force. Validate that
    // multi-GPU cluster forces equal single-GPU cluster forces (same kernel,
    // so only float summation order differs -> tight tolerance). A looser
    // cross-check vs brute force confirms the cluster path is physically sane.
    auto g = make_grid_n(8, 24.0); // 512 atoms, spacing 3.0
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
    }
    const double cut = 5.0, eps = 1.0, sig = 2.5;

    // Reference: single-GPU cluster path.
    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.setup_force_engine(cut, eps, sig);
        mg1.reneighbor();
        mg1.compute_lj_nlist();
        mg1.gather_forces(ref);
    }
    // Sanity: single-GPU brute force should roughly agree with the cluster path.
    cpu::Frame bf(n);
    for (int i = 0; i < n; ++i)
        bf.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.build_halo(cut);
        mg1.compute_lj_bruteforce(cut, eps, sig);
        mg1.gather_forces(bf);
    }
    double maxabs = 0, maxdiff_bf = 0;
    for (int i = 0; i < n; ++i) {
        maxabs = std::max(maxabs, std::abs(ref.span(kAtom_fx)[i]));
        maxdiff_bf = std::max(
            maxdiff_bf, std::abs(ref.span(kAtom_fx)[i] - bf.span(kAtom_fx)[i]));
    }
    EXPECT_GT(maxabs, 1e-2) << "reference forces should be nonzero";
    EXPECT_LT(maxdiff_bf, 5e-2) << "cluster path should match brute force";

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g, 2.0);
    mg.setup_force_engine(cut, eps, sig);
    mg.reneighbor();
    mg.compute_lj_nlist();
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather_forces(mf);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_fx)[i], ref.span(kAtom_fx)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fy)[i], ref.span(kAtom_fy)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fz)[i], ref.span(kAtom_fz)[i], 2e-3)
            << "atom " << i;
    }
}

TEST(MultiGpu, LjForceParityVsSingleGpu4Way) {
    // The 4-GPU decomposition is 2x2x1, so atoms near the (x=6, y=6) edge have
    // neighbors owned by the DIAGONAL subdomain. A single forward halo pass
    // would never deliver those corner atoms; only the 3-axis cascade (x-pass
    // then y-pass forwarding x-ghosts) does. This test fails without the
    // cascade and passes with it — owned-atom LJ forces must match single-GPU.
    cuda::md::MultiGpu mg(4);
    if (mg.count() < 4)
        GTEST_SKIP() << "needs 4 GPUs";
    ASSERT_EQ(mg.devices().count(), 4);

    auto g = make_grid(12.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
    }
    const double cut = 3.5, eps = 1.0, sig = 2.5;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g);
        mg1.build_halo(cut);
        mg1.compute_lj_bruteforce(cut, eps, sig);
        mg1.gather_forces(ref);
    }

    mg.partition(g); // creates the SpatialGrid

    // Confirm the decomposition genuinely needs the cascade (>1 axis split).
    const auto dims = mg.domain().dims();
    int split_axes = (dims[0] > 1) + (dims[1] > 1) + (dims[2] > 1);
    ASSERT_GE(split_axes, 2) << "4-GPU layout should split >=2 axes";

    mg.build_halo(cut);
    mg.compute_lj_bruteforce(cut, eps, sig);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather_forces(mf);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_fx)[i], ref.span(kAtom_fx)[i], 1e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fy)[i], ref.span(kAtom_fy)[i], 1e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_fz)[i], ref.span(kAtom_fz)[i], 1e-3)
            << "atom " << i;
    }
}

TEST(MultiGpu, NveTrajectoryParity) {
    // Run a short NVE trajectory single-GPU vs split-across-GPUs and require the
    // owned positions to agree — validating the full step loop (halo + force +
    // integrate) end to end. Small dt / few steps so atoms stay in their slab
    // (no migration) and float chaos stays bounded.
    auto g = make_grid(12.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = 0.02 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.02 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.02 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 3.5, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 10;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g);
        mg1.run_nve(K, dt, cut, eps, sig);
        mg1.gather(ref);
    }

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g);
    mg.run_nve(K, dt, cut, eps, sig);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 5e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 5e-3);
        EXPECT_NEAR(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 5e-3);
    }
}

TEST(MultiGpu, NveTrajectoryParityNlist) {
    // End-to-end Verlet-cadence loop: single-GPU vs multi-GPU run_nve_nlist must
    // agree on owned positions after a short trajectory. Exercises reneighbor
    // (halo rebuild + neighbor-list rebuild) and forward_comm position refresh.
    auto g = make_grid_n(8, 24.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.4 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.4 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.4 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = 0.01 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.01 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.01 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 5.0, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 12, reneigh = 4;

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
        mg1.gather(ref);
    }

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g, 2.0);
    mg.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 5e-3)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 5e-3);
        EXPECT_NEAR(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 5e-3);
    }
}

TEST(MultiGpu, MigrateReassignsOwners) {
    // Shift every owned atom +6 in x (box 24, dims {2,1,1}, boundary x=12) so a
    // large fraction crosses into the neighbor subdomain, then migrate(). After
    // migration: (a) owned counts per subdomain match owner_of on the shifted
    // positions, (b) total owned atoms conserved, (c) every atom's full state
    // (shifted position + velocity, resolved by tag) is preserved.
    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    auto g = make_grid_n(8, 24.0);
    const int n = g.nlocal();
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_vx)[i] = 0.013 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.017 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.019 * std::sin(1.1 * i);
    }
    mg.partition(g, 2.0);
    ASSERT_EQ(mg.domain().dims(), (std::array<int, 3>{2, 1, 1}));

    const float delta = 6.0f;
    for (int s = 0; s < 2; ++s) {
        int nl = mg.n_local(s);
        cudaSetDevice(mg.devices().device_id(s));
        int block = 128, grid = (nl + block - 1) / block;
        shift_owned_x<<<grid, block>>>(mg.frame(s).atoms.x, nl, delta);
    }
    for (int s = 0; s < 2; ++s) {
        cudaSetDevice(mg.devices().device_id(s));
        cudaDeviceSynchronize();
    }

    mg.migrate();

    // (a) per-subdomain owned counts vs owner_of on the shifted positions.
    const auto& dd = mg.domain();
    std::vector<int> expect(2, 0);
    for (int i = 0; i < n; ++i)
        expect[dd.owner_of(g.span(kAtom_x)[i] + 6.0, g.span(kAtom_y)[i],
                           g.span(kAtom_z)[i])]++;
    int total = 0;
    for (int s = 0; s < 2; ++s) {
        EXPECT_EQ(mg.n_local(s), expect[s]) << "slab " << s;
        total += mg.n_local(s);
    }
    EXPECT_EQ(total, n) << "atoms conserved across migration";

    // (c) full state preserved, resolved by tag.
    cpu::Frame out(n);
    for (int i = 0; i < n; ++i)
        out.span(kAtom_tag)[i] = i;
    mg.gather(out);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out.span(kAtom_x)[i], g.span(kAtom_x)[i] + 6.0, 2e-3)
            << "atom " << i;
        EXPECT_NEAR(out.span(kAtom_y)[i], g.span(kAtom_y)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(out.span(kAtom_z)[i], g.span(kAtom_z)[i], 2e-3)
            << "atom " << i;
        EXPECT_NEAR(out.span(kAtom_vx)[i], g.span(kAtom_vx)[i], 1e-4)
            << "atom " << i;
        EXPECT_NEAR(out.span(kAtom_vz)[i], g.span(kAtom_vz)[i], 1e-4)
            << "atom " << i;
    }
}

TEST(MultiGpu, NveMigrationParity) {
    // Long-run correctness: a uniform +x drift (Galilean boost — leaves all
    // relative positions and forces unchanged) carries atoms across the x=12
    // boundary during the run, so migrate() fires repeatedly inside reneighbor.
    // Multi-GPU (partition + migrate) must still match single-GPU positions.
    auto g = make_grid_n(8, 24.0);
    const int n = g.nlocal();
    const double drift = 40.0;
    for (int i = 0; i < n; ++i) {
        g.span(kAtom_x)[i] += 0.3 * std::sin(0.7 * i + 0.1);
        g.span(kAtom_y)[i] += 0.3 * std::sin(1.3 * i + 0.2);
        g.span(kAtom_z)[i] += 0.3 * std::sin(1.9 * i + 0.3);
        g.span(kAtom_vx)[i] = drift + 0.01 * std::sin(0.5 * i);
        g.span(kAtom_vy)[i] = 0.01 * std::sin(0.9 * i);
        g.span(kAtom_vz)[i] = 0.01 * std::sin(1.1 * i);
        g.span(kAtom_mass)[i] = 1.0;
    }
    const double cut = 5.0, eps = 1.0, sig = 2.5, dt = 0.002;
    const int K = 20, reneigh = 4; // 20*0.002*40 = 1.6 -> atoms cross x=12

    cpu::Frame ref(n);
    for (int i = 0; i < n; ++i)
        ref.span(kAtom_tag)[i] = i;
    {
        cuda::md::MultiGpu mg1(1);
        mg1.partition(g, 2.0);
        mg1.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
        mg1.gather(ref);
    }

    cuda::md::MultiGpu mg(2);
    if (mg.count() < 2)
        GTEST_SKIP() << "needs >= 2 GPUs";
    mg.partition(g, 2.0);
    mg.run_nve_nlist(K, dt, cut, eps, sig, -1.0, reneigh);
    cpu::Frame mf(n);
    for (int i = 0; i < n; ++i)
        mf.span(kAtom_tag)[i] = i;
    mg.gather(mf);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(mf.span(kAtom_x)[i], ref.span(kAtom_x)[i], 2e-2)
            << "atom " << i;
        EXPECT_NEAR(mf.span(kAtom_y)[i], ref.span(kAtom_y)[i], 2e-2);
        EXPECT_NEAR(mf.span(kAtom_z)[i], ref.span(kAtom_z)[i], 2e-2);
    }
}

TEST(MultiGpu, EachAtomOnOwnerSlab) {
    auto g = make_grid(12.0);
    cuda::md::MultiGpu mg;
    mg.partition(g);
    if (mg.count() < 2)
        GTEST_SKIP() << "single GPU: trivially one slab";
    // Owner count per slab should match SpatialGrid::owner_of.
    const auto& dd = mg.domain();
    std::vector<int> expect(mg.count(), 0);
    for (int i = 0; i < g.nlocal(); ++i)
        expect[dd.owner_of(g.span(kAtom_x)[i], g.span(kAtom_y)[i],
                           g.span(kAtom_z)[i])]++;
    for (int s = 0; s < mg.count(); ++s)
        EXPECT_EQ(mg.n_local(s), expect[s]) << "slab " << s;
}
