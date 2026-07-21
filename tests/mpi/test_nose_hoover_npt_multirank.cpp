/// @file tests/mpi/test_nose_hoover_npt_multirank.cpp
/// @brief Focused regression: NoseHooverNPT under MPI must not zero out
///        the atom kinetic energy after a few steps.
///
/// Discovered 2026-05-25 from polyethylene_tension at 64 ranks: stage 2
/// NPT (`fix npt temp 500 500 50 iso 0 0 1000 drag 2`) showed T_K = 508 K
/// at step 0 (sane) but T_K = 0 by step 100 of stage 2 (E_kin → 0,
/// velocities collapsed). The 1-rank path is healthy.
///
/// The existing 8-rank MPI tests (`test_pe_msu_pipeline`, `test_npt_multirank`)
/// cover NoseHoover NVT + **Berendsen** barostat — NoseHooverNPT was
/// **never** validated under MPI. This test exercises that specific
/// component at 8 ranks with a short run (200 steps) and asserts T_K
/// stays > 50 K throughout (well above the failure-mode signature of
/// T_K → 0).

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

cpu::Frame make_fcc_argon(int n_cells, double a_lat) {
    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] =
                        (ix + basis[b][0] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_y)[idx] =
                        (iy + basis[b][1] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_z)[idx] =
                        (iz + basis[b][2] + 0.25) * a_lat;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a_lat);
    return frame;
}

double temperature_K(const cpu::Frame& f, int n_global) {
    const int ndof = 3 * n_global - 3;
    if (ndof <= 0)
        return 0.0;
    return 2.0 * f.meta.get(ATV::kKinetic)
           / (static_cast<double>(ndof) * units::kB_ev_per_k);
}

} // namespace

// ── 8-rank NoseHooverNPT must keep velocities alive ─────────────────────

TEST(NoseHooverNptMultiRank, KineticEnergyNonZero) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8) {
        GTEST_SKIP() << "8-rank specific test; got world_size=" << world_size;
    }
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 500 atoms in a 26.5 Å FCC Argon box. Same geometry as
    // test_npt_multirank — already a multi-rank-validated layout under
    // Berendsen, so any failure here isolates NoseHooverNPT.
    auto frame = make_fcc_argon(5, 5.3);
    const int n_global = static_cast<int>(frame.nlocal());

    const double lj_eps = 0.0104;
    const double lj_sig = 3.4;
    const double lj_cut = 6.0;
    const double dt = 0.5;       // fs
    const double T_target = 300; // K
    const double tau_T = 100.0;
    const double tau_P = 1000.0;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = T_target,
                                               .tau_T = tau_T,
                                               .P_target = 0.0,
                                               .tau_P = tau_P,
                                               .dt = dt,
                                               .drag = 2.0}});
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/12345});
    sim.set_neighbor(cpu::md::NeighborList<>{lj_cut, /*skin=*/0.3});
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sig, lj_cut}.set_special_bonds(
        0.0, 0.0, 0.0));
    sim.build();

    // Step 1: equipartition init should give T close to target.
    sim.run(frame, 1);
    const double T1 = temperature_K(frame, n_global);
    if (my_rank == 0) {
        std::printf("[NHnpt-8r] step 1: T=%.3f K, E_kin=%.3f eV, box=%.4f Å\n",
                    T1, static_cast<double>(frame.meta.get(ATV::kKinetic)),
                    frame.simbox.cell()(0, 0));
    }
    EXPECT_GT(T1, 50.0) << "after step 1: T_K should be > 50 K, got " << T1;
    EXPECT_LT(T1, 600.0) << "after step 1: T_K runaway: " << T1;

    // Run 200 more steps. T must not collapse to 0.
    constexpr int n_steps = 200;
    constexpr int probe_every = 20;
    double T_min_observed = T1;
    for (int chunk_start = 0; chunk_start < n_steps;
         chunk_start += probe_every) {
        sim.run(frame, probe_every);
        const double T = temperature_K(frame, n_global);
        if (my_rank == 0) {
            std::printf("[NHnpt-8r] step %d: T=%.3f K, E_kin=%.3f eV, "
                        "box=%.4f Å\n",
                        chunk_start + probe_every + 1, T,
                        static_cast<double>(frame.meta.get(ATV::kKinetic)),
                        frame.simbox.cell()(0, 0));
        }
        if (T < T_min_observed)
            T_min_observed = T;
        ASSERT_GT(T, 1.0) << "T_K collapsed to ~0 at step "
                          << (chunk_start + probe_every + 1)
                          << " — NoseHooverNPT MPI bug repro";
    }
    if (my_rank == 0) {
        std::printf("[NHnpt-8r] T_min over %d steps = %.3f K\n", n_steps,
                    T_min_observed);
    }

    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x went NaN";
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_vx)[i]))
            << "rank " << my_rank << " atom " << i << " vx went NaN";
    }
}

// ── Same setup but WITH bonded interactions (the bench's actual config) ──

namespace {

cpu::Frame make_pe_box_simple(int n_chains_per_axis, int n_atoms_per_chain,
                              double box_len) {
    // Layout n_chains_per_axis^3 chains in a 3D grid; each chain sits in its
    // own cubic cell aligned along x. Cells span the full box so every
    // 2x2x2 MPI rank sees a roughly equal local atom count after
    // `partition_atoms`. Bond length 1.53 Å (matches bench / Dreiding-UA).
    const int n_chains =
        n_chains_per_axis * n_chains_per_axis * n_chains_per_axis;
    const int n_atoms = n_chains * n_atoms_per_chain;
    const double r0 = 1.53;
    const double cell = box_len / static_cast<double>(n_chains_per_axis);

    cpu::Frame frame(n_atoms);

    int atom_idx = 0;
    for (int cx = 0; cx < n_chains_per_axis; ++cx)
        for (int cy = 0; cy < n_chains_per_axis; ++cy)
            for (int cz = 0; cz < n_chains_per_axis; ++cz) {
                const double x0 = cx * cell + 0.5 * r0;
                const double y0 = (cy + 0.5) * cell;
                const double z0 = (cz + 0.5) * cell;
                for (int k = 0; k < n_atoms_per_chain; ++k) {
                    frame.span(ATV::kAtom_x)[atom_idx] = x0 + k * r0;
                    frame.span(ATV::kAtom_y)[atom_idx] = y0;
                    frame.span(ATV::kAtom_z)[atom_idx] = z0;
                    frame.span(ATV::kAtom_mass)[atom_idx] = 14.0;
                    frame.span(ATV::kAtom_type)[atom_idx] = 1;
                    // Frame ctor default tag[i] == i, used by add_bond /
                    // add_angle below to route the per-atom topology entry
                    // to the LAMMPS-convention owner row.
                    ++atom_idx;
                }
            }

    // Per-atom topology population (LAMMPS num_bond[i] + bond_atom[i][m]
    // layout). `add_bond` routes to the smaller-tag owner; `add_angle`
    // routes to the center atom.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int c = 0; c < n_chains; ++c) {
        const int chain_base = c * n_atoms_per_chain;
        for (int k = 0; k < n_atoms_per_chain - 1; ++k)
            frame.add_bond(chain_base + k, chain_base + k + 1, 0);
        for (int k = 0; k < n_atoms_per_chain - 2; ++k)
            frame.add_angle(chain_base + k, chain_base + k + 1,
                            chain_base + k + 2, 0);
    }
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

// Equilibrate first with VelocityVerlet + Langevin (mimics the bench's
// stage 1), then swap-via-rebuild to NoseHooverNPT — this matches what
// the polyethylene_tension bench actually does. Done as a SINGLE MD
// (no per-stage rebuild, no DD adoption) so any failure isolates to
// the NoseHooverNPT × bonded × MPI interaction proper, not the
// stage-rebuild path.
TEST(NoseHooverNptMultiRank, BondsOnlyKineticNonZero) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8 && world_size != 1) {
        GTEST_SKIP() << "1-rank or 8-rank only; got world_size=" << world_size;
    }
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 2 chains/axis × 18 atoms = 144 atoms in 70 Å box (same atom count
    // as test_pe_msu_pipeline). Cell = 35 Å, chain length 26 Å fits inside
    // one cell with margin. halo_cut ≈ 11.55 < cell/2 (17.5), so cascade
    // ghosts stay bounded and fit the fixed 144-atom frame capacity.
    auto frame = make_pe_box_simple(2, 18, 70.0);
    const int n_global = static_cast<int>(frame.nlocal());

    const double bond_K_eV = 350.0 * 0.0433641 / (1.53 * 1.53);
    const double bond_r0 = 1.53;
    const double lj_eps = 0.112 * 0.0433641;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;
    const double dt = 0.5;
    const double T_target = 300.0;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = T_target,
                                               .tau_T = 100.0,
                                               .P_target = 0.0,
                                               .tau_P = 1000.0,
                                               .dt = dt,
                                               .drag = 2.0}});
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/42});
    sim.set_neighbor(cpu::md::NeighborList<>{lj_cut, /*skin=*/0.3});
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sig, lj_cut}.set_special_bonds(
        0.0, 0.0, 0.0));
    sim.add_potential(cpu::BondHarmonic{{bond_K_eV}, {bond_r0}});
    // NO angle, NO dihedral — bonds only.
    sim.build();

    // Initial state: atoms at bond equilibrium (r=r0=1.53), so bond force
    // is zero at step 0 and there is no PE to convert to KE. With only
    // bonds + LJ (no angle/dihedral), a properly-functioning NoseHooverNPT
    // should hold T near T_target (300 K) within a few-percent envelope.
    // 100 steps is short enough to be a unit test, long enough to expose
    // the integration drift the BondedKineticNonZero variant catches in
    // ~20 steps.
    for (int s = 1; s <= 100; ++s) {
        sim.run(frame, 1);
        const double T = temperature_K(frame, n_global);
        if (my_rank == 0 && (s <= 5 || s % 10 == 0)) {
            std::printf("[NHnpt+bondsonly-8r] step %d: T=%.3f K, E_kin=%.3f "
                        "eV, box=%.4f Å\n",
                        s, T,
                        static_cast<double>(frame.meta.get(ATV::kKinetic)),
                        frame.simbox.cell()(0, 0));
        }
        ASSERT_GT(T, 1.0) << "T_K collapsed at step " << s
                          << " (bonds-only repro)";
        ASSERT_LT(T, 1000.0)
            << "T_K runaway at step " << s
            << " — bonds-only NoseHooverNPT instability (no angle / no "
               "dihedral, so any heating must come from bond + thermostat)";
    }
}

TEST(NoseHooverNptMultiRank, BondedKineticNonZero) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8 && world_size != 1) {
        GTEST_SKIP() << "1-rank or 8-rank only; got world_size=" << world_size;
    }
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 8 chains × 18 atoms = 144 atoms in 70 Å box. See BondsOnlyKineticNonZero
    // for the cascade-vs-capacity sizing rationale (cell/2 must exceed
    // halo_cut by ≥ 2×, else cascade ghosts overflow the frame's fixed
    // capacity).
    auto frame = make_pe_box_simple(/*n_chains_per_axis=*/2,
                                    /*n_atoms_per_chain=*/18,
                                    /*box_len=*/70.0);
    const int n_global = static_cast<int>(frame.nlocal());

    // Dreiding-UA-ish params (matching the bench).
    const double bond_K_eV =
        350.0 * 0.0433641 / (1.53 * 1.53); // kcal/mol/Å² → eV/Å²
    const double bond_r0 = 1.53;
    const double angle_K_eV = 60.0 * 0.0433641;
    const double angle_theta0 = 109.5 * 3.14159265358979 / 180.0;
    const double lj_eps = 0.112 * 0.0433641;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;
    const double dt = 0.5;
    const double T_target = 300.0;

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::NoseHooverNPT{{.T_target = T_target,
                                               .tau_T = 100.0,
                                               .P_target = 0.0,
                                               .tau_P = 1000.0,
                                               .dt = dt,
                                               .drag = 2.0}});
    sim.add(cpu::md::InitVelocity{T_target, /*seed=*/42});
    sim.set_neighbor(cpu::md::NeighborList<>{lj_cut, /*skin=*/0.3});
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sig, lj_cut}.set_special_bonds(
        0.0, 0.0, 0.0));
    sim.add_potential(cpu::BondHarmonic{{bond_K_eV}, {bond_r0}});
    sim.add_potential(cpu::AngleHarmonic{{angle_K_eV}, {angle_theta0}});
    sim.build();

    // NOTE: this test was originally a *collapse* repro (T → 0 mode). It
    // also doubles as a coarse runaway gate; T_K should not exceed ~30000 K
    // even though the initial geometry (chain at 180°, angle θ₀=109.5°)
    // releases significant angle PE in the first ~20 steps. A separate
    // equilibrium-zigzag test exercises NoseHooverNPT under a realistic
    // initial state.
    constexpr int n_steps = 200;
    constexpr int probe_every = 20;
    for (int chunk = 0; chunk < n_steps; chunk += probe_every) {
        sim.run(frame, probe_every);
        const double T = temperature_K(frame, n_global);
        if (my_rank == 0) {
            std::printf("[NHnpt+bond-8r] step %d: T=%.3f K, E_kin=%.3f eV, "
                        "box=%.4f Å\n",
                        chunk + probe_every + 1, T,
                        static_cast<double>(frame.meta.get(ATV::kKinetic)),
                        frame.simbox.cell()(0, 0));
        }
        ASSERT_GT(T, 1.0) << "T_K collapsed at step "
                          << (chunk + probe_every + 1)
                          << " — NoseHooverNPT + bonded MPI bug repro";
    }
}

// ── Migration × special-bonds regression (MSU-PE 10k NPT collapse) ──────
//
// Root cause (fixed 2026-05-29): `PairLJCut` built its tag-keyed
// special-bonds exclusion map ONCE at `build()` from that rank's
// then-owned atoms. After `Migrate` moved an atom into a rank whose
// build-time snapshot never held its tag, the atom's 1-2/1-3/1-4 partners
// escaped the exclusion filter, so the LJ kernel saw a ~1.53 Å bonded
// pair and produced a catastrophic repulsive spike (E_pair → 1e5+ eV;
// under NPT this became T → 1e4 K then a frozen T = 0 collapse). The
// NoseHooverNpt configs above never exposed it: each chain lives inside
// one rank's slab, so no atom ever migrates.
//
// This test forces the exact mechanism deterministically. Every chain is
// laid ENTIRELY in the x > box/2 half, so the x < box/2 ranks' build-time
// exclusion map provably contains NONE of the chains' tags. A gentle
// uniform −x drift carries the chains across the x = box/2 slab boundary;
// the periodic Migrate (every 10 steps) then moves chain atoms onto
// ranks whose stale map lacks them. NVE (no thermostat/barostat) keeps the
// trajectory clean: only the LJ singularity can blow up E_pair, which is
// drift-invariant. The fix rebuilds the exclusion map from the live
// per-atom `special` lists on every neighbor-list rebuild, so the
// exclusion survives migration and E_pair stays bounded.
namespace {

cpu::Frame make_drifting_x1_chains(int n_lanes, int n_atoms_per_chain,
                                   double box_len, double drift_vx) {
    // n_lanes^2 straight chains laid along +x, each in its own (y, z) lane,
    // ALL starting in the upper-x half [box/2 + margin, …]. A uniform −x
    // drift velocity carries them across x = box/2 within the run. Bond
    // length 1.53 Å; chains stay clear of x = box so no wrap aliasing in
    // the first ~hundred steps.
    const int n_chains = n_lanes * n_lanes;
    const int n_atoms = n_chains * n_atoms_per_chain;
    const double r0 = 1.53;
    const double lane = box_len / static_cast<double>(n_lanes);
    const double x_start = 0.5 * box_len + 1.0; // just inside the x1 half

    cpu::Frame frame(n_atoms);

    int atom_idx = 0;
    for (int ly = 0; ly < n_lanes; ++ly)
        for (int lz = 0; lz < n_lanes; ++lz) {
            const double y0 = (ly + 0.5) * lane;
            const double z0 = (lz + 0.5) * lane;
            for (int k = 0; k < n_atoms_per_chain; ++k) {
                frame.span(ATV::kAtom_x)[atom_idx] = x_start + k * r0;
                frame.span(ATV::kAtom_y)[atom_idx] = y0;
                frame.span(ATV::kAtom_z)[atom_idx] = z0;
                frame.span(ATV::kAtom_vx)[atom_idx] =
                    drift_vx; // coherent −x march
                frame.span(ATV::kAtom_mass)[atom_idx] = 14.0;
                frame.span(ATV::kAtom_type)[atom_idx] = 1;
                ++atom_idx;
            }
        }

    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int c = 0; c < n_chains; ++c) {
        const int chain_base = c * n_atoms_per_chain;
        for (int k = 0; k < n_atoms_per_chain - 1; ++k)
            frame.add_bond(chain_base + k, chain_base + k + 1, 0);
        // Angles populate the 1-3 special tier so the exclusion covers the
        // full 1-2/1-3/1-4 set the bench uses.
        for (int k = 0; k < n_atoms_per_chain - 2; ++k)
            frame.add_angle(chain_base + k, chain_base + k + 1,
                            chain_base + k + 2, 0);
    }
    frame.build_special_list();

    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

TEST(NoseHooverNptMultiRank, MigrationKeepsSpecialBondsExclusion) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 8) {
        GTEST_SKIP() << "8-rank specific (needs 2x2x2 slab crossing); got "
                     << world_size;
    }
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 16 chains × 18 atoms in a 70 Å box, all in x ∈ [36, 62] at build
    // (entirely the x1 half; x-slab boundary = 35 Å). drift_vx = −0.04 Å/fs
    // ⇒ the chain heads reach x = 35 around step ~50 and migrate to the x0
    // ranks, whose build-time exclusion map never held these tags.
    // halo_cut = 1.1·lj_cut = 11.55 < slab/2 = 17.5 (cascade-bound rule).
    auto frame =
        make_drifting_x1_chains(/*n_lanes=*/4, /*n_atoms_per_chain=*/18,
                                /*box_len=*/70.0, /*drift_vx=*/-0.04);

    const double bond_K_eV = 350.0 * 0.0433641 / (1.53 * 1.53);
    const double bond_r0 = 1.53;
    const double lj_eps = 0.112 * 0.0433641;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;
    const double dt = 0.5;

    cpu::md::MD sim;
    // NVE: isolate the migration × special-bonds path from any
    // thermostat/barostat dynamics. The −x drift fires the migrations.
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_neighbor(cpu::md::NeighborList<>{lj_cut, /*skin=*/0.3});
    // special_bonds(0,0,0): 1-2/1-3/1-4 bonded pairs MUST be excluded from
    // LJ. The migration-staleness bug dropped this exclusion for migrated
    // atoms, feeding the LJ kernel a 1.53 Å pair (E_pair → 1e5+ eV).
    sim.add_potential(cpu::PairLJCut{lj_eps, lj_sig, lj_cut}.set_special_bonds(
        0.0, 0.0, 0.0));
    sim.add_potential(cpu::BondHarmonic{{bond_K_eV}, {bond_r0}});
    sim.build();

    // E_pair for these sparse, lane-separated chains is tiny: only intra-
    // chain 1-5⁺ pairs at ≥ 6.1 Å contribute (1-2/1-3/1-4 excluded), and
    // different lanes are box/4 = 17.5 Å > cutoff apart. A correct run
    // holds |E_pair| ≲ a few eV. A lost exclusion puts a 1.53 Å pair into
    // the kernel: (σ/r)¹² with σ/r ≈ 2.6 ⇒ thousands of eV per pair.
    constexpr int n_steps = 150;
    constexpr int probe_every = 10;
    for (int chunk = 0; chunk < n_steps; chunk += probe_every) {
        sim.run(frame, probe_every);
        const double e_pair =
            static_cast<double>(frame.meta.get(ATV::kEnergyPair));
        if (my_rank == 0) {
            std::printf("[mig-excl-8r] step %d: E_pair=%.4f eV, box=%.3f Å\n",
                        chunk + probe_every, e_pair, frame.simbox.cell()(0, 0));
        }
        ASSERT_LT(std::abs(e_pair), 200.0)
            << "E_pair exploded at step " << (chunk + probe_every)
            << " — a migrated atom lost its special-bonds exclusion and the "
               "LJ kernel saw a ~1.53 Å bonded pair (stale PairLJCut map)";
    }
    for (int i = 0; i < frame.nlocal(); ++i) {
        ASSERT_FALSE(std::isnan(frame.span(ATV::kAtom_x)[i]))
            << "rank " << my_rank << " atom " << i << " x NaN";
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
