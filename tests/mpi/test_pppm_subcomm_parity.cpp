/// @file tests/mpi/test_pppm_subcomm_parity.cpp
/// @brief Production gate: the PPPM mesh-arm selector and its FFT subgroup.
///
/// Pins the third arm of the `KSpacePPPM` mesh selector (spec
/// `kspace-subcomm`): the FFT slab lives on a communicator of
/// `pppm_fft_ranks(K0, nprocs)` ranks derived from `dd->cart_comm`, so at
/// `world == 4` with `K0 == 6` exactly one rank is a non-member.
///
///   * world 1 → `mesh_arm() == Replicated`, explicit `fft_ranks = 1` is a
///     no-op on the energy (same code path ⇒ bitwise identical);
///   * world 2 → `pppm_fft_ranks(6, 2) == 2 == nprocs`, so the subgroup
///     derivation must **alias** `cart_comm`, not split it → `BrickWorld`;
///   * world 4 → `fft_ranks() == 3`, `mesh_arm() == BrickSubgroup`, and the
///     physics is invariant: total energy and tag-aligned per-atom forces
///     match a `fft_ranks = 4` `BrickWorld` run on the same system, and the
///     reduced energy is rank-consistent.
///
/// Geometry (grilling-corrected): cubic L = 18 Å with spacing 3.2 Å gives
/// `ceil(18 / 3.2) = 6`, `good_fft_size(6) = 6`; order 5 needs K ≥ 6 so the
/// order clamp leaves it alone, and dim 2 is even so the evenness clamp does
/// too. Eight ±1 e point charges, net neutral, off every cell edge and off the
/// 2×2×1 domain cuts at 9 Å.
///
/// One static evaluation per arm — no convergence loop, no timing, no external
/// engine, and (deliberately) no literature oracle: Madelung-style pins belong
/// to `regressions/`.
///
/// Requires ATV_BUILD_KSPACE. Multi-rank needs ATV_HAS_FFTW_MPI.

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme_spline.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

using namespace ATV;

namespace {

constexpr double kBoxL = 18.0;   ///< cubic cell edge [Å]
constexpr double kSpacing = 3.2; ///< target mesh spacing [Å] → K = 6 per dim
constexpr int kOrder = 5;        ///< B-spline order; needs K ≥ order + 1 = 6
constexpr double kAlpha = 0.35;  ///< Ewald splitting [1/Å]
constexpr double kCutoff = 6.0;  ///< real-space cutoff [Å] (< L/2 at 2 cuts)
constexpr double kSkin = 0.3;    ///< neighbor skin [Å]

constexpr int kExpectedK0 = 6;          ///< mesh planes along dim 0
constexpr int kNAtoms = 8;              ///< eight ±1 e charges, net neutral
constexpr double kEnergyRel = 1e-9;     ///< arm-vs-arm total energy [relative]
constexpr double kForceAbs = 1e-9;      ///< arm-vs-arm per-atom force [eV/Å]
constexpr double kRankAgreeRel = 1e-10; ///< reduced energy across ranks [rel]

struct Site {
    double x, y, z; ///< position [Å]
    double q;       ///< charge [e]
    double mass;    ///< mass [amu]
    int type;
};

/// Corners of an inset cube (quadrant centres of the 2×2×1 decomposition at
/// world 4), nudged by hard-coded offsets so the reciprocal-space forces do
/// not vanish by symmetry — a zero-force system would make force parity
/// vacuous. Charges alternate: Σq = 0.
constexpr Site kSites[kNAtoms] = {
    {4.61, 4.42, 4.73, +1.0, 22.9898, 0},
    {4.38, 4.65, 13.29, -1.0, 35.453, 1},
    {4.72, 13.51, 4.36, -1.0, 35.453, 1},
    {4.45, 13.28, 13.62, +1.0, 22.9898, 0},
    {13.36, 4.57, 4.28, -1.0, 35.453, 1},
    {13.62, 4.31, 13.44, +1.0, 22.9898, 0},
    {13.29, 13.46, 4.55, +1.0, 22.9898, 0},
    {13.55, 13.39, 13.31, -1.0, 35.453, 1},
};

/// @brief Mesh planes along dim 0 for this test's box/spacing, from the real
/// sizing function the solver uses (`ensure_grid`, `pppm.cpp:135`).
int resolved_k0() {
    return longrange::good_fft_size(
        static_cast<int>(std::ceil(kBoxL / kSpacing)));
}

std::string arm_label(cpu::PppmMeshArm arm) {
    switch (arm) {
    case cpu::PppmMeshArm::Replicated:
        return "Replicated";
    case cpu::PppmMeshArm::BrickWorld:
        return "BrickWorld";
    case cpu::PppmMeshArm::BrickSubgroup:
        return "BrickSubgroup";
    }
    return "<invalid>";
}

cpu::Frame make_charges() {
    cpu::Frame frame(kNAtoms);
    for (int i = 0; i < kNAtoms; ++i) {
        frame.span(ATV::kAtom_x)[i] = kSites[i].x;
        frame.span(ATV::kAtom_y)[i] = kSites[i].y;
        frame.span(ATV::kAtom_z)[i] = kSites[i].z;
        frame.span(ATV::kAtom_charge)[i] = kSites[i].q;
        frame.span(ATV::kAtom_mass)[i] = kSites[i].mass;
        frame.span(ATV::kAtom_type)[i] = kSites[i].type;
        frame.span(ATV::kAtom_tag)[i] = i + 1;
        frame.span(ATV::kAtom_vx)[i] = 0.0;
        frame.span(ATV::kAtom_vy)[i] = 0.0;
        frame.span(ATV::kAtom_vz)[i] = 0.0;
    }
    frame.box = cpu::Box::cubic(kBoxL);
    return frame;
}

/// @brief One static PPPM evaluation on a freshly built system.
struct ArmRun {
    double energy = 0.0; ///< frame total energy [eV]
    int fft_ranks = 0;   ///< self-reported FFT subgroup size
    cpu::PppmMeshArm arm = cpu::PppmMeshArm::Replicated; ///< self-reported arm
    cpu::md::mpi::GatheredFrame gathered{}; ///< tag-ordered, root only
};

/// @brief Build CoulLong + KSpacePPPM, run exactly one static step, report.
/// @param fft_ranks 0 = derive from the mesh (default arm), else pinned.
ArmRun run_arm(int fft_ranks) {
    cpu::Frame frame = make_charges();
    // Held by reference so the accessors can be read after the run: a moved-in
    // evaluator would leave this object empty.
    cpu::KSpacePPPM kspace{kAlpha, kSpacing, kOrder, fft_ranks};

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kCutoff});
    sim.set_evaluator_ref(kspace);
    sim.build();
    sim.run(frame, 1);

    ArmRun out;
    out.energy = frame.meta.get(ATV::kEnergy);
    out.fft_ranks = kspace.fft_ranks();
    out.arm = kspace.mesh_arm();

    auto dd = sim.release_dd();
    if (dd != nullptr)
        out.gathered = cpu::md::mpi::gather_to_root(frame, *dd);
    return out;
}

int world_size() {
    int n = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

int world_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

/// @brief Skip unless this launch has `want` ranks and multi-rank FFT is built.
/// @return true when the caller should keep running.
bool arm_case_enabled(int want) {
    if (world_size() != want)
        return false;
#ifndef ATV_HAS_FFTW_MPI
    if (want > 1)
        return false;
#endif
    return true;
}

} // namespace

// ── world = 1: the replicated arm, and fft_ranks=1 is a no-op ──────────────

TEST(PppmSubcommParity, SingleRankReportsReplicatedArm) {
    if (world_size() != 1)
        GTEST_SKIP() << "1-rank specific; got world_size=" << world_size();

    const ArmRun run = run_arm(/*fft_ranks=*/0);
    EXPECT_EQ(run.arm, cpu::PppmMeshArm::Replicated)
        << "serial PPPM must report the replicated arm, got "
        << arm_label(run.arm);
    // `fft_ranks()` is documented as 0 only *before* the first compute(); after
    // one solve it must report the resolved group, which at world 1 is
    // pppm_fft_ranks(K0, 1) == 1. Without this the arm check above passes on a
    // default-constructed member alone.
    EXPECT_EQ(run.fft_ranks, cpu::pppm_fft_ranks(kExpectedK0, 1))
        << "the resolved FFT group was never published after compute()";
}

TEST(PppmSubcommParity, SingleRankExplicitFftRanksOneIsBitwiseIdentical) {
    if (world_size() != 1)
        GTEST_SKIP() << "1-rank specific; got world_size=" << world_size();

    const ArmRun derived = run_arm(/*fft_ranks=*/0);
    const ArmRun pinned = run_arm(/*fft_ranks=*/1);
    EXPECT_EQ(pinned.arm, cpu::PppmMeshArm::Replicated)
        << "pinning fft_ranks=1 must not leave the replicated arm, got "
        << arm_label(pinned.arm);
    // Same arm, same mesh, same summation order ⇒ no float slack is allowed.
    EXPECT_EQ(derived.energy, pinned.energy)
        << "fft_ranks=1 took a different code path: derived=" << derived.energy
        << " pinned=" << pinned.energy;
}

// ── world = 2: the group is the whole world, so it must alias, not split ───

TEST(PppmSubcommParity, TwoRanksAliasWorldInsteadOfSplitting) {
    if (!arm_case_enabled(2))
        GTEST_SKIP() << "2-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();
    ASSERT_EQ(resolved_k0(), kExpectedK0)
        << "geometry premise broken: L/spacing must resolve to K0=6";
    ASSERT_EQ(cpu::pppm_fft_ranks(kExpectedK0, 2), 2)
        << "premise: 2 divides 6, so the group is the whole world";

    const ArmRun run = run_arm(/*fft_ranks=*/0);
    EXPECT_EQ(run.fft_ranks, 2);
    EXPECT_EQ(run.arm, cpu::PppmMeshArm::BrickWorld)
        << "fft_ranks == nprocs must alias cart_comm (no MPI_Comm_split), got "
        << arm_label(run.arm);
}

TEST(PppmSubcommParity, TwoRanksAgreeOnReducedEnergy) {
    if (!arm_case_enabled(2))
        GTEST_SKIP() << "2-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();

    const ArmRun run = run_arm(/*fft_ranks=*/0);
    double e_min = run.energy, e_max = run.energy;
    MPI_Allreduce(MPI_IN_PLACE, &e_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &e_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NEAR(e_min, e_max,
                kRankAgreeRel * std::max(1.0, std::fabs(run.energy)))
        << "ranks disagree on the reduced energy (missing/duplicated reduce?)";
}

// ── world = 4: K0 = 6, group = 3 ranks, one non-member ─────────────────────

TEST(PppmSubcommParity, FourRanksSelectSubgroupArmWithThreeFftRanks) {
    if (!arm_case_enabled(4))
        GTEST_SKIP() << "4-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();
    ASSERT_EQ(resolved_k0(), kExpectedK0)
        << "geometry premise broken: L/spacing must resolve to K0=6";
    ASSERT_EQ(cpu::pppm_fft_ranks(kExpectedK0, 4), 3)
        << "premise: largest divisor of 6 that is ≤ 4 is 3";

    const ArmRun run = run_arm(/*fft_ranks=*/0);
    EXPECT_EQ(run.fft_ranks, 3)
        << "the FFT slab must live on 3 of the 4 ranks (6 planes, 2 each)";
    EXPECT_EQ(run.arm, cpu::PppmMeshArm::BrickSubgroup)
        << "one rank is a non-member, so this is the subgroup arm, got "
        << arm_label(run.arm);
}

TEST(PppmSubcommParity, FourRankSubgroupEnergyMatchesWorldArm) {
    if (!arm_case_enabled(4))
        GTEST_SKIP() << "4-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();

    const ArmRun subgroup = run_arm(/*fft_ranks=*/0);
    // fft_ranks = 4 does not divide 6: today's world arm, ragged FFTW-MPI
    // slabs (default dim-0 block ceil(6/4) = 2 -> local_n0 = 2,2,2,0). Energy
    // is a functional of the Hamiltonian, so only the summation order may
    // differ.
    const ArmRun world = run_arm(/*fft_ranks=*/4);

    const double rel = std::fabs(subgroup.energy - world.energy)
                       / std::max(1.0, std::fabs(world.energy));
    EXPECT_LT(rel, kEnergyRel)
        << "subgroup FFT changed the energy: subgroup=" << subgroup.energy
        << " world=" << world.energy << " rel=" << rel;
    // Anti-false-green: without these an implementation that silently stays on
    // the world arm would pass the parity comparison trivially.
    EXPECT_EQ(subgroup.arm, cpu::PppmMeshArm::BrickSubgroup)
        << "default arm is not the subgroup arm: " << arm_label(subgroup.arm);
    EXPECT_EQ(world.arm, cpu::PppmMeshArm::BrickWorld)
        << "pinned fft_ranks=4 is not the world arm: " << arm_label(world.arm);
}

TEST(PppmSubcommParity, FourRankSubgroupForcesMatchWorldArm) {
    if (!arm_case_enabled(4))
        GTEST_SKIP() << "4-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();

    const ArmRun subgroup = run_arm(/*fft_ranks=*/0);
    const ArmRun world = run_arm(/*fft_ranks=*/4);
    // Anti-false-green (every rank, before the root-only comparison).
    EXPECT_EQ(subgroup.arm, cpu::PppmMeshArm::BrickSubgroup)
        << "default arm is not the subgroup arm: " << arm_label(subgroup.arm);

    if (world_rank() != 0)
        return;
    ASSERT_EQ(subgroup.gathered.n_total, kNAtoms)
        << "each atom must be owned exactly once";
    ASSERT_EQ(world.gathered.n_total, kNAtoms);

    // Both gathers are tag-ordered, so index i is the same atom in both.
    double fmax = 0.0;
    double dmax = 0.0;
    for (int i = 0; i < kNAtoms; ++i) {
        ASSERT_EQ(subgroup.gathered.tag[static_cast<std::size_t>(i)],
                  world.gathered.tag[static_cast<std::size_t>(i)]);
        const std::size_t k = static_cast<std::size_t>(i);
        fmax =
            std::max(fmax, std::max(std::fabs(world.gathered.fx[k]),
                                    std::max(std::fabs(world.gathered.fy[k]),
                                             std::fabs(world.gathered.fz[k]))));
        dmax = std::max(
            dmax,
            std::max(std::fabs(subgroup.gathered.fx[k] - world.gathered.fx[k]),
                     std::max(std::fabs(subgroup.gathered.fy[k]
                                        - world.gathered.fy[k]),
                              std::fabs(subgroup.gathered.fz[k]
                                        - world.gathered.fz[k]))));
    }
    // The sites are deliberately off-symmetry, so a zero force field would
    // mean the mesh path never ran and the comparison below is vacuous.
    ASSERT_GT(fmax, 1e-3) << "reciprocal forces vanished; parity is vacuous";
    EXPECT_LT(dmax, kForceAbs)
        << "subgroup FFT changed the forces: max|ΔF| = " << dmax << " eV/Å";
}

TEST(PppmSubcommParity, FourRanksAgreeOnReducedEnergy) {
    if (!arm_case_enabled(4))
        GTEST_SKIP() << "4-rank + ATV_HAS_FFTW_MPI specific; got world_size="
                     << world_size();

    const ArmRun run = run_arm(/*fft_ranks=*/0);
    // The non-member rank must still see the reduced energy: the subgroup
    // Allreduce plus the existing world-level reduction has to leave every
    // rank on the same number. Reduce before any EXPECT so a one-rank failure
    // cannot strand the others in a collective.
    double e_min = run.energy, e_max = run.energy;
    MPI_Allreduce(MPI_IN_PLACE, &e_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &e_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(run.arm, cpu::PppmMeshArm::BrickSubgroup)
        << "default arm is not the subgroup arm: " << arm_label(run.arm);
    EXPECT_NEAR(e_min, e_max,
                kRankAgreeRel * std::max(1.0, std::fabs(run.energy)))
        << "ranks disagree on the reduced energy (non-member rank left out?)";
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
