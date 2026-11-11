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
///     reduced energy is rank-consistent;
///   * world 2 and 4 → each brick arm equals the replicated arm
///     (`ATV_PPPM_ALLREDUCE=1`) for odd and even orders, which pins the
///     ghost extents of LAMMPS's centred stencil (`Grid3d::ghost_grid` with
///     `set_stencil_atom(-nlower, nupper)` and the odd-order 0.5 shift).
///
/// Geometry (grilling-corrected): cubic L = 18 Å with spacing 3.2 Å gives
/// `ceil(18 / 3.2) = 6`, `pppm_fft_size(6) = 6`; order 5 needs K ≥ 6 so the
/// order clamp leaves it alone. Eight ±1 e point charges, net neutral, off every cell edge and off the
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
#include <cstdlib>
#include <string>
#include <vector>

#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/cpu/communication/gather.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
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
/// sizing function the solver uses (`KSpacePPPM::ensure_grid`).
int resolved_k0() {
    return longrange::pppm_fft_size(
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
/// @param order Charge-assignment order.
ArmRun run_arm(int fft_ranks, int order = kOrder) {
    cpu::Frame frame = make_charges();
    // Held by reference so the accessors can be read after the run: a moved-in
    // evaluator would leave this object empty.
    cpu::KSpacePPPM kspace{kAlpha, kSpacing, order, fft_ranks};

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

// ── world = 2, 4: brick arms against the replicated arm, odd and even ──────

TEST(PppmSubcommParity, BrickArmsMatchReplicatedForOddAndEvenOrders) {
    if (!arm_case_enabled(2) && !arm_case_enabled(4))
        GTEST_SKIP() << "2- or 4-rank + ATV_HAS_FFTW_MPI specific; got "
                        "world_size="
                     << world_size();
    struct AllreduceEnv {
        bool present = std::getenv("ATV_PPPM_ALLREDUCE") != nullptr;
        std::string value = present ? std::getenv("ATV_PPPM_ALLREDUCE") : "";
        ~AllreduceEnv() {
            if (present)
                setenv("ATV_PPPM_ALLREDUCE", value.c_str(), 1);
            else
                unsetenv("ATV_PPPM_ALLREDUCE");
        }
    } restore;
    for (int order : {3, 4, 5, 6, 7}) {
        SCOPED_TRACE("order " + std::to_string(order));
        unsetenv("ATV_PPPM_ALLREDUCE");
        const ArmRun brick = run_arm(/*fft_ranks=*/0, order);
        setenv("ATV_PPPM_ALLREDUCE", "1", 1);
        const ArmRun replicated = run_arm(/*fft_ranks=*/0, order);
        unsetenv("ATV_PPPM_ALLREDUCE");
        EXPECT_NE(brick.arm, cpu::PppmMeshArm::Replicated)
            << "the default multi-rank arm must be a brick arm";
        EXPECT_EQ(replicated.arm, cpu::PppmMeshArm::Replicated);
        EXPECT_NEAR(brick.energy, replicated.energy,
                    kEnergyRel * std::max(1.0, std::fabs(replicated.energy)));
        if (world_rank() != 0)
            continue;
        ASSERT_EQ(brick.gathered.n_total, kNAtoms);
        ASSERT_EQ(replicated.gathered.n_total, kNAtoms);
        double fmax = 0.0, dmax = 0.0;
        for (int i = 0; i < kNAtoms; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            ASSERT_EQ(brick.gathered.tag[k], replicated.gathered.tag[k]);
            const double r[3] = {replicated.gathered.fx[k],
                                 replicated.gathered.fy[k],
                                 replicated.gathered.fz[k]};
            const double b[3] = {brick.gathered.fx[k], brick.gathered.fy[k],
                                 brick.gathered.fz[k]};
            for (int d = 0; d < 3; ++d) {
                fmax = std::max(fmax, std::fabs(r[d]));
                dmax = std::max(dmax, std::fabs(b[d] - r[d]));
            }
        }
        ASSERT_GT(fmax, 1e-3) << "reciprocal forces vanished; parity vacuous";
        EXPECT_LT(dmax, kForceAbs)
            << "brick arm changed the forces: max|ΔF| = " << dmax << " eV/Å";
    }
}

// ── ghost depth: LAMMPS set_distance(0.5·skin) ─────────────────────────────

namespace {

constexpr double kDriftSpacing = 1.0; ///< mesh spacing [Å] → K = 18
constexpr double kDriftCutoff = 3.0;  ///< real-space cutoff [Å]
constexpr double kDriftSkin = 5.0;    ///< skin [Å]: owned drift up to 2.5 Å
constexpr double kDriftStart = 8.85;  ///< [Å], 0.15 Å inside the 9 Å cut
constexpr double kDriftStep = 2.3;    ///< displacement in one 1 fs step [Å]

/// @brief One MD step in which site 0 crosses a subdomain cut by 2.15 Å
/// (2.15 mesh cells) without reneighbouring (2.3 Å < skin/2 = 2.5 Å,
/// delay 0, check yes); the step's forces are taken at the displaced
/// position.
ArmRun run_drift(int axis, int order) {
    cpu::Frame frame = make_charges();
    StorageReal* r[3] = {frame.span(ATV::kAtom_x), frame.span(ATV::kAtom_y),
                         frame.span(ATV::kAtom_z)};
    StorageReal* v[3] = {frame.span(ATV::kAtom_vx), frame.span(ATV::kAtom_vy),
                         frame.span(ATV::kAtom_vz)};
    r[axis][0] = kDriftStart;
    v[axis][0] = kDriftStep; // Å/fs, dt = 1 fs
    cpu::KSpacePPPM kspace{kAlpha, kDriftSpacing, order};
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kDriftCutoff, kDriftSkin,
                                             /*every=*/1, /*delay=*/0,
                                             /*check=*/true});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kDriftCutoff});
    sim.set_evaluator_ref(kspace);
    sim.build();
    sim.run(frame, 1);
    ArmRun out;
    out.energy = frame.meta.get(ATV::kEnergy);
    out.arm = kspace.mesh_arm();
    auto dd = sim.release_dd();
    if (dd != nullptr)
        out.gathered = cpu::md::mpi::gather_to_root(frame, *dd);
    return out;
}

/// @brief First axis the default decomposition cuts (dims > 1), or -1.
int cut_axis() {
    cpu::Frame frame = make_charges();
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kDriftCutoff, kDriftSkin});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kDriftCutoff});
    sim.set_evaluator(cpu::KSpacePPPM{kAlpha, kDriftSpacing, kOrder});
    sim.build();
    sim.run(frame, 1);
    auto dd = sim.release_dd();
    if (dd == nullptr)
        return -1;
    for (int d = 0; d < 3; ++d)
        if (dd->dims[d] > 1)
            return d;
    return -1;
}

} // namespace

TEST(PppmSubcommParity, OwnedAtomDriftingTwoCellsPastItsCutUsesSkinGhosts) {
    if (!arm_case_enabled(2) && !arm_case_enabled(4))
        GTEST_SKIP() << "2- or 4-rank + ATV_HAS_FFTW_MPI specific; got "
                        "world_size="
                     << world_size();
    const int axis = cut_axis();
    ASSERT_GE(axis, 0) << "no cut axis in a multi-rank decomposition";
    // The first cut of 2 or 4 ranks sits at kBoxL/2 on that axis.
    ASSERT_GT(kDriftStart + kDriftStep - 0.5 * kBoxL, 2.0 * kDriftSpacing);
    ASSERT_LT(kDriftStep, 0.5 * kDriftSkin);
    struct AllreduceEnv {
        bool present = std::getenv("ATV_PPPM_ALLREDUCE") != nullptr;
        std::string value = present ? std::getenv("ATV_PPPM_ALLREDUCE") : "";
        ~AllreduceEnv() {
            if (present)
                setenv("ATV_PPPM_ALLREDUCE", value.c_str(), 1);
            else
                unsetenv("ATV_PPPM_ALLREDUCE");
        }
    } restore;
    for (int order : {5, 6}) {
        SCOPED_TRACE("order " + std::to_string(order));
        unsetenv("ATV_PPPM_ALLREDUCE");
        // A one-cell ghost margin threw "maps to mesh cells outside this
        // rank's ghost-extended brick" here.
        ArmRun brick;
        ASSERT_NO_THROW(brick = run_drift(axis, order));
        setenv("ATV_PPPM_ALLREDUCE", "1", 1);
        const ArmRun replicated = run_drift(axis, order);
        unsetenv("ATV_PPPM_ALLREDUCE");
        EXPECT_NE(brick.arm, cpu::PppmMeshArm::Replicated);
        EXPECT_NEAR(brick.energy, replicated.energy,
                    kEnergyRel * std::max(1.0, std::fabs(replicated.energy)));
        if (world_rank() != 0)
            continue;
        ASSERT_EQ(brick.gathered.n_total, kNAtoms);
        double dmax = 0.0, fmax = 0.0;
        for (int i = 0; i < kNAtoms; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            ASSERT_EQ(brick.gathered.tag[k], replicated.gathered.tag[k]);
            const double rf[3] = {replicated.gathered.fx[k],
                                  replicated.gathered.fy[k],
                                  replicated.gathered.fz[k]};
            const double bf[3] = {brick.gathered.fx[k], brick.gathered.fy[k],
                                  brick.gathered.fz[k]};
            for (int d = 0; d < 3; ++d) {
                fmax = std::max(fmax, std::fabs(rf[d]));
                dmax = std::max(dmax, std::fabs(bf[d] - rf[d]));
            }
        }
        ASSERT_GT(fmax, 1e-3);
        EXPECT_LT(dmax, kForceAbs);
    }
}

TEST(PppmSubcommParity, OutOfRangeAtomThrowsOnEveryRank) {
    if (!arm_case_enabled(2) && !arm_case_enabled(4))
        GTEST_SKIP() << "2- or 4-rank + ATV_HAS_FFTW_MPI specific; got "
                        "world_size="
                     << world_size();
    const int axis = cut_axis();
    ASSERT_GE(axis, 0) << "no cut axis in a multi-rank decomposition";
    // A 0.2 Å skin gives 0.1 Å of ghost cells beyond the cut, and the
    // neighbor list never rebuilds, so the 2.3 Å step leaves the owning
    // rank's brick. Only that rank detects it; every rank must throw
    // rather than wait in the next mesh collective.
    cpu::Frame frame = make_charges();
    StorageReal* r[3] = {frame.span(ATV::kAtom_x), frame.span(ATV::kAtom_y),
                         frame.span(ATV::kAtom_z)};
    StorageReal* v[3] = {frame.span(ATV::kAtom_vx), frame.span(ATV::kAtom_vy),
                         frame.span(ATV::kAtom_vz)};
    r[axis][0] = kDriftStart;
    v[axis][0] = kDriftStep; // Å/fs, dt = 1 fs
    cpu::KSpacePPPM kspace{kAlpha, kDriftSpacing, kOrder};
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.set_neighbor(cpu::md::NeighborList<>{kDriftCutoff, /*skin=*/0.2,
                                             /*every=*/1000, /*delay=*/0,
                                             /*check=*/false});
    sim.on(md::Phase::force).add(cpu::CoulLong{kAlpha, kDriftCutoff});
    sim.set_evaluator_ref(kspace);
    sim.build();
    EXPECT_THROW(sim.run(frame, 1), std::runtime_error);
}
