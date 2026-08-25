/// @file tests/mpi/test_evaluate_halo_parity.cpp
/// @brief Pin `cpu::Evaluate::install_defaults` against the MPI comm contract.
///
/// `cpu::md::MD::install_defaults`
/// (`include/atomiverse/cpu/dynamics/md.h:440-447`) and
/// `cpu::minimizer::Minimize::install_defaults`
/// (`include/atomiverse/cpu/minimize/minimize.h:212-238`) both auto-inject the
/// four communication components a multi-rank force evaluation needs:
/// `comm::Migrate`, `comm::HaloSend`, `comm::HaloRecv`, `md::mpi::Reduce`.
/// `cpu::Evaluate::install_defaults`
/// (`include/atomiverse/cpu/evaluate/evaluate.h:111`) injects NONE of them: it
/// only scans for a neighbor list and throws if one is missing.
///
/// The consequence is silent, not loud. `Runtime::set_frame` still partitions
/// the frame into per-rank slabs (`src/cpu/communication/domain_runtime.cpp:73`),
/// so every rank ends up owning a slab with ZERO ghost atoms; the neighbor list
/// then enumerates intra-slab pairs only and every cross-boundary interaction
/// is dropped outright — no throw, no warning, just a systematically too-small
/// energy that gets worse the more ranks you use (measured: 2.3-5% per bonded
/// component on a 24k-atom GAFF system at np=2). Nothing reduces the per-rank
/// partials either, so `frame.meta.get(kEnergy)` is a rank-local number
/// masquerading as a system total.
///
/// The two cases below state the contract Evaluate must satisfy, mirroring
/// Minimize's single-phase shape on `evaluate::Phase::evaluate` (which is also
/// this workflow's `neighbor_phase`, `include/atomiverse/cpu/driver/driver.h:128`):
///
///   1. `TotalEnergyMatchesBruteForce` — the energy a user reads off the Frame
///      after `eval.run(frame)` equals the O(N^2) minimum-image oracle,
///      independent of rank count. Fails today at 2 ranks (missing ghosts AND
///      missing reduction), passes at 1 rank (the serial guardrail: the fix
///      must not perturb the single-rank number by so much as a ULP).
///   2. `GhostShellIsPopulated` — the halo shell is non-empty at 2 ranks and
///      the partition conserves the lattice. This is the halo-specific
///      statement: a fix that only added the reduction would satisfy neither
///      case, and a fix that only added the halo would satisfy this one alone.
///
/// Oracle: in-file O(N^2) minimum-image sum over a pristine copy of the global
/// frame. No third-party read, no RNG, no wall clock — the lattice is fixed and
/// the whole file runs in well under a second.
///
/// Geometry: 6^3 simple-cubic lattice, a0 = 5.26 A (box 31.56 A), cutoff 6.0 A.
/// The auto skin (0.3 * cutoff) makes `neighbor_reach()` 7.8 A, so a
/// Minimize-shaped shell is `max(1.1 * 6.0, 7.8) = 7.8 A`. At 2 ranks the split
/// span is 15.78 A > 2 * 7.8 A, so neither face self-overlaps — the dual-image
/// branch that `test_halo_dual_image_ghosts.cpp` exists for is deliberately NOT
/// exercised here, and cutoff 6.0 A <= L/2 keeps the flat min-image oracle
/// legal.

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <string>

#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104;    // [eV]
constexpr double kSigma = 3.4;     // [A]
constexpr double kCutoff = 6.0;    // [A]
constexpr double kLatticeA = 5.26; // [A]
constexpr int kSide = 6;           // 6^3 = 216 sites
constexpr double kEnergyRelativeTolerance = 1.0e-9;

/// Simple-cubic n^3 lattice, spacing a0, cubic box n*a0. Sites sit at
/// `(i + 0.5) * a0`, i.e. mid-cell, so a decomposition whose cuts fall on
/// multiples of a0 never lands a site on a partition plane.
/// (Local copy of the `make_sc_lattice` shape in
/// `tests/mpi/test_halo_dual_image_ghosts.cpp` — test files do not include
/// each other.)
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
    frame.box = ATV::cpu::Box::cubic(n_side * a0);
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

/// O(N^2) minimum-image total LJ energy oracle over the GLOBAL frame; legal
/// exactly while `cutoff <= L_d / 2` on every axis.
double brute_force_energy(const cpu::Frame& frame, double cutoff) {
    const int n = frame.nlocal();
    const double lx = frame.box.length(0);
    const double ly = frame.box.length(1);
    const double lz = frame.box.length(2);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx =
                frame.span(ATV::kAtom_x)[i] - frame.span(ATV::kAtom_x)[j];
            double dy =
                frame.span(ATV::kAtom_y)[i] - frame.span(ATV::kAtom_y)[j];
            double dz =
                frame.span(ATV::kAtom_z)[i] - frame.span(ATV::kAtom_z)[j];
            dx -= lx * std::round(dx / lx);
            dy -= ly * std::round(dy / ly);
            dz -= lz * std::round(dz / lz);
            total += lj_pair_energy(dx * dx + dy * dy + dz * dz, cutoff);
        }
    return total;
}

/// @brief Reads this rank's owned / ghost counts out of the decomposition
/// after the evaluate phase has run. `dd` is engine-private, so the
/// user-extension path (a `Component` reading `ctx.runtime.md.dd`) is how a
/// test observes it — same shape as `GhostCountProbe` in
/// `tests/mpi/test_halo_dual_image_ghosts.cpp`.
class GhostCountProbe : public cpu::Component {
  public:
    std::string name() const override { return "evaluate_ghost_count_probe"; }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto* dd = ctx.runtime.md.dd;
        if (dd == nullptr)
            return;
        n_local = dd->n_local;
        n_ghost = dd->n_ghost;
        ++launches;
    }

    int n_local = -1; ///< owned rows on this rank at the last launch
    int n_ghost = -1; ///< ghost rows on this rank at the last launch
    int launches = 0; ///< guards against a vacuous pass
};

/// The minimal public Evaluate pipeline: neighbor list + LJ evaluator, one
/// pass. Nothing communication-related is registered — auto-injecting it is
/// exactly the responsibility under test.
void run_single_point(cpu::Frame& frame, GhostCountProbe* probe = nullptr) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCutoff});
    eval.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    if (probe != nullptr)
        eval.on(ATV::evaluate::Phase::evaluate).add_ref(*probe, +500);
    eval.build();
    eval.run(frame);
}

} // namespace

// ── 1. The energy on the Frame is the system energy, at any rank count ──
//
// RED at world == 2: with no auto-injected halo the cross-slab pairs never
// enter the CSR, and with no auto-injected reduction the number left on the
// Frame is this rank's partial. GREEN guardrail at world == 1, where every
// injected component is a documented no-op.
TEST(EvaluateHaloParity, TotalEnergyMatchesBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1 && world != 2)
        GTEST_SKIP() << "single-split shape covers 1 and 2 ranks";

    cpu::Frame frame = make_sc_lattice(kSide, kLatticeA);
    const cpu::Frame ref = frame; // pristine global copy for the oracle
    const double e_bf = brute_force_energy(ref, kCutoff);
    ASSERT_GT(std::abs(e_bf), 1e-9) << "degenerate oracle energy";

    run_single_point(frame);

    const double e_frame = frame.meta.get(ATV::kEnergy);

    // Diagnostic only: with the reduction missing, the sum of the per-rank
    // partials is what the distributed pipeline actually computed, so the
    // ghost deficit is readable from a failing run rather than inferred.
    double e_sum = 0.0;
    MPI_Allreduce(&e_frame, &e_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const double rel = std::abs(e_frame - e_bf) / std::abs(e_bf);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "rank " << rank << " world=" << world << ": Evaluate left "
        << e_frame << " on the Frame vs brute force " << e_bf << " (rel=" << rel
        << "); sum of per-rank partials " << e_sum << " (deficit "
        << (100.0 * (e_sum - e_bf) / std::abs(e_bf))
        << "%) — cpu::Evaluate::install_defaults injects no "
           "Migrate/HaloSend/HaloRecv/Reduce";
}

// ── 2. The halo shell exists and the partition conserves the lattice ────
//
// The energy identity alone cannot say WHICH missing piece broke it. This
// case is the halo-specific statement: at 2 ranks some rank must hold ghost
// rows, and the owned rows must still sum to the global lattice.
TEST(EvaluateHaloParity, GhostShellIsPopulated) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 1 && world != 2)
        GTEST_SKIP() << "single-split shape covers 1 and 2 ranks";

    cpu::Frame frame = make_sc_lattice(kSide, kLatticeA);
    const int n_atoms = frame.nlocal();

    GhostCountProbe probe;
    run_single_point(frame, &probe);
    ASSERT_EQ(probe.launches, 1) << "the probe never ran: test is vacuous";

    int owned_total = 0;
    MPI_Allreduce(&probe.n_local, &owned_total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_EQ(owned_total, n_atoms)
        << "atoms lost or duplicated by the Evaluate decomposition";

    if (world == 1)
        return; // single rank owns everything; there is no shell to fill

    int ghost_max = 0;
    MPI_Allreduce(&probe.n_ghost, &ghost_max, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_GT(ghost_max, 0)
        << "no ghosts anywhere at world=" << world
        << ": cpu::Evaluate::install_defaults auto-injects no comm::HaloSend, "
           "so every cross-boundary interaction is silently dropped";
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
