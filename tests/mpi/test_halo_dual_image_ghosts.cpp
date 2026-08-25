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
#include <cstdlib>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/swap_list.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

constexpr double kEps = 0.0104; // [eV]
constexpr double kSigma = 3.4;  // [A]
constexpr double kEnergyRelativeTolerance = 1.0e-9;

/// Simple-orthorhombic nx*ny*nz lattice, per-axis spacing, box (n_d * a_d).
/// Sites sit at `(i + 0.5) * a_d`, i.e. mid-cell, so a decomposition whose
/// cuts fall on multiples of a_d never lands a site on a partition plane.
cpu::Frame make_sc_lattice(int nx, int ny, int nz, double ax, double ay,
                           double az) {
    const int n = nx * ny * nz;
    cpu::Frame frame(n);
    int idx = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                frame.span(ATV::kAtom_x)[idx] = (i + 0.5) * ax;
                frame.span(ATV::kAtom_y)[idx] = (j + 0.5) * ay;
                frame.span(ATV::kAtom_z)[idx] = (k + 0.5) * az;
                frame.span(ATV::kAtom_mass)[idx] = 39.948;
                frame.span(ATV::kAtom_tag)[idx] = idx;
                ++idx;
            }
    frame.box = ATV::cpu::Box::orthorhombic(nx * ax, ny * ay, nz * az);
    return frame;
}

/// Simple-cubic n^3 lattice, spacing a0, cubic box n*a0.
cpu::Frame make_sc_lattice(int n_side, double a0) {
    return make_sc_lattice(n_side, n_side, n_side, a0, a0, a0);
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

/// O(N^2) minimum-image total LJ energy oracle. Per-axis edge lengths, so the
/// same oracle serves the cubic pins and the orthorhombic production grid;
/// legal exactly while `cutoff <= L_d / 2` on every axis.
double brute_force_energy(const cpu::Frame& frame, double cutoff) {
    const int n = frame.nlocal();
    const double Lx = frame.box.length(0);
    const double Ly = frame.box.length(1);
    const double Lz = frame.box.length(2);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx =
                frame.span(ATV::kAtom_x)[i] - frame.span(ATV::kAtom_x)[j];
            double dy =
                frame.span(ATV::kAtom_y)[i] - frame.span(ATV::kAtom_y)[j];
            double dz =
                frame.span(ATV::kAtom_z)[i] - frame.span(ATV::kAtom_z)[j];
            dx -= Lx * std::round(dx / Lx);
            dy -= Ly * std::round(dy / Ly);
            dz -= Lz * std::round(dz / Lz);
            total += lj_pair_energy(dx * dx + dy * dy + dz * dz, cutoff);
        }
    return total;
}

/// Build the standard LJ MD pipeline and run one frozen step.
void run_one_frozen_step(cpu::Frame& frame, double cutoff) {
    cpu::md::MD sim;
    sim.set_integrator(
        cpu::md::VelocityVerlet{0.0}); // dt = 0: positions frozen
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{cutoff}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, cutoff});
    sim.build();
    sim.run(frame, 1);
}

// ── narrow-subdomain (multi-hop) support ───────────────────────────────
//
// Spec `.claude/specs/comm-topology-02-multihop.md`: with the ghost shell
// wider than one subdomain the halo must fill it by forwarding ghosts over
// several rounds instead of refusing the decomposition.

/// @brief Scoped `ATV_FORCE_DIMS` override — the single parse point is
/// `DomainDecomposition::choose_dims` (`src/cpu/communication/domain.cpp:74`),
/// read when the runtime builds its topology, so the variable must be set
/// before the `MD` is built and cleared afterwards (an early `ASSERT_*`
/// return must not leak it into the next TEST). In-tree precedent:
/// `regressions/md/const-pot-charge-halo-mpi/reg_const-pot-charge-halo-mpi.cpp:200`.
class ForcedDims {
  public:
    explicit ForcedDims(const char* grid) {
        ::setenv("ATV_FORCE_DIMS", grid, 1);
    }
    ~ForcedDims() { ::unsetenv("ATV_FORCE_DIMS"); }
    ForcedDims(const ForcedDims&) = delete;
    ForcedDims& operator=(const ForcedDims&) = delete;
};

/// @brief Reads this rank's owned / ghost counts out of the decomposition
/// after the halo has run. `dd` is engine-private, so the user-extension
/// path (a `Component` reading `ctx.runtime.md.dd`) is how a test observes
/// it — same shape as the const-pot regression's `GhostChargeChecker`.
class GhostCountProbe : public cpu::Component {
  public:
    std::string name() const override { return "ghost_count_probe"; }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto* dd = ctx.runtime.md.dd;
        if (dd == nullptr)
            return;
        n_local = dd->n_local;
        n_ghost = dd->n_ghost;
        // The published swap table is how a case states which shell it
        // actually got: `max_hops(d)` is the derived round count of axis d
        // (0 on an undecomposed axis, by the `sendself` convention).
        if (const auto* swaps = ctx.runtime.md.swap_list)
            for (int d = 0; d < 3; ++d)
                hops[d] = swaps->max_hops(d);
        ++launches;
    }

    int n_local = -1;           ///< owned rows on this rank at the last launch
    int n_ghost = -1;           ///< ghost rows on this rank at the last launch
    int hops[3] = {-1, -1, -1}; ///< derived forwarding rounds per axis
    int launches = 0;
};

/// Same pipeline as `run_one_frozen_step`, with the probe bound after the
/// pair kernel so it sees the post-halo counts of the step it reports on.
///
/// @param skin Verlet skin [A]; negative = the `NeighborList` auto default
///        (0.3 * cutoff). The skin is what sizes the auto-injected ghost
///        shell — `MD::install_defaults` takes
///        `max(1.1 * cutoff, NeighborList::neighbor_reach())`
///        (`include/atomiverse/cpu/dynamics/md.h:438`) — so a case that
///        states its halo width in the comment must state its skin here.
void run_one_frozen_step(cpu::Frame& frame, double cutoff,
                         GhostCountProbe& probe, double skin = -1.0) {
    cpu::md::MD sim;
    sim.set_integrator(
        cpu::md::VelocityVerlet{0.0}); // dt = 0: positions frozen
    sim.on(ATV::md::Phase::force)
        .add(cpu::md::NeighborList<>{cutoff, skin}, -10);
    sim.on(ATV::md::Phase::force).add(cpu::PairLJCut{kEps, kSigma, cutoff});
    sim.on(ATV::md::Phase::force).add_ref(probe, +500);
    sim.build();
    sim.run(frame, 1);
}

/// @brief One narrow-subdomain case: force `grid`, run one frozen step of
/// `frame` at cutoff 6 A, and check the energy against the in-file O(N^2)
/// minimum-image oracle.
///
/// Every rank runs the same geometry, so the checks are collective by
/// construction: `n_local` is summed (the lattice must be conserved across
/// the decomposition, including ranks that own nothing) and `n_ghost` is
/// max-reduced (a rank with zero owners still receives ghosts, but the
/// meaningful statement is that the shell is not empty anywhere).
///
/// @param grid `ATV_FORCE_DIMS` value, e.g. "4,1,1".
/// @param frame Lattice to run; a pristine copy feeds the oracle.
/// @param skin Verlet skin [A] (negative = the NeighborList auto default).
/// @param out_probe Optional: receives this rank's post-halo counts and the
///        derived per-axis hop counts, so a caller can pin the shell shape
///        its comment claims.
void expect_narrow_subdomain_matches_brute_force(
    const char* grid, cpu::Frame frame, double skin = -1.0,
    GhostCountProbe* out_probe = nullptr) {
    ForcedDims forced{grid};

    const cpu::Frame ref = frame; // pristine copy for the oracle
    const int n_atoms = ref.nlocal();
    const double e_bf = brute_force_energy(ref, 6.0);
    ASSERT_GT(std::abs(e_bf), 1e-9);

    GhostCountProbe local_probe;
    GhostCountProbe& probe = (out_probe != nullptr) ? *out_probe : local_probe;
    run_one_frozen_step(frame, 6.0, probe, skin);
    ASSERT_EQ(probe.launches, 1) << "the probe never ran: test is vacuous";

    const double rel =
        std::abs(frame.meta.get(ATV::kEnergy) - e_bf) / std::abs(e_bf);
    EXPECT_LT(rel, kEnergyRelativeTolerance)
        << "ATV_FORCE_DIMS=" << grid << ": distributed "
        << frame.meta.get(ATV::kEnergy) << " vs brute force " << e_bf;

    int owned_total = 0;
    MPI_Allreduce(&probe.n_local, &owned_total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_EQ(owned_total, n_atoms)
        << "atoms lost or duplicated by the narrow decomposition";

    int ghost_max = 0;
    MPI_Allreduce(&probe.n_ghost, &ghost_max, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_GT(ghost_max, 0) << "no ghosts: the multi-hop shell is empty";
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

// ── narrow subdomains: the ghost shell spans several ranks ─────────────
//
// Same 3^3 lattice and same 6 A cutoff (auto halo 6.6 A) as the dims == 2
// case above, cut finer along x. The shell is now WIDER than a subdomain,
// so filling it needs ghosts forwarded through intermediate ranks —
// `max_hops = ceil(6.6 / sub)` rounds per direction. comm-topology-02
// fills the shell by forwarding ghosts over that many rounds; the only
// remaining refusal is the minimum-image bound
// (`SwapList` refuse_minimum_image). The `1e-9` oracle window is the
// same one the n1 / n2 cases use: spatial decomposition may not move
// the total energy at all.

// 4 ranks: sub = 15.78 / 4 = 3.945 A < 6.6 A => max_hops = 2 along x.
TEST(HaloDualImageGhosts, NarrowSubdomainFourRanksMatchBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 4)
        GTEST_SKIP() << "narrow 4-rank x split; needs exactly 4 ranks";

    expect_narrow_subdomain_matches_brute_force("4,1,1",
                                                make_sc_lattice(3, 5.26));
}

// 8 ranks: sub = 15.78 / 8 = 1.9725 A => max_hops = 4 along x, still inside
// the minimum-image bound (4 <= dims - 1 = 7). 27 atoms over 8 ranks also
// covers the zero-owned-rank edge: the x planes of the lattice sit at
// 2.63 / 7.89 / 13.15 A, so several slabs own nothing and must still take
// part in every swap round.
TEST(HaloDualImageGhosts, NarrowSubdomainEightRanksMatchBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 8)
        GTEST_SKIP() << "narrow 8-rank x split; needs exactly 8 ranks";

    expect_narrow_subdomain_matches_brute_force("8,1,1",
                                                make_sc_lattice(3, 5.26));
}

// ── production proxy: TWO decomposed axes at once ──────────────────────
//
// The 8x8x4 / 256-rank shape of `comm-topology-03` reduced to 8 ranks:
// `ATV_FORCE_DIMS=4,2,1` over a 24 x 14 x 14 A box at cutoff 6.0 A with an
// EXPLICIT 0.6 A skin, so the auto-injected shell is
// `max(1.1 * 6.0, 6.0 + 0.6) = 6.6 A` — the number the geometry below is
// argued from. The two decomposed axes hit two DIFFERENT branches in the
// same run, which is the point of the case:
//
//   * x: dims 4, span 24/4 = 6.0 A < 6.6 A  => `max_hops = 2`. The shell
//     cannot be filled by one face exchange; round 2 forwards what round 1
//     received (comm-topology-02), and 2 <= dims - 1 = 3 keeps it inside
//     the minimum-image bound.
//   * y: dims 2, span 14/2 = 7.0 A >= 6.6 A => a SINGLE hop, but
//     2 * 6.6 = 13.2 A > 7.0 A, so the +y and -y partner are the same rank
//     and every atom in the 6.2 A overlap band ships through BOTH faces —
//     the dual-image branch this file exists for.
//   * z: dims 1 => zero hops; periodicity there is the pair kernel's
//     minimum image.
//
// Why 14 A on y and not the 12 A of a 24x12x12 box: at dims_y = 2 the deepest
// legal hop is dims - 1 = 1 (round 2 would wrap onto this rank's own periodic
// image), so `SwapList::derive_hops` refuses `ceil(6.6 / 6.0) = 2`
// (`src/cpu/communication/swap_list.cpp:191`) with
//   "the requested ghost shell (6.6 A) violates the minimum-image convention
//    on periodic axis 1 ... the bound is shell <= L - w_self = 12 - 6 = 6 A".
// A 2-hop x axis and a 2-hop y axis at dims_y = 2 therefore cannot coexist;
// span_y >= halo is the legality condition, and 7.0 A is the narrowest span
// that keeps y inside it while still overlapping its own two faces.
//
// Lattice: 8 x 4 x 4 sites at (3.0, 3.5, 3.5) A spacing = 128 atoms. Two
// sites per x subdomain (planes at 1.5 + 3k A vs cuts at 0/6/12/18 A: no
// site on a partition plane), and cutoff 6.0 A <= L_d/2 on every axis
// (12 / 7 / 7 A) keeps the flat minimum-image oracle legal.
TEST(HaloDualImageGhosts, ProductionGridTwoNarrowAxesMatchBruteForce) {
    int world = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 8)
        GTEST_SKIP() << "4x2x1 production proxy; needs exactly 8 ranks";

    GhostCountProbe probe;
    expect_narrow_subdomain_matches_brute_force(
        "4,2,1", make_sc_lattice(8, 4, 4, 3.0, 3.5, 3.5), /*skin=*/0.6, &probe);

    // Without this the case would still pass on a SINGLE-hop x shell — the
    // energy identity alone cannot tell which shell produced it.
    EXPECT_GE(probe.hops[0], 2)
        << "x shell degraded to a single hop: span 6.0 A vs halo 6.6 A must "
           "derive ceil(6.6 / 6.0) = 2 forwarding rounds";
    EXPECT_EQ(probe.hops[1], 1)
        << "y (dims 2, span 7.0 A >= halo 6.6 A) must be a single hop; a "
           "second round there would wrap onto this rank itself";
    EXPECT_EQ(probe.hops[2], 0) << "z is undecomposed: zero hops";
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
