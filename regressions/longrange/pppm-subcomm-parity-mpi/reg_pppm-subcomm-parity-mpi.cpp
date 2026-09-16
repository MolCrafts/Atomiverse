/// @file regressions/longrange/pppm-subcomm-parity-mpi/reg_pppm-subcomm-parity-mpi.cpp
/// @brief END-TO-END: rock-salt NaCl reproduces the cited Madelung energy, and
///        confining the SPME FFT slabs to a SUBGROUP of the Cartesian
///        communicator changes nothing but the order of the summands.
///
/// See REFERENCE.md. Public headers + link `atomiverse` only. `oracle:
/// literature` — the NaCl Madelung constant M = 1.747564594633182 (Kittel 8th
/// ed. ch. 3; Sherman, Chem. Rev. 11, 93 (1932), DOI 10.1021/cr60038a002) is a
/// hard-coded literal in this file, and the arm-vs-arm criteria are the
/// *identity* "a decomposition is a partition of the same sum" (Plimpton,
/// J. Comput. Phys. 117, 1 (1995), DOI 10.1006/jcph.1995.1039). No third-party
/// engine is imported, spawned or read at build or run time. `perf_exempt`:
/// this slug makes no performance claim — the three-arm wall-clock matrix
/// belongs to `tools/bench_gaff_lammps/README.md`.
///
/// One binary, two modes:
///  - parent (default): re-launches itself through the configured mpiexec
///    (compile-time pin ATV_MPIEXEC) three times — np = 1, np = 8, and np = 8
///    with `ATV_PPPM_BRICK=1` — then gates the three child reports against the
///    literature oracle and against each other.
///  - child (`--child <out>`): MPI_Init, one static `cpu::Evaluate` pass of
///    CoulLong + KSpacePPPM on 216 rock-salt ions under a forced 2x2x2
///    decomposition, then — from an out-of-library Component at
///    `evaluate::Phase::finalize`, i.e. after `md::mpi::Reduce` has made the
///    Frame scalars global — allgathers every owned row by tag and reports the
///    total energy, the per-atom forces, and the solver's OWN account of which
///    mesh arm it took.
///
/// Why these three legs (spec `kspace-subcomm`, ac-008 / ac-009):
///   * np = 1  -> `PppmMeshArm::Replicated`: the serial control that carries
///     the literature window.
///   * np = 8  -> `PppmMeshArm::BrickSubgroup`: K0 = 12 mesh planes and
///     `pppm_fft_ranks(12, 8) = 6`, so the FFTW-MPI slab lives on 6 of the 8
///     ranks (2 planes each) and TWO ranks never enter an FFT collective. They
///     still spread, halo, remap and interpolate — which is exactly the path
///     that a wrong `fft_owner` image or a subgroup-scoped energy reduction
///     would corrupt.
///   * np = 8 + `ATV_PPPM_BRICK=1` -> `PppmMeshArm::BrickWorld`: the previous
///     two-arm behaviour (ragged FFTW-MPI slabs 2,2,2,2,2,2,0,0 over all 8
///     ranks) on the SAME frozen binary, so the subgroup arm is compared
///     against a real alternative decomposition rather than against itself.
///
/// Anti-false-green: every leg self-reports `mesh_arm()` and `fft_ranks()`
/// straight off the `KSpacePPPM` instance, min/max-reduced across ranks, and
/// the parent FAILS on a mismatch. Without it an implementation that silently
/// fell back to the world arm would print PASS on every physics gate.
///
/// Geometry rationale: 6x6x6 simple-cubic alternating +/-1 e sites at
/// d = 2.82 A give L = 16.92 A and 216 atoms (Sum q = 0). The real-space
/// cutoff is 7.0 A: the auto-injected ghost shell is
/// max(1.1 * 7.0, 7.0 + 0.3) = 7.7 A, which fits inside the 8.46 A slabs of
/// the forced 2x2x2 split (`ceil(7.7 * 2 / 16.92) = 1` hop, the deepest legal
/// hop at dims = 2), and 7.0 A <= L/2 = 8.46 A keeps the pair kernel's minimum
/// image well defined. The mesh spacing 1.45 A resolves to
/// `good_fft_size(ceil(16.92 / 1.45)) = good_fft_size(12) = 12` planes per
/// dimension — asserted in-code below, because the whole point of the case is
/// that 12 has a proper divisor <= 8.
///
/// The lattice is rigidly displaced by exactly ONE mesh cell (L / 12 = 1.41 A)
/// along each axis. The exact Coulomb force vanishes by rock-salt symmetry
/// wherever the lattice sits, so the displacement is not physics — it moves
/// every ion off the periodic boundary and off the 8.46 A domain cut planes
/// (nearest ion 1.41 A away) while keeping d = 2.82 A an exact multiple of the
/// mesh spacing, i.e. every ion still on a mesh node. That is what makes gate
/// (4) sharp rather than approximate: with the B-spline stencil
/// inversion-symmetric about each ion the mesh force cancels to the ULP, so a
/// mesh pipeline that lost a plane, mis-addressed an FFT owner or skipped a
/// halo would break the cancellation and show up as a force.
///
/// Off-node ions would NOT do: `cpu::KSpacePPPM` interpolates forces by pure
/// analytic differentiation (`src/cpu/longrange/pppm.cpp:1178`) with no
/// self-force correction (LAMMPS `PPPM::compute_sf_precoeff` /
/// `fieldforce_ad`), so an ion displaced to a generic sub-cell fraction feels a
/// spurious force proportional to q_i^2 — same sign on every ion, ~1.4e-3 eV/A
/// each at this resolution, summing to ~0.3 eV/A over 216 ions. That is a
/// pre-existing accuracy artifact of the AD scheme, identical on all three
/// arms, and it is not what this case is about.

#include <mpi.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme_spline.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;

namespace {

// ── Crystal: rock-salt NaCl, conventional simple-cubic description ──────
constexpr int kSide = 6;                       // ions per edge
constexpr double kD = 2.82;                    // Na-Cl nearest neighbour [A]
constexpr double kBoxL = kSide * kD;           // 16.92 A cubic cell
constexpr int kNAtoms = kSide * kSide * kSide; // 216
constexpr double kMassNa = 22.9898, kMassCl = 35.453; // [amu]

// ── Ewald split ────────────────────────────────────────────────────────
constexpr double kCutoff = 7.0; // real-space cutoff [A]; <= L/2 = 8.46
constexpr double kSkin = 0.3;   // Verlet skin [A]; shell = max(1.1 rc, rc+skin)
constexpr double kAlpha = 0.40; // Ewald splitting [1/A]; alpha*rc = 2.8
constexpr double kSpacing = 1.45; // target mesh spacing [A] -> K = 12 per dim
constexpr int kOrder = 5;         // B-spline order (needs K >= order + 1 = 6)

// ── Decomposition ──────────────────────────────────────────────────────
constexpr int kRanksX = 2, kRanksY = 2, kRanksZ = 2;     // 8.46 A slabs
constexpr int kNpParallel = kRanksX * kRanksY * kRanksZ; // 8
constexpr int kExpectedK0 = 12;      // mesh planes along dim 0
constexpr int kExpectedFftRanks = 6; // pppm_fft_ranks(12, 8)

/// One mesh cell, 16.92 / 12 = 1.41 A. Used as the lattice origin so every ion
/// sits on a mesh node (d = 2.82 A is exactly two cells) yet no ion sits on the
/// periodic boundary or on the 8.46 A domain cut. See the file header.
constexpr double kOrigin = kBoxL / kExpectedK0;

// ── Literature oracle (hard-coded literals; see REFERENCE.md) ──────────
/// NaCl rock-salt Madelung constant. Kittel, *Introduction to Solid State
/// Physics*, 8th ed., ch. 3; J. Sherman, Chem. Rev. 11, 93 (1932),
/// DOI 10.1021/cr60038a002.
constexpr double kMadelung = 1.747564594633182;
/// Coulomb constant [eV*A/e^2]. Mirrors `ATV::units::coulomb`; the equality is
/// checked at run time so this literal cannot silently drift from the engine's.
constexpr double kCoulomb = 14.399645;

// ── Gates (hard-coded; see REFERENCE.md) ───────────────────────────────
/// Grid-discretization window of a K = 12 / order-5 SPME mesh vs the exact
/// lattice sum. NOT a float tolerance — it is the physical size of the mesh
/// truncation error at this resolution.
constexpr double kMadelungRelTol = 2.0e-3;
/// Decomposition invariance: two decompositions evaluate the SAME sum, so only
/// the floating-point association may differ.
constexpr double kDecompRelTol = 1.0e-9;
constexpr double kDecompForceTol = 1.0e-9; // [eV/A], tag-aligned
/// Newton's third law over the whole cell.
constexpr double kNetForceTol = 1.0e-10; // [eV/A]
/// Residual per-ion force: zero by rock-salt symmetry up to the mesh artifact.
constexpr double kMaxForceTol = 5.0e-3; // [eV/A]

/// @brief Human-readable arm name for the report lines.
const char* arm_label(int arm) {
    switch (arm) {
    case 0:
        return "Replicated";
    case 1:
        return "BrickWorld";
    case 2:
        return "BrickSubgroup";
    default:
        return "<invalid>";
    }
}

/// @brief The system: 216 rock-salt ions, alternating +/-1 e, net neutral.
///
/// Site (i,j,k) carries q = (-1)^(i+j+k), so the 108 cations and 108 anions
/// tile the conventional NaCl cell and the lattice sum of
/// `E = -1/2 N M C / d` is the exact electrostatic energy of this
/// configuration under 3D periodicity.
///
/// @return Frame with positions [A], charges [e], masses [amu] and 1-based
///         tags; no topology (an ionic crystal has no bonds, so CoulLong needs
///         no special-bond lists).
cpu::Frame make_rocksalt() {
    cpu::Frame f(kNAtoms);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    auto mass = f.span(kAtom_mass);
    auto charge = f.span(kAtom_charge);
    auto type = f.span(kAtom_type);
    auto tag = f.span(kAtom_tag);

    int idx = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k) {
                const bool cation = ((i + j + k) % 2 == 0);
                x[idx] = static_cast<StorageReal>(kOrigin + i * kD);
                y[idx] = static_cast<StorageReal>(kOrigin + j * kD);
                z[idx] = static_cast<StorageReal>(kOrigin + k * kD);
                vx[idx] = vy[idx] = vz[idx] = 0.0;
                charge[idx] = static_cast<StorageReal>(cation ? 1.0 : -1.0);
                mass[idx] =
                    static_cast<StorageReal>(cation ? kMassNa : kMassCl);
                type[idx] = cation ? 0 : 1;
                tag[idx] = idx + 1;
                ++idx;
            }
    f.box = cpu::Box::cubic(kBoxL);
    return f;
}

/// @brief Out-of-library `finalize` Component: assemble the distributed state.
///
/// Bound at `evaluate::Phase::finalize` @ +900, i.e. after `md::mpi::Reduce`
/// (evaluate @ +20) has turned the Frame scalars into system totals and long
/// after `CommBrick` (evaluate @ +10) folded every ghost force back onto its
/// owner. Every rank ends up holding the same tag-ordered arrays, so the
/// child's report is independent of which rank writes it.
class RockSaltChecker : public cpu::Component {
  public:
    std::string name() const override { return "pppm_subcomm_checker"; }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto& f = ctx.frame;
        auto* dd = ctx.runtime.md.dd;
        const int nloc = dd ? dd->n_local : f.nlocal();

        const auto tag_col = f.span(kAtom_tag);
        const auto fx_col = f.span(kAtom_fx);
        const auto fy_col = f.span(kAtom_fy);
        const auto fz_col = f.span(kAtom_fz);

        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        std::vector<int> counts3(static_cast<std::size_t>(size), 0);
        std::vector<int> displs3(static_cast<std::size_t>(size), 0);
        int total = 0;
        for (int r = 0; r < size; ++r) {
            const std::size_t u = static_cast<std::size_t>(r);
            displs[u] = total;
            displs3[u] = 3 * total;
            counts3[u] = 3 * counts[u];
            total += counts[u];
        }
        n_gathered_ = total;

        std::vector<long long> my_tags(static_cast<std::size_t>(nloc));
        std::vector<double> my_rows(static_cast<std::size_t>(3 * nloc));
        for (int i = 0; i < nloc; ++i) {
            const std::size_t u = static_cast<std::size_t>(i);
            my_tags[u] = static_cast<long long>(tag_col[i]);
            my_rows[3 * u + 0] = static_cast<double>(fx_col[i]);
            my_rows[3 * u + 1] = static_cast<double>(fy_col[i]);
            my_rows[3 * u + 2] = static_cast<double>(fz_col[i]);
        }
        std::vector<long long> all_tags(static_cast<std::size_t>(total));
        std::vector<double> all_rows(static_cast<std::size_t>(3 * total));
        MPI_Allgatherv(my_tags.data(), nloc, MPI_LONG_LONG, all_tags.data(),
                       counts.data(), displs.data(), MPI_LONG_LONG,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(my_rows.data(), 3 * nloc, MPI_DOUBLE, all_rows.data(),
                       counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        energy_ = f.meta.get(kEnergy);
        ++launches_;
        if (total != kNAtoms) {
            ++assembly_errors_;
            return;
        }
        const std::size_t n = static_cast<std::size_t>(kNAtoms);
        force_.assign(3 * n, 0.0);
        std::vector<int> seen(n, 0);
        for (int i = 0; i < total; ++i) {
            const long long t = all_tags[static_cast<std::size_t>(i)];
            if (t < 1 || t > kNAtoms) {
                ++assembly_errors_;
                return;
            }
            const std::size_t u = static_cast<std::size_t>(t - 1);
            ++seen[u];
            const std::size_t s = 3 * static_cast<std::size_t>(i);
            force_[3 * u + 0] = all_rows[s + 0];
            force_[3 * u + 1] = all_rows[s + 1];
            force_[3 * u + 2] = all_rows[s + 2];
        }
        for (std::size_t u = 0; u < n; ++u)
            if (seen[u] != 1)
                ++assembly_errors_;

        double sfx = 0.0, sfy = 0.0, sfz = 0.0, fmax = 0.0;
        for (std::size_t u = 0; u < n; ++u) {
            sfx += force_[3 * u + 0];
            sfy += force_[3 * u + 1];
            sfz += force_[3 * u + 2];
            fmax = std::fmax(
                fmax, std::sqrt(force_[3 * u + 0] * force_[3 * u + 0]
                                + force_[3 * u + 1] * force_[3 * u + 1]
                                + force_[3 * u + 2] * force_[3 * u + 2]));
        }
        net_force_ = std::sqrt(sfx * sfx + sfy * sfy + sfz * sfz);
        max_force_ = fmax;
    }

    std::vector<double> force_;     ///< forces by tag, xyz [eV/A]
    double energy_ = 0.0;           ///< reduced total energy [eV]
    double net_force_ = -1.0;       ///< ||sum_i F_i|| [eV/A]
    double max_force_ = -1.0;       ///< max_i |F_i| [eV/A]
    long long assembly_errors_ = 0; ///< lost / duplicated owned tags
    int n_gathered_ = -1;           ///< owned rows assembled, must be 216
    int launches_ = 0;
};

int run_child(const char* out_path) {
    // Deterministic 2x2x2 split: 8.46 A slabs against a 7.7 A ghost shell.
    // Ignored at np = 1 (the override is only honoured when dx*dy*dz equals
    // the world size), which is what makes the np = 1 leg a serial control.
    ::setenv("ATV_FORCE_DIMS", "2,2,2", 1);
    MPI_Init(nullptr, nullptr);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Precondition, not a gate: if the box/spacing pair stopped resolving to
    // 12 planes the subgroup arm would never be reachable and every physics
    // gate below would be measuring the world arm twice.
    const int k0 =
        longrange::good_fft_size(static_cast<int>(std::ceil(kBoxL / kSpacing)));
    if (k0 != kExpectedK0) {
        if (rank == 0)
            std::printf("FAIL: child mesh premise broken: good_fft_size("
                        "ceil(%.2f / %.2f)) = %d, want %d\n",
                        kBoxL, kSpacing, k0, kExpectedK0);
        MPI_Finalize();
        return 1;
    }

    auto frame = make_rocksalt();

    // Held by reference: `mesh_arm()` / `fft_ranks()` are read off THIS object
    // after the pass, so the evaluator must not be moved into the driver.
    cpu::KSpacePPPM kspace{kAlpha, kSpacing, kOrder};

    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    eval.set_evaluator(cpu::CoulLong{kAlpha, kCutoff});
    eval.set_evaluator_ref(kspace);

    RockSaltChecker checker;
    eval.on(ATV::evaluate::Phase::finalize).add_ref(checker, +900);

    eval.build();
    eval.run(frame);

    // The arm is a property of the SOLVE, so it must agree on every rank —
    // a per-rank divergence here is a split that some ranks did not join.
    int arm_min = static_cast<int>(kspace.mesh_arm());
    int arm_max = arm_min;
    int fft_min = kspace.fft_ranks();
    int fft_max = fft_min;
    MPI_Allreduce(MPI_IN_PLACE, &arm_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &arm_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fft_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &fft_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    double e_min = checker.energy_, e_max = checker.energy_;
    MPI_Allreduce(MPI_IN_PLACE, &e_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &e_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    long long errs = checker.assembly_errors_;
    MPI_Allreduce(MPI_IN_PLACE, &errs, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out) {
            std::printf("FAIL: child cannot open %s\n", out_path);
            rc = 1;
        } else {
            std::fprintf(out,
                         "%.17e %.17e %.17e %.17e %.17e %d %d %d %d %d %lld "
                         "%d %d\n",
                         checker.energy_, e_min, e_max, checker.net_force_,
                         checker.max_force_, arm_min, arm_max, fft_min, fft_max,
                         checker.n_gathered_, errs, checker.launches_, size);
            for (int i = 0; i < kNAtoms; ++i) {
                const std::size_t u = static_cast<std::size_t>(i);
                const bool have = checker.force_.size() >= 3 * u + 3;
                std::fprintf(out, "%.17e %.17e %.17e\n",
                             have ? checker.force_[3 * u + 0] : 0.0,
                             have ? checker.force_[3 * u + 1] : 0.0,
                             have ? checker.force_[3 * u + 2] : 0.0);
            }
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

/// @brief One child report; every field is measured, never a default that
///        could pass a gate by accident.
struct Report {
    double energy = 0.0;
    double e_min = 0.0, e_max = 0.0;
    double net_force = -1.0, max_force = -1.0;
    int arm_min = -1, arm_max = -1;
    int fft_min = -1, fft_max = -1;
    int n_gathered = -1;
    long long assembly_errors = -1;
    int launches = -1, world = -1;
    std::vector<double> force; ///< 3 * kNAtoms, by tag [eV/A]
};

/// @brief Re-launch this binary under mpiexec and read back the child report.
/// @param self Absolute path of this executable.
/// @param np Rank count.
/// @param force_world_arm When true, export `ATV_PPPM_BRICK=1` for the child
///        so the solver takes the world-communicator brick arm.
/// @param out Parsed report.
/// @return true on a clean child exit with a well-formed report.
bool run_under_mpiexec(const std::string& self, int np, bool force_world_arm,
                       Report& out) {
    const std::string path =
        "/tmp/atv-pppm-subcomm-parity-np" + std::to_string(np)
        + (force_world_arm ? "-world-" : "-default-")
        + std::to_string(static_cast<long>(::getpid())) + ".txt";
    // The debug forcing travels through the environment (Hydra exports the
    // launching environment to every rank), so all three legs run the SAME
    // frozen binary — an arm difference cannot come from a different build.
    if (force_world_arm)
        ::setenv("ATV_PPPM_BRICK", "1", 1);
    else
        ::unsetenv("ATV_PPPM_BRICK");

    const std::string cmd = std::string(ATV_MPIEXEC) + " -n "
                            + std::to_string(np) + " \"" + self
                            + "\" --child \"" + path + "\"";
    std::printf("[pppm-subcomm-parity] launching: %s%s\n",
                force_world_arm ? "ATV_PPPM_BRICK=1 " : "", cmd.c_str());
    std::fflush(stdout);
    const int sys_rc = std::system(cmd.c_str());
    ::unsetenv("ATV_PPPM_BRICK");
    if (sys_rc != 0) {
        std::printf("FAIL: np=%d child exited nonzero\n", np);
        return false;
    }
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing np=%d child report %s\n", np, path.c_str());
        return false;
    }
    const int got = std::fscanf(
        in, "%lf %lf %lf %lf %lf %d %d %d %d %d %lld %d %d", &out.energy,
        &out.e_min, &out.e_max, &out.net_force, &out.max_force, &out.arm_min,
        &out.arm_max, &out.fft_min, &out.fft_max, &out.n_gathered,
        &out.assembly_errors, &out.launches, &out.world);
    int force_read = 0;
    out.force.assign(3 * static_cast<std::size_t>(kNAtoms), 0.0);
    for (int i = 0; i < kNAtoms; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        force_read += std::fscanf(in, "%lf %lf %lf", &out.force[3 * u + 0],
                                  &out.force[3 * u + 1], &out.force[3 * u + 2]);
    }
    std::fclose(in);
    std::remove(path.c_str());
    if (got != 13 || force_read != 3 * kNAtoms) {
        std::printf("FAIL: malformed np=%d child report (%d header fields, "
                    "%d force fields)\n",
                    np, got, force_read);
        return false;
    }
    return true;
}

/// @brief Print one numeric gate line and tally it.
/// @return 0 when the gate passes, 1 when it fails.
int gate(const char* label, double got, double tol, bool ok) {
    std::printf("  %-42s got %.3e  want <= %.0e  -> %s\n", label, got, tol,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// @brief Print one integral gate line and tally it.
/// @return 0 when the gate passes, 1 when it fails.
int gate_int(const char* label, long long got, long long want, bool ok) {
    std::printf("  %-42s got %lld  want %lld           -> %s\n", label, got,
                want, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// @brief Structural sanity of one leg: every atom owned exactly once, the
///        pass really ran, and every rank agreed on the reduced energy.
int gate_assembly(const char* leg, const Report& r) {
    const bool ok = r.n_gathered == kNAtoms && r.assembly_errors == 0
                    && r.launches == 1
                    && std::fabs(r.e_max - r.e_min)
                           <= 1.0e-10 * std::fmax(1.0, std::fabs(r.energy));
    std::printf("  %-42s got %d / %lld / %d / %.3e  -> %s\n",
                (std::string(leg) + " atoms/err/pass/rank-spread").c_str(),
                r.n_gathered, r.assembly_errors, r.launches,
                std::fabs(r.e_max - r.e_min), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// @brief Per-leg physical invariants: Newton's third law and the rock-salt
///        symmetry bound on the residual per-ion force.
int gate_forces(const char* leg, const Report& r) {
    int bad = 0;
    bad +=
        gate((std::string(leg) + " ||sum_i F_i|| [eV/A]").c_str(), r.net_force,
             kNetForceTol, r.net_force >= 0.0 && r.net_force <= kNetForceTol);
    bad +=
        gate((std::string(leg) + " max_i |F_i| [eV/A]").c_str(), r.max_force,
             kMaxForceTol, r.max_force >= 0.0 && r.max_force <= kMaxForceTol);
    return bad;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);

    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::printf("FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }
    self_buf[n] = '\0';
    const std::string self(self_buf);

    ::setenv("OMP_NUM_THREADS", "1", 1);

    int failures = 0;

    // ── (0) Preconditions. These are pure functions of the public header, so
    //        they are evaluated here, in the parent, before a single rank is
    //        launched: if 12 stopped being the mesh extent, or if 6 stopped
    //        being the largest divisor of 12 that fits 8 ranks, then the
    //        "subgroup" leg would silently be a world leg and every physics
    //        gate below would compare the same arm with itself.
    std::printf("[pppm-subcomm-parity] preconditions "
                "(%d ions, d = %.2f A, L = %.2f A, spacing %.2f A)\n",
                kNAtoms, kD, kBoxL, kSpacing);
    const int k0 =
        longrange::good_fft_size(static_cast<int>(std::ceil(kBoxL / kSpacing)));
    failures += gate_int("good_fft_size(ceil(L / spacing))", k0, kExpectedK0,
                         k0 == kExpectedK0);
    const int fft_ranks = cpu::pppm_fft_ranks(kExpectedK0, kNpParallel);
    failures += gate_int("pppm_fft_ranks(12, 8)", fft_ranks, kExpectedFftRanks,
                         fft_ranks == kExpectedFftRanks);
    const int want_arm = static_cast<int>(
        cpu::pppm_select_mesh_arm(kExpectedK0, kNpParallel, fft_ranks));
    failures +=
        gate_int("pppm_select_mesh_arm -> BrickSubgroup", want_arm,
                 static_cast<int>(cpu::PppmMeshArm::BrickSubgroup),
                 want_arm == static_cast<int>(cpu::PppmMeshArm::BrickSubgroup));
    // The Madelung literal below is multiplied by the Coulomb constant spelled
    // out in this file; if the engine's constant ever moved, the "literature"
    // energy would quietly stop being the engine's own units.
    failures += gate("|kCoulomb - units::coulomb| [eV*A/e^2]",
                     std::fabs(kCoulomb - static_cast<double>(units::coulomb)),
                     0.0, kCoulomb == static_cast<double>(units::coulomb));

    Report serial, subgroup, world;
    if (!run_under_mpiexec(self, 1, /*force_world_arm=*/false, serial))
        return 1;
    if (!run_under_mpiexec(self, kNpParallel, /*force_world_arm=*/false,
                           subgroup))
        return 1;
    if (!run_under_mpiexec(self, kNpParallel, /*force_world_arm=*/true, world))
        return 1;

    // ── (1) Literature oracle: the exact rock-salt lattice sum.
    //        E = -1/2 N M C / d, with M the cited Madelung constant.
    const double e_madelung = -0.5 * kNAtoms * kMadelung * kCoulomb / kD;
    std::printf("[pppm-subcomm-parity] stage 1 — literature oracle "
                "(E_Madelung = %.10f eV)\n",
                e_madelung);
    failures += gate_assembly("np=1", serial);
    const double rel_madelung = std::fabs(serial.energy / e_madelung - 1.0);
    failures += gate("|E(np1) / E_Madelung - 1|", rel_madelung, kMadelungRelTol,
                     rel_madelung <= kMadelungRelTol);

    // ── (2) Decomposition invariance of the energy: np = 8 subgroup vs np = 1.
    std::printf("[pppm-subcomm-parity] stage 2 — subgroup arm vs the serial "
                "control (np=%d, ATV_FORCE_DIMS=%d,%d,%d)\n",
                kNpParallel, kRanksX, kRanksY, kRanksZ);
    failures += gate_assembly("np=8 sub", subgroup);
    const double rel_sub_serial =
        std::fabs(subgroup.energy - serial.energy) / std::fabs(serial.energy);
    failures += gate("|E(np8,sub) - E(np1)| / |E(np1)|", rel_sub_serial,
                     kDecompRelTol, rel_sub_serial <= kDecompRelTol);

    // ── (3) Arm vs arm on the same rank count: the ONLY difference between
    //        these two runs is which communicator the FFT slabs live on.
    std::printf("[pppm-subcomm-parity] stage 3 — subgroup arm vs world arm "
                "(ATV_PPPM_BRICK=1), same np, same binary\n");
    failures += gate_assembly("np=8 world", world);
    const double rel_arms =
        std::fabs(subgroup.energy - world.energy) / std::fabs(world.energy);
    failures += gate("|E(np8,sub) - E(np8,world)| / |E|", rel_arms,
                     kDecompRelTol, rel_arms <= kDecompRelTol);
    double max_df = 0.0;
    int worst_tag = -1;
    for (int i = 0; i < 3 * kNAtoms; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        const double d = std::fabs(subgroup.force[u] - world.force[u]);
        if (d > max_df) {
            max_df = d;
            worst_tag = i / 3 + 1;
        }
    }
    failures += gate("max |F_i(sub) - F_i(world)| [eV/A]", max_df,
                     kDecompForceTol, max_df <= kDecompForceTol);
    if (max_df > kDecompForceTol)
        std::printf("    worst tag %d\n", worst_tag);

    // ── (4) Per-leg physical invariants.
    std::printf("[pppm-subcomm-parity] stage 4 — momentum conservation and "
                "rock-salt symmetry, per leg\n");
    failures += gate_forces("np=1", serial);
    failures += gate_forces("np=8 sub", subgroup);
    failures += gate_forces("np=8 world", world);

    // ── (5) Anti-false-green: the solver's own account of the arm it took.
    std::printf("[pppm-subcomm-parity] stage 5 — arm self-report "
                "(0=Replicated, 1=BrickWorld, 2=BrickSubgroup)\n");
    auto gate_arm = [&](const char* leg, const Report& r, int want_arm_id,
                        int want_fft) {
        const bool ok = r.arm_min == want_arm_id && r.arm_max == want_arm_id
                        && r.fft_min == want_fft && r.fft_max == want_fft
                        && r.world == (want_arm_id == 0 ? 1 : kNpParallel);
        std::printf("  %-42s got arm %s[%d,%d] fft_ranks [%d,%d] world %d  "
                    "want %s / %d -> %s\n",
                    leg, arm_label(r.arm_min), r.arm_min, r.arm_max, r.fft_min,
                    r.fft_max, r.world, arm_label(want_arm_id), want_fft,
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    };
    failures += gate_arm("np=1 self-report", serial,
                         static_cast<int>(cpu::PppmMeshArm::Replicated), 1);
    failures += gate_arm("np=8 default self-report", subgroup,
                         static_cast<int>(cpu::PppmMeshArm::BrickSubgroup),
                         kExpectedFftRanks);
    failures +=
        gate_arm("np=8 ATV_PPPM_BRICK=1 self-report", world,
                 static_cast<int>(cpu::PppmMeshArm::BrickWorld), kNpParallel);

    std::printf("[pppm-subcomm-parity] E(np=1)         = %.16f eV\n",
                serial.energy);
    std::printf("[pppm-subcomm-parity] E(np=8, sub)    = %.16f eV\n",
                subgroup.energy);
    std::printf("[pppm-subcomm-parity] E(np=8, world)  = %.16f eV\n",
                world.energy);
    std::printf("[pppm-subcomm-parity] E_Madelung      = %.16f eV\n",
                e_madelung);
    if (failures == 0)
        std::printf("PASS\n");
    else
        std::printf("FAIL (%d gate%s)\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
