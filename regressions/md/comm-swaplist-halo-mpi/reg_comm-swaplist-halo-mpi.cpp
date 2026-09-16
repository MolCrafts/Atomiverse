/// @file regressions/md/comm-swaplist-halo-mpi/reg_comm-swaplist-halo-mpi.cpp
/// @brief END-TO-END: the swap-list halo reproduces the single-image
///        O(N^2) minimum-image LJ sum under MPI domain decomposition.
///
/// See REFERENCE.md. Public headers + link atomiverse only. oracle=literature:
/// the criterion is the *identity* "a domain decomposition is a partition of
/// the same minimum-image sum", evaluated in-file — no third-party engine is
/// imported, spawned or read at build or run time. perf_exempt: this slug
/// makes no performance claim.
///
/// One binary, two modes:
///  - parent (default): re-launches itself through the configured mpiexec
///    (compile-time pin ATV_MPIEXEC) at np = 2 and at np = 1, then gates the
///    two child reports.
///  - child (`--child <out>`): MPI_Init, run 21 MD steps of a 512-atom
///    simple-cubic argon lattice (PairLJCut + NeighborList + VelocityVerlet,
///    halo AUTO-INJECTED by the MD preset), then — from an out-of-library
///    Component at end_of_step — allgather every owned row by tag, recompute
///    the total energy and every per-atom force as a flat O(N^2) min-image
///    sum, and report the deviations plus the published swap-table tallies.
///
/// Geometry rationale (the reason this case exists at all): forcing a 2x1x1
/// decomposition of a 28.936 A cubic box gives 14.468 A subdomains against a
/// 9.35 A ghost shell (1.1 x the 8.5 A pair cutoff, the MD preset's rule), so
///   * need_x = ceil(9.35 / 14.468) = 1        -> the single-hop swap table,
///   * 2 x 9.35 = 18.7 > 14.468                -> the lo-face and hi-face
///     bands OVERLAP, so the same owner reaches its single partner twice, as
///     two distinct periodic images. That dual-image path is the one a
///     "tidy-up" tag dedup would silently break, and it is exercised here.
/// 21 steps at NeighborList every=10 / delay=10 / check no (LAMMPS nflag)
/// cross >= 2 FULL rebuilds (borders) with forward_comm refreshes in
/// between; reverse (post_force) is gated separately by |sum_i F_i|.

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;

namespace {

// ── System: Rahman (1964) argon on a simple-cubic lattice ───────────────
constexpr int kCells = 8;                         // 8^3 sites
constexpr int kNAtoms = kCells * kCells * kCells; // 512
constexpr double kA0 = 3.617;                     // lattice constant [A]
constexpr double kBoxL = kCells * kA0;            // 28.936 A cubic box
constexpr double kEps = 0.010341;                 // 120 k_B [eV]
constexpr double kSigma = 3.4;                    // [A]
constexpr double kCut = 2.5 * kSigma;             // 8.5 A
constexpr double kSkin = 0.85;                    // Verlet skin [A]
constexpr int kNeighEvery = 10;                   // reneighbor cadence [steps]
constexpr double kMass = 39.95;                   // argon [amu]
constexpr double kDt = 1.0;                       // [fs]
constexpr int kSteps = 21;
constexpr double kNudge = 0.35; // symmetry-breaking shift [A]

// ── Gates (hard-coded; see REFERENCE.md) ────────────────────────────────
constexpr double kEnergyTolPerAtom = 1.0e-9; // [eV/atom]
constexpr double kForceTol = 1.0e-9;         // [eV/A]
constexpr double kNetForceTol = 1.0e-10;     // [eV/A]
constexpr int kExpectedNswap = 6;            // 2 * sum_d maxneed_d, maxneed=1
constexpr int kMinRebuilds = 2;           // full borders passes over 21 steps
constexpr double kDecompRelTol = 1.0e-12; // |E(np2)-E(np1)| / |E|

/// @brief Simple-cubic argon lattice, one site nudged off the rank cut.
///
/// Sites sit at (i + 0.5) * a0 so no site lands exactly on the x = L/2
/// partition cut. The site at (ix, iy, iz) = (3, 3, 3) — x = 12.6595 A, i.e.
/// 1.81 A inside rank 0 and deep inside its high-face halo band — is shifted
/// by kNudge along (1,1,1)/sqrt(3). On a pristine lattice every net force
/// vanishes by symmetry, which would make the force gates vacuous; the nudge
/// is what makes them bite.
cpu::Frame make_sc_argon() {
    cpu::Frame f(kNAtoms);
    auto x = f.span(kAtom_x);
    auto y = f.span(kAtom_y);
    auto z = f.span(kAtom_z);
    auto vx = f.span(kAtom_vx);
    auto vy = f.span(kAtom_vy);
    auto vz = f.span(kAtom_vz);
    auto mass = f.span(kAtom_mass);
    auto type = f.span(kAtom_type);
    auto tag = f.span(kAtom_tag);
    int idx = 0;
    for (int ix = 0; ix < kCells; ++ix)
        for (int iy = 0; iy < kCells; ++iy)
            for (int iz = 0; iz < kCells; ++iz) {
                x[idx] = (static_cast<double>(ix) + 0.5) * kA0;
                y[idx] = (static_cast<double>(iy) + 0.5) * kA0;
                z[idx] = (static_cast<double>(iz) + 0.5) * kA0;
                vx[idx] = vy[idx] = vz[idx] = 0.0;
                mass[idx] = kMass;
                type[idx] = 0;
                tag[idx] = idx + 1;
                ++idx;
            }
    const int nudged = (3 * kCells + 3) * kCells + 3;
    const double d = kNudge / std::sqrt(3.0);
    x[nudged] = static_cast<StorageReal>(static_cast<double>(x[nudged]) + d);
    y[nudged] = static_cast<StorageReal>(static_cast<double>(y[nudged]) + d);
    z[nudged] = static_cast<StorageReal>(static_cast<double>(z[nudged]) + d);
    f.box = cpu::Box::cubic(kBoxL);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

/// @brief The oracle, evaluated in this file: the flat O(N^2) minimum-image
///        LJ 12-6 sum over one assembled global configuration.
///
/// Energies are cutoff-shifted exactly as `cpu::PairLJCut` shifts them
/// (u(r) - u(r_c)); forces are the unshifted analytic gradient
/// f_pair = 24 eps / r^2 * (2 s^12 - s^6), s = sigma / r — the shift is a
/// constant and does not enter the force.
///
/// @param x,y,z Assembled positions, index = tag - 1 [Angstrom].
/// @param fx,fy,fz Output forces, resized to N [eV/Angstrom].
/// @return Total potential energy [eV].
double brute_force_lj(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z, std::vector<double>& fx,
                      std::vector<double>& fy, std::vector<double>& fz) {
    const std::size_t n = x.size();
    fx.assign(n, 0.0);
    fy.assign(n, 0.0);
    fz.assign(n, 0.0);

    const double sigma6 = kSigma * kSigma * kSigma * kSigma * kSigma * kSigma;
    const double rc2 = kCut * kCut;
    const double rc6inv = (1.0 / (rc2 * rc2 * rc2)) * sigma6;
    const double v_shift = 4.0 * kEps * (rc6inv * rc6inv - rc6inv);

    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            // Minimum image (multi-wrap safe); kBoxL / 2 = 14.468 A > kCut.
            while (dx > 0.5 * kBoxL)
                dx -= kBoxL;
            while (dx < -0.5 * kBoxL)
                dx += kBoxL;
            while (dy > 0.5 * kBoxL)
                dy -= kBoxL;
            while (dy < -0.5 * kBoxL)
                dy += kBoxL;
            while (dz > 0.5 * kBoxL)
                dz -= kBoxL;
            while (dz < -0.5 * kBoxL)
                dz += kBoxL;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2)
                continue;
            const double r2inv = 1.0 / r2;
            const double r6inv = r2inv * r2inv * r2inv * sigma6;
            e += 4.0 * kEps * (r6inv * r6inv - r6inv) - v_shift;
            const double fpair =
                24.0 * kEps * r2inv * (2.0 * r6inv * r6inv - r6inv);
            fx[i] += fpair * dx;
            fy[i] += fpair * dy;
            fz[i] += fpair * dz;
            fx[j] -= fpair * dx;
            fy[j] -= fpair * dy;
            fz[j] -= fpair * dz;
        }
    }
    return e;
}

/// @brief Out-of-library end_of_step Component: assemble the global state and
///        confront the decomposed result with the in-file oracle.
///
/// Bound at end_of_step @ +900, i.e. AFTER `mpi::Reduce` (-50) has published
/// the globally summed potential energy and long after `CommBrick` (post_force
/// @ -1000) has folded every ghost force back onto its owner. Every launch
/// overwrites the stored deviations, so what the report carries is the state
/// of the LAST step; the swap-table tallies are accumulated as extrema over
/// all launches, so a single bad step cannot hide.
class HaloTableChecker : public cpu::Component {
  public:
    std::string name() const override { return "swaplist_halo_checker"; }

    void launch(const cpu::LaunchContext& ctx) override {
        const auto& f = ctx.frame;
        auto* dd = ctx.runtime.md.dd;
        const int nloc = dd ? dd->n_local : f.nlocal();
        const int ngh = dd ? dd->n_ghost : 0;

        // Swap table is private to CommBrick; n_ghost is the published
        // observable. recv_mismatch stays 0 (no unpublished table to disagree).
        if (ngh > 0) {
            if (nghost_min_ < 0 || ngh < nghost_min_)
                nghost_min_ = ngh;
            if (total_recv_min_ < 0 || ngh < total_recv_min_)
                total_recv_min_ = ngh;
        }
        if (dd && dd->ghost_shell_width > 0.0) {
            const auto need = cpu::comm::Halo::derive_hops(
                *dd, ctx.frame.box, dd->ghost_shell_width);
            int nswap = 0;
            for (int d = 0; d < 3; ++d) {
                const int h = std::max(need.minus[d], need.plus[d]);
                nswap += 2 * std::max(h, 1);
            }
            if (nswap_min_ < 0 || nswap < nswap_min_)
                nswap_min_ = nswap;
            if (nswap > nswap_max_)
                nswap_max_ = nswap;
        }
        if (dd)
            rebuilds_ = static_cast<unsigned long long>(dd->halo_generation);

        // ── Assemble the global configuration from every rank's OWNED rows.
        const auto tag_col = f.span(kAtom_tag);
        const auto x_col = f.span(kAtom_x);
        const auto y_col = f.span(kAtom_y);
        const auto z_col = f.span(kAtom_z);
        const auto fx_col = f.span(kAtom_fx);
        const auto fy_col = f.span(kAtom_fy);
        const auto fz_col = f.span(kAtom_fz);

        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
        std::vector<int> displs(static_cast<std::size_t>(size), 0);
        std::vector<int> displs6(static_cast<std::size_t>(size), 0);
        std::vector<int> counts6(static_cast<std::size_t>(size), 0);
        int total = 0;
        for (int r = 0; r < size; ++r) {
            displs[static_cast<std::size_t>(r)] = total;
            displs6[static_cast<std::size_t>(r)] = 6 * total;
            counts6[static_cast<std::size_t>(r)] =
                6 * counts[static_cast<std::size_t>(r)];
            total += counts[static_cast<std::size_t>(r)];
        }
        n_gathered_ = total;

        std::vector<long long> my_tags(static_cast<std::size_t>(nloc));
        std::vector<double> my_rows(static_cast<std::size_t>(6 * nloc));
        for (int i = 0; i < nloc; ++i) {
            const std::size_t u = static_cast<std::size_t>(i);
            my_tags[u] = static_cast<long long>(tag_col[i]);
            my_rows[6 * u + 0] = static_cast<double>(x_col[i]);
            my_rows[6 * u + 1] = static_cast<double>(y_col[i]);
            my_rows[6 * u + 2] = static_cast<double>(z_col[i]);
            my_rows[6 * u + 3] = static_cast<double>(fx_col[i]);
            my_rows[6 * u + 4] = static_cast<double>(fy_col[i]);
            my_rows[6 * u + 5] = static_cast<double>(fz_col[i]);
        }
        std::vector<long long> all_tags(static_cast<std::size_t>(total));
        std::vector<double> all_rows(static_cast<std::size_t>(6 * total));
        MPI_Allgatherv(my_tags.data(), nloc, MPI_LONG_LONG, all_tags.data(),
                       counts.data(), displs.data(), MPI_LONG_LONG,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(my_rows.data(), 6 * nloc, MPI_DOUBLE, all_rows.data(),
                       counts6.data(), displs6.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (total != kNAtoms) {
            ++assembly_errors_;
            ++launches_;
            return;
        }
        const std::size_t n = static_cast<std::size_t>(kNAtoms);
        std::vector<double> px(n, 0.0), py(n, 0.0), pz(n, 0.0);
        std::vector<double> ex(n, 0.0), ey(n, 0.0), ez(n, 0.0);
        std::vector<int> seen(n, 0);
        for (int i = 0; i < total; ++i) {
            const long long t = all_tags[static_cast<std::size_t>(i)];
            if (t < 1 || t > kNAtoms) {
                ++assembly_errors_;
                ++launches_;
                return;
            }
            const std::size_t u = static_cast<std::size_t>(t - 1);
            ++seen[u];
            const std::size_t s = 6 * static_cast<std::size_t>(i);
            px[u] = all_rows[s + 0];
            py[u] = all_rows[s + 1];
            pz[u] = all_rows[s + 2];
            ex[u] = all_rows[s + 3];
            ey[u] = all_rows[s + 4];
            ez[u] = all_rows[s + 5];
        }
        for (std::size_t u = 0; u < n; ++u)
            if (seen[u] != 1)
                ++assembly_errors_;

        // ── The oracle and the three physics deviations.
        std::vector<double> bx, by, bz;
        e_brute_ = brute_force_lj(px, py, pz, bx, by, bz);
        e_dd_ = f.meta.get(kEnergy);

        double max_df = 0.0;
        double sfx = 0.0, sfy = 0.0, sfz = 0.0;
        for (std::size_t u = 0; u < n; ++u) {
            max_df = std::fmax(max_df, std::fabs(ex[u] - bx[u]));
            max_df = std::fmax(max_df, std::fabs(ey[u] - by[u]));
            max_df = std::fmax(max_df, std::fabs(ez[u] - bz[u]));
            sfx += ex[u];
            sfy += ey[u];
            sfz += ez[u];
        }
        max_dforce_ = max_df;
        net_force_ = std::sqrt(sfx * sfx + sfy * sfy + sfz * sfz);
        // Non-vacuous force gate: the nudged lattice must actually carry force.
        double max_f = 0.0;
        for (std::size_t u = 0; u < n; ++u)
            max_f = std::fmax(max_f, std::sqrt(bx[u] * bx[u] + by[u] * by[u]
                                               + bz[u] * bz[u]));
        max_force_magnitude_ = max_f;
        ++launches_;
    }

    double e_dd_ = 0.0;                ///< reduced engine potential energy [eV]
    double e_brute_ = 0.0;             ///< in-file O(N^2) min-image sum [eV]
    double max_dforce_ = -1.0;         ///< max_i |F_i,dd - F_i,brute| [eV/A]
    double net_force_ = -1.0;          ///< ||sum_i F_i,dd|| [eV/A]
    double max_force_magnitude_ = 0.0; ///< max_i |F_i,brute| [eV/A]
    int nswap_min_ = -1;               ///< min swaps over launches
    int nswap_max_ = -1;               ///< max swaps over launches
    long long total_recv_min_ = -1;    ///< min sum_i nrecv over launches
    long long nghost_min_ = -1;        ///< min dd->n_ghost over launches
    long long recv_mismatch_ = 0;      ///< launches where sum nrecv != n_ghost
    long long assembly_errors_ = 0;    ///< lost / duplicated owned tags
    unsigned long long rebuilds_ = 0;  ///< dd->halo_generation at the last step
    int n_gathered_ = -1;              ///< owned rows assembled, must be 512
    int launches_ = 0;
};

int run_child(const char* out_path) {
    // Deterministic 2x1x1 split along x (ignored at np = 1: the override is
    // only honoured when dx*dy*dz == world_size).
    ::setenv("ATV_FORCE_DIMS", "2,1,1", 1);
    MPI_Init(nullptr, nullptr);

    auto frame = make_sc_argon();

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCut});
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, kNeighEvery,
                                             kNeighEvery, /*check=*/false});
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.enable_timing(false);

    HaloTableChecker checker;
    sim.on(ATV::md::Phase::end_of_step).add_ref(checker, +900);

    sim.build();
    sim.run(frame, kSteps);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out) {
            std::printf("FAIL: child cannot open %s\n", out_path);
            rc = 1;
        } else {
            std::fprintf(out,
                         "%.17e %.17e %.17e %.17e %.17e %d %d %lld %lld %lld "
                         "%lld %llu %d %d\n",
                         checker.e_dd_, checker.e_brute_, checker.max_dforce_,
                         checker.net_force_, checker.max_force_magnitude_,
                         checker.nswap_min_, checker.nswap_max_,
                         checker.total_recv_min_, checker.nghost_min_,
                         checker.recv_mismatch_, checker.assembly_errors_,
                         checker.rebuilds_, checker.n_gathered_,
                         checker.launches_);
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

/// @brief One child report; every field is a measured quantity, never a
///        default that could pass a gate by accident.
struct Report {
    double e_dd = 0.0, e_brute = 0.0;
    double max_dforce = -1.0, net_force = -1.0, max_force = -1.0;
    int nswap_min = -1, nswap_max = -1;
    long long total_recv_min = -1, nghost_min = -1;
    long long recv_mismatch = -1, assembly_errors = -1;
    unsigned long long rebuilds = 0;
    int n_gathered = -1, launches = -1;
};

/// @brief Re-launch this binary under mpiexec at `np` ranks and read back the
///        child's report.
/// @param self Absolute path of this executable.
/// @param np Rank count.
/// @param out Parsed report.
/// @return true on a clean child exit with a well-formed report.
bool run_under_mpiexec(const std::string& self, int np, Report& out) {
    const std::string path =
        "/tmp/atv-comm-swaplist-halo-np" + std::to_string(np) + "-"
        + std::to_string(static_cast<long>(::getpid())) + ".txt";
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n "
                            + std::to_string(np) + " \"" + self
                            + "\" --child \"" + path + "\"";
    std::printf("[comm-swaplist-halo-mpi] launching: %s\n", cmd.c_str());
    std::fflush(stdout);
    if (std::system(cmd.c_str()) != 0) {
        std::printf("FAIL: np=%d child exited nonzero\n", np);
        return false;
    }
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing np=%d child report %s\n", np, path.c_str());
        return false;
    }
    const int got = std::fscanf(
        in, "%lf %lf %lf %lf %lf %d %d %lld %lld %lld %lld %llu %d %d",
        &out.e_dd, &out.e_brute, &out.max_dforce, &out.net_force,
        &out.max_force, &out.nswap_min, &out.nswap_max, &out.total_recv_min,
        &out.nghost_min, &out.recv_mismatch, &out.assembly_errors,
        &out.rebuilds, &out.n_gathered, &out.launches);
    std::fclose(in);
    std::remove(path.c_str());
    if (got != 14) {
        std::printf("FAIL: malformed np=%d child report (%d fields)\n", np,
                    got);
        return false;
    }
    return true;
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
    Report dd, serial;
    if (!run_under_mpiexec(self, 2, dd))
        return 1;
    if (!run_under_mpiexec(self, 1, serial))
        return 1;

    int failures = 0;
    std::printf("[comm-swaplist-halo-mpi] gates (np=2, 512 atoms, 21 steps)\n");

    // (0) The assembly itself must be sound, or every gate below is vacuous.
    const bool ok_assembly = dd.n_gathered == kNAtoms && dd.assembly_errors == 0
                             && dd.launches >= kSteps;
    std::printf("  atoms assembled / errors / launches got %d / %lld / %d  "
                "want %d / 0 / >=%d  -> %s\n",
                dd.n_gathered, dd.assembly_errors, dd.launches, kNAtoms, kSteps,
                ok_assembly ? "PASS" : "FAIL");
    if (!ok_assembly)
        ++failures;

    // (1) Energy: the decomposed sum IS the min-image sum.
    const double de_per_atom =
        std::fabs(dd.e_dd - dd.e_brute) / static_cast<double>(kNAtoms);
    const bool ok_energy = de_per_atom <= kEnergyTolPerAtom;
    std::printf("  |E_dd - E_brute| / N            got %.3e  want 0  "
                "tol %.0e eV/atom  -> %s\n",
                de_per_atom, kEnergyTolPerAtom, ok_energy ? "PASS" : "FAIL");
    if (!ok_energy)
        ++failures;

    // (2) Forces: every owned component, gathered by tag.
    const bool ok_force = dd.max_dforce >= 0.0 && dd.max_dforce <= kForceTol;
    std::printf("  max_i |F_i,dd - F_i,brute|      got %.3e  want 0  "
                "tol %.0e eV/A  -> %s\n",
                dd.max_dforce, kForceTol, ok_force ? "PASS" : "FAIL");
    if (!ok_force)
        ++failures;

    // (3) Newton's third law across the cut: reverse ran exactly once.
    const bool ok_net = dd.net_force >= 0.0 && dd.net_force <= kNetForceTol;
    std::printf("  ||sum_i F_i||                   got %.3e  want 0  "
                "tol %.0e eV/A  -> %s\n",
                dd.net_force, kNetForceTol, ok_net ? "PASS" : "FAIL");
    if (!ok_net)
        ++failures;

    // (3b) Non-vacuous: the nudged lattice really does carry force.
    const bool ok_live = dd.max_force > 1.0e-4;
    std::printf("  max_i |F_i,brute| (non-vacuous) got %.3e  want > 1e-04 "
                "eV/A          -> %s\n",
                dd.max_force, ok_live ? "PASS" : "FAIL");
    if (!ok_live)
        ++failures;

    // (4) The published swap table: 6 single-hop swaps, all rebuilds.
    const bool ok_nswap =
        dd.nswap_min == kExpectedNswap && dd.nswap_max == kExpectedNswap;
    std::printf("  nswap (min/max over steps)      got %d/%d  want %d/%d"
                "               -> %s\n",
                dd.nswap_min, dd.nswap_max, kExpectedNswap, kExpectedNswap,
                ok_nswap ? "PASS" : "FAIL");
    if (!ok_nswap)
        ++failures;

    // (5) Slot bookkeeping: sum_i nrecv == n_ghost > 0 on every step.
    const bool ok_recv = dd.recv_mismatch == 0 && dd.total_recv_min > 0
                         && dd.nghost_min == dd.total_recv_min;
    std::printf("  sum nrecv / n_ghost (min)       got %lld / %lld  "
                "mismatches %lld  want equal and > 0  -> %s\n",
                dd.total_recv_min, dd.nghost_min, dd.recv_mismatch,
                ok_recv ? "PASS" : "FAIL");
    if (!ok_recv)
        ++failures;

    // (6) >= 2 FULL rebuilds, so borders AND in-step forward both ran.
    const bool ok_rebuilds =
        dd.rebuilds >= static_cast<unsigned long long>(kMinRebuilds);
    std::printf("  full halo rebuilds              got %llu  want >= %d"
                "                    -> %s\n",
                dd.rebuilds, kMinRebuilds, ok_rebuilds ? "PASS" : "FAIL");
    if (!ok_rebuilds)
        ++failures;

    // (7) Decomposition invariance: the same run at np = 1.
    const double rel_np =
        std::fabs(dd.e_dd - serial.e_dd) / std::fabs(serial.e_dd);
    const bool ok_np = rel_np <= kDecompRelTol;
    std::printf("  |E(np2) - E(np1)| / |E|         got %.3e  want 0  "
                "tol %.0e        -> %s\n",
                rel_np, kDecompRelTol, ok_np ? "PASS" : "FAIL");
    if (!ok_np)
        ++failures;

    std::printf("[comm-swaplist-halo-mpi] E(np=2) = %.12f eV, "
                "E_brute = %.12f eV, E(np=1) = %.12f eV\n",
                dd.e_dd, dd.e_brute, serial.e_dd);
    if (failures == 0)
        std::printf("PASS\n");
    else
        std::printf("FAIL (%d gate%s)\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
