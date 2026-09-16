/// @file regressions/pw/mpi-pw-scaling-4rank/reg_mpi-pw-scaling-4rank.cpp
/// @brief END-TO-END: MPI PW k-mesh round-robin scaling gate at 4 ranks.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW + ATV_USE_MPI.
///
/// One binary, two modes:
///  - parent (default): re-launches itself through the configured
///    `mpiexec` (compile-time pin ATV_MPIEXEC) at np = 1 and np = 4 with
///    equal OMP threads per rank, then gates
///      (a) both energies inside the external ABINIT-class pin band,
///      (b) rank-count invariance |E(1) - E(4)| <= 1e-8 Ry,
///      (c) a real 4-way round-robin split (8 k-points -> 2 per rank),
///      (d) the scaling wall ratio T(1)/T(4) >= 1.2.
///  - child (`--child <out>`): MPI_Init, run the pw-lda-si Si diamond
///    twin, time `run()` with MPI_Wtime, reduce the per-rank k counts,
///    rank 0 writes one result line, MPI_Finalize.
///
/// Exit non-zero on failure with got / want / tol.

#include <mpi.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif
#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

// Twin settings — identical to regressions/pw/pw-lda-si (see REFERENCE.md).
constexpr double kA = 5.43;      // Å, diamond Si conventional lattice constant
constexpr double kEcutRy = 15.0; // tutorial-class cutoff
constexpr int kMesh = 2;         // 2x2x2 MP -> 8 k-points

// External pin: PseudoDojo Si.djrepo deltafactor E_min (ABINIT-class), Ry.
constexpr double kEminAbinitRy = -16.916316;
constexpr double kBandHeadroomRy = 1.50; // incomplete ecut/k window
constexpr double kParityTolRy = 1e-8;    // rank-count invariance
constexpr double kSpeedupFloor = 1.2;    // weak industrial scaling floor

struct ChildReport {
    double energy_ry = 0;
    int iterations = 0;
    int converged = 0;
    int n_kpts = 0;
    int nk_local_min = 0;
    int nk_local_max = 0;
    double wall_s = 0;
};

int run_child(const char* out_path) {
    MPI_Init(nullptr, nullptr);

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = kA / 4.0;
    frame.span(kAtom_y)[1] = kA / 4.0;
    frame.span(kAtom_z)[1] = kA / 4.0;
    const double h = kA / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.tol = 1e-6;
    method.max_iter = 50;

    method.build(frame);
    const double t0 = MPI_Wtime();
    const PWResult& r = method.run(frame);
    const double wall = MPI_Wtime() - t0;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nk_min = 0, nk_max = 0;
    MPI_Allreduce(&r.n_kpts_local, &nk_min, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&r.n_kpts_local, &nk_max, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    double wall_max = 0; // slowest rank defines the run wall
    MPI_Allreduce(&wall, &wall_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out) {
            std::printf("FAIL: child cannot open %s\n", out_path);
            rc = 1;
        } else {
            std::fprintf(out, "%.12f %d %d %d %d %d %.6f\n", r.energy,
                         r.iterations, r.converged ? 1 : 0, r.n_kpts, nk_min,
                         nk_max, wall_max);
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

bool launch_and_read(const std::string& self, int np, const std::string& out,
                     ChildReport& rep) {
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n "
                            + std::to_string(np) + " \"" + self
                            + "\" --child \"" + out + "\"";
    std::printf("[mpi-pw-scaling-4rank] launching: %s\n", cmd.c_str());
    std::fflush(stdout);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::printf("FAIL: child launch (np=%d) exited with %d\n", np, rc);
        return false;
    }
    std::FILE* in = std::fopen(out.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing child report %s\n", out.c_str());
        return false;
    }
    const int got =
        std::fscanf(in, "%lf %d %d %d %d %d %lf", &rep.energy_ry,
                    &rep.iterations, &rep.converged, &rep.n_kpts,
                    &rep.nk_local_min, &rep.nk_local_max, &rep.wall_s);
    std::fclose(in);
    std::remove(out.c_str());
    if (got != 7) {
        std::printf("FAIL: malformed child report %s (%d fields)\n",
                    out.c_str(), got);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);

    // Resolve our own binary for re-launch under mpiexec.
    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::printf("FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }
    self_buf[n] = '\0';
    const std::string self(self_buf);

    // Equal threads per rank in BOTH runs — the wall ratio is then a pure
    // k-point-distribution scaling measurement (4 ranks x 4 threads).
    ::setenv("OMP_NUM_THREADS", "4", 1);

    const std::string tag = std::to_string(static_cast<long>(::getpid()));
    ChildReport r1, r4;
    if (!launch_and_read(self, 1, "/tmp/atv-mpi-pw-scaling-" + tag + "-np1.txt",
                         r1))
        return 1;
    if (!launch_and_read(self, 4, "/tmp/atv-mpi-pw-scaling-" + tag + "-np4.txt",
                         r4))
        return 1;

    int failures = 0;

    // (a) Convergence + external ABINIT-class pin band, both rank counts.
    const double lo = kEminAbinitRy;
    const double hi = kEminAbinitRy + kBandHeadroomRy;
    const ChildReport* reports[2] = {&r1, &r4};
    const int nps[2] = {1, 4};
    for (int i = 0; i < 2; ++i) {
        const ChildReport& r = *reports[i];
        if (!r.converged) {
            std::printf("FAIL: np=%d SCF did not converge (iters=%d)\n", nps[i],
                        r.iterations);
            ++failures;
        }
        if (!std::isfinite(r.energy_ry)
            || !(r.energy_ry >= lo && r.energy_ry <= hi)) {
            std::printf("FAIL: np=%d energy_ry=%.9f outside pin band "
                        "[%.6f, %.6f]\n",
                        nps[i], r.energy_ry, lo, hi);
            ++failures;
        }
    }

    // (b) Rank-count invariance (deterministic Davidson seeding).
    const double de = std::fabs(r1.energy_ry - r4.energy_ry);
    if (!(de <= kParityTolRy)) {
        std::printf("FAIL: |E(np=1) - E(np=4)| = %.3e Ry > tol %.1e Ry\n", de,
                    kParityTolRy);
        ++failures;
    }
    if (r1.iterations != r4.iterations) {
        std::printf("FAIL: iteration count differs (np1=%d np4=%d) — "
                    "trajectory not rank-count invariant\n",
                    r1.iterations, r4.iterations);
        ++failures;
    }

    // (c) Real 4-way round-robin split: 8 k-points -> exactly 2 per rank.
    if (r4.n_kpts != 8 || r4.nk_local_min != 2 || r4.nk_local_max != 2) {
        std::printf("FAIL: np=4 distribution n_kpts=%d local=[%d,%d] "
                    "(want 8 -> [2,2])\n",
                    r4.n_kpts, r4.nk_local_min, r4.nk_local_max);
        ++failures;
    }
    if (r1.n_kpts != 8 || r1.nk_local_min != 8 || r1.nk_local_max != 8) {
        std::printf("FAIL: np=1 distribution n_kpts=%d local=[%d,%d] "
                    "(want 8 -> [8,8])\n",
                    r1.n_kpts, r1.nk_local_min, r1.nk_local_max);
        ++failures;
    }

    // (d) Scaling wall ratio (queue metric): weak industrial floor.
    const double speedup = (r4.wall_s > 0) ? r1.wall_s / r4.wall_s : 0.0;
    if (!(speedup >= kSpeedupFloor)) {
        std::printf("FAIL: speedup T(np1)/T(np4) = %.2f < floor %.2f "
                    "(T1=%.3fs T4=%.3fs)\n",
                    speedup, kSpeedupFloor, r1.wall_s, r4.wall_s);
        ++failures;
    }

    std::printf("[mpi-pw-scaling-4rank] E(np1)=%.9f Ry E(np4)=%.9f Ry "
                "|dE|=%.2e Ry iters=%d/%d\n",
                r1.energy_ry, r4.energy_ry, de, r1.iterations, r4.iterations);
    std::printf("[mpi-pw-scaling-4rank] walls: T1=%.3f s T4=%.3f s "
                "speedup=%.2fx (floor %.2f)\n",
                r1.wall_s, r4.wall_s, speedup, kSpeedupFloor);

    if (failures) {
        std::printf("FAILED: mpi-pw-scaling-4rank (%d checks)\n", failures);
        return 1;
    }
    std::printf("PASSED: mpi-pw-scaling-4rank — 4-rank k-point round-robin "
                "reproduces the pinned serial energy and scales\n");
    return 0;
}
