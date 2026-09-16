/// @file regressions/gpw/mpi-gpw-scaling-4rank/reg_mpi-gpw-scaling-4rank.cpp
/// @brief Industrial gate: multi-rank GPW k-point energy invariance
///        (parent self-launches mpiexec np=4; serial n=1 baseline).
///
/// Out-of-library consumer: public GPW headers only (+ MPI when worker).
/// Spec: `.claude/specs/mpi-gpw-scaling-4rank/`.
///
/// Modes:
///   default (parent): serial SCF energy + wall; spawn
///     `mpiexec -n 4 <self> --mpi-worker`; pin |E1−E4| and record wall ratio.
///   `--mpi-worker`: MPI_Init, distributed k-RR SCF, rank-agreement, print
///     ENERGY= / WALL_S= / N_KPTS= / N_KPTS_LOCAL= for the parent parser.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#ifndef ATV_MPIEXEC
#define ATV_MPIEXEC "mpiexec"
#endif

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

// Diamond Si bct 4-atom cell (same geometry family as gpw-kpoints-si).
std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr) {
    using namespace ATV;
    const double at = a_bohr / std::sqrt(2.0);
    const cpu::gpw::GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
    const cpu::gpw::GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

ATV::cpu::gpw::GpwDft si_method(double a_bohr) {
    ATV::cpu::gpw::GpwDft m;
    m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
    m.cell[2] = a_bohr;
    // Lightweight twin for self-launch wall (≤ 60 s total parent+worker).
    m.cutoff_ry = 80.0;
    m.n_occ = 8;
    m.max_iter = 40;
    m.tol = 1e-7;
    m.xc = ATV::cpu::gpw::XcKind::PBE;
    m.eps_default = 1e-6;
    m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2; // TR → 4 k
    m.k_time_reversal = true;
    return m;
}

struct ScfOut {
    double e_total = 0.0;
    bool converged = false;
    int n_kpts = 0;
    int n_kpts_local = 0;
    int iters = 0;
    double wall_s = 0.0;
};

ScfOut run_scf() {
    const double a = 5.43 * kA2B;
    auto atoms = si_bct(a);
    auto method = si_method(a);
    method.build(atoms);
    const auto t0 = std::chrono::steady_clock::now();
    const auto& r = method.run(atoms);
    const auto t1 = std::chrono::steady_clock::now();
    ScfOut o;
    o.e_total = r.e_total;
    o.converged = r.converged;
    o.n_kpts = r.n_kpts;
    o.n_kpts_local = r.n_kpts_local;
    o.iters = r.iters;
    o.wall_s = std::chrono::duration<double>(t1 - t0).count();
    return o;
}

#ifdef ATV_HAS_MPI
int run_worker() {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const ScfOut o = run_scf();

    double e_min = 0.0, e_max = 0.0;
    MPI_Allreduce(&o.e_total, &e_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&o.e_total, &e_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int n_local_sum = 0;
    MPI_Allreduce(&o.n_kpts_local, &n_local_sum, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    int local_fail = 0;
    if (!o.converged || !std::isfinite(o.e_total))
        local_fail = 1;
    if (std::abs(e_min - e_max) > 1e-12)
        local_fail = 1;
    if (o.n_kpts != 4)
        local_fail = 1;
    if (n_local_sum != o.n_kpts)
        local_fail = 1;

    int any_fail = 0;
    MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("ENERGY=%.16e\n", o.e_total);
        std::printf("WALL_S=%.6f\n", o.wall_s);
        std::printf("N_KPTS=%d\n", o.n_kpts);
        std::printf("N_KPTS_LOCAL_SUM=%d\n", n_local_sum);
        std::printf("NPROCS=%d\n", nprocs);
        std::printf("ITERS=%d\n", o.iters);
        std::printf("RANK_E_SPREAD=%.3e\n", std::abs(e_min - e_max));
        if (any_fail)
            std::printf("WORKER_STATUS=FAIL\n");
        else
            std::printf("WORKER_STATUS=OK\n");
        std::fflush(stdout);
    }
    MPI_Finalize();
    return any_fail ? 1 : 0;
}
#endif

// Parse ENERGY= and WALL_S= from worker stdout captured in a temp file.
bool parse_worker_file(const char* path, double& energy, double& wall_s,
                       int& n_kpts, std::string& status) {
    FILE* f = std::fopen(path, "r");
    if (!f)
        return false;
    char line[512];
    energy = 0.0;
    wall_s = 0.0;
    n_kpts = 0;
    status.clear();
    bool got_e = false;
    while (std::fgets(line, sizeof(line), f)) {
        double v = 0.0;
        int iv = 0;
        char buf[128];
        if (std::sscanf(line, "ENERGY=%lf", &v) == 1) {
            energy = v;
            got_e = true;
        } else if (std::sscanf(line, "WALL_S=%lf", &v) == 1) {
            wall_s = v;
        } else if (std::sscanf(line, "N_KPTS=%d", &iv) == 1) {
            n_kpts = iv;
        } else if (std::sscanf(line, "WORKER_STATUS=%127s", buf) == 1) {
            status = buf;
        }
    }
    std::fclose(f);
    return got_e;
}

int run_parent(const char* self_path) {
    std::printf("=== mpi-gpw-scaling-4rank parent (serial n=1 baseline) ===\n");
    const ScfOut serial = run_scf();
    std::printf("serial ENERGY=%.16e  WALL_S=%.3f  n_kpts=%d n_local=%d "
                "iters=%d converged=%d\n",
                serial.e_total, serial.wall_s, serial.n_kpts,
                serial.n_kpts_local, serial.iters, serial.converged ? 1 : 0);

    check(serial.converged, "A1 serial SCF converged");
    check(std::isfinite(serial.e_total), "A1 serial finite e_total");
    check(serial.n_kpts == 4, "A4 serial n_kpts == 4 (MP 2x2x2 TR)");
    check(serial.n_kpts_local == serial.n_kpts,
          "A4 serial n_kpts_local == n_kpts");

#ifndef ATV_HAS_MPI
    std::printf(
        "ATV_HAS_MPI off — serial-only gate; skip mpiexec self-launch\n");
    return g_fail ? 1 : 0;
#else
    // Self-launch mpiexec -n 4 worker mode.
    char out_path[] = "/tmp/atv_mpi_gpw_scaling_XXXXXX";
    const int fd = mkstemp(out_path);
    if (fd < 0) {
        std::fprintf(stderr, "FAIL: mkstemp for worker log\n");
        return 1;
    }
    close(fd);

    std::string cmd = std::string(ATV_MPIEXEC) + " -n 4 \"" + self_path
                      + "\" --mpi-worker > \"" + out_path + "\" 2>&1";
    std::printf("launch: %s\n", cmd.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = std::system(cmd.c_str());
    const auto t1 = std::chrono::steady_clock::now();
    const double parent_wall = std::chrono::duration<double>(t1 - t0).count();

    double e4 = 0.0, wall4 = 0.0;
    int nk4 = 0;
    std::string status;
    const bool parsed = parse_worker_file(out_path, e4, wall4, nk4, status);
    // Keep the log on failure for diagnosis.
    if (parsed && status == "OK" && rc == 0)
        std::remove(out_path);
    else {
        std::fprintf(stderr, "worker log retained at %s (rc=%d status=%s)\n",
                     out_path, rc, status.c_str());
        // Dump log to stderr for ctest/regression visibility.
        if (FILE* f = std::fopen(out_path, "r")) {
            char line[512];
            while (std::fgets(line, sizeof(line), f))
                std::fputs(line, stderr);
            std::fclose(f);
        }
    }

    check(rc == 0 && parsed && status == "OK", "A1 n=4 worker completed OK");
    check(std::isfinite(e4), "A1 n=4 finite energy");
    check(nk4 == 4, "A4 n=4 n_kpts == 4");

    const double de = std::abs(serial.e_total - e4);
    std::printf("rank-count: E1=%.16e  E4=%.16e  |Δ|=%.3e Ha\n", serial.e_total,
                e4, de);
    check(de <= 1e-8, "A3 rank-count invariance |E1-E4| <= 1e-8 Ha");

    const double ratio = (wall4 > 0.0) ? serial.wall_s / wall4 : 0.0;
    std::printf("wall: T1=%.3f s  T4_worker=%.3f s  parent_launch=%.3f s  "
                "ratio T1/T4=%.3f  (perf_exempt — k-RR foothold only)\n",
                serial.wall_s, wall4, parent_wall, ratio);
    std::printf("PERF_EXEMPT=true ratio=%.3f\n", ratio);

    return g_fail ? 1 : 0;
#endif
}

} // namespace

int main(int argc, char** argv) {
    bool worker = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mpi-worker") == 0)
            worker = true;
    }
    if (worker) {
#ifdef ATV_HAS_MPI
        return run_worker();
#else
        std::fprintf(stderr, "FAIL: --mpi-worker requires ATV_HAS_MPI\n");
        return 1;
#endif
    }
    return run_parent(argv[0]);
}
