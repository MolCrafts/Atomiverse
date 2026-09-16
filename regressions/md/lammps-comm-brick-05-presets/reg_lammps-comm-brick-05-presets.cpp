/// @file reg_lammps-comm-brick-05-presets.cpp
/// @brief Evaluate + PairLJCut + NeighborList, no dead halo Plan plugins.
///
/// np=2 vs in-file O(N²) min-image LJ identity. Public headers + link
/// atomiverse. Literature: Plimpton JCP 117 (1995); Thompson CPC 271
/// (2022); Rahman PR 136 (1964).

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/plan.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

using namespace ATV;

namespace {

constexpr double kEps = 0.010341; // 120 k_B [eV]
constexpr double kSigma = 3.4;    // [A]
constexpr double kCutoff = 6.0;   // [A]
constexpr double kSkin = 0.6;     // [A]
constexpr double kLatticeA = 5.26;
constexpr int kSide = 6;
constexpr double kEnergyRelTol = 1.0e-9;

const std::unordered_set<std::string> kDead{
    "halo_send",      "halo_recv",      "migrate",     "halo_refresh_x",
    "halo_refresh_v", "halo_refresh_q", "halo_refresh"};

cpu::Frame make_sc_lattice() {
    const int n = kSide * kSide * kSide;
    cpu::Frame frame(n);
    int idx = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k) {
                frame.span(kAtom_x)[idx] = (i + 0.5) * kLatticeA;
                frame.span(kAtom_y)[idx] = (j + 0.5) * kLatticeA;
                frame.span(kAtom_z)[idx] = (k + 0.5) * kLatticeA;
                frame.span(kAtom_mass)[idx] = 39.948;
                frame.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    frame.box = cpu::Box::cubic(kSide * kLatticeA);
    return frame;
}

double lj_pair_energy(double r2) {
    const double rc2 = kCutoff * kCutoff;
    if (r2 >= rc2)
        return 0.0;
    auto e = [](double rr2) {
        const double s2 = (kSigma * kSigma) / rr2;
        const double s6 = s2 * s2 * s2;
        return 4.0 * kEps * (s6 * s6 - s6);
    };
    return e(r2) - e(rc2);
}

double brute_force_energy(const cpu::Frame& frame) {
    const int n = frame.nlocal();
    const double lx = frame.box.length(0);
    const double ly = frame.box.length(1);
    const double lz = frame.box.length(2);
    double total = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx = frame.span(kAtom_x)[i] - frame.span(kAtom_x)[j];
            double dy = frame.span(kAtom_y)[i] - frame.span(kAtom_y)[j];
            double dz = frame.span(kAtom_z)[i] - frame.span(kAtom_z)[j];
            dx -= lx * std::round(dx / lx);
            dy -= ly * std::round(dy / ly);
            dz -= lz * std::round(dz / lz);
            total += lj_pair_energy(dx * dx + dy * dy + dz * dz);
        }
    return total;
}

bool plan_has_reduce_no_dead(const Plan<cpu::Component>& plan,
                             std::string& dead_hit) {
    bool reduce = false;
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            if (!op.component)
                continue;
            const std::string n = op.component->name();
            if (n == "reduce")
                reduce = true;
            if (kDead.count(n) != 0) {
                dead_hit = n;
                return false;
            }
        }
    }
    return reduce;
}

int check_plan() {
    cpu::Evaluate eval;
    eval.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    eval.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    eval.build();
    std::string dead;
    if (!plan_has_reduce_no_dead(eval.plan(), dead)) {
        if (!dead.empty()) {
            std::fprintf(stderr, "FAIL: Plan still contains %s\n",
                         dead.c_str());
        } else {
            std::fprintf(stderr, "FAIL: Plan missing reduce\n");
        }
        return EXIT_FAILURE;
    }
    std::printf("plan reduce yes dead none\n");
    return EXIT_SUCCESS;
}

#ifdef ATV_HAS_MPI

int run_child(const char* out_path) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    ::setenv("ATV_FORCE_DIMS", "2,1,1", 1);
    cpu::Frame frame = make_sc_lattice();
    const cpu::Frame ref = frame;
    const double e_bf = brute_force_energy(ref);

    cpu::Evaluate eval;
    eval.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
    eval.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
    eval.build();
    std::string dead;
    const int plan_ok = plan_has_reduce_no_dead(eval.plan(), dead) ? 1 : 0;
    eval.run(frame);
    const double e_dd = frame.meta.get(kEnergy);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = EXIT_SUCCESS;
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out)
            rc = EXIT_FAILURE;
        else {
            std::fprintf(out, "%.17e %.17e %d\n", e_dd, e_bf, plan_ok);
            std::fclose(out);
        }
        if (plan_ok == 0)
            rc = EXIT_FAILURE;
        if (!(std::fabs(e_bf) > 1e-12))
            rc = EXIT_FAILURE;
        else if (std::fabs(e_dd - e_bf) / std::fabs(e_bf) > kEnergyRelTol)
            rc = EXIT_FAILURE;
    }
    MPI_Finalize();
    return rc;
}

bool run_under_mpiexec(const std::string& self, double& e_dd, double& e_bf,
                       int& plan_ok) {
    const std::string path = "/tmp/atv-comm-brick-05-np2-"
                             + std::to_string(static_cast<long>(::getpid()))
                             + ".txt";
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n 2 \"" + self
                            + "\" --child \"" + path + "\"";
    std::printf("[lammps-comm-brick-05-presets] launching: %s\n", cmd.c_str());
    std::fflush(stdout);
    if (std::system(cmd.c_str()) != 0) {
        std::printf("FAIL: np=2 child exited nonzero\n");
        return false;
    }
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in) {
        std::printf("FAIL: missing child report %s\n", path.c_str());
        return false;
    }
    const int got = std::fscanf(in, "%lf %lf %d", &e_dd, &e_bf, &plan_ok);
    std::fclose(in);
    std::remove(path.c_str());
    return got == 3;
}

#endif

} // namespace

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);
#endif

    if (check_plan() != EXIT_SUCCESS)
        return EXIT_FAILURE;

#ifdef ATV_HAS_MPI
    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::printf("FAIL: cannot resolve /proc/self/exe\n");
        return EXIT_FAILURE;
    }
    self_buf[n] = '\0';
    ::setenv("OMP_NUM_THREADS", "1", 1);
    double e_dd = 0.0, e_bf = 0.0;
    int plan_ok = 0;
    if (!run_under_mpiexec(self_buf, e_dd, e_bf, plan_ok))
        return EXIT_FAILURE;
    if (plan_ok == 0) {
        std::printf("FAIL: child Plan still contains a dead comm plugin\n");
        return EXIT_FAILURE;
    }
    if (!(std::fabs(e_bf) > 1e-12)) {
        std::printf("FAIL: degenerate brute-force energy\n");
        return EXIT_FAILURE;
    }
    const double rel = std::fabs(e_dd - e_bf) / std::fabs(e_bf);
    std::printf("|E-E_brute|/|E| %.3e want <= %.0e  -> %s\n", rel,
                kEnergyRelTol, rel <= kEnergyRelTol ? "PASS" : "FAIL");
    return rel <= kEnergyRelTol ? EXIT_SUCCESS : EXIT_FAILURE;
#else
    (void)argc;
    (void)argv;
    std::printf("FAIL: ATV_HAS_MPI required\n");
    return EXIT_FAILURE;
#endif
}
