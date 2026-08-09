/// @file tools/bench_cuda_pw_cell.cu
/// @brief Manual CPU-vs-CUDA plane-wave wall harness over Si supercells
///        (parity-gap slug `cuda-pw-large-cell-measure`). NOT a ctest and
///        NOT a regression: no literature window, no pass/fail pin — it
///        prints same-settings walls so the large-cell measurement note
///        (`.claude/notes/cuda-pw-large-cell.md`) can be written from a
///        GH200 job log. Public API only: `cpu::pw::PW` / `cuda::pw::PW`
///        + `Frame`; settings live inside the CUDA LSDA envelope
///        (restricted, fixed occupations, `XCType::LSDA`, no smearing).
///
/// Build: compiled by `tools/gpu_job_cuda_pw_large_cell.sh` with the exact
/// compile/device-link/link commands CMake generates for
/// `reg_pw-cuda-lda-si` (harvested via `make -n`), so no tracked CMake file
/// changes and the flags match the regression bit-for-bit.
///
/// Run:
///   bench_cuda_pw_cell <upf> <n_rep> <cpu|cuda|both>
///                      [ecut_ry=15] [k_mesh=1] [tol=1e-6] [max_iter=100]
///
/// The cell is the 8-atom conventional diamond-Si cube (a = 5.43 Å)
/// replicated n_rep per axis: n_rep = 1/2/3 -> 8/64/216 atoms. Timed region
/// is `run(frame)` only (geometry-dependent setup + full SCF, both
/// backends); `build()` (pseudopotential parse) and CUDA context creation
/// (`cudaFree(0)`) are excluded on both legs.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

using namespace ATV;

namespace {

struct Opts {
    std::string upf;
    int n_rep = 1;
    std::string device = "both";
    double ecut_ry = 15.0;
    int k_mesh = 1;
    double tol = 1e-6;
    int max_iter = 100;
};

/// @brief n×n×n replication of the 8-atom conventional diamond-Si cube.
/// @param a Cubic lattice constant [Å].
/// @param n Repetitions per axis (dimensionless).
cpu::Frame make_si_supercell(double a, int n) {
    constexpr double frac[8][3] = {
        {0.00, 0.00, 0.00}, {0.00, 0.50, 0.50}, {0.50, 0.00, 0.50},
        {0.50, 0.50, 0.00}, {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75},
        {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
    const int nat = 8 * n * n * n;
    cpu::Frame f(nat);
    int i = 0;
    for (int ix = 0; ix < n; ++ix)
        for (int iy = 0; iy < n; ++iy)
            for (int iz = 0; iz < n; ++iz)
                for (int b = 0; b < 8; ++b) {
                    f.span(kAtom_type)[i] = 14;
                    f.span(kAtom_mass)[i] = 28.0855;
                    f.span(kAtom_x)[i] = (ix + frac[b][0]) * a;
                    f.span(kAtom_y)[i] = (iy + frac[b][1]) * a;
                    f.span(kAtom_z)[i] = (iz + frac[b][2]) * a;
                    ++i;
                }
    const double L = a * n;
    Mat3 cell;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            cell(r, c) = (r == c) ? L : 0.0;
    f.box.set_cell(cell);
    return f;
}

/// @brief One same-settings SCF leg on one backend; wall covers run() only.
/// @tparam Method `cpu::pw::PW` or `cuda::pw::PW` (identical member surface).
/// @param frame  Supercell geometry [Å].
/// @param o      Shared settings.
/// @param out    Out: copied PWResult of the leg.
/// @return Wall time of `run(frame)` [s].
template <class Method>
double run_leg(const cpu::Frame& frame, const Opts& o, cpu::pw::PWResult& out) {
    Method m;
    m.upf_path = o.upf;
    m.ecut_ry = o.ecut_ry;
    m.k_mesh = o.k_mesh;
    m.tol = o.tol;
    m.max_iter = o.max_iter;
    m.xc = cpu::scf::XCType::LSDA; // the CUDA device envelope
    m.build(frame);
    const auto t0 = std::chrono::steady_clock::now();
    out = m.run(frame);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int report(const char* dev, const Opts& o, int nat,
           const cpu::pw::PWResult& r, double wall_s) {
    std::printf("BENCH_RESULT device=%s n_rep=%d atoms=%d n_kpts=%d "
                "ecut_ry=%.1f kmesh=%d tol=%.1e max_iter=%d iters=%d "
                "converged=%d E_ry=%.8f E_per_atom_ry=%.8f wall_s=%.3f\n",
                dev, o.n_rep, nat, r.n_kpts, o.ecut_ry, o.k_mesh, o.tol,
                o.max_iter, r.iterations, r.converged ? 1 : 0, r.energy,
                r.energy / nat, wall_s);
    std::fflush(stdout);
    if (!std::isfinite(r.energy)) {
        std::fprintf(stderr, "FAIL [%s]: non-finite energy\n", dev);
        return 1;
    }
    return r.converged ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <upf> <n_rep> <cpu|cuda|both> "
                     "[ecut_ry=15] [k_mesh=1] [tol=1e-6] [max_iter=100]\n",
                     argv[0]);
        return 1;
    }
    Opts o;
    o.upf = argv[1];
    o.n_rep = std::atoi(argv[2]);
    o.device = argv[3];
    if (argc > 4) o.ecut_ry = std::atof(argv[4]);
    if (argc > 5) o.k_mesh = std::atoi(argv[5]);
    if (argc > 6) o.tol = std::atof(argv[6]);
    if (argc > 7) o.max_iter = std::atoi(argv[7]);
    if (o.n_rep < 1 || (o.device != "cpu" && o.device != "cuda" &&
                        o.device != "both")) {
        std::fprintf(stderr, "FAIL: bad n_rep=%d or device=%s\n", o.n_rep,
                     o.device.c_str());
        return 1;
    }

    constexpr double kA = 5.43; // Å, conventional diamond-Si cube
    const cpu::Frame frame = make_si_supercell(kA, o.n_rep);
    const int nat = 8 * o.n_rep * o.n_rep * o.n_rep;
    std::printf("BENCH_SETUP n_rep=%d atoms=%d a=%.2f A cell=%.2f A "
                "ecut_ry=%.1f kmesh=%d xc=LSDA upf=%s\n",
                o.n_rep, nat, kA, kA * o.n_rep, o.ecut_ry, o.k_mesh,
                o.upf.c_str());

    int rc = 0;
    bool have_cpu = false, have_cuda = false;
    cpu::pw::PWResult r_cpu, r_cuda;

    if (o.device == "cpu" || o.device == "both") {
        const double w = run_leg<cpu::pw::PW>(frame, o, r_cpu);
        rc |= report("cpu", o, nat, r_cpu, w);
        have_cpu = true;
    }
    if (o.device == "cuda" || o.device == "both") {
        cudaFree(nullptr); // create the CUDA context outside the timed region
        const double w = run_leg<cuda::pw::PW>(frame, o, r_cuda);
        rc |= report("cuda", o, nat, r_cuda, w);
        have_cuda = true;
    }
    if (have_cpu && have_cuda) {
        std::printf("BENCH_PARITY n_rep=%d atoms=%d |dE|_ry=%.3e "
                    "|dE|_per_atom_ry=%.3e\n",
                    o.n_rep, nat, std::abs(r_cuda.energy - r_cpu.energy),
                    std::abs(r_cuda.energy - r_cpu.energy) / nat);
    }
    return rc;
}
