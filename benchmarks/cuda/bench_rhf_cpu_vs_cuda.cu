/// @file benchmarks/cuda/bench_rhf_cpu_vs_cuda.cu
/// @brief Fair CPU vs CUDA RHF wall-time + AO-size scaling (STO-3G).
///
/// Built as `bench_rhf_cpu_vs_cuda` with ATV_BUILD_BENCHMARKS+CUDA+HF (see
/// benchmarks/cuda/CMakeLists.txt). Run on a GPU node so both backends share
/// the same hardware.
///
/// Usage:
///   OMP_NUM_THREADS=8 ./bin/bench_rhf_cpu_vs_cuda [n_rep_energy=12] [n_rep_force=8]
///
/// Reports mean/min wall ms, SCF iters, energy [Ha], and CUDA/CPU speedup.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cuda/scf/rhf.cuh"

using namespace ATV;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static void set_atom(cpu::Frame& f, int i, int Z, double x, double y,
                     double z) {
    f.span(kAtom_type)[i] = Z;
    f.span(kAtom_x)[i] = x;
    f.span(kAtom_y)[i] = y;
    f.span(kAtom_z)[i] = z;
    // rough mass by Z (only for Frame completeness)
    f.span(kAtom_mass)[i] = (Z == 1)   ? 1.008
                            : (Z == 6) ? 12.011
                            : (Z == 7) ? 14.007
                            : (Z == 8) ? 15.999
                                       : static_cast<double>(Z);
}

// ── Geometries (Å), closed-shell, STO-3G ──────────────────────────────

static cpu::Frame mol_h2() {
    cpu::Frame f(2);
    set_atom(f, 0, 1, 0.0, 0.0, 0.0);
    set_atom(f, 1, 1, 0.74, 0.0, 0.0);
    f.box = cpu::Box::cubic(40.0);
    return f;
}

static cpu::Frame mol_h2o() {
    cpu::Frame f(3);
    set_atom(f, 0, 8, 0.0, 0.0, 0.1173);
    set_atom(f, 1, 1, 0.0, 0.7572, -0.4692);
    set_atom(f, 2, 1, 0.0, -0.7572, -0.4692);
    f.box = cpu::Box::cubic(40.0);
    return f;
}

static cpu::Frame mol_ch4() {
    // Tetrahedral CH4, C–H ≈ 1.09 Å
    cpu::Frame f(5);
    const double r = 1.09;
    const double a = r / std::sqrt(3.0);
    set_atom(f, 0, 6, 0, 0, 0);
    set_atom(f, 1, 1, a, a, a);
    set_atom(f, 2, 1, a, -a, -a);
    set_atom(f, 3, 1, -a, a, -a);
    set_atom(f, 4, 1, -a, -a, a);
    f.box = cpu::Box::cubic(40.0);
    return f;
}

static cpu::Frame mol_c2h4() {
    // Planar ethylene, C=C 1.33 Å, C–H 1.09 Å, HCH ~117°
    cpu::Frame f(6);
    set_atom(f, 0, 6, -0.665, 0.0, 0.0);
    set_atom(f, 1, 6, 0.665, 0.0, 0.0);
    set_atom(f, 2, 1, -1.23, 0.93, 0.0);
    set_atom(f, 3, 1, -1.23, -0.93, 0.0);
    set_atom(f, 4, 1, 1.23, 0.93, 0.0);
    set_atom(f, 5, 1, 1.23, -0.93, 0.0);
    f.box = cpu::Box::cubic(40.0);
    return f;
}

static cpu::Frame mol_c2h6() {
    // Staggered ethane, C–C 1.54 Å
    cpu::Frame f(8);
    set_atom(f, 0, 6, -0.77, 0.0, 0.0);
    set_atom(f, 1, 6, 0.77, 0.0, 0.0);
    set_atom(f, 2, 1, -1.15, 1.02, 0.0);
    set_atom(f, 3, 1, -1.15, -0.51, 0.88);
    set_atom(f, 4, 1, -1.15, -0.51, -0.88);
    set_atom(f, 5, 1, 1.15, -1.02, 0.0);
    set_atom(f, 6, 1, 1.15, 0.51, 0.88);
    set_atom(f, 7, 1, 1.15, 0.51, -0.88);
    f.box = cpu::Box::cubic(40.0);
    return f;
}

static cpu::Frame mol_c6h6() {
    // Benzene, planar, C–C 1.40 Å, C–H 1.08 Å
    cpu::Frame f(12);
    const double rc = 1.40;
    const double rh = 1.40 + 1.08;
    for (int i = 0; i < 6; ++i) {
        const double th = 2.0 * 3.141592653589793 * i / 6.0;
        set_atom(f, i, 6, rc * std::cos(th), rc * std::sin(th), 0.0);
        set_atom(f, 6 + i, 1, rh * std::cos(th), rh * std::sin(th), 0.0);
    }
    f.box = cpu::Box::cubic(50.0);
    return f;
}

/// Linear H_n (pure-s), bond 0.74 Å — device force path + energy.
static cpu::Frame mol_h_chain(int n) {
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i)
        set_atom(f, i, 1, 0.74 * i, 0.0, 0.0);
    f.box = cpu::Box::cubic(std::max(40.0, 0.74 * n + 20.0));
    return f;
}

// STO-3G AO count: H=1, C/N/O=5
static int count_ao_sto3g(const cpu::Frame& f) {
    int nao = 0;
    auto Z = f.span(kAtom_type);
    for (int i = 0; i < static_cast<int>(f.nlocal()); ++i) {
        const int z = static_cast<int>(Z[i]);
        if (z == 1)
            nao += 1;
        else if (z >= 3 && z <= 10)
            nao += 5; // Li–Ne STO-3G
        else
            nao += 5; // conservative
    }
    return nao;
}

struct Stat {
    double mean_ms = 0, min_ms = 1e300, max_ms = 0;
    double e_ha = 0;
    int iters = 0;
    int n_rep = 0;
    bool ok = false;
};

template <typename Fn>
static Stat time_reps(int n_rep, int n_warm, Fn&& body) {
    for (int w = 0; w < n_warm; ++w)
        body();
    Stat s;
    s.n_rep = n_rep;
    double sum = 0;
    for (int i = 0; i < n_rep; ++i) {
        auto t0 = clk::now();
        body();
        const double dt = ms_since(t0);
        sum += dt;
        s.min_ms = std::min(s.min_ms, dt);
        s.max_ms = std::max(s.max_ms, dt);
    }
    s.mean_ms = sum / std::max(1, n_rep);
    s.ok = true;
    return s;
}

static Stat bench_cpu_energy(const cpu::Frame& proto, int n_rep, double tol) {
    Stat s = time_reps(n_rep, /*warm=*/1, [&] {
        auto f = proto;
        cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        rhf.run(f);
        s.e_ha = rhf.energy();
        s.iters = 0; // CPU may not expose iterations the same way
    });
    // one more for energy/iters
    {
        auto f = proto;
        cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        rhf.run(f);
        s.e_ha = rhf.energy();
        s.ok = std::isfinite(s.e_ha);
    }
    return s;
}

static Stat bench_cuda_energy(const cpu::Frame& proto, int n_rep, double tol) {
    Stat s = time_reps(n_rep, 1, [&] {
        auto f = proto;
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        rhf.with_gradient(false);
        rhf.run(f);
        cudaDeviceSynchronize();
        s.e_ha = rhf.energy();
        s.iters = rhf.iterations();
    });
    s.ok = std::isfinite(s.e_ha);
    return s;
}

static Stat bench_cpu_forces(const cpu::Frame& proto, int n_rep, double tol) {
    Stat s = time_reps(n_rep, 1, [&] {
        auto f = proto;
        cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        auto g = rhf.forces(f);
        s.e_ha = g.energy / units::kHartreeInEv;
    });
    s.ok = std::isfinite(s.e_ha);
    return s;
}

static Stat bench_cuda_forces(const cpu::Frame& proto, int n_rep, double tol) {
    Stat s = time_reps(n_rep, 1, [&] {
        auto f = proto;
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, tol, 100};
        auto g = rhf.forces(f); // forces() enables compute_force
        cudaDeviceSynchronize();
        s.e_ha = rhf.energy();
        s.iters = rhf.iterations();
        (void)g;
    });
    s.ok = std::isfinite(s.e_ha);
    return s;
}

static void print_row(const char* name, int natm, int nao, const Stat& cpu,
                      const Stat& gpu, const char* note = "") {
    const double spd =
        (gpu.mean_ms > 0 && cpu.mean_ms > 0) ? cpu.mean_ms / gpu.mean_ms : 0;
    std::printf(
        "%-10s  N=%2d  nAO=%3d | CPU %8.2f ms (min %7.2f)  E=%10.6f | "
        "CUDA %8.2f ms (min %7.2f) it=%3d E=%10.6f | speedup %6.2fx  %s\n",
        name, natm, nao, cpu.mean_ms, cpu.min_ms, cpu.e_ha, gpu.mean_ms,
        gpu.min_ms, gpu.iters, gpu.e_ha, spd, note);
}

int main(int argc, char** argv) {
    int n_rep_e = 12;
    int n_rep_f = 8;
    if (argc > 1)
        n_rep_e = std::max(1, std::atoi(argv[1]));
    if (argc > 2)
        n_rep_f = std::max(1, std::atoi(argv[2]));

    const double tol = 1e-9;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("GPU: %s  CC %d.%d\n", prop.name, prop.major, prop.minor);
    std::printf("OMP_NUM_THREADS=%d  (omp_get_max_threads=%d)\n",
                omp_get_max_threads(), omp_get_max_threads());
    std::printf("n_rep energy=%d  force=%d  tol=%.1e  basis=STO-3G\n\n",
                n_rep_e, n_rep_f, tol);

    struct Case {
        const char* name;
        cpu::Frame (*make)();
    };

    // ── A. Energy-only: s/p molecules (CUDA device integrals) ─────────
    std::printf("=== ENERGY-ONLY (with_gradient=false on CUDA) ===\n");
    std::printf("%-10s  %-12s | %-36s | %-42s | %s\n", "mol", "size", "CPU",
                "CUDA", "CUDA/CPU");

    const Case energy_cases[] = {
        {"H2", mol_h2},     {"H2O", mol_h2o},   {"CH4", mol_ch4},
        {"C2H4", mol_c2h4}, {"C2H6", mol_c2h6}, {"C6H6", mol_c6h6},
    };
    for (const auto& c : energy_cases) {
        auto f = c.make();
        const int natm = static_cast<int>(f.nlocal());
        const int nao = count_ao_sto3g(f);
        // Fewer reps for larger systems
        const int rep = (nao >= 30)   ? std::max(3, n_rep_e / 3)
                        : (nao >= 14) ? std::max(5, n_rep_e / 2)
                                      : n_rep_e;
        Stat cpu = bench_cpu_energy(f, rep, tol);
        Stat gpu = bench_cuda_energy(f, rep, tol);
        const char* note = "";
        if (std::abs(cpu.e_ha - gpu.e_ha) > 1e-4)
            note = "WARN |E|";
        print_row(c.name, natm, nao, cpu, gpu, note);
    }

    // ── B. Pure-s H-chain energy scaling ───────────────────────────────
    std::printf("\n=== ENERGY-ONLY pure-s H_n (device path, n_AO = n) ===\n");
    for (int n : {2, 4, 6, 8, 12, 16, 24}) {
        auto f = mol_h_chain(n);
        // even electron count only
        if (n % 2)
            continue;
        const int rep = (n >= 16) ? std::max(3, n_rep_e / 3) : n_rep_e;
        Stat cpu = bench_cpu_energy(f, rep, tol);
        Stat gpu = bench_cuda_energy(f, rep, tol);
        char name[16];
        std::snprintf(name, sizeof(name), "H%d", n);
        print_row(name, n, n, cpu, gpu);
    }

    // ── C. Forces: pure-s (CUDA device forces) vs CPU ──────────────────
    std::printf("\n=== FORCES (CUDA pure-s device forces; CPU analytic) ===\n");
    for (int n : {2, 4, 6, 8, 12}) {
        if (n % 2)
            continue;
        auto f = mol_h_chain(n);
        const int rep = (n >= 8) ? std::max(3, n_rep_f / 2) : n_rep_f;
        Stat cpu = bench_cpu_forces(f, rep, tol);
        Stat gpu = bench_cuda_forces(f, rep, tol);
        char name[16];
        std::snprintf(name, sizeof(name), "H%d", n);
        print_row(name, n, n, cpu, gpu, "forces");
    }

    // H2O forces: CUDA falls back to host for p-shells — still time it
    std::printf("\n=== FORCES H2O (CUDA uses host force fallback for p) ===\n");
    {
        auto f = mol_h2o();
        Stat cpu = bench_cpu_forces(f, n_rep_f, tol);
        Stat gpu = bench_cuda_forces(f, std::max(2, n_rep_f / 2), tol);
        print_row("H2O", 3, 7, cpu, gpu, "CUDA host-F");
    }

    std::printf("\nDONE\n");
    return 0;
}
