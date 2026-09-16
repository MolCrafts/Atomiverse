/// @file benchmarks/scf/validation/ri_speed/bench_ri_speed.cpp
/// @brief Glycine / def2-SVP RHF wall-clock: exact-ERI vs RI-JK.
///
/// Out-of-library consumer (public headers + link atomiverse). Runs each
/// method `repeats` times on one fixed geometry and prints per-run wall
/// times [s] plus the converged energies [Ha] — the run.sh driver pairs
/// this with a PySCF density-fit timing for the three-column table in
/// docs/tutorial/ch19-hf-large-systems.md. NOT part of ctest.

#include <chrono>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

/// Gas-phase glycine (NH2-CH2-COOH), 10 atoms, Å — the same literal
/// geometry is pinned in run.sh for the PySCF timing.
ATV::cpu::Frame glycine() {
    using namespace ATV;
    const int Z[10] = {7, 6, 6, 8, 8, 1, 1, 1, 1, 1};
    const double xyz[10][3] = {
        {1.9310, 0.0900, -0.0340},  {0.5610, -0.4000, 0.0080},
        {-0.4400, 0.7360, -0.0100}, {-0.1560, 1.9160, -0.0330},
        {-1.7100, 0.3210, 0.0180},  {2.0350, 0.8420, 0.6390},
        {2.5810, -0.6500, 0.2130},  {0.4110, -1.0180, 0.9010},
        {0.3480, -1.0480, -0.8500}, {-2.2670, 1.1120, 0.0090},
    };
    cpu::Frame f(10);
    for (int i = 0; i < 10; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
    }
    f.box = cpu::Box::cubic(30.0);
    return f;
}

double run_once(bool use_ri, double& energy) {
    using clock = std::chrono::steady_clock;
    ATV::cpu::Frame f = glycine();
    ATV::cpu::scf::RHF rhf;
    rhf.basis = ATV::cpu::scf::BasisSet::Def2SVP;
    rhf.tol = 1e-9;
    rhf.max_iter = 200;
    if (use_ri)
        rhf.ri = ATV::cpu::scf::RI::JK;
    const auto t0 = clock::now();
    rhf.run(f);
    const auto t1 = clock::now();
    energy = rhf.energy();
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int main() {
    const int repeats = 3;
    double e_exact = 0.0, e_ri = 0.0;
    std::printf("# glycine/def2-SVP RHF, tol=1e-9, %d repeats\n", repeats);
    for (int r = 0; r < repeats; ++r) {
        const double t = run_once(false, e_exact);
        std::printf("exact  run %d: %.3f s  E = %.9f Ha\n", r, t, e_exact);
    }
    for (int r = 0; r < repeats; ++r) {
        const double t = run_once(true, e_ri);
        std::printf("ri-jk  run %d: %.3f s  E = %.9f Ha\n", r, t, e_ri);
    }
    std::printf("delta E (ri - exact) = %.3e Ha\n", e_ri - e_exact);
    return 0;
}
