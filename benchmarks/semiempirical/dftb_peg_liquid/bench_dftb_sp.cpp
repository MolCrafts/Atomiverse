/// @file benchmarks/semiempirical/dftb_peg_liquid/bench_dftb_sp.cpp
/// @brief Wall-clock of one SCC-DFTB2 single point (energy + forces) on a
///        `.gen` geometry, for the fair comparison against DFTB+.
///
/// NOT part of ctest. Out-of-library consumer: public headers + link
/// `atomiverse`. Methodology and the measured table are in README.md.
///
///   bench_dftb_sp <geo.gen> [scc_tol] [repeat]
///
/// The `.gen` geometry kind picks the boundary the way DFTB+ reads it: `S`/`F`
/// is a lattice, `C` is a cluster.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

int z_of(const std::string& s) {
    if (s == "H")
        return 1;
    if (s == "C")
        return 6;
    if (s == "N")
        return 7;
    if (s == "O")
        return 8;
    std::fprintf(stderr, "bench_dftb_sp: unsupported species '%s'\n",
                 s.c_str());
    std::exit(2);
}

double mass_of(int z) {
    return z == 1 ? 1.008 : z == 6 ? 12.011 : z == 7 ? 14.007 : 15.999;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: bench_dftb_sp <geo.gen> [scc_tol] [repeat]\n");
        return 2;
    }
    const double scc_tol = argc > 2 ? std::atof(argv[2]) : 1e-5;
    const int repeat = argc > 3 ? std::atoi(argv[3]) : 3;

    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "bench_dftb_sp: cannot open %s\n", argv[1]);
        return 2;
    }
    std::string line;
    std::getline(in, line);
    std::istringstream head(line);
    int nat = 0;
    std::string kind;
    head >> nat >> kind;
    std::getline(in, line);
    std::vector<std::string> spec;
    {
        std::istringstream s(line);
        std::string t;
        while (s >> t)
            spec.push_back(t);
    }
    std::vector<int> Z(static_cast<std::size_t>(nat));
    std::vector<double> X(static_cast<std::size_t>(nat));
    std::vector<double> Y(static_cast<std::size_t>(nat));
    std::vector<double> Zc(static_cast<std::size_t>(nat));
    for (int i = 0; i < nat; ++i) {
        std::getline(in, line);
        std::istringstream s(line);
        int idx = 0, sp = 0;
        s >> idx >> sp >> X[static_cast<std::size_t>(i)]
            >> Y[static_cast<std::size_t>(i)]
            >> Zc[static_cast<std::size_t>(i)];
        Z[static_cast<std::size_t>(i)] =
            z_of(spec[static_cast<std::size_t>(sp - 1)]);
    }
    const bool periodic =
        kind == "S" || kind == "s" || kind == "F" || kind == "f";
    Mat3 cell{};
    if (periodic) {
        std::getline(in, line); // origin
        for (int r = 0; r < 3; ++r) {
            std::getline(in, line);
            std::istringstream s(line);
            s >> cell(r, 0) >> cell(r, 1) >> cell(r, 2);
        }
    }

    cpu::Frame frame(nat);
    for (int i = 0; i < nat; ++i) {
        const std::size_t u = static_cast<std::size_t>(i);
        frame.span(kAtom_type)[i] = Z[u];
        frame.span(kAtom_x)[i] = X[u];
        frame.span(kAtom_y)[i] = Y[u];
        frame.span(kAtom_z)[i] = Zc[u];
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = mass_of(Z[u]);
    }
    if (periodic)
        frame.box.set_cell(cell);
    else
        frame.box = cpu::Box::cubic(100.0);

    std::vector<double> walls;
    double energy = 0;
    int iters = 0;
    for (int r = 0; r < repeat; ++r) {
        cpu::Frame f = frame;
        f.meta.get(kEnergy) = 0;
        cpu::semiempirical::DFTB2 dftb;
        dftb.boundary = periodic ? cpu::semiempirical::DftbBoundary::lattice
                                 : cpu::semiempirical::DftbBoundary::cluster;
        dftb.scc_tol = scc_tol;
        dftb.max_scc = 200;
        dftb.with_forces = true;

        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add_ref(dftb);
        eval.build();
        const auto t0 = std::chrono::steady_clock::now();
        eval.run(f);
        walls.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count());
        energy = dftb.last_energy();
        iters = dftb.last_scc_iterations();
    }
    std::sort(walls.begin(), walls.end());
    const double median = walls[walls.size() / 2];

    std::printf("{\"n_atoms\": %d, \"periodic\": %s, \"scc_tol\": %g,"
                " \"scc_iterations\": %d, \"energy_eV\": %.10f,"
                " \"wall_s_median\": %.4f, \"wall_s_min\": %.4f,"
                " \"repeat\": %d}\n",
                nat, periodic ? "true" : "false", scc_tol, iters, energy,
                median, walls.front(), repeat);
    return 0;
}
