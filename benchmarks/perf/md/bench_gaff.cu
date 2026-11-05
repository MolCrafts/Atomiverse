/// @file benchmarks/perf/md/bench_gaff.cu
/// @brief Public-API GAFF PEO correctness and short NVE timing consumer.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ATV;
int main(int argc, char** argv) try {
    if (argc < 3)
        throw std::runtime_error(
            "usage: bench_gaff input.txt output.txt [steps=100] [warmup=10] "
            "[all|nonbonded|bond|angle|dihedral] [samples=3] "
            "[drift_every=0]");
    std::ifstream in(argv[1]);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    int n, nt, nbt, nat, ndt, nb, na, nd, order, nv;
    double lx, ly, lz, xy, xz, yz, alpha, cutoff, spacing, dt;
    in >> n >> lx >> ly >> lz >> xy >> xz >> yz >> nt >> nbt >> nat >> ndt >> nb
        >> na >> nd >> alpha >> cutoff >> spacing >> order >> dt >> nv;
    if (nv != 0 && nv != n)
        throw std::invalid_argument("velocity rows must be 0 or N");
    std::vector<double> eps(nt), sig(nt), bk(nbt), br(nbt), ak(nat), at(nat);
    std::vector<std::array<double, 5>> dc(ndt);
    const std::string term = argc > 5 ? argv[5] : "all";
    if (term != "all" && term != "nonbonded" && term != "bond"
        && term != "angle" && term != "dihedral")
        throw std::invalid_argument("unknown force term");
    for (int i = 0; i < nt; ++i)
        in >> eps[i] >> sig[i];
    // Diagonal rows only: unlisted pairs mix Lorentz-Berthelot, which keeps
    // the per-atom fast tile path. OpenMM compares unshifted LJ.
    std::vector<LJTypeCoeff> lj;
    for (int i = 0; i < nt; ++i)
        lj.push_back({i, i, eps[i], sig[i]});
    for (int i = 0; i < nbt; ++i)
        in >> bk[i] >> br[i];
    for (int i = 0; i < nat; ++i)
        in >> ak[i] >> at[i];
    for (auto& c : dc)
        for (double& v : c)
            in >> v;
    cpu::Frame f(n);
    // LAMMPS-form H (lattice vectors are the columns): a = (lx,0,0),
    // b = (xy,ly,0), c = (xz,yz,lz).
    Mat3 cell = Mat3::diagonal(lx, ly, lz);
    cell(0, 1) = xy;
    cell(0, 2) = xz;
    cell(1, 2) = yz;
    f.box.set_cell(cell);
    for (int i = 0; i < n; ++i) {
        in >> f.span(kAtom_x)[i] >> f.span(kAtom_y)[i] >> f.span(kAtom_z)[i]
            >> f.span(kAtom_charge)[i] >> f.span(kAtom_type)[i]
            >> f.span(kAtom_mass)[i];
        f.span(kAtom_tag)[i] = i;
    }
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int b = 0; b < nb; ++b) {
        int i, j, t;
        in >> i >> j >> t;
        f.add_bond(i, j, t);
    }
    for (int a = 0; a < na; ++a) {
        int i, j, k, t;
        in >> i >> j >> k >> t;
        f.add_angle(i, j, k, t);
    }
    for (int d = 0; d < nd; ++d) {
        int i, j, k, l, t;
        in >> i >> j >> k >> l >> t;
        f.add_dihedral(i, j, k, l, t);
    }
    for (int i = 0; i < nv; ++i) // initial velocities [Angstrom/fs]
        in >> f.span(kAtom_vx)[i] >> f.span(kAtom_vy)[i] >> f.span(kAtom_vz)[i];
    auto bind = [&](auto& sim, auto phase) {
        if (term == "all" || term == "nonbonded") {
            sim.set_neighbor(ATV::cuda::NeighborList{cutoff, 0.08 * cutoff});
            sim.set_evaluator(ATV::cuda::PairLJCutCoulLong{lj, alpha, cutoff}
                                  .set_special_bonds_amber()
                                  .set_energy_shift(false));
            sim.on(phase).add(ATV::cuda::KSpacePME{alpha, spacing, order}, 100);
        }
        if (term == "all" || term == "bond")
            sim.set_evaluator(ATV::cuda::BondHarmonic{bk, br});
        if (term == "all" || term == "angle")
            sim.set_evaluator(ATV::cuda::AngleHarmonic{ak, at});
        if (term == "all" || term == "dihedral")
            sim.set_evaluator(ATV::cuda::DihedralMultiHarmonic{dc});
    };
    {
        ATV::cuda::Evaluate eval;
        bind(eval, evaluate::Phase::evaluate);
        eval.build();
        eval.run(f);
        std::ofstream out(argv[2]);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << std::setprecision(17) << f.meta.get(kEnergy) << '\n';
        for (int i = 0; i < n; ++i)
            out << f.span(kAtom_fx)[i] << ' ' << f.span(kAtom_fy)[i] << ' '
                << f.span(kAtom_fz)[i] << '\n';
    }
    const int steps = argc > 3 ? std::stoi(argv[3]) : 100,
              warm = argc > 4 ? std::stoi(argv[4]) : 10,
              samples = argc > 6 ? std::stoi(argv[6]) : 3,
              drift_every = argc > 7 ? std::stoi(argv[7]) : 0;
    if (steps < 1 || warm < 1 || samples < 1 || drift_every < 0)
        throw std::invalid_argument("positive step counts required");
    // MD keeps the nonbonded pieces by reference so their rebuild and tile
    // statistics can be reported after timing; they are declared first so
    // they outlive the driver that borrows them.
    ATV::cuda::NeighborList md_neighbor{cutoff, 0.08 * cutoff};
    auto md_pair = ATV::cuda::PairLJCutCoulLong{lj, alpha, cutoff}
                       .set_special_bonds_amber()
                       .set_energy_shift(false);
    ATV::cuda::md::MD md;
    if (term == "all" || term == "nonbonded") {
        md.set_neighbor_ref(md_neighbor);
        md.set_evaluator_ref(md_pair);
        md.on(ATV::md::Phase::force)
            .add(ATV::cuda::KSpacePME{alpha, spacing, order}, 100);
    }
    if (term == "all" || term == "bond")
        md.set_evaluator(ATV::cuda::BondHarmonic{bk, br});
    if (term == "all" || term == "angle")
        md.set_evaluator(ATV::cuda::AngleHarmonic{ak, at});
    if (term == "all" || term == "dihedral")
        md.set_evaluator(ATV::cuda::DihedralMultiHarmonic{dc});
    if (std::getenv("ATV_BENCH_TIMING"))
        md.enable_timing();
    md.set_integrator(ATV::cuda::md::VelocityVerlet{dt});
    md.build();
    if (drift_every > 0) {
        // NVE energy series from t = 0: "step potential kinetic" [eV].
        std::ofstream drift(std::string(argv[2]) + ".drift");
        drift.exceptions(std::ios::failbit | std::ios::badbit);
        drift << std::setprecision(17);
        md.run(f, 0);
        // run(0) prepares forces and energy only; KE at t = 0 is evaluated
        // here from the input velocities (later records come from MD).
        double ke0 = 0;
        for (int i = 0; i < n; ++i)
            ke0 += 0.5 * units::md_mvv2e * f.span(kAtom_mass)[i]
                   * (f.span(kAtom_vx)[i] * f.span(kAtom_vx)[i]
                      + f.span(kAtom_vy)[i] * f.span(kAtom_vy)[i]
                      + f.span(kAtom_vz)[i] * f.span(kAtom_vz)[i]);
        for (int done = 0;; done += drift_every) {
            drift << done << ' ' << f.meta.get(kEnergy) << ' '
                  << (done == 0 ? ke0 : f.meta.get(kKinetic)) << '\n';
            if (done >= steps)
                break;
            md.run(f, drift_every);
        }
        std::cout << "engine=atv N=" << n << " drift_steps=" << steps << '\n';
        return 0;
    }
    md.run(f, warm);
    std::vector<double> times;
    for (int sample = 0; sample < samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        md.run(f, steps);
        times.push_back(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count()
                        / steps);
    }
    std::sort(times.begin(), times.end());
    const double ms = 0.5 * (times[(samples - 1) / 2] + times[samples / 2]);
    if (!std::isfinite(f.meta.get(kEnergy)))
        throw std::runtime_error("non-finite final potential energy");
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(f.span(kAtom_x)[i])
            || !std::isfinite(f.span(kAtom_y)[i])
            || !std::isfinite(f.span(kAtom_z)[i]))
            throw std::runtime_error("non-finite final position");
    }
    std::ofstream final_out(std::string(argv[2]) + ".final");
    final_out.exceptions(std::ios::failbit | std::ios::badbit);
    final_out << std::setprecision(17);
    for (int i = 0; i < n; ++i)
        final_out << f.span(kAtom_x)[i] << ' ' << f.span(kAtom_y)[i] << ' '
                  << f.span(kAtom_z)[i] << '\n';
    // The *_bytes keys are parsed by benchmarks/compare/gaff-peo-openmm/compare.py:
    // stored coordinate, force math, reduction and integration widths.
    std::cout << std::setprecision(10) << "engine=atv N=" << n
              << " steps=" << steps << " ms_per_step=" << ms
              << " samples=" << samples
              << " real_bytes=" << sizeof(ATV::cuda::pos_t)
              << " work_bytes=" << sizeof(ATV::cuda::force_t)
              << " accum_bytes=" << sizeof(ATV::cuda::acc_t)
              << " mixed_bytes=" << sizeof(ATV::cuda::mixed_t)
              << " energy=" << f.meta.get(kEnergy)
              << " rebuilds=" << md_neighbor.rebuild_count()
              << " candidate_tiles=" << md_neighbor.candidate_tiles()
              << " valid_tiles=" << md_pair.valid_tiles()
              << " packed_tiles=" << md_pair.packed_tiles() << '\n';
    if (std::getenv("ATV_BENCH_TIMING"))
        std::cout << md.summary();
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
