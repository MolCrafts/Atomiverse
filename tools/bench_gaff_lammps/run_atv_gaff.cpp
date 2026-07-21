/// @file tools/bench_gaff_lammps/run_atv_gaff.cpp
/// @brief MPI GAFF MD timing bench vs LAMMPS paper config.
///
/// Aligns with figures/flexible (Gudla–Zhang) / ref_lammps packmol_n100:
///   units real → ATV eV/Å/fs
///   atom_style full, special_bonds amber
///   pair_style lj/cut/coul/long 10 10  (fused PairLJCutCoulLong)
///   kspace_style pppm 1e-4             (KSpacePPPM)
///   neighbor 2.0 bin; neigh_modify delay 0 every 1 check yes
///   fix nvt temp 400.0 400.0 100.0; timestep 1.0
///
/// Usage:
///   mpirun -np N run_atv_gaff --data melt.data --steps 1000 [--warmup 100]
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

struct Args {
    std::string data =
        "/home/jicli594/work/molpack-paper/figures/relaxation/"
        "ref_lammps/runs/packmol_n100_r2/melt/melt.data";
    int warmup = 50;
    int steps = 500;
    double cutoff = 10.0;
    double accuracy = 1.0e-4;
    double grid_spacing = -1.0; // <0 → from_accuracy default (rcut/2.5)
    double T = 400.0;
    double tau_T = 100.0;
    int seed = 4928;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--data"))
            a.data = next("--data");
        else if (!std::strcmp(argv[i], "--steps"))
            a.steps = std::atoi(next("--steps"));
        else if (!std::strcmp(argv[i], "--warmup"))
            a.warmup = std::atoi(next("--warmup"));
        else if (!std::strcmp(argv[i], "--cutoff"))
            a.cutoff = std::atof(next("--cutoff"));
        else if (!std::strcmp(argv[i], "--accuracy"))
            a.accuracy = std::atof(next("--accuracy"));
        else if (!std::strcmp(argv[i], "--grid-spacing"))
            a.grid_spacing = std::atof(next("--grid-spacing"));
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf(
                "run_atv_gaff --data PATH [--steps N] [--warmup N] "
                "[--cutoff 10] [--accuracy 1e-4] [--grid-spacing Å]\n");
            std::exit(0);
        }
    }
    return a;
}

// GAFF2 PEO type tables matching melt.ff / type labels in melt.data
// types 0..4 = c3, h1, hc, ho, oh
std::vector<cpu::LJTypeCoeff> peo_lj(double cut_unused) {
    (void)cut_unused;
    using cpu::kcal_per_mol_to_ev;
    return {
        {0, 0, kcal_per_mol_to_ev(0.1078), 3.39771},
        {1, 1, kcal_per_mol_to_ev(0.0208), 2.421997},
        {2, 2, kcal_per_mol_to_ev(0.0208), 2.600177},
        {3, 3, kcal_per_mol_to_ev(0.0047), 0.537925},
        {4, 4, kcal_per_mol_to_ev(0.0930), 3.242871},
    };
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nrank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    const Args args = parse(argc, argv);

    // Load on rank 0 then... actually load_lammps_data is serial full-system;
    // DomainDecomposition will scatter on build. All ranks load (simple).
    cpu::Frame frame = cpu::load_lammps_data(args.data);
    const int n_atoms = frame.nlocal();
    if (rank == 0) {
        std::printf(
            "[atv] ranks=%d OMP_NUM_THREADS=%s atoms=%d L=%.3f cut=%.1f "
            "acc=%.1e steps=%d warmup=%d\n",
            nrank,
            std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS")
                                           : "1",
            n_atoms, frame.simbox.length(0), args.cutoff, args.accuracy,
            args.steps, args.warmup);
        std::printf("[atv] data=%s\n", args.data.c_str());
    }

    using cpu::kcal_per_mol_to_ev;
    using cpu::degree_to_rad;

    // Bond types: c3-c3, c3-h1, c3-hc, c3-oh, ho-oh
    std::vector<double> Kbond = {
        kcal_per_mol_to_ev(228.89), kcal_per_mol_to_ev(344.22),
        kcal_per_mol_to_ev(345.25), kcal_per_mol_to_ev(285.15),
        kcal_per_mol_to_ev(535.51)};
    std::vector<double> r0 = {1.5354, 1.0969, 1.0962, 1.4242, 0.9725};
    // Angle types 1..9 from data labels
    std::vector<double> Kang = {
        kcal_per_mol_to_ev(43.28),  kcal_per_mol_to_ev(43.27),
        kcal_per_mol_to_ev(76.79),  kcal_per_mol_to_ev(88.334),
        kcal_per_mol_to_ev(65.13),  kcal_per_mol_to_ev(35.64),
        kcal_per_mol_to_ev(56.29),  kcal_per_mol_to_ev(35.80),
        kcal_per_mol_to_ev(62.80)};
    std::vector<double> th0 = {
        degree_to_rad(109.590047), degree_to_rad(109.680047),
        degree_to_rad(109.660047), degree_to_rad(113.600049),
        degree_to_rad(107.390046), degree_to_rad(108.550047),
        degree_to_rad(110.380047), degree_to_rad(107.730046),
        degree_to_rad(109.500047)};
    // Dihedral types (fourier single term → Charmm)
    std::vector<double> Kd = {
        kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.060000),
        kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.123000),
        kcal_per_mol_to_ev(0.155556), kcal_per_mol_to_ev(0.090000),
        kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.250000),
        kcal_per_mol_to_ev(1.000000)};
    std::vector<int> nper = {3, 3, 3, 3, 3, 3, 3, 1, 1};
    std::vector<double> dphase(9, 0.0);

    const double alpha =
        cpu::pppm_alpha_from_accuracy(args.accuracy, args.cutoff);
    const double grid_sp =
        args.grid_spacing > 0.0 ? args.grid_spacing : (args.cutoff / 2.5);
    if (rank == 0)
        std::printf("[atv] alpha=%.6f (from accuracy) grid_spacing=%.3f "
                    "(LAMMPS pppm 1e-4 ≈ 4 Å → 24³ on L~100)\n",
                    alpha, grid_sp);

    cpu::md::MD sim;
    // Match paper in.workflow: neighbor 2.0 bin; neigh_modify delay 0 every 1
    // check yes. atom_modify sort: LAMMPS default nfreq=1000 (paper does not
    // set sort 1); every-reneighbor sort thrash-rebuilds BondedTupleCache.
    sim.set_neighbor(
        cpu::md::NeighborList<>{args.cutoff, /*skin=*/2.0, /*every=*/1,
                                /*delay=*/0, /*check=*/true}
            .set_atom_sort({.nfreq = 1000, .binsize = 0.0, .enabled = true}));
    // NVT: skip bonded virial (barostat-only). Pair still needs energy;
    // stress off saves ~3 outer products per torsion and pair virial.
    sim.add_potential(
        cpu::PairLJCutCoulLong{peo_lj(args.cutoff), alpha, args.cutoff}
            .set_special_bonds_amber()
            .set_compute_stress(false));
    sim.add(cpu::KSpacePPPM::from_accuracy(args.accuracy, args.cutoff, grid_sp));
    sim.add_potential(cpu::BondHarmonic{Kbond, r0});
    sim.add_potential(cpu::AngleHarmonic{Kang, th0});
    sim.add_potential(
        cpu::DihedralCharmm(Kd, nper, dphase).set_compute_stress(false));
    sim.set_integrator(cpu::md::VelocityVerlet{1.0});
    sim.set_thermostat(cpu::md::NoseHoover{args.T, args.tau_T, 1.0});
    sim.add(cpu::md::InitVelocity{args.T, static_cast<unsigned>(args.seed)});
    sim.build();

    // Warmup (not timed in our wall clock; still contributes to driver summary)
    if (args.warmup > 0)
        sim.run(frame, args.warmup);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    sim.run(frame, args.steps);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();
    const double wall =
        std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        const double steps_s = args.steps / wall;
        const double matom = static_cast<double>(n_atoms) * steps_s / 1e6;
        // dt = 1 fs → ns/day
        const double ns_day = steps_s * 1e-6 * 86400.0;
        std::printf(
            "[atv] PERFORMANCE  wall=%.4f s  steps/s=%.3f  "
            "Matom-step/s=%.3f  ns/day=%.3f  ranks=%d\n",
            wall, steps_s, matom, ns_day, nrank);
        std::printf("[atv] E=%.6f pair=%.6f (eV)\n", frame.meta.get(kEnergy),
                    frame.meta.get(kEnergyPair));
        std::printf(
            "[atv] NOTE: Timing breakdown table above uses LAMMPS section "
            "names: Pair Bond Kspace Neigh Comm Output Modify Other\n");
    }

    MPI_Finalize();
    return 0;
}
