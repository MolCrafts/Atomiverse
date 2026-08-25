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
///   mpirun -np N run_atv_gaff --data system.data --ff system.ff [--npt]
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_coeffs.h"
#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

using namespace ATV;

namespace {

struct Args {
    std::string data = "/home/jicli594/work/molpack-paper/figures/relaxation/"
                       "ref_lammps/runs/packmol_n100_r2/melt/melt.data";
    std::string ff; // coefficient file; empty → built-in PEO tables
    int warmup = 50;
    int steps = 500;
    double cutoff = 10.0;       // overridden by --ff pair_style cutoff
    double accuracy = 1.0e-4;   // overridden by --ff kspace_style accuracy
    double alpha = -1.0;        // <0 → pppm_alpha_from_accuracy(accuracy)
    double grid_spacing = -1.0; // <0 → from_accuracy default (rcut/2.5)
    double T = 400.0;
    double tau_T = 100.0;
    bool npt = false;
    double P_atm = 1.0;
    double tau_P = 1000.0;
    bool single_point = false; // one force evaluation, print components, exit
    int tail_override = -1;    // -1 auto (from ff), 0 off, 1 on
    int shift_override = -1;   // -1 auto (!tail), 0 off, 1 on
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
        else if (!std::strcmp(argv[i], "--ff"))
            a.ff = next("--ff");
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
        else if (!std::strcmp(argv[i], "--alpha"))
            a.alpha = std::atof(next("--alpha"));
        else if (!std::strcmp(argv[i], "--T"))
            a.T = std::atof(next("--T"));
        else if (!std::strcmp(argv[i], "--tau-T"))
            a.tau_T = std::atof(next("--tau-T"));
        else if (!std::strcmp(argv[i], "--npt"))
            a.npt = true;
        else if (!std::strcmp(argv[i], "--P"))
            a.P_atm = std::atof(next("--P"));
        else if (!std::strcmp(argv[i], "--tau-P"))
            a.tau_P = std::atof(next("--tau-P"));
        else if (!std::strcmp(argv[i], "--single-point"))
            a.single_point = true;
        else if (!std::strcmp(argv[i], "--tail"))
            a.tail_override = std::atoi(next("--tail"));
        else if (!std::strcmp(argv[i], "--shift"))
            a.shift_override = std::atoi(next("--shift"));
        else if (!std::strcmp(argv[i], "--seed"))
            a.seed = std::atoi(next("--seed"));
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf(
                "run_atv_gaff --data PATH [--ff PATH] [--steps N] [--warmup N]"
                " [--cutoff 10] [--accuracy 1e-4] [--grid-spacing Å]"
                " [--T 400] [--tau-T 100] [--npt] [--P atm] [--tau-P 1000]"
                " [--single-point] [--seed N]\n");
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
            n_atoms, frame.box.length(0), args.cutoff, args.accuracy,
            args.steps, args.warmup);
        std::printf("[atv] data=%s\n", args.data.c_str());
    }

    using cpu::degree_to_rad;
    using cpu::kcal_per_mol_to_ev;

    // ── Coefficient tables: --ff file, or the built-in PEO GAFF2 tables ──
    std::vector<cpu::LJTypeCoeff> lj_coeffs;
    std::vector<double> Kbond, r0, Kang, th0;
    std::vector<double> Kd, dphase;
    std::vector<int> nper;
    std::vector<std::vector<double>> multi_Cn;
    bool use_charmm = true, use_multi = false;
    double sp_lj[3] = {0.0, 0.0, 0.5};
    double sp_coul[3] = {0.0, 0.0, 5.0 / 6.0};
    double cutoff = args.cutoff;
    double accuracy = args.accuracy;
    bool tail = false; // pair_modify tail yes → tail on, energy shift off

    if (!args.ff.empty()) {
        const cpu::LammpsCoeffs ff = cpu::load_lammps_coeffs(args.ff);
        if (ff.pair_style != "lj/cut/coul/long")
            throw std::runtime_error("run_atv_gaff: --ff pair_style must be "
                                     "lj/cut/coul/long, got '"
                                     + ff.pair_style + "'");
        lj_coeffs = ff.lj;
        Kbond = ff.bond_K;
        r0 = ff.bond_r0;
        Kang = ff.angle_K;
        th0 = ff.angle_theta0;
        Kd = ff.dihedral_charmm_K;
        nper = ff.dihedral_charmm_n;
        dphase = ff.dihedral_charmm_delta;
        multi_Cn = ff.dihedral_multi_Cn;
        use_charmm = ff.has_charmm;
        use_multi = ff.has_multi;
        if (ff.has_special) {
            std::copy(ff.special_lj, ff.special_lj + 3, sp_lj);
            std::copy(ff.special_coul, ff.special_coul + 3, sp_coul);
        }
        if (ff.pair_cut_lj > 0.0)
            cutoff = ff.pair_cut_lj;
        if (ff.kspace_accuracy > 0.0)
            accuracy = ff.kspace_accuracy;
        tail = ff.tail_correction;
        if (rank == 0)
            std::printf("[atv] ff=%s types: lj=%zu bond=%zu angle=%zu "
                        "dihedral=%zu (charmm=%d multi=%d) cutoff=%.2f "
                        "accuracy=%.1e tail=%d\n",
                        args.ff.c_str(), ff.lj.size(), ff.bond_K.size(),
                        ff.angle_K.size(), ff.dihedral_charmm_K.size(),
                        int(use_charmm), int(use_multi), cutoff, accuracy,
                        int(ff.tail_correction));
    } else {
        lj_coeffs = peo_lj(cutoff);
        // Bond types: c3-c3, c3-h1, c3-hc, c3-oh, ho-oh
        Kbond = {kcal_per_mol_to_ev(228.89), kcal_per_mol_to_ev(344.22),
                 kcal_per_mol_to_ev(345.25), kcal_per_mol_to_ev(285.15),
                 kcal_per_mol_to_ev(535.51)};
        r0 = {1.5354, 1.0969, 1.0962, 1.4242, 0.9725};
        // Angle types 1..9 from data labels
        Kang = {kcal_per_mol_to_ev(43.28), kcal_per_mol_to_ev(43.27),
                kcal_per_mol_to_ev(76.79), kcal_per_mol_to_ev(88.334),
                kcal_per_mol_to_ev(65.13), kcal_per_mol_to_ev(35.64),
                kcal_per_mol_to_ev(56.29), kcal_per_mol_to_ev(35.80),
                kcal_per_mol_to_ev(62.80)};
        th0 = {degree_to_rad(109.590047), degree_to_rad(109.680047),
               degree_to_rad(109.660047), degree_to_rad(113.600049),
               degree_to_rad(107.390046), degree_to_rad(108.550047),
               degree_to_rad(110.380047), degree_to_rad(107.730046),
               degree_to_rad(109.500047)};
        // Dihedral types (fourier single term → Charmm)
        Kd = {kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.060000),
              kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.123000),
              kcal_per_mol_to_ev(0.155556), kcal_per_mol_to_ev(0.090000),
              kcal_per_mol_to_ev(0.166667), kcal_per_mol_to_ev(0.250000),
              kcal_per_mol_to_ev(1.000000)};
        nper = {3, 3, 3, 3, 3, 3, 3, 1, 1};
        dphase.assign(9, 0.0);
    }

    const double alpha = args.alpha > 0.0
                             ? args.alpha
                             : cpu::pppm_alpha_from_accuracy(accuracy, cutoff);
    const double grid_sp =
        args.grid_spacing > 0.0 ? args.grid_spacing : (cutoff / 2.5);
    if (rank == 0)
        std::printf("[atv] alpha=%.6f (%s) grid_spacing=%.3f\n", alpha,
                    args.alpha > 0.0 ? "explicit" : "from accuracy", grid_sp);

    // Velocities: keep the data file's if present, else Maxwell-Boltzmann.
    bool have_velocities = false;
    {
        const StorageReal* vx = frame.span(kAtom_vx);
        const StorageReal* vy = frame.span(kAtom_vy);
        const StorageReal* vz = frame.span(kAtom_vz);
        for (int i = 0; i < n_atoms; ++i) {
            if (vx[i] != 0.0 || vy[i] != 0.0 || vz[i] != 0.0) {
                have_velocities = true;
                break;
            }
        }
    }

    // The barostat needs the pair/bonded virial; NVT benchmark skips it.
    const bool stress = args.npt;
    if (args.tail_override >= 0)
        tail = args.tail_override != 0;
    const bool shift =
        args.shift_override >= 0 ? args.shift_override != 0 : !tail;

    auto install = [&](auto& sim) {
        sim.set_evaluator(cpu::PairLJCutCoulLong{lj_coeffs, alpha, cutoff}
                              .set_special_bonds(sp_lj[0], sp_lj[1], sp_lj[2],
                                                 sp_coul[0], sp_coul[1],
                                                 sp_coul[2])
                              .set_tail_correction(tail)
                              .set_energy_shift(shift)
                              .set_compute_stress(stress));
        sim.set_evaluator(cpu::KSpacePPPM{alpha, grid_sp, /*order=*/5});
        sim.set_evaluator(cpu::BondHarmonic{Kbond, r0});
        sim.set_evaluator(cpu::AngleHarmonic{Kang, th0});
        if (use_charmm)
            sim.set_evaluator(cpu::DihedralCharmm(Kd, nper, dphase)
                                  .set_compute_stress(stress));
        if (use_multi)
            sim.set_evaluator(cpu::DihedralMultiHarmonic{multi_Cn});
    };

    // ── Single-point: one evaluation, component energies, exit ──────────
    if (args.single_point) {
        // Same skin the MD path hands NeighborList below; the geometry
        // report reuses it as the ghost-shell lower bound.
        const double skin = 2.0;
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
        install(eval);
        eval.build();
        eval.run(frame);
        // The component energies on the Frame are already GLOBAL:
        // `Evaluate::install_defaults` binds `md::mpi::Reduce` at
        // `evaluate @ +20`, which allreduces kEnergy / kEnergyPair /
        // kEnergyBond / kEnergyAngle / kEnergyDihedral (plus improper,
        // charging work, kinetic and both stress tensors) in one batched
        // call. Re-reducing them here would multiply every printed total by
        // the rank count.

        // Rank-invariant force scalars. Only OWNER rows are summed:
        // `partition_atoms` sets `frame.nlocal()` to this rank's
        // `DomainDecomposition::n_local`, and ghost rows live above it in
        // `[nlocal, nlocal + nghost)`. A ghost's force is a partial sum that
        // the reverse halo adds into its owner's row, so counting it here
        // would double it. `fmax` reduces with MAX over |f_i|² and takes one
        // sqrt at the end (the max of the squares is the square of the max);
        // `fnorm` reduces the sum of squares with SUM, then takes the sqrt.
        //
        // The reverse halo this relies on is auto-injected: `Evaluate` now
        // binds `comm::Migrate` / `comm::HaloSend` / `comm::HaloRecv` /
        // `md::mpi::Reduce` around the evaluate phase exactly as
        // `cpu::minimizer::Minimize` does. Before that fix this branch ran
        // with ZERO ghost atoms at np > 1 and every quantity here silently
        // lost its cross-rank part (MIL0PEGM18 24k: E = 1684.070 eV at np=1
        // vs 1241.720 eV at np=8); `widebox_parity.sh` is what reports it.
        {
            const AccReal* fx = frame.span(kAtom_fx);
            const AccReal* fy = frame.span(kAtom_fy);
            const AccReal* fz = frame.span(kAtom_fz);
            const int n_owned = frame.nlocal();
            double fmax2 = 0.0, fsum2 = 0.0;
            for (int i = 0; i < n_owned; ++i) {
                const double f2 = fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i];
                fsum2 += f2;
                fmax2 = std::max(fmax2, f2);
            }
            MPI_Allreduce(MPI_IN_PLACE, &fmax2, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &fsum2, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            if (rank == 0)
                std::printf("[atv] SINGLE-POINT FORCES: fmax=%.10e "
                            "fnorm=%.10e\n",
                            std::sqrt(fmax2), std::sqrt(fsum2));
        }

        // Processor grid geometry, from the same pure function the
        // DomainDecomposition itself calls — so this reports the grid that
        // was actually used, `ATV_FORCE_DIMS` override included.
        //
        // `halo` here is the GEOMETRIC LOWER BOUND on the ghost shell:
        // `cutoff + skin` only. Bonded reach (bond/angle/dihedral spans) is
        // added engine-side on top of it, so a printed `hops=2` means "at
        // least 2 hops" — never fewer.
        if (rank == 0) {
            int dims[3] = {1, 1, 1};
            cpu::md::mpi::DomainDecomposition::choose_dims(
                nrank, frame.box.length(0), frame.box.length(1),
                frame.box.length(2), dims);
            const double halo = cutoff + skin;
            double span[3];
            int hops[3];
            for (int d = 0; d < 3; ++d) {
                span[d] = frame.box.length(d) / dims[d];
                hops[d] = dims[d] == 1
                              ? 0
                              : static_cast<int>(std::ceil(halo / span[d]));
            }
            std::printf("[atv] GRID dims=%d,%d,%d span=%.6f,%.6f,%.6f "
                        "halo=%.6f hops=%d,%d,%d\n",
                        dims[0], dims[1], dims[2], span[0], span[1], span[2],
                        halo, hops[0], hops[1], hops[2]);
        }

        if (rank == 0) {
            std::printf("[atv] SINGLE-POINT (eV): E=%.8f pair+kspace=%.8f "
                        "bond=%.8f angle=%.8f dihedral=%.8f\n",
                        frame.meta.get(kEnergy), frame.meta.get(kEnergyPair),
                        frame.meta.get(kEnergyBond),
                        frame.meta.get(kEnergyAngle),
                        frame.meta.get(kEnergyDihedral));
            const double kcal = 1.0 / ATV::cpu::kKcalPerMolInEv;
            std::printf("[atv] SINGLE-POINT (kcal/mol): E=%.6f "
                        "pair+kspace=%.6f bond=%.6f angle=%.6f "
                        "dihedral=%.6f\n",
                        frame.meta.get(kEnergy) * kcal,
                        frame.meta.get(kEnergyPair) * kcal,
                        frame.meta.get(kEnergyBond) * kcal,
                        frame.meta.get(kEnergyAngle) * kcal,
                        frame.meta.get(kEnergyDihedral) * kcal);
        }
        MPI_Finalize();
        return 0;
    }

    cpu::md::MD sim;
    // Match paper in.workflow: neighbor 2.0 bin; neigh_modify delay 0 every 1
    // check yes. atom_modify sort: LAMMPS default nfreq=1000 (paper does not
    // set sort 1); every-reneighbor sort thrash-rebuilds BondedTupleCache.
    sim.set_neighbor(
        cpu::md::NeighborList<>{cutoff, /*skin=*/2.0, /*every=*/1,
                                /*delay=*/0, /*check=*/true}
            .set_atom_sort({.nfreq = 1000, .binsize = 0.0, .enabled = true}));
    install(sim);
    if (args.npt) {
        // 1 atm = 6.324209e-7 eV/Å³ (LAMMPS units real pressure is atm)
        const double P_evA3 = args.P_atm * 6.3242090e-7;
        sim.set_integrator(cpu::md::NoseHooverNPT{args.T, args.tau_T, P_evA3,
                                                  args.tau_P, 1.0});
    } else {
        sim.set_integrator(cpu::md::VelocityVerlet{1.0});
        sim.set_thermostat(cpu::md::NoseHoover{args.T, args.tau_T, 1.0});
    }
    if (!have_velocities)
        sim.set_observer(
            cpu::md::InitVelocity{args.T, static_cast<unsigned>(args.seed)});
    else if (rank == 0)
        std::printf("[atv] using velocities from data file\n");
    sim.build();

    // Warmup (not timed in our wall clock; still contributes to driver summary)
    if (args.warmup > 0)
        sim.run(frame, args.warmup);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::steady_clock::now();
    sim.run(frame, args.steps);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        const double steps_s = args.steps / wall;
        const double matom = static_cast<double>(n_atoms) * steps_s / 1e6;
        // dt = 1 fs → ns/day
        const double ns_day = steps_s * 1e-6 * 86400.0;
        std::printf("[atv] PERFORMANCE  wall=%.4f s  steps/s=%.3f  "
                    "Matom-step/s=%.3f  ns/day=%.3f  ranks=%d\n",
                    wall, steps_s, matom, ns_day, nrank);
        std::printf("[atv] E=%.6f pair=%.6f (eV)\n", frame.meta.get(kEnergy),
                    frame.meta.get(kEnergyPair));
        std::printf(
            "[atv] NOTE: Timing breakdown table above uses LAMMPS section "
            "names: Pair Bond Kspace Neigh Comm Output Modify Other\n");
    }
    std::fflush(stdout);

    MPI_Finalize();
    return 0;
}
