/// @file tools/debug_amorphous/run_atomiverse.cpp
/// @brief Reproduces the LAMMPS `amorphous.in` Stage 1 + Stage 2 runs on
///        the Atomiverse CPU + MPI backend, with thermo output in the
///        same cadence as LAMMPS so the two logs can be diffed line by line.
///
/// LAMMPS protocol (see tools/debug_amorphous/amorphous.in):
///   Stage 1: NVE/limit 0.05 + Langevin (T=500K), dt=1 fs, 10 000 steps.
///   Stage 2: NPT       (T=500K, P=0, drag=2),    dt=0.5 fs, 50 000 steps.
///
/// Atomiverse mapping (LAMMPS-equivalent algorithms in the source tree):
///   Stage 1: cpu::md::NVELimit     (dt=1 fs, xmax=0.05 Å)    // == fix nve/limit
///            cpu::md::FixLangevin  (T=500K, damp=10 fs)      // == fix langevin
///   Stage 2: cpu::md::NoseHooverNPT(T=500K, τ_T=50 fs,       // == fix npt iso drag 2
///                                   P=0,    τ_P=1000 fs,
///                                   drag=2, mtchain=3,       //    (LAMMPS defaults)
///                                   mpchain=3, dt=0.5 fs)
///            cpu::md::ZeroMomentum{1}                        // == fix momentum 1 linear 1 1 1
///
/// The point is to compare temperature / pressure trajectories on
/// **identical input atoms, force field, box, cutoff, integrator, and
/// thermostat/barostat algorithm** — only the random number stream
/// differs (Atomiverse uses a per-(seed,step,tag) splitmix; LAMMPS uses
/// RanMars). Ensemble statistics ⟨P⟩, ⟨T⟩ should match.

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "atomiverse/core/units.h"

#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include "gather_frame.h"
#include "lammps_loader.h"

using namespace ATV;

namespace {

// ── Unit conversions: LAMMPS "real" → Atomiverse internal ──────────────
//
// kcal/mol → eV:
inline constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
// degrees → rad:
inline double deg2rad(double d) { return d * units::pi / 180.0; }
// LAMMPS pressure unit is atmosphere; Atomiverse internal stress is
// eV / Å³ . 1 atm ≈ 1.013 25e5 Pa = 1.013 25e5 J / m³ → 6.324209e-7
// eV / Å³  (Pa × m³/eV·Å³).  For pressure printing we go the other way:
// Atomiverse pressure [eV/Å³] → LAMMPS "atm" so the log is comparable.
inline constexpr double kEvPerA3ToAtm = 1.0 / 6.32420911905e-7;

// 1 amu × Å⁻³ → g·cm⁻³. Comes from 1 amu = 1.660 539 067e-24 g and
// 1 Å³ = 1e-24 cm³, so the 1e-24 factors cancel.
inline constexpr double kAmuPerA3ToGPerCc = 1.660'539'066'60;

// ── Async observer: read globally-reduced energy/stress/KE on rank 0
//    every `period` steps and print a LAMMPS-style thermo line. -----
//
// Bound at end_of_step at order >= +20 so it runs after ReduceEnergy (-50)
// and after Barostat (-45). Reads typed Frame energy/kinetic/stress metadata
// — all already globally reduced.
//
// Columns match LAMMPS `thermo_style custom step temp press pe ke etotal
// density vol lx ly lz` so the two logs can be diffed line by line
// (LAMMPS pe/ke/etotal are in kcal/mol and we convert to eV in
// compare.py). ThermoPrinter prints energies in eV.
class ThermoPrinter : public cpu::Component {
  public:
    ThermoPrinter(int period, std::ostream& out, int n_atoms_global,
                  double total_mass_amu)
        : period_(period), out_(&out), n_global_(n_atoms_global),
          total_mass_amu_(total_mass_amu) {}

    std::string name() const override { return "thermo_printer"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +20)};
    }

    void launch(const cpu::LaunchContext& ctx) override {
        if (ctx.runtime.mpi_rank != 0) {
            ++step_;
            return;
        }
        const auto& f = ctx.frame;

        if (step_ % period_ == 0) {
            const double Lx = f.box.length(0);
            const double Ly = f.box.length(1);
            const double Lz = f.box.length(2);
            const double V = Lx * Ly * Lz;
            const double kinetic = f.meta.get(kKinetic);
            const double energy = f.meta.get(kEnergy);
            const auto& stress = f.meta.get(kStress);
            const double T =
                (n_global_ > 1)
                    ? (2.0 * kinetic)
                          / (3.0 * (n_global_ - 1) * units::kB_ev_per_k)
                    : 0.0;
            // Atomiverse `frame.stress` stores the LAMMPS-style virial
            // directly (virial[0] = Σ r_ij,x · F_ij,x accumulated with
            // compression-positive sign), so the physical pressure is
            // just `(kin + pair virial)/(3V)` with NO sign flip. Sign
            // must match LAMMPS log (+ = compression).
            const double P_evA3 =
                +(stress[0] + stress[1] + stress[2]) / (3.0 * V);
            const double P_atm = P_evA3 * kEvPerA3ToAtm;

            const double density =
                (V > 0.0) ? total_mass_amu_ * kAmuPerA3ToGPerCc / V : 0.0;
            const double etotal = energy + kinetic;

            (*out_) << std::setw(10) << step_ << ' ' << std::setw(14)
                    << std::setprecision(8) << T << ' ' << std::setw(14)
                    << std::setprecision(8) << P_atm << ' ' << std::setw(14)
                    << std::setprecision(8) << energy << ' ' << std::setw(14)
                    << std::setprecision(8) << kinetic << ' ' << std::setw(14)
                    << std::setprecision(8) << etotal << ' ' << std::setw(12)
                    << std::setprecision(6) << density << ' ' << std::setw(14)
                    << std::setprecision(8) << V << ' ' << std::setw(10)
                    << std::setprecision(7) << Lx << ' ' << std::setw(10)
                    << std::setprecision(7) << Ly << ' ' << std::setw(10)
                    << std::setprecision(7) << Lz << '\n';
            out_->flush();
        }
        ++step_;
    }

    void reset_step() { step_ = 0; }

  private:
    int period_;
    std::ostream* out_;
    int n_global_;
    double total_mass_amu_;
    long long step_ = 0;
};

void print_thermo_header(std::ostream& out, const std::string& tag) {
    out << "# " << tag << '\n';
    out << "#       Step           Temp     Press[atm]       "
           "PotE[eV]       KinE[eV]     EtotE[eV]    Dens[g/cc]      "
           "Vol[A^3]        Lx        Ly        Lz\n";
}

} // namespace

// ── main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string data_path = "/Users/roykid/work/molcrafts/Atomiverse/"
                            "tools/debug_amorphous/PE_nc10_cl1000.dat";
    std::string log_path = "/Users/roykid/work/molcrafts/Atomiverse/"
                           "tools/debug_amorphous/log.atomiverse.txt";
    int n_steps_stage1 = 10000;
    int n_steps_stage2 = 50000;
    bool short_run = false;
    bool run_stage1 = true;
    bool run_stage2 = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--data" && i + 1 < argc)
            data_path = argv[++i];
        else if (a == "--log" && i + 1 < argc)
            log_path = argv[++i];
        else if (a == "--steps-stage1" && i + 1 < argc)
            n_steps_stage1 = std::stoi(argv[++i]);
        else if (a == "--steps-stage2" && i + 1 < argc)
            n_steps_stage2 = std::stoi(argv[++i]);
        else if (a == "--stage1-only")
            run_stage2 = false;
        else if (a == "--stage2-only")
            run_stage1 = false;
        else if (a == "--short") {
            short_run = true;
            n_steps_stage1 = 1000;
            n_steps_stage2 = 2000;
        }
    }

    if (rank == 0) {
        std::cerr << "[atomiverse] world_size = " << world_size << '\n';
        std::cerr << "[atomiverse] data_path  = " << data_path << '\n';
        std::cerr << "[atomiverse] log_path   = " << log_path << '\n';
        std::cerr << "[atomiverse] stage 1    = " << n_steps_stage1
                  << " steps × 1.0 fs\n";
        std::cerr << "[atomiverse] stage 2    = " << n_steps_stage2
                  << " steps × 0.5 fs\n";
        if (short_run)
            std::cerr << "[atomiverse] (--short)\n";
    }

    // ── Force field (LAMMPS amorphous.in, kcal/mol → eV) ────────────
    const double bond_K_kcal = 350.0;
    const double bond_r0 = 1.53;
    const double angle_K_kcal = 60.0;
    const double angle_theta0 = deg2rad(109.5);
    const std::vector<double> dihedral_kcal = {1.73, -4.49, 0.776, 6.99, 0.0};

    const double bond_K_ev = bond_K_kcal * kKcalPerMolToEv;
    const double angle_K_ev = angle_K_kcal * kKcalPerMolToEv;
    std::vector<double> dihedral_ev;
    dihedral_ev.reserve(dihedral_kcal.size());
    for (double v : dihedral_kcal)
        dihedral_ev.push_back(v * kKcalPerMolToEv);

    const double lj_eps_kcal = 0.112;
    const double lj_sig = 4.01;
    const double lj_cut = 10.5;
    const double lj_eps_ev = lj_eps_kcal * kKcalPerMolToEv;

    // ── Load the LAMMPS data file (every rank reads the full file
    //    — DomainDecomposition partitions atoms in Backend::setup) ───
    cpu::Frame frame = debug_amorphous::load_lammps_data(data_path);

    if (rank == 0) {
        std::cerr << "[atomiverse] n_atoms     = " << frame.nlocal() << '\n';
        // Per-atom topology counts (the flat sub-structs are deleted).
        long nb = 0, na = 0, nd = 0;
        const auto num_bond = frame.span(kAtom_num_bond);
        const auto num_angle = frame.span(kAtom_num_angle);
        const auto num_dihedral = frame.span(kAtom_num_dihedral);
        for (int i = 0; i < frame.nlocal(); ++i) {
            nb += num_bond[i];
            na += num_angle[i];
            nd += num_dihedral[i];
        }
        std::cerr << "[atomiverse] n_bonds     = " << nb << '\n';
        std::cerr << "[atomiverse] n_angles    = " << na << '\n';
        std::cerr << "[atomiverse] n_dihedrals = " << nd << '\n';
        std::cerr << "[atomiverse] box (Å)     = " << frame.box.length(0) << ' '
                  << frame.box.length(1) << ' ' << frame.box.length(2) << '\n';
    }

    const int n_atoms_global = frame.nlocal();

    // Total mass snapshot — taken before Backend::set_frame partitions
    // the atoms, so every rank agrees on the global system mass. Used
    // by ThermoPrinter to print density in g/cc without any inter-stage
    // MPI communication.
    double total_mass_amu = 0.0;
    const auto mass = frame.span(kAtom_mass);
    for (int i = 0; i < frame.nlocal(); ++i) {
        total_mass_amu += mass[i];
    }
    if (rank == 0) {
        std::cerr << "[atomiverse] total_mass = " << total_mass_amu << " amu\n";
    }

    int stage1_n_local = frame.nlocal(); // assume all-on-rank-0 default

    std::ofstream log_stream;
    if (rank == 0) {
        log_stream.open(log_path);
        if (!log_stream)
            std::cerr << "[atomiverse] WARNING: cannot open " << log_path
                      << " for writing\n";
    }

    // ── Stage 1: nve/limit + Langevin at 500 K, dt = 1 fs ──────────
    if (run_stage1 && n_steps_stage1 > 0) {
        const double dt = 1.0;
        const double T_tgt = 500.0;
        const double tau_lang = 10.0; // LAMMPS-equivalent damp
        const double xmax = 0.05;     // LAMMPS-equivalent cap

        cpu::md::MD sim;
        sim.set_integrator(cpu::md::NVELimit{dt, xmax});
        sim.set_observer(cpu::md::InitVelocity{T_tgt, /*seed=*/1231});
        sim.set_thermostat(cpu::md::FixLangevin{T_tgt, /*T_stop=*/T_tgt,
                                                tau_lang, dt,
                                                /*seed=*/904297});
        sim.set_neighbor(
            cpu::md::NeighborList<>{lj_cut, /*skin=*/0.4, /*every=*/10});
        sim.on(ATV::md::Phase::force)
            .add(cpu::PairLJCut{lj_eps_ev, lj_sig, lj_cut}.set_special_bonds(
                0.0, 0.0, 0.0))
            .add(cpu::BondHarmonic{{bond_K_ev}, {bond_r0}})
            .add(cpu::AngleHarmonic{{angle_K_ev}, {angle_theta0}})
            .add(cpu::DihedralMultiHarmonic{{dihedral_ev}});
        sim.set_observer(cpu::md::Virial{});

        ThermoPrinter thermo{100, rank == 0 ? log_stream : std::cout,
                             n_atoms_global, total_mass_amu};
        sim.set_observer_ref(thermo);

        // Stash this rank's final n_local so we can gather the frame
        // back to every rank between stages (Stage 2's driver scans the
        // whole array on first partition; without a gather it'd see
        // stale ghost rows).
        debug_amorphous::CaptureNLocal capture{&stage1_n_local};
        sim.set_observer_ref(capture);

        sim.build();

        if (rank == 0) {
            print_thermo_header(
                log_stream,
                "Stage 1: NVE/limit + Langevin (T=500 K, damp=10 fs), dt=1 fs, "
                    + std::to_string(n_steps_stage1) + " steps");
        }

        const auto t0 = std::chrono::steady_clock::now();
        sim.run(frame, n_steps_stage1);
        const auto t1 = std::chrono::steady_clock::now();

        if (rank == 0) {
            const double secs = std::chrono::duration<double>(t1 - t0).count();
            std::cerr << "[atomiverse] stage 1 wall time : " << secs << " s ("
                      << (n_steps_stage1 / secs) << " steps/s)\n";
            log_stream << "# stage1 wall_seconds = " << secs << '\n';
        }
    }

    // ── Inter-stage gather: replicate the full frame on every rank ──
    if (run_stage1 && n_steps_stage1 > 0 && run_stage2 && world_size > 1) {
        debug_amorphous::gather_full_frame_inplace(frame, stage1_n_local,
                                                   MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "[atomiverse] inter-stage gather complete "
                      << "(n_local on rank 0 was " << stage1_n_local << ")\n";
        }
    }

    // ── Stage 2: NoseHooverNPT (MTK-iso with M-chain NHC) + ZeroMomentum.
    //
    // Mirrors LAMMPS `fix npt iso 500 500 50 iso 0 0 1000 drag 2` +
    // `fix momentum 1 linear 1 1 1`. Exact algorithmic match: Martyna-
    // Tobias-Klein isotropic NPT, M-chain Nose-Hoover for both the atom
    // and barostat thermostat (LAMMPS default `mtchain=3`, `mpchain=3`),
    // empirical drag attenuation on the innermost chain momentum
    // (LAMMPS `drag 2`). Strict rule: the reference and our run use the
    // same thermostat family (NH, not Langevin) and the same barostat
    // family (MTK, not Berendsen), so the two trajectories differ only
    // in the RNG-free deterministic evolution from the same starting
    // conditions — ⟨T⟩, ⟨P⟩, ⟨ρ⟩ should match pointwise after the
    // common equilibration transient, not just in ensemble average.
    if (run_stage2) {
        const double dt = 0.5;
        const double T_tgt = 500.0;
        const double tau_T = 50.0; // LAMMPS Tdamp
        const double P_tgt = 0.0;
        const double tau_P = 1000.0; // LAMMPS Pdamp

        cpu::md::MD sim;
        sim.set_integrator(cpu::md::NoseHooverNPT{
            */ T_tgt, */ tau_T, */ P_tgt, */ tau_P, */ dt, */ 2.0, */ 3, */ 3});
        // LAMMPS `fix momentum 1 linear 1 1 1` equivalent.
        sim.set_observer(cpu::md::ZeroMomentum{1});
        sim.set_neighbor(
            cpu::md::NeighborList<>{lj_cut, /*skin=*/0.4, /*every=*/10});
        sim.on(ATV::md::Phase::force)
            .add(cpu::PairLJCut{lj_eps_ev, lj_sig, lj_cut}.set_special_bonds(
                0.0, 0.0, 0.0))
            .add(cpu::BondHarmonic{{bond_K_ev}, {bond_r0}})
            .add(cpu::AngleHarmonic{{angle_K_ev}, {angle_theta0}})
            .add(cpu::DihedralMultiHarmonic{{dihedral_ev}});
        sim.set_observer(cpu::md::Virial{});

        ThermoPrinter thermo{100, rank == 0 ? log_stream : std::cout,
                             n_atoms_global, total_mass_amu};
        thermo.reset_step();
        sim.set_observer_ref(thermo);

        sim.build();

        if (rank == 0) {
            print_thermo_header(
                log_stream, "Stage 2: NoseHooverNPT (T=500 K, tau_T=50 fs, "
                            "P=0, tau_P=1 ps, drag=2, M=3), dt=0.5 fs, "
                                + std::to_string(n_steps_stage2) + " steps");
        }

        const auto t0 = std::chrono::steady_clock::now();
        sim.run(frame, n_steps_stage2);
        const auto t1 = std::chrono::steady_clock::now();

        if (rank == 0) {
            const double secs = std::chrono::duration<double>(t1 - t0).count();
            std::cerr << "[atomiverse] stage 2 wall time : " << secs << " s ("
                      << (n_steps_stage2 / secs) << " steps/s)\n";
            log_stream << "# stage2 wall_seconds = " << secs << '\n';
        }
    }

    if (rank == 0)
        log_stream.close();
    MPI_Finalize();
    return 0;
}
