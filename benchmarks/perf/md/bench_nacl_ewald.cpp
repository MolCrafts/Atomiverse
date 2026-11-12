/// @file benchmarks/perf/md/bench_nacl_ewald.cpp
/// @brief Public-API CPU timing of a small, dense Coulomb system: 512-ion
///        molten NaCl (Petravic & Delhommelle, J. Chem. Phys. 118, 7477
///        (2003)) under `PairBorn` (Tosi–Fumi) + `CoulLong` + `KSpaceEwald`,
///        integrated by `GearPC` with the Gauss isokinetic or the
///        configurational thermostat. Never registered with ctest.
///
/// Usage:
///   bench_nacl_ewald STATE SETTING THREADS STEPS WARM
///                    [--thermostat gauss|config] [--lammps PREFIX]
///
///  - STATE   `liquid` (1500 K, L = 25.4904 Å, 1 fs) or `fluid` (3200 K,
///            L = 49.8957 Å, 0.5 fs): the paper's two state points.
///  - SETTING `paper`: κ = 1.8π/L, Born and real-space cutoff L/2, reciprocal
///            sphere |k| ≤ 6·2π/L (accuracy e^{−100/9}), skin 2 Å.
///            `accuracy`: cutoff 8 Å, α = 3/8 Å⁻¹, accuracy 1e-6, skin
///            0.5 Å (the reciprocal sphere then grows with the volume).
///  - THREADS `Driver::set_threads`; 0 keeps the OpenMP default.
///  - `--thermostat` `gauss` (default): Gauss isokinetic (Petravic eq. 9).
///            `config`: configurational Nosé–Hoover (eqs. 12a–c, Q = 100
///            Å²/ps²), whose T_C1 needs every force term's curvature. It
///            needs ∇²U > 0, which the lattice start has in `liquid` only:
///            the expanded `fluid` lattice has ∇²U < 0 and GearPC refuses
///            to start.
///  - `--lammps PREFIX` writes PREFIX.data (the initial state) and
///            PREFIX.lammps (the same model and Ewald parameters as LAMMPS
///            commands) for `benchmarks/compare/nacl-ewald-lammps`.
///
/// Prints ms/step per component (timing registry, after WARM untimed
/// steps) and one machine-readable `BENCH` line. `thermostat` is the Gauss
/// multiplier and projection, or the curvature evaluation (T_C1 and its
/// gradient) of the configurational thermostat, both inside `gear_pc`;
/// `integrate` is the rest of `gear_pc`.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/timer.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/longrange/ewald_sum.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_born.h"

namespace {

using namespace ATV;

constexpr int kSide = 8; // 8³ simple-cubic sites with alternating charge
constexpr int kN = kSide * kSide * kSide;
constexpr double kMass[2] = {22.9898, 35.453}; // Na, Cl [g/mol]
constexpr double kCharge[2] = {1.0, -1.0};

/// Tosi–Fumi NaCl in eV/Å (`PairBorn` adds +D/r⁸, so D carries the minus
/// sign of the published −D/r⁸): the published table, as a LAMMPS
/// `born/coul/long` pair_coeff set uses it.
std::vector<cpu::BornTypeCoeff> tosi_fumi() {
    return {{0, 0, 0.263705848, 0.317, 2.340, 1.048583006, -0.49932529},
            {0, 1, 0.210964679, 0.317, 2.755, 6.99055303, -8.675775756},
            {1, 1, 0.158223509, 0.317, 3.170, 72.40215779, -145.4284714}};
}

struct StatePoint {
    std::string name;
    double length;      ///< cubic edge [Å]
    double temperature; ///< [K]
    double dt;          ///< [ps]
};

struct EwaldSetting {
    std::string name;
    double cutoff; ///< Born and real-space Coulomb cutoff [Å]
    double alpha;  ///< Ewald splitting [1/Å]
    double accuracy;
    double skin; ///< neighbor skin [Å]
};

StatePoint state_point(const std::string& name) {
    if (name == "liquid")
        return {name, 25.4904, 1500.0, 0.001};
    if (name == "fluid")
        return {name, 49.8957, 3200.0, 0.0005};
    throw std::invalid_argument("STATE must be liquid or fluid");
}

EwaldSetting ewald_setting(const std::string& name, double length) {
    if (name == "paper")
        // κ = 1.8π/L and |k| ≤ 6·2π/L: 2κ·sqrt(−ln δ) = 12π/L ⇔ δ = e^{−100/9}
        // (the repro decision rounds it to 1.4947e-5).
        return {name, 0.5 * length, 1.8 * units::pi / length, 1.4947e-5, 2.0};
    if (name == "accuracy")
        return {name, 8.0, 3.0 / 8.0, 1e-6, 0.5};
    throw std::invalid_argument("SETTING must be paper or accuracy");
}

/// Rock salt (8³ sites, alternating charge), Maxwell velocities at T with
/// zero total momentum.
cpu::Frame rock_salt(const StatePoint& sp, unsigned seed) {
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    const double spacing = sp.length / kSide;
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    cpu::Frame f(kN);
    double p[3] = {0, 0, 0}, mass_sum = 0;
    int row = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k, ++row) {
                const int type = (i + j + k) % 2;
                const double m = kMass[type];
                const double sd =
                    std::sqrt(metal.boltz * sp.temperature / (m * metal.mvv2e));
                f.span(kAtom_x)[row] = (i + 0.25) * spacing;
                f.span(kAtom_y)[row] = (j + 0.25) * spacing;
                f.span(kAtom_z)[row] = (k + 0.25) * spacing;
                f.span(kAtom_vx)[row] = sd * gauss(rng);
                f.span(kAtom_vy)[row] = sd * gauss(rng);
                f.span(kAtom_vz)[row] = sd * gauss(rng);
                f.span(kAtom_charge)[row] = kCharge[type];
                f.span(kAtom_mass)[row] = m;
                f.span(kAtom_type)[row] = type;
                f.span(kAtom_tag)[row] = row + 1;
                p[0] += m * f.span(kAtom_vx)[row];
                p[1] += m * f.span(kAtom_vy)[row];
                p[2] += m * f.span(kAtom_vz)[row];
                mass_sum += m;
            }
    for (int i = 0; i < kN; ++i) {
        f.span(kAtom_vx)[i] -= p[0] / mass_sum;
        f.span(kAtom_vy)[i] -= p[1] / mass_sum;
        f.span(kAtom_vz)[i] -= p[2] / mass_sum;
    }
    f.box = cpu::Box::cubic(sp.length);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

/// Reciprocal vectors `KSpaceEwald` keeps for this cell and setting, and the
/// radius of its sphere in units of 2π/L.
std::size_t k_vector_count(const StatePoint& sp, const EwaldSetting& es) {
    longrange::EwaldSum sum(es.alpha, es.accuracy);
    sum.rebuild(cpu::Box::cubic(sp.length).inv_cell());
    return sum.size();
}

double k_sphere_radius(const StatePoint& sp, const EwaldSetting& es) {
    const double gcut = 2.0 * es.alpha * std::sqrt(-std::log(es.accuracy));
    return gcut * sp.length / (2.0 * units::pi);
}

/// The initial state and the same model as LAMMPS input, for the compare
/// script. LAMMPS can only cut a cubic cell's sphere at an integer radius in
/// units of 2π/L (`kmax/ewald`), boundary shell included, so the radius is
/// rounded to the nearest integer (`paper`: 5.99997 → 6); the script reports
/// both k-vector counts.
void write_lammps(const std::string& prefix, const cpu::Frame& f,
                  const StatePoint& sp, const EwaldSetting& es) {
    std::ofstream data(prefix + ".data");
    data.exceptions(std::ios::badbit | std::ios::failbit);
    data << std::setprecision(17)
         << "bench_nacl_ewald initial state; metal units\n\n"
         << kN << " atoms\n2 atom types\n\n0 " << sp.length << " xlo xhi\n0 "
         << sp.length << " ylo yhi\n0 " << sp.length
         << " zlo zhi\n\nMasses\n\n1 " << kMass[0] << "\n2 " << kMass[1]
         << "\n\nAtoms # charge\n\n";
    for (int i = 0; i < kN; ++i)
        data << f.span(kAtom_tag)[i] << ' ' << f.span(kAtom_type)[i] + 1 << ' '
             << f.span(kAtom_charge)[i] << ' ' << f.span(kAtom_x)[i] << ' '
             << f.span(kAtom_y)[i] << ' ' << f.span(kAtom_z)[i] << '\n';
    data << "\nVelocities\n\n";
    for (int i = 0; i < kN; ++i)
        data << f.span(kAtom_tag)[i] << ' ' << f.span(kAtom_vx)[i] << ' '
             << f.span(kAtom_vy)[i] << ' ' << f.span(kAtom_vz)[i] << '\n';

    std::ofstream in(prefix + ".lammps");
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in << std::setprecision(17) << "pair_style born/coul/long " << es.cutoff
       << '\n';
    for (const auto& c : tosi_fumi())
        in << "pair_coeff " << c.type_i + 1 << ' ' << c.type_j + 1 << ' ' << c.A
           << ' ' << c.rho << ' ' << c.sigma << ' ' << c.C << ' ' << c.D
           << '\n';
    const long kmax = std::lround(k_sphere_radius(sp, es));
    in << "pair_modify shift yes\n"
       << "kspace_style ewald " << es.accuracy << '\n'
       << "kspace_modify gewald " << es.alpha << " kmax/ewald " << kmax << ' '
       << kmax << ' ' << kmax << '\n'
       << "neighbor " << es.skin << " bin\n"
       << "neigh_modify every 1 delay 0 check yes\n"
       << "timestep " << sp.dt << '\n'
       << "variable T equal " << sp.temperature << '\n';
}

/// Component wall [s] split into the report's buckets.
struct Buckets {
    double pair = 0, kspace = 0, integrate = 0, thermostat = 0, neigh = 0,
           other = 0;
};

Buckets bucket(const TimerRegistry& reg) {
    Buckets b;
    std::map<std::string, double> unknown;
    for (std::size_t i = 0; i < reg.keys().size(); ++i) {
        const std::string& name = reg.keys()[i].component;
        const double s = reg.per_component()[i].total_s;
        if (name == "born" || name == "coul/long")
            b.pair += s;
        else if (name == "ewald")
            b.kspace += s;
        else if (name == "gear_pc")
            b.integrate += s;
        else if (name == "neighbor_list")
            b.neigh += s;
        else
            b.other += s;
    }
    // The thermostat runs inside gear_pc's launches.
    for (const char* key : {"gear_pc/gauss", "gear_pc/curvature"}) {
        const auto it = reg.scoped().find(key);
        if (it != reg.scoped().end())
            b.thermostat += it->second.total_s;
    }
    b.integrate -= b.thermostat;
    return b;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const char* usage =
            "usage: bench_nacl_ewald liquid|fluid paper|accuracy THREADS "
            "STEPS WARM [--thermostat gauss|config] [--lammps PREFIX]";
        if (argc < 6)
            throw std::invalid_argument(usage);
        std::string thermostat = "gauss", prefix;
        for (int a = 6; a < argc; a += 2) {
            const std::string flag = argv[a];
            if (a + 1 >= argc)
                throw std::invalid_argument(usage);
            if (flag == "--thermostat")
                thermostat = argv[a + 1];
            else if (flag == "--lammps")
                prefix = argv[a + 1];
            else
                throw std::invalid_argument(usage);
        }
        if (thermostat != "gauss" && thermostat != "config")
            throw std::invalid_argument("--thermostat must be gauss or config");
        const StatePoint sp = state_point(argv[1]);
        const EwaldSetting es = ewald_setting(argv[2], sp.length);
        const int threads = std::stoi(argv[3]);
        const int steps = std::stoi(argv[4]);
        const int warm = std::stoi(argv[5]);
        if (threads < 0 || steps < 1 || warm < 0)
            throw std::invalid_argument(
                "require THREADS >= 0, STEPS >= 1, WARM >= 0");

        cpu::Frame frame = rock_salt(sp, 20031);
        if (!prefix.empty())
            write_lammps(prefix, frame, sp, es);

        cpu::md::MD sim;
        sim.set_units("metal");
        sim.set_neighbor(
            cpu::md::NeighborList<>{es.cutoff, es.skin, 1, 0, true});
        sim.set_evaluator(cpu::PairBorn{tosi_fumi(), es.cutoff});
        sim.set_evaluator(cpu::CoulLong{es.alpha, es.cutoff});
        sim.set_evaluator(cpu::KSpaceEwald{es.alpha, es.accuracy});
        if (thermostat == "gauss")
            sim.set_integrator(cpu::md::GearPC{
                sp.dt, cpu::md::GaussThermostat{cpu::md::GaussKind::Kinetic,
                                                sp.temperature, 0}});
        else
            sim.set_integrator(cpu::md::GearPC{
                sp.dt,
                cpu::md::ConfigurationalThermostat{sp.temperature, 100.0}});
        if (threads > 0)
            sim.set_threads(threads);
        sim.enable_timing(true);
        sim.build();
        sim.run(frame, warm);
        sim.reset_timing();
        const auto t0 = std::chrono::steady_clock::now();
        sim.run(frame, steps);
        const double wall_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();

        const TimerRegistry& reg = *sim.timing();
        const Buckets b = bucket(reg);
        const double to_ms = 1e3 / steps;
        const double loop_ms = reg.loop_seconds() * to_ms;
        std::printf("bench_nacl_ewald: %d ions, state=%s L=%.4f A T=%.0f K "
                    "dt=%g fs, thermostat=%s, setting=%s rc=%.4f A "
                    "alpha=%.6f 1/A "
                    "accuracy=%.4g skin=%.1f A, k-vectors=%zu (|k| <= %.3f "
                    "2pi/L), threads=%d, %d warm + %d timed steps\n",
                    kN, sp.name.c_str(), sp.length, sp.temperature, sp.dt * 1e3,
                    thermostat.c_str(), es.name.c_str(), es.cutoff, es.alpha,
                    es.accuracy, es.skin, k_vector_count(sp, es),
                    k_sphere_radius(sp, es), threads, warm, steps);
        std::printf("  %-12s %10s\n", "component", "ms/step");
        const std::pair<const char*, double> rows[] = {
            {"pair", b.pair},           {"kspace", b.kspace},
            {"integrate", b.integrate}, {"thermostat", b.thermostat},
            {"neighbor", b.neigh},      {"other", b.other}};
        for (const auto& [label, s] : rows)
            std::printf("  %-12s %10.4f\n", label, s * to_ms);
        std::printf("  %-12s %10.4f\n  %-12s %10.4f\n", "loop", loop_ms, "wall",
                    wall_s * to_ms);
        std::printf("BENCH engine=ATV state=%s setting=%s thermostat=%s "
                    "threads=%d steps=%d "
                    "nk=%zu loop_ms=%.6f wall_ms=%.6f pair_ms=%.6f "
                    "kspace_ms=%.6f integrate_ms=%.6f thermostat_ms=%.6f "
                    "neigh_ms=%.6f other_ms=%.6f energy=%.17g\n",
                    sp.name.c_str(), es.name.c_str(), thermostat.c_str(),
                    threads, steps, k_vector_count(sp, es), loop_ms,
                    wall_s * to_ms, b.pair * to_ms, b.kspace * to_ms,
                    b.integrate * to_ms, b.thermostat * to_ms, b.neigh * to_ms,
                    b.other * to_ms, frame.meta.get(kEnergy));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "bench_nacl_ewald: %s\n", error.what());
        return 1;
    }
    return 0;
}
