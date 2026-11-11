/// @file tests/cpu/dynamics/test_lammps_units_parity.cpp
/// @brief Classical MD in LAMMPS `units lj|real|metal|si|electron` against
///        stock LAMMPS on the same input decks (tests/data/lammps_units).
///
/// Each style's system (data.lmp + system.ff, coefficients in that style's
/// own units, read without any conversion) is run with `Driver::set_units`
/// of the style and the deck's settings, then compared with the LAMMPS
/// thermo/dump references written by make_refs.sh:
///   - single: `run 0` forces, energy terms, kinetic energy, temperature,
///     pressure tensor;
///   - nve / langevin (T = 0, deterministic drag) / npt / nvt: the state
///     after 100 steps (positions, velocities, forces, pe, ke, pressure,
///     volume).
/// The numbers only agree if every factor the engine multiplies (ftm2v,
/// mvv2e, boltz, qqrd2e, nktv2p) is the selected style's.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_coeffs.h"
#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pppm.h"
#endif

using namespace ATV;

namespace {

std::filesystem::path data_dir(const std::string& style) {
    return std::filesystem::path(__FILE__)
               .parent_path()
               .parent_path()
               .parent_path()
           / "data" / "lammps_units" / style;
}

/// The tokens of the first deck line whose first token is `cmd` (and, when
/// given, whose second token is `arg`).
std::vector<std::string> deck_line(const std::filesystem::path& deck,
                                   const std::string& cmd,
                                   const std::string& arg = "") {
    std::ifstream in(deck);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<std::string> t;
        for (std::string w; iss >> w;)
            t.push_back(w);
        if (!t.empty() && t[0] == cmd && (arg.empty() || t[1] == arg))
            return t;
    }
    return {};
}

/// Value of keyword `key` inside a token list (`gewald 0.39 mesh 15 ...`).
double after(const std::vector<std::string>& t, const std::string& key,
             int offset = 1) {
    for (std::size_t i = 0; i + offset < t.size(); ++i)
        if (t[i] == key)
            return std::stod(t[i + offset]);
    throw std::runtime_error("deck keyword '" + key + "' missing");
}

/// LAMMPS thermo rows by step, columns by header name.
std::map<long, std::map<std::string, double>>
read_thermo(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::vector<std::string> head;
    std::map<long, std::map<std::string, double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<std::string> t;
        for (std::string w; iss >> w;)
            t.push_back(w);
        if (t.empty())
            continue;
        if (t[0] == "Step") {
            head = t;
            continue;
        }
        if (head.empty() || t.size() != head.size())
            continue;
        std::map<std::string, double> row;
        for (std::size_t i = 0; i < t.size(); ++i)
            row[head[i]] = std::stod(t[i]);
        rows[static_cast<long>(row["Step"])] = row;
    }
    return rows;
}

/// LAMMPS dump columns per atom id.
std::map<int, std::map<std::string, double>>
read_dump(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::string line;
    std::vector<std::string> cols;
    while (std::getline(in, line))
        if (line.rfind("ITEM: ATOMS", 0) == 0) {
            std::istringstream iss(line.substr(11));
            for (std::string w; iss >> w;)
                cols.push_back(w);
            break;
        }
    std::map<int, std::map<std::string, double>> out;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::map<std::string, double> row;
        for (const auto& c : cols) {
            double v = 0;
            if (!(iss >> v))
                break;
            row[c] = v;
        }
        if (row.size() == cols.size())
            out[static_cast<int>(row["id"])] = row;
    }
    return out;
}

struct System {
    std::string style;
    units::UnitSystem u;
    cpu::Frame frame{0};
    cpu::LammpsCoeffs ff;
};

System load(const std::string& style) {
    System s;
    s.style = style;
    s.u = units::from_style(style);
    s.frame = cpu::load_lammps_data((data_dir(style) / "data.lmp").string());
    s.ff = cpu::load_lammps_coeffs((data_dir(style) / "system.ff").string());
    return s;
}

/// Mount the deck's force field, neighbor list and the system stress.
void mount_forces(cpu::md::MD& md, const System& s,
                  const std::filesystem::path& deck) {
    const auto& ff = s.ff;
    const double skin = std::stod(deck_line(deck, "neighbor").at(1));
    const double cutoff = ff.pair_cut_lj;
    // neigh_modify every 1 delay 0 check no
    md.set_neighbor(cpu::md::NeighborList<>{cutoff, skin, 1, 0, false});
    // Kinetic stress from the end-of-step velocities, as LAMMPS thermo.
    md.on(ATV::md::Phase::end_of_step).add(cpu::md::Virial{}, 50);
    const double* sl = ff.special_lj;
    const double* sc = ff.special_coul;
    if (ff.pair_style == "lj/cut/coul/long") {
#ifdef ATV_HAS_KSPACE
        const auto k = deck_line(deck, "kspace_modify");
        const double alpha = after(k, "gewald");
        const int mesh = static_cast<int>(after(k, "mesh"));
        const int order = static_cast<int>(after(k, "order"));
        md.set_evaluator(
            cpu::PairLJCutCoulLong{ff.lj, alpha, cutoff}
                .set_special_bonds(sl[0], sl[1], sl[2], sc[0], sc[1], sc[2])
                .set_energy_shift(ff.energy_shift));
        md.set_evaluator(cpu::KSpacePPPM{alpha, {mesh, mesh, mesh}, order});
#endif
    } else {
        md.set_evaluator(cpu::PairLJCut{ff.lj, cutoff}
                             .set_special_bonds(sl[0], sl[1], sl[2])
                             .set_energy_shift(ff.energy_shift));
    }
    if (!ff.bond_K.empty())
        md.set_evaluator(cpu::BondHarmonic{ff.bond_K, ff.bond_r0});
    if (!ff.angle_K.empty())
        md.set_evaluator(cpu::AngleHarmonic{ff.angle_K, ff.angle_theta0});
    if (ff.has_charmm)
        md.set_evaluator(cpu::DihedralCharmm{ff.dihedral_charmm_K,
                                             ff.dihedral_charmm_n,
                                             ff.dihedral_charmm_delta});
    if (ff.has_improper_cvff)
        md.set_evaluator(
            cpu::ImproperCvff{ff.improper_K, ff.improper_d, ff.improper_n});
}

/// Run the deck `in.<name>.lmp` of `style` for `steps` steps.
cpu::Frame run_case(System& s, const std::string& name, int steps) {
    const auto deck = data_dir(s.style) / ("in." + name + ".lmp");
    cpu::md::MD md;
    md.set_units(s.style);
    md.set_threads(1);
    mount_forces(md, s, deck);
    const double dt = std::stod(deck_line(deck, "timestep").at(1));
    const auto nvt = deck_line(deck, "fix", "1");
    if (name.rfind("npt", 0) == 0) {
        // fix npt temp T T Tdamp iso P P Pdamp (tchain 3, pchain 3, mtk)
        md.set_integrator(cpu::md::NoseHooverNPT{
            after(nvt, "temp"), after(nvt, "temp", 3), after(nvt, "iso"),
            after(nvt, "iso", 3), dt});
    } else {
        md.set_integrator(cpu::md::VelocityVerlet{dt});
        if (name == "nvt")
            md.set_thermostat(cpu::md::NoseHoover{after(nvt, "temp"),
                                                  after(nvt, "temp", 3), dt});
        if (name == "langevin") {
            const auto lang = deck_line(deck, "fix", "2");
            md.set_thermostat(cpu::md::FixLangevin{
                std::stod(lang.at(4)), std::stod(lang.at(5)),
                std::stod(lang.at(6)), dt, 1u});
        }
    }
    md.build();
    cpu::Frame f = s.frame; // fresh copy of the data-file state
    md.run(f, steps);
    // LAMMPS refuses `units` once the box exists; so does the driver.
    const char* other = s.style == "metal" ? "real" : "metal";
    EXPECT_THROW(md.set_units(other), std::logic_error);
    return f;
}

double rel(double got, double want, double scale) {
    return std::fabs(got - want) / (std::fabs(want) + scale);
}

/// Pressure tensor (Voigt) in the style's unit, as LAMMPS compute pressure.
std::array<double, 6> pressure(const cpu::Frame& f,
                               const units::UnitSystem& u) {
    std::array<double, 6> p{};
    const double V = f.box.volume();
    for (int a = 0; a < 6; ++a)
        p[a] = (f.meta.get(kStress)[a] + f.meta.get(kStressKinetic)[a]) / V
               * u.nktv2p;
    return p;
}

/// Largest per-atom force relative to the largest reference force.
double force_error(const cpu::Frame& f,
                   const std::map<int, std::map<std::string, double>>& ref) {
    double worst = 0, fmax = 0;
    const auto tag = f.span(kAtom_tag);
    const AccReal* F[3] = {f.span(kAtom_fx), f.span(kAtom_fy),
                           f.span(kAtom_fz)};
    const char* key[3] = {"fx", "fy", "fz"};
    for (int i = 0; i < f.nlocal(); ++i) {
        const auto& r = ref.at(static_cast<int>(tag[i]) + 1);
        for (int d = 0; d < 3; ++d) {
            worst = std::max(worst, std::fabs(F[d][i] - r.at(key[d])));
            fmax = std::max(fmax, std::fabs(r.at(key[d])));
        }
    }
    return worst / fmax;
}

/// Largest position (minimum image vs the unwrapped reference) and
/// velocity error, relative to the length unit (Å in the style) and the
/// largest reference speed.
std::pair<double, double>
state_error(const cpu::Frame& f,
            const std::map<int, std::map<std::string, double>>& ref,
            const units::UnitSystem& u) {
    double dx_max = 0, dv_max = 0, vmax = 0;
    const auto tag = f.span(kAtom_tag);
    for (int i = 0; i < f.nlocal(); ++i) {
        const auto& r = ref.at(static_cast<int>(tag[i]) + 1);
        double d[3] = {f.span(kAtom_x)[i] - r.at("xu"),
                       f.span(kAtom_y)[i] - r.at("yu"),
                       f.span(kAtom_z)[i] - r.at("zu")};
        f.box.min_image(d[0], d[1], d[2]);
        const double v[3] = {f.span(kAtom_vx)[i] - r.at("vx"),
                             f.span(kAtom_vy)[i] - r.at("vy"),
                             f.span(kAtom_vz)[i] - r.at("vz")};
        for (int k = 0; k < 3; ++k) {
            dx_max = std::max(dx_max, std::fabs(d[k]));
            dv_max = std::max(dv_max, std::fabs(v[k]));
        }
        vmax = std::max({vmax, std::fabs(r.at("vx")), std::fabs(r.at("vy")),
                         std::fabs(r.at("vz"))});
    }
    return {dx_max / u.angstrom, dv_max / vmax};
}

// LAMMPS `pair_modify table 0` evaluates erfc with the Abramowitz-Stegun
// polynomial (|error| ~ 1e-7); Atomiverse evaluates it exactly. That sets
// the floor of the charged styles' force and pair-energy agreement.

class LammpsUnitsParity : public testing::TestWithParam<std::string> {};

TEST_P(LammpsUnitsParity, SinglePointMatchesLammpsRun0) {
    System s = load(GetParam());
#ifndef ATV_HAS_KSPACE
    if (s.ff.pair_style == "lj/cut/coul/long")
        GTEST_SKIP() << "PPPM needs ATV_BUILD_KSPACE";
#endif
    const cpu::Frame f = run_case(s, "single", 0);
    const auto th = read_thermo(data_dir(s.style) / "ref.single.thermo").at(0);
    const auto dump = read_dump(data_dir(s.style) / "ref.single.dump");
    const double escale =
        std::fabs(th.at("PotEng")) + std::fabs(th.at("KinEng"));

    EXPECT_LT(force_error(f, dump), 1e-6) << s.style;
    EXPECT_LT(rel(f.meta.get(kEnergy), th.at("PotEng"), escale), 1e-6);
    EXPECT_LT(rel(f.meta.get(kEnergyPair),
                  th.at("E_vdwl") + th.at("E_coul") + th.at("E_long"), escale),
              1e-6);
    EXPECT_LT(rel(f.meta.get(kEnergyBond), th.at("E_bond"), escale), 1e-12);
    EXPECT_LT(rel(f.meta.get(kEnergyAngle), th.at("E_angle"), escale), 1e-12);
    EXPECT_LT(rel(f.meta.get(kEnergyDihedral), th.at("E_dihed"), escale),
              1e-12);
    EXPECT_LT(rel(f.meta.get(kEnergyImproper), th.at("E_impro"), escale),
              1e-12);
    // mvv2e and boltz: kinetic energy and temperature (dof 3N - 3).
    EXPECT_LT(rel(f.meta.get(kKinetic), th.at("KinEng"), 0.0), 1e-12);
    const double ndof = 3.0 * f.nlocal() - 3.0;
    EXPECT_LT(rel(2.0 * f.meta.get(kKinetic) / (ndof * s.u.boltz),
                  th.at("Temp"), 0.0),
              1e-10);
    // nktv2p: the pressure tensor in the style's pressure unit.
    const auto p = pressure(f, s.u);
    const char* key[6] = {"Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"};
    double pscale = 0;
    for (const char* k : key)
        pscale = std::max(pscale, std::fabs(th.at(k)));
    for (int a = 0; a < 6; ++a)
        EXPECT_LT(rel(p[a], th.at(key[a]), pscale), 1e-6)
            << s.style << " " << key[a];
}

/// Agreement bounds of a 100-step case (relative; `state` < 0 skips the
/// per-atom comparison).
struct Tol {
    double state, energy, pressure, volume;
};

/// The state after 100 steps of the deck, against LAMMPS.
void expect_trajectory(const std::string& style, const std::string& name,
                       const Tol& tol) {
    System s = load(style);
#ifndef ATV_HAS_KSPACE
    if (s.ff.pair_style == "lj/cut/coul/long")
        GTEST_SKIP() << "PPPM needs ATV_BUILD_KSPACE";
#endif
    const cpu::Frame f = run_case(s, name, 100);
    const auto base = data_dir(style) / ("ref." + name);
    const auto thermo = read_thermo(base.string() + ".thermo");
    const auto th = thermo.at(100);
    if (tol.state > 0) {
        const auto dump = read_dump(base.string() + ".dump");
        const auto [dx, dv] = state_error(f, dump, s.u);
        EXPECT_LT(dx, tol.state) << style << " " << name << " positions";
        EXPECT_LT(dv, tol.state) << style << " " << name << " velocities";
        EXPECT_LT(force_error(f, dump), tol.state) << style << " " << name;
    }
    const double escale =
        std::fabs(th.at("PotEng")) + std::fabs(th.at("KinEng"));
    EXPECT_LT(rel(f.meta.get(kEnergy), th.at("PotEng"), escale), tol.energy)
        << style << " " << name;
    EXPECT_LT(rel(f.meta.get(kKinetic), th.at("KinEng"), escale), tol.energy)
        << style << " " << name;
    const auto p = pressure(f, s.u);
    const double press = (p[0] + p[1] + p[2]) / 3.0;
    // The pressure is a small difference of large virial and kinetic terms:
    // judge it on the scale of the run's pressure excursions.
    double pscale = 0;
    for (const auto& [step, row] : thermo)
        pscale = std::max(pscale, std::fabs(row.at("Press")));
    EXPECT_LT(rel(press, th.at("Press"), pscale), tol.pressure)
        << style << " " << name;
    if (th.count("Volume")) {
        // The barostat's response: the volume change of the run.
        const double v0 = thermo.at(0).at("Volume");
        EXPECT_LT(std::fabs((f.box.volume() - v0) - (th.at("Volume") - v0))
                      / std::fabs(th.at("Volume") - v0),
                  tol.volume)
            << style << " " << name << " volume change";
    }
}

TEST_P(LammpsUnitsParity, NveTrajectoryMatchesLammps) {
    expect_trajectory(GetParam(), "nve", {5e-6, 1e-6, 1e-5, 0});
}

INSTANTIATE_TEST_SUITE_P(Styles, LammpsUnitsParity,
                         testing::Values("lj", "real", "metal", "si",
                                         "electron"),
                         [](const auto& info) { return info.param; });

class LammpsUnitsThermostats : public testing::TestWithParam<std::string> {};

TEST_P(LammpsUnitsThermostats, LangevinDragMatchesLammpsAtZeroTemperature) {
    // fix langevin 0 0 damp: the random force is exactly zero, so the drag
    // -m v / (damp ftm2v) is the whole thermostat and the run is
    // deterministic.
    expect_trajectory(GetParam(), "langevin", {5e-6, 3e-6, 1e-5, 0});
}

TEST_P(LammpsUnitsThermostats, NptMatchesLammpsFixNpt) {
    // NoseHooverNPT fuses LAMMPS's box remap into one analytic step and uses
    // the MTK mass (N_f + 3) kT tau_P^2 where fix npt has (N + 1)·3 kT
    // tau_P^2: thermodynamics agree, atom trajectories drift apart.
    expect_trajectory(GetParam(), "npt", {-1, 1e-4, 1e-2, 0.1});
}

TEST_P(LammpsUnitsThermostats, NoseHooverTracksLammpsFixNvt) {
    // NoseHoover updates xi once per half step where fix nvt (tchain 1)
    // splits it into dt/4 kicks around the velocity scaling: the two agree
    // to O((dt/tdamp)^2) per step, not to round-off.
    expect_trajectory(GetParam(), "nvt", {3e-3, 1e-3, 1e-2, 0});
}

INSTANTIATE_TEST_SUITE_P(Styles, LammpsUnitsThermostats,
                         testing::Values("lj", "real", "metal"),
                         [](const auto& info) { return info.param; });

/// A 20000 atm / bar target drives the volume: the barostat reads the
/// target in the style's pressure unit (p_target / nktv2p).
class LammpsUnitsPressureTarget : public testing::TestWithParam<std::string> {};

TEST_P(LammpsUnitsPressureTarget, VolumeResponseToTheTargetMatchesLammps) {
    expect_trajectory(GetParam(), "npt_hp", {-1, 3e-3, 3e-2, 0.05});
}

INSTANTIATE_TEST_SUITE_P(Styles, LammpsUnitsPressureTarget,
                         testing::Values("real", "metal"),
                         [](const auto& info) { return info.param; });

} // namespace
