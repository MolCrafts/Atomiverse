/// @file app/atomiverse/input_script.cpp
/// @brief Parse and execute a LAMMPS-shaped native input (LJ MD first slice).

#include "input_script.h"

#include "atomiverse/core/frame_schema.h" // kAtom_*, kEnergy
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace app {
namespace {

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tok;
    std::string cur;
    for (char c : line) {
        if (c == '#')
            break;
        if (c == ' ' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                tok.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty())
        tok.push_back(cur);
    return tok;
}

double req_d(const std::vector<std::string>& t, std::size_t i,
             const char* cmd) {
    if (i >= t.size())
        throw std::invalid_argument(std::string(cmd) + ": missing number");
    return std::stod(t[i]);
}

int req_i(const std::vector<std::string>& t, std::size_t i, const char* cmd) {
    if (i >= t.size())
        throw std::invalid_argument(std::string(cmd) + ": missing integer");
    return std::stoi(t[i]);
}

} // namespace

InputDeck parse_input_script(std::istream& in) {
    InputDeck d;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        auto t = tokenize(line);
        if (t.empty())
            continue;
        const std::string& c = t[0];
        try {
            if (c == "units") {
                if (t.size() < 2 || (t[1] != "metal" && t[1] != "atomiverse"))
                    throw std::invalid_argument(
                        "units: only 'metal' or 'atomiverse' "
                        "(eV, Å, fs, amu) — no other LAMMPS unit sets");
            } else if (c == "dimension") {
                if (req_i(t, 1, "dimension") != 3)
                    throw std::invalid_argument("dimension: only 3");
            } else if (c == "boundary") {
                // accepted; Box is periodic
            } else if (c == "atom_style") {
                if (t.size() < 2 || t[1] != "atomic")
                    throw std::invalid_argument(
                        "atom_style: only 'atomic' on this CLI slice");
            } else if (c == "lattice") {
                if (t.size() < 3 || t[1] != "sc")
                    throw std::invalid_argument("lattice: only 'sc <a>'");
                d.lattice_a = req_d(t, 2, "lattice");
                if (!(d.lattice_a > 0))
                    throw std::invalid_argument("lattice: a must be > 0");
            } else if (c == "region") {
                if (t.size() < 9 || t[2] != "block")
                    throw std::invalid_argument(
                        "region: only 'region box block xlo xhi ylo yhi zlo "
                        "zhi'");
                d.xlo = req_d(t, 3, "region");
                d.xhi = req_d(t, 4, "region");
                d.ylo = req_d(t, 5, "region");
                d.yhi = req_d(t, 6, "region");
                d.zlo = req_d(t, 7, "region");
                d.zhi = req_d(t, 8, "region");
                d.have_region = true;
            } else if (c == "create_box") {
                d.ntypes = req_i(t, 1, "create_box");
                if (d.ntypes < 1)
                    throw std::invalid_argument("create_box: ntypes >= 1");
            } else if (c == "create_atoms") {
                d.type_id = req_i(t, 1, "create_atoms");
                d.created_atoms = true;
            } else if (c == "mass") {
                // mass * m   or  mass 1 m
                d.mass = req_d(t, t.size() >= 3 ? 2 : 1, "mass");
                if (!(d.mass > 0))
                    throw std::invalid_argument("mass: must be > 0");
            } else if (c == "pair_style") {
                if (t.size() < 3 || t[1] != "lj/cut")
                    throw std::invalid_argument(
                        "pair_style: only 'lj/cut <cutoff>' on this CLI slice");
                d.pair_cut = req_d(t, 2, "pair_style");
            } else if (c == "pair_coeff") {
                // pair_coeff * * eps sigma
                if (t.size() < 5)
                    throw std::invalid_argument(
                        "pair_coeff: 'pair_coeff * * eps sigma'");
                d.pair_eps = req_d(t, 3, "pair_coeff");
                d.pair_sigma = req_d(t, 4, "pair_coeff");
                d.have_pair = true;
            } else if (c == "neighbor") {
                d.neighbor_skin = req_d(t, 1, "neighbor");
            } else if (c == "neigh_modify") {
                // accepted no-op (defaults match LAMMPS)
            } else if (c == "timestep") {
                d.timestep = req_d(t, 1, "timestep");
                if (!(d.timestep > 0))
                    throw std::invalid_argument("timestep: must be > 0");
            } else if (c == "thermo") {
                d.thermo_every = req_i(t, 1, "thermo");
            } else if (c == "thermo_style") {
                // accepted no-op
            } else if (c == "velocity") {
                // velocity all create T seed
                if (t.size() < 5 || t[2] != "create")
                    throw std::invalid_argument(
                        "velocity: only 'velocity all create T seed'");
                d.velocity_T = req_d(t, 3, "velocity");
                d.velocity_seed =
                    static_cast<unsigned>(req_i(t, 4, "velocity"));
            } else if (c == "run") {
                d.run_steps = req_i(t, 1, "run");
                if (d.run_steps < 0)
                    throw std::invalid_argument("run: steps >= 0");
            } else if (c == "log" || c == "echo" || c == "processor"
                       || c == "comm_modify" || c == "newton") {
                // accepted no-op (LAMMPS noise)
            } else {
                throw std::invalid_argument("unknown command '" + c + "'");
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument("line " + std::to_string(lineno) + ": "
                                        + e.what());
        }
    }
    return d;
}

void run_input_deck(const InputDeck& d, std::ostream& log) {
    if (!d.have_region)
        throw std::invalid_argument("input: missing 'region box block …'");
    if (!d.created_atoms)
        throw std::invalid_argument("input: missing 'create_atoms'");
    if (!(d.lattice_a > 0))
        throw std::invalid_argument("input: missing 'lattice sc <a>'");
    if (!d.have_pair || !(d.pair_cut > 0))
        throw std::invalid_argument("input: missing pair_style/pair_coeff");

    const double Lx = d.xhi - d.xlo;
    const double Ly = d.yhi - d.ylo;
    const double Lz = d.zhi - d.zlo;
    if (!(Lx > 0 && Ly > 0 && Lz > 0))
        throw std::invalid_argument("input: region extents must be positive");

    const int nx = static_cast<int>(std::floor(Lx / d.lattice_a + 1e-12));
    const int ny = static_cast<int>(std::floor(Ly / d.lattice_a + 1e-12));
    const int nz = static_cast<int>(std::floor(Lz / d.lattice_a + 1e-12));
    const int n = nx * ny * nz;
    if (n < 1)
        throw std::invalid_argument("input: lattice does not fit in the box");

    ATV::cpu::Frame frame(n);
    auto x = frame.span(ATV::kAtom_x);
    auto y = frame.span(ATV::kAtom_y);
    auto z = frame.span(ATV::kAtom_z);
    auto type = frame.span(ATV::kAtom_type);
    auto mass = frame.span(ATV::kAtom_mass);
    int i = 0;
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
                x[i] = d.xlo + (ix + 0.5) * d.lattice_a;
                y[i] = d.ylo + (iy + 0.5) * d.lattice_a;
                z[i] = d.zlo + (iz + 0.5) * d.lattice_a;
                type[i] = d.type_id;
                mass[i] = d.mass;
                ++i;
            }
    frame.box = ATV::cpu::Box::orthorhombic(Lx, Ly, Lz);

    ATV::cpu::md::MD sim;
    sim.set_evaluator(
        ATV::cpu::PairLJCut{d.pair_eps, d.pair_sigma, d.pair_cut});
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{d.pair_cut, d.neighbor_skin});
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{d.timestep});
    if (d.velocity_T > 0)
        sim.set_observer(
            ATV::cpu::md::InitVelocity{d.velocity_T, d.velocity_seed});
    if (d.thermo_every > 0)
        sim.set_thermo(d.thermo_every);
    sim.build();

    log << "atomiverse native input: n=" << n << "  box=" << Lx << " " << Ly
        << " " << Lz << " Å  dt=" << d.timestep << " fs  steps=" << d.run_steps
        << "\n";
    if (d.run_steps > 0)
        sim.run(frame, d.run_steps);
    const double pe = frame.meta.get(ATV::kEnergy);
    log << "PE = " << pe << " eV\n";
}

int run_input_script(const std::string& path, const std::string& log_path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("atomiverse: cannot read '" + path + "'");
    const InputDeck deck = parse_input_script(in);
    if (log_path.empty() || log_path == "none") {
        run_input_deck(deck, std::cout);
        return 0;
    }
    std::ofstream log(log_path);
    if (!log)
        throw std::runtime_error("atomiverse: cannot write log '" + log_path
                                 + "'");
    run_input_deck(deck, log);
    return 0;
}

} // namespace app
} // namespace ATV
