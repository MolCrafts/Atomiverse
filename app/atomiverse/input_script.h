/// @file app/atomiverse/input_script.h
/// @brief Native LAMMPS-shaped input deck for the `atomiverse` CLI.
///
/// App-only. The library never includes this header (no app → lib cycle).
/// Commands map to public ctor args / categorical setters — no Config bag.
#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace ATV {
namespace app {

/// @brief Parsed native input (units: eV, Å, fs, amu).
struct InputDeck {
    double lattice_a = 0; ///< sc lattice constant [Å]
    double xlo = 0, xhi = 0;
    double ylo = 0, yhi = 0;
    double zlo = 0, zhi = 0;
    bool have_region = false;
    int ntypes = 1;
    bool created_atoms = false;
    double mass = 39.948; ///< type-1 mass [amu]
    int type_id = 1;
    double pair_cut = 0;   ///< lj/cut cutoff [Å]
    double pair_eps = 0;   ///< [eV]
    double pair_sigma = 0; ///< [Å]
    bool have_pair = false;
    double neighbor_skin = -1.0; ///< [Å]; negative = NeighborList auto
    double timestep = 1.0;       ///< [fs]
    int thermo_every = 0;
    double velocity_T = -1.0; ///< <0 = no InitVelocity
    unsigned velocity_seed = 42;
    int run_steps = 0;
};

/// @brief Parse a native input stream. Does not run physics.
/// @throws std::invalid_argument on unknown/illegal commands.
InputDeck parse_input_script(std::istream& in);

/// @brief Run a parsed deck (MD LJ only on this slice).
/// @throws std::invalid_argument / std::runtime_error on incomplete deck.
void run_input_deck(const InputDeck& deck, std::ostream& log);

/// @brief Open `path`, parse, run. Optional log file (empty = stdout).
int run_input_script(const std::string& path, const std::string& log_path);

} // namespace app
} // namespace ATV
