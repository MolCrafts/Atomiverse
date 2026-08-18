/// @file app/atomiverse/cli_args.h
/// @brief Command line of the `atomiverse` binary (LAMMPS-shaped).
#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

namespace ATV {
namespace app {

/// @brief Parsed argv for the native / launcher binary.
struct CliArgs {
    bool help = false;
    std::string in_path;     ///< native input (`-in` or a non-.py file)
    std::string log_path;    ///< optional `-log`
    std::string py_script;   ///< Python script (launcher path)
    int py_argc = 0;
    char** py_argv = nullptr;
};

inline bool ends_with(const char* s, const char* suf) {
    const std::size_t n = std::strlen(s);
    const std::size_t m = std::strlen(suf);
    return n >= m && std::strcmp(s + (n - m), suf) == 0;
}

/// @brief Parse LAMMPS-like flags. Does not open files.
inline CliArgs parse_cli_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "-help") == 0
            || std::strcmp(arg, "--help") == 0) {
            a.help = true;
            return a;
        }
        if (std::strcmp(arg, "-in") == 0 || std::strcmp(arg, "--in") == 0) {
            if (i + 1 >= argc)
                throw std::runtime_error("atomiverse: -in requires a file");
            a.in_path = argv[++i];
            continue;
        }
        if (std::strcmp(arg, "-log") == 0 || std::strcmp(arg, "--log") == 0) {
            if (i + 1 >= argc)
                throw std::runtime_error("atomiverse: -log requires a file");
            a.log_path = argv[++i];
            continue;
        }
        if (arg[0] == '-')
            throw std::runtime_error(std::string("atomiverse: unknown flag ")
                                     + arg);
        // Positional: .py → Python host; anything else → native input.
        if (ends_with(arg, ".py")) {
            a.py_script = arg;
            a.py_argc = argc - i;
            a.py_argv = argv + i;
            return a;
        }
        a.in_path = arg;
    }
    return a;
}

inline const char* cli_help() {
    return
        "atomiverse — native driver (LAMMPS-shaped CLI)\n"
        "\n"
        "  atomiverse -in in.lj                 run a native input script\n"
        "  atomiverse -in in.lj -log log.atv    same, write log file\n"
        "  atomiverse in.lj                     positional input file\n"
        "  atomiverse script.py [args…]         run a Python script "
        "(needs ATV_BUILD_LAUNCHER)\n"
        "\n"
        "Native input (first slice): units metal, atom_style atomic,\n"
        "lattice sc, region box block, create_box, create_atoms,\n"
        "mass, pair_style lj/cut, pair_coeff, neighbor, timestep,\n"
        "thermo, velocity all create, run.\n"
        "Units are always eV, Å, fs, amu.\n";
}

} // namespace app
} // namespace ATV
