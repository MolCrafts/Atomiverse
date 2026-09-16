/// @file tools/debug_amorphous/lammps_loader.h
/// @brief Minimal LAMMPS "atom_style molecular" data file → cpu::Frame loader.
///
/// Self-contained text parser. Reads the header (atom/bond/angle/dihedral
/// counts + box bounds) then the Atoms / Bonds / Angles / Dihedrals
/// sections. Skips the Masses section. Tag (LAMMPS atom-id) is preserved
/// 1-to-1 in the atoms Block tag column (LAMMPS atom-id 1 → tag 0). Bond / angle /
/// dihedral atom indices are translated from LAMMPS 1-based atom-ids to
/// 0-based local rows by `id - 1`.
///
/// Shipped only with tools/debug_amorphous/run_atomiverse — not part of the
/// installed library.
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace debug_amorphous {

inline ATV::cpu::Frame load_lammps_data(const std::string& path,
                                        double mass_amu = 14.02) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);

    auto next_line = [&](std::string& s) -> bool {
        while (std::getline(in, s)) {
            // Strip a trailing comment "# ..." but keep numeric content.
            const auto hash = s.find('#');
            if (hash != std::string::npos)
                s.erase(hash);
            // Trim trailing whitespace.
            while (!s.empty()
                   && std::isspace(static_cast<unsigned char>(s.back())))
                s.pop_back();
            if (!s.empty())
                return true;
        }
        return false;
    };

    int n_atoms = 0, n_bonds = 0, n_angles = 0, n_dihedrals = 0;
    double xlo = 0, xhi = 0, ylo = 0, yhi = 0, zlo = 0, zhi = 0;

    std::string line;
    // Header: read until a section keyword appears.
    auto starts_with_section = [](const std::string& s) {
        // LAMMPS section labels start with an uppercase letter at column 0
        // (after any leading whitespace). Tokenize and check the first one.
        std::istringstream is(s);
        std::string t;
        if (!(is >> t))
            return false;
        return std::isupper(static_cast<unsigned char>(t[0])) != 0;
    };

    std::string section;
    while (next_line(line)) {
        if (starts_with_section(line)) {
            section = line;
            break;
        }
        std::istringstream is(line);
        long n;
        std::string a, b;
        is >> n >> a;
        if (a == "atoms")
            n_atoms = static_cast<int>(n);
        else if (a == "bonds")
            n_bonds = static_cast<int>(n);
        else if (a == "angles")
            n_angles = static_cast<int>(n);
        else if (a == "dihedrals")
            n_dihedrals = static_cast<int>(n);
        else if (a == "atom" || a == "bond" || a == "angle"
                 || a == "dihedral") {
            // "1 atom types" — ignored, we trust input file.
        } else {
            // Try box bounds: "<lo> <hi> xlo xhi" / "ylo yhi" / "zlo zhi"
            std::istringstream is2(line);
            double lo, hi;
            std::string tag1, tag2;
            if ((is2 >> lo >> hi >> tag1 >> tag2) && tag1.size() == 3) {
                if (tag1 == "xlo") {
                    xlo = lo;
                    xhi = hi;
                } else if (tag1 == "ylo") {
                    ylo = lo;
                    yhi = hi;
                } else if (tag1 == "zlo") {
                    zlo = lo;
                    zhi = hi;
                }
            }
        }
    }

    if (n_atoms <= 0)
        throw std::runtime_error("LAMMPS data: missing 'atoms' header");

    ATV::cpu::Frame frame(n_atoms);
    frame.box.set_cell(ATV::Mat3::diagonal(xhi - xlo, yhi - ylo, zhi - zlo));

    // Default: every atom gets the supplied mass and type 0.
    auto initial_mass = frame.span(kAtom_mass);
    auto initial_type = frame.span(kAtom_type);
    for (int i = 0; i < n_atoms; ++i) {
        initial_mass[i] = mass_amu;
        initial_type[i] = 0;
    }

    auto trim_section = [](std::string& s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
            s.pop_back();
    };

    // Per-atom topology (LAMMPS Atom layout): the tag map must be
    // populated before the section handlers call add_bond/add_angle/
    // add_dihedral (owner rows are resolved by global tag; the file's
    // 1-based ids map to identity tags 0..n_atoms-1).
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, n_atoms);

    auto read_section = [&](const std::string& label, auto handler) {
        // Caller has already read the section header line into `section`.
        if (section.find(label) == std::string::npos)
            return false;
        while (next_line(line)) {
            trim_section(line);
            if (starts_with_section(line)) {
                section = line;
                return true;
            }
            handler(line);
        }
        section.clear();
        return true;
    };

    int n_atoms_read = 0, n_bonds_read = 0, n_angles_read = 0,
        n_dihedrals_read = 0;

    auto consume_section = [&]() {
        if (section.find("Atoms") != std::string::npos) {
            auto x_col = frame.span(kAtom_x);
            auto y_col = frame.span(kAtom_y);
            auto z_col = frame.span(kAtom_z);
            auto tag_col = frame.span(kAtom_tag);
            auto type_col = frame.span(kAtom_type);
            read_section("Atoms", [&](const std::string& l) {
                std::istringstream is(l);
                long aid, mol;
                int type;
                double x, y, z;
                if (!(is >> aid >> mol >> type >> x >> y >> z))
                    throw std::runtime_error("Atoms parse: " + l);
                const int row = static_cast<int>(aid) - 1;
                if (row < 0 || row >= n_atoms)
                    throw std::runtime_error("Atoms id out of range: " + l);
                x_col[row] = x;
                y_col[row] = y;
                z_col[row] = z;
                tag_col[row] = static_cast<TagInt>(row);
                type_col[row] = type - 1; // LAMMPS 1-based → 0-based
                ++n_atoms_read;
            });
        } else if (section.find("Masses") != std::string::npos) {
            auto type_col = frame.span(kAtom_type);
            auto mass_col = frame.span(kAtom_mass);
            read_section("Masses", [&](const std::string& l) {
                std::istringstream is(l);
                int t;
                double m;
                if (is >> t >> m) {
                    // Apply the per-type mass to every atom of that type.
                    const int t0 = t - 1;
                    for (int i = 0; i < n_atoms; ++i)
                        if (type_col[i] == t0)
                            mass_col[i] = m;
                }
            });
        } else if (section.find("Bonds") != std::string::npos) {
            read_section("Bonds", [&](const std::string& l) {
                std::istringstream is(l);
                long bid;
                int btype;
                long ai, aj;
                if (!(is >> bid >> btype >> ai >> aj))
                    throw std::runtime_error("Bonds parse: " + l);
                ++n_bonds_read;
                frame.add_bond(static_cast<int>(ai) - 1,
                               static_cast<int>(aj) - 1, btype - 1);
            });
        } else if (section.find("Angles") != std::string::npos) {
            read_section("Angles", [&](const std::string& l) {
                std::istringstream is(l);
                long aid;
                int atype;
                long ai, aj, ak;
                if (!(is >> aid >> atype >> ai >> aj >> ak))
                    throw std::runtime_error("Angles parse: " + l);
                ++n_angles_read;
                frame.add_angle(static_cast<int>(ai) - 1,
                                static_cast<int>(aj) - 1,
                                static_cast<int>(ak) - 1, atype - 1);
            });
        } else if (section.find("Dihedrals") != std::string::npos) {
            read_section("Dihedrals", [&](const std::string& l) {
                std::istringstream is(l);
                long did;
                int dtype;
                long ai, aj, ak, al;
                if (!(is >> did >> dtype >> ai >> aj >> ak >> al))
                    throw std::runtime_error("Dihedrals parse: " + l);
                ++n_dihedrals_read;
                frame.add_dihedral(static_cast<int>(ai) - 1,
                                   static_cast<int>(aj) - 1,
                                   static_cast<int>(ak) - 1,
                                   static_cast<int>(al) - 1, dtype - 1);
            });
        } else {
            // Unrecognized section — drain it until next section or EOF.
            while (next_line(line)) {
                trim_section(line);
                if (starts_with_section(line)) {
                    section = line;
                    return;
                }
            }
            section.clear();
        }
    };

    while (!section.empty())
        consume_section();

    if (n_atoms_read != n_atoms || n_bonds_read != n_bonds
        || n_angles_read != n_angles || n_dihedrals_read != n_dihedrals) {
        std::ostringstream os;
        os << "LAMMPS data: count mismatch  "
           << "atoms " << n_atoms_read << "/" << n_atoms << "  bonds "
           << n_bonds_read << "/" << n_bonds << "  angles " << n_angles_read
           << "/" << n_angles << "  dihedrals " << n_dihedrals_read << "/"
           << n_dihedrals;
        throw std::runtime_error(os.str());
    }

    frame.build_special_list();
    return frame;
}

} // namespace debug_amorphous
} // namespace ATV
