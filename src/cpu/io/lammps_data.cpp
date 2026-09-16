/// @file cpu/io/lammps_data.cpp
/// @brief Self-contained C++ reader for LAMMPS atom_style full and molecular.
///
/// We parse the file directly rather than going through molrs because molrs's
/// atom-style auto-detection misclassifies molecular 6-column lines (third
/// token looks like a float → charge branch → mol_id becomes type). Full
/// (7-column, with q) is detected from the first Atoms data row token count.

#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ATV {
namespace cpu {

namespace {

bool starts_with(const std::string& s, const char* prefix) {
    const std::size_t n = std::char_traits<char>::length(prefix);
    return s.size() >= n && std::strncmp(s.data(), prefix, n) == 0;
}

std::string ltrim(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    return s.substr(i);
}

// Throw with file context.
[[noreturn]] void die(const std::string& path, const std::string& msg) {
    throw std::runtime_error("load_lammps_data(" + path + "): " + msg);
}

// Read the next non-blank, non-comment line. Returns false at EOF.
bool next_data_line(std::ifstream& in, std::string& out) {
    while (std::getline(in, out)) {
        const std::string t = ltrim(out);
        if (t.empty() || t[0] == '#')
            continue;
        out = t;
        return true;
    }
    return false;
}

} // namespace

Frame load_lammps_data(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        die(path, "cannot open file");

    // ── Pass 1: header ─────────────────────────────────────────────────
    int n_atoms = 0, n_bonds = 0, n_angles = 0, n_dihedrals = 0;
    double xlo = 0.0, xhi = 0.0, ylo = 0.0, yhi = 0.0, zlo = 0.0, zhi = 0.0;

    std::string line;
    // Skip title line.
    std::getline(in, line);

    while (next_data_line(in, line)) {
        // Section headers end the header block.
        if (starts_with(line, "Masses") || starts_with(line, "Atoms")
            || starts_with(line, "Bonds") || starts_with(line, "Angles")
            || starts_with(line, "Dihedrals") || starts_with(line, "Impropers")
            || starts_with(line, "Velocities")) {
            // Push the section line back by seeking; simpler: process it as
            // the section dispatcher below.
            break;
        }
        std::istringstream iss(line);
        if (line.find("atoms") != std::string::npos
            && line.find("atom types") == std::string::npos) {
            iss >> n_atoms;
        } else if (line.find("bonds") != std::string::npos
                   && line.find("bond types") == std::string::npos) {
            iss >> n_bonds;
        } else if (line.find("angles") != std::string::npos
                   && line.find("angle types") == std::string::npos) {
            iss >> n_angles;
        } else if (line.find("dihedrals") != std::string::npos
                   && line.find("dihedral types") == std::string::npos) {
            iss >> n_dihedrals;
        } else if (line.find("xlo xhi") != std::string::npos) {
            iss >> xlo >> xhi;
        } else if (line.find("ylo yhi") != std::string::npos) {
            iss >> ylo >> yhi;
        } else if (line.find("zlo zhi") != std::string::npos) {
            iss >> zlo >> zhi;
        }
        // Ignore "N atom types", "N bond types", etc. — single type assumed.
    }

    if (n_atoms <= 0)
        die(path, "header reports zero atoms");

    Frame frame(n_atoms);

    std::unordered_map<int, double> type_to_mass;

    // ── Pass 2: sections — `line` currently holds the first section header
    auto consume_n = [&](int n, auto handler) {
        int read = 0;
        while (read < n) {
            if (!next_data_line(in, line))
                die(path, "unexpected EOF in section");
            handler(line);
            ++read;
        }
    };

    auto is_section = [&](const std::string& s) {
        return starts_with(line, s.c_str());
    };

    // LAMMPS atom-id (1-indexed) → 0-indexed row in the atoms Block.
    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(n_atoms);
    int next_idx = 0;

    do {
        if (is_section("Masses")) {
            // Masses: <type> <mass>
            // Type count is unbounded in the header; consume until next header.
            while (next_data_line(in, line)) {
                if (starts_with(line, "Atoms") || starts_with(line, "Bonds")
                    || starts_with(line, "Angles")
                    || starts_with(line, "Dihedrals")
                    || starts_with(line, "Velocities")
                    || starts_with(line, "Impropers")) {
                    break;
                }
                std::istringstream iss(line);
                int t;
                double m;
                if (iss >> t >> m)
                    type_to_mass[t] = m;
            }
            continue; // `line` now holds the next section header
        }
        if (is_section("Atoms")) {
            // Detect style from the first data row token count:
            //   full:      id mol type q x y z   (≥7 tokens)
            //   molecular: id mol type x y z     (6 tokens)
            // Do NOT stream 6-field extraction on a 7-field line — that
            // silently maps charge → x and shifts coordinates.
            if (!next_data_line(in, line))
                die(path, "unexpected EOF in Atoms section");
            {
                std::istringstream count_iss(line);
                std::string tok;
                int ntok = 0;
                while (count_iss >> tok)
                    ++ntok;
                if (ntok < 6)
                    die(path, "malformed Atoms row (need full or molecular): "
                                  + line);
            }
            std::istringstream probe(line);
            int id0 = 0, mol0 = 0, type0 = 0;
            double a = 0, b = 0, c = 0, d = 0;
            if (!(probe >> id0 >> mol0 >> type0 >> a >> b >> c))
                die(path, "malformed Atoms row: " + line);
            const bool is_full = static_cast<bool>(probe >> d);

            StorageReal* __restrict sp_x = frame.span(kAtom_x);
            StorageReal* __restrict sp_y = frame.span(kAtom_y);
            StorageReal* __restrict sp_z = frame.span(kAtom_z);
            int* __restrict sp_type = frame.span(kAtom_type);
            StorageReal* __restrict sp_mass = frame.span(kAtom_mass);
            StorageReal* __restrict sp_q = frame.span(kAtom_charge);

            auto store_atom = [&](int id, int /*mol_id*/, int atype, double q,
                                  double x, double y, double z) {
                const int idx = next_idx++;
                id_to_idx[id] = idx;
                // ATV's box spans [0, L) per axis; LAMMPS positions live in
                // [lo, hi). Shift by the header origin (harmless when lo=0).
                sp_x[idx] = static_cast<StorageReal>(x - xlo);
                sp_y[idx] = static_cast<StorageReal>(y - ylo);
                sp_z[idx] = static_cast<StorageReal>(z - zlo);
                sp_type[idx] = atype - 1; // 0-indexed
                sp_q[idx] = static_cast<StorageReal>(q);
                auto it = type_to_mass.find(atype);
                // No fallback: mass 0 makes the first integration step
                // produce inf/NaN velocities with nothing to trace it to.
                if (it == type_to_mass.end())
                    throw std::runtime_error(
                        "load_lammps_data: atom type " + std::to_string(atype)
                        + " appears in the Atoms section but has no entry in "
                          "Masses. Add it to the Masses section.");
                sp_mass[idx] = static_cast<StorageReal>(it->second);
            };

            // First row (already in `line`).
            if (is_full) {
                store_atom(id0, mol0, type0, a, b, c, d);
            } else {
                store_atom(id0, mol0, type0, /*q=*/0.0, a, b, c);
            }

            for (int i = 1; i < n_atoms; ++i) {
                if (!next_data_line(in, line))
                    die(path, "unexpected EOF in Atoms section");
                std::istringstream iss(line);
                int id, mol_id, atype;
                if (is_full) {
                    double q, x, y, z;
                    if (!(iss >> id >> mol_id >> atype >> q >> x >> y >> z))
                        die(path, "malformed Atoms row: " + line);
                    store_atom(id, mol_id, atype, q, x, y, z);
                } else {
                    double x, y, z;
                    if (!(iss >> id >> mol_id >> atype >> x >> y >> z))
                        die(path, "malformed Atoms row: " + line);
                    store_atom(id, mol_id, atype, 0.0, x, y, z);
                }
            }
            if (!next_data_line(in, line))
                line.clear();
            continue;
        }
        // Topology sections — scatter into the per-atom Block layout.
        // `id_to_idx` maps the LAMMPS atom-id (1-indexed) to the row in
        // the atoms Block (0-indexed); the row IS the global tag (Frame ctor
        // sets tag[i] == i). The `add_*` helpers route the entry to the
        // LAMMPS owner atom's slot via the Frame tag index.
        if (is_section("Bonds")) {
            frame.clear_tag_index();
            frame.rebuild_tag_index(0, frame.nlocal());
            consume_n(n_bonds, [&](const std::string& l) {
                std::istringstream iss(l);
                int bid, btype, ai, aj;
                if (!(iss >> bid >> btype >> ai >> aj)) {
                    die(path, "malformed Bonds row: " + l);
                }
                frame.add_bond(id_to_idx[ai], id_to_idx[aj], btype - 1);
            });
            if (!next_data_line(in, line))
                line.clear();
            continue;
        }
        if (is_section("Angles")) {
            consume_n(n_angles, [&](const std::string& l) {
                std::istringstream iss(l);
                int aid, atype, ai, aj, ak;
                if (!(iss >> aid >> atype >> ai >> aj >> ak)) {
                    die(path, "malformed Angles row: " + l);
                }
                frame.add_angle(id_to_idx[ai], id_to_idx[aj], id_to_idx[ak],
                                atype - 1);
            });
            if (!next_data_line(in, line))
                line.clear();
            continue;
        }
        if (is_section("Dihedrals")) {
            consume_n(n_dihedrals, [&](const std::string& l) {
                std::istringstream iss(l);
                int did, dtype, ai, aj, ak, al;
                if (!(iss >> did >> dtype >> ai >> aj >> ak >> al)) {
                    die(path, "malformed Dihedrals row: " + l);
                }
                frame.add_dihedral(id_to_idx[ai], id_to_idx[aj], id_to_idx[ak],
                                   id_to_idx[al], dtype - 1);
            });
            if (!next_data_line(in, line))
                line.clear();
            continue;
        }
        if (is_section("Velocities")) {
            // Velocities: <id> <vx> <vy> <vz> — LAMMPS `units real` velocity
            // is Å/fs, identical to ATV engineering units; copy through.
            if (id_to_idx.empty())
                die(path, "Velocities section before Atoms is not supported");
            StorageReal* __restrict sp_vx = frame.span(kAtom_vx);
            StorageReal* __restrict sp_vy = frame.span(kAtom_vy);
            StorageReal* __restrict sp_vz = frame.span(kAtom_vz);
            consume_n(n_atoms, [&](const std::string& l) {
                std::istringstream iss(l);
                int id;
                double vx, vy, vz;
                if (!(iss >> id >> vx >> vy >> vz))
                    die(path, "malformed Velocities row: " + l);
                const auto it = id_to_idx.find(id);
                if (it == id_to_idx.end())
                    die(path, "Velocities row for unknown atom id "
                                  + std::to_string(id));
                sp_vx[it->second] = static_cast<StorageReal>(vx);
                sp_vy[it->second] = static_cast<StorageReal>(vy);
                sp_vz[it->second] = static_cast<StorageReal>(vz);
            });
            if (!next_data_line(in, line))
                line.clear();
            continue;
        }
        // Skip Impropers — no improper styles are wired into this loader yet.
        if (is_section("Impropers")) {
            // Best-effort skip: advance until the next section header or EOF.
            while (next_data_line(in, line)) {
                if (starts_with(line, "Atoms") || starts_with(line, "Bonds")
                    || starts_with(line, "Angles")
                    || starts_with(line, "Dihedrals")
                    || starts_with(line, "Masses")
                    || starts_with(line, "Velocities")
                    || starts_with(line, "Impropers"))
                    break;
            }
            continue;
        }
        // Unknown section header — bail out.
        break;
    } while (!line.empty());

    // ── Box: orthogonal ────────────────────────────────────────────────
    Mat3 cell;
    cell(0, 0) = xhi - xlo;
    cell(1, 1) = yhi - ylo;
    cell(2, 2) = zhi - zlo;
    frame.box.set_cell(cell);

    // Build 1-2 / 1-3 / 1-4 exclusion list once topology is scattered.
    frame.build_special_list();

    return frame;
}

} // namespace cpu
} // namespace ATV
