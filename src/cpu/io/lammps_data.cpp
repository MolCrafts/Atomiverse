/// @file cpu/io/lammps_data.cpp
/// @brief Self-contained C++ reader for LAMMPS atom_style full and molecular.
///
/// We parse the file directly rather than going through molpy because molpy's
/// atom-style auto-detection misclassifies molecular 6-column lines (third
/// token looks like a float → charge branch → mol_id becomes type). Full
/// (7-column, with q) is detected from the first Atoms data row token count.
///
/// Section boundaries follow the file format itself: a data row starts with
/// a number, a section header with a word. Nothing below relies on a list of
/// "known next sections", so a section this reader skips can never be read
/// as rows of the one before it.

#include "atomiverse/cpu/io/lammps_data.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ATV {
namespace cpu {

namespace {

std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;
    return s.substr(b, e - b);
}

/// Line without its `#` comment, trimmed.
std::string strip_comment(const std::string& s) {
    const std::size_t hash = s.find('#');
    return trim(hash == std::string::npos ? s : s.substr(0, hash));
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

/// A data/header row starts with a number; a section header with a word.
bool is_numeric_row(const std::string& s) {
    if (s.empty())
        return false;
    const char c = s[0];
    return std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+'
           || c == '.';
}

[[noreturn]] void die(const std::string& path, const std::string& msg) {
    throw std::runtime_error("load_lammps_data(" + path + "): " + msg);
}

/// Line reader that hands back non-blank lines with comments stripped.
class DataReader {
  public:
    explicit DataReader(const std::string& path) : path_(path), in_(path) {
        if (!in_)
            die(path_, "cannot open file");
    }

    /// Next non-blank, comment-stripped line; false at EOF.
    bool next(std::string& out) {
        std::string raw;
        while (std::getline(in_, raw)) {
            out = strip_comment(raw);
            if (!out.empty())
                return true;
        }
        return false;
    }

    /// Next row of a section that must have `what` rows left.
    std::string row(const std::string& what) {
        std::string line;
        if (!next(line))
            die(path_, "unexpected EOF in " + what + " section");
        if (!is_numeric_row(line))
            die(path_, what + " section ended early at '" + line + "'");
        return line;
    }

    /// Skip rows until the next section header (left in `line`) or EOF.
    void skip_section(std::string& line) {
        while (next(line))
            if (!is_numeric_row(line))
                return;
        line.clear();
    }

    std::ifstream& stream() { return in_; }

  private:
    std::string path_;
    std::ifstream in_;
};

struct Header {
    long long atoms = 0, bonds = 0, angles = 0, dihedrals = 0, impropers = 0;
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {0.0, 0.0, 0.0};
};

/// Parse the header; returns the first section header line in `line`.
Header read_header(DataReader& in, const std::string& path, std::string& line) {
    Header h;
    std::string title;
    std::getline(in.stream(), title); // first line is always the title
    static const char* const kAxes[3][2] = {
        {"xlo", "xhi"}, {"ylo", "yhi"}, {"zlo", "zhi"}};
    while (in.next(line)) {
        if (!is_numeric_row(line))
            return h;
        std::istringstream iss(line);
        std::vector<std::string> tok;
        for (std::string t; iss >> t;)
            tok.push_back(t);
        const std::string keyword = tok.size() >= 2
                                        ? tok[tok.size() - 2] + " " + tok.back()
                                        : std::string();
        const std::string last = tok.back();
        if (tok.size() == 2 && last == "atoms")
            h.atoms = std::stoll(tok[0]);
        else if (tok.size() == 2 && last == "bonds")
            h.bonds = std::stoll(tok[0]);
        else if (tok.size() == 2 && last == "angles")
            h.angles = std::stoll(tok[0]);
        else if (tok.size() == 2 && last == "dihedrals")
            h.dihedrals = std::stoll(tok[0]);
        else if (tok.size() == 2 && last == "impropers")
            h.impropers = std::stoll(tok[0]);
        else if (tok.size() == 5 && last == "yz" && std::stod(tok[0]) == 0.0
                 && std::stod(tok[1]) == 0.0 && std::stod(tok[2]) == 0.0)
            continue; // explicit zero tilt is an orthogonal box
        else if (last == "yz")
            die(path, "triclinic boxes (non-zero xy xz yz) are not supported "
                      "by this reader");
        else {
            bool axis = false;
            for (int d = 0; d < 3 && !axis; ++d)
                if (tok.size() == 4 && tok[2] == kAxes[d][0]
                    && tok[3] == kAxes[d][1]) {
                    h.lo[d] = std::stod(tok[0]);
                    h.hi[d] = std::stod(tok[1]);
                    axis = true;
                }
            // Remaining header lines ("N atom types", "N extra bond per
            // atom", ...) size LAMMPS arrays; topology rows carry their own
            // type ids and the Frame grows per atom, so they are not needed.
        }
    }
    line.clear();
    return h;
}

void read_atoms(DataReader& in, const std::string& path, Frame& frame,
                const Header& h,
                const std::unordered_map<int, double>& type_to_mass,
                std::unordered_map<long long, int>& id_to_idx) {
    StorageReal* __restrict sp_x = frame.span(kAtom_x);
    StorageReal* __restrict sp_y = frame.span(kAtom_y);
    StorageReal* __restrict sp_z = frame.span(kAtom_z);
    int* __restrict sp_type = frame.span(kAtom_type);
    StorageReal* __restrict sp_mass = frame.span(kAtom_mass);
    StorageReal* __restrict sp_q = frame.span(kAtom_charge);
    ImageInt* __restrict sp_image = frame.span(kAtom_image).data();

    int next_idx = 0;
    bool style_known = false, is_full = false;
    for (long long n = 0; n < h.atoms; ++n) {
        const std::string l = in.row("Atoms");
        std::istringstream iss(l);
        std::vector<double> v;
        for (double x; iss >> x;)
            v.push_back(x);
        if (!style_known) {
            // full: id mol type q x y z [ix iy iz]; molecular: id mol type
            // x y z [ix iy iz]. Never stream 6 fields out of a 7-field row:
            // that silently maps the charge to x.
            if (v.size() == 7 || v.size() == 10)
                is_full = true;
            else if (v.size() == 6 || v.size() == 9)
                is_full = false;
            else
                die(path, "Atoms row is neither full nor molecular: " + l);
            style_known = true;
        }
        const std::size_t want = is_full ? 7 : 6;
        if (v.size() != want && v.size() != want + 3)
            die(path, "malformed Atoms row: " + l);
        const long long id = static_cast<long long>(v[0]);
        const int atype = static_cast<int>(v[2]);
        const double q = is_full ? v[3] : 0.0;
        const std::size_t p = is_full ? 4 : 3;
        const auto mass = type_to_mass.find(atype);
        // No fallback: mass 0 makes the first integration step produce
        // inf/NaN velocities with nothing to trace it to.
        if (mass == type_to_mass.end())
            die(path, "atom type " + std::to_string(atype)
                          + " appears in Atoms but has no entry in Masses");
        if (!id_to_idx.emplace(id, next_idx).second)
            die(path, "duplicate atom id " + std::to_string(id));
        const int idx = next_idx++;
        // ATV's box spans [0, L) per axis; LAMMPS positions live in [lo, hi).
        sp_x[idx] = static_cast<StorageReal>(v[p] - h.lo[0]);
        sp_y[idx] = static_cast<StorageReal>(v[p + 1] - h.lo[1]);
        sp_z[idx] = static_cast<StorageReal>(v[p + 2] - h.lo[2]);
        sp_type[idx] = atype - 1; // 0-indexed
        sp_q[idx] = static_cast<StorageReal>(q);
        sp_mass[idx] = static_cast<StorageReal>(mass->second);
        // Trailing `ix iy iz` (LAMMPS Atom::data_atoms): the unwrapped
        // position is x + H·image, so dropping them would silently reset
        // every unwrapped consumer (MSD, FixEfield energy) to the wrapped
        // start. Flags must be integers in the ImageInt range.
        if (v.size() == want + 3)
            for (int d = 0; d < 3; ++d) {
                const double flag = v[want + static_cast<std::size_t>(d)];
                if (flag != std::trunc(flag)
                    || flag < std::numeric_limits<ImageInt>::min()
                    || flag > std::numeric_limits<ImageInt>::max())
                    die(path, "image flag is not an int32 integer: " + l);
                sp_image[3 * static_cast<std::size_t>(idx) + d] =
                    static_cast<ImageInt>(flag);
            }
    }
}

} // namespace

Frame load_lammps_data(const std::string& path) {
    DataReader in(path);
    std::string line;
    const Header h = read_header(in, path, line);
    if (h.atoms <= 0)
        die(path, "header reports zero atoms");
    Frame frame(static_cast<int>(h.atoms));
    // A data file names force-field types, never elements: the Atoms type
    // column lands in kAtom_type and the Frame carries no species column (a
    // mass is not an element — Z is never guessed from it). A consumer that
    // needs Z throws with the how-to (`atom_field_how_to`).
    frame.block("atoms").erase(kAtom_atomic_number.key);

    std::unordered_map<int, double> type_to_mass;
    std::unordered_map<long long, int> id_to_idx;
    id_to_idx.reserve(static_cast<std::size_t>(h.atoms));
    bool have_atoms = false;

    // LAMMPS atom id → 0-indexed row; the row IS the global tag (the Frame
    // ctor sets tag[i] == i), and `add_*` route each tuple to its owner atom.
    auto tag_of = [&](long long id, const char* what) -> TagInt {
        if (!have_atoms)
            die(path, std::string(what) + " section before Atoms");
        const auto it = id_to_idx.find(id);
        if (it == id_to_idx.end())
            die(path, std::string(what) + " row names unknown atom id "
                          + std::to_string(id));
        return static_cast<TagInt>(it->second);
    };
    auto read_tuples = [&](const char* what, long long count, int arity,
                           auto add) {
        for (long long n = 0; n < count; ++n) {
            const std::string l = in.row(what);
            std::istringstream iss(l);
            long long id = 0, type = 0, a[4] = {0, 0, 0, 0};
            if (!(iss >> id >> type))
                die(path, std::string("malformed ") + what + " row: " + l);
            for (int k = 0; k < arity; ++k)
                if (!(iss >> a[k]))
                    die(path, std::string("malformed ") + what + " row: " + l);
            TagInt t[4] = {0, 0, 0, 0};
            for (int k = 0; k < arity; ++k)
                t[k] = tag_of(a[k], what);
            add(t, static_cast<int>(type) - 1);
        }
    };

    while (!line.empty()) {
        const std::string section = line;
        if (section == "Masses") {
            // `type mass` rows up to the next section header.
            while (in.next(line) && is_numeric_row(line)) {
                std::istringstream iss(line);
                int t = 0;
                double m = 0.0;
                if (!(iss >> t >> m) || !(m > 0.0))
                    die(path, "malformed Masses row: " + line);
                type_to_mass[t] = m;
            }
            if (in.stream().eof() && is_numeric_row(line))
                line.clear();
            continue;
        }
        if (section == "Atoms") {
            read_atoms(in, path, frame, h, type_to_mass, id_to_idx);
            have_atoms = true;
            frame.clear_tag_index();
            frame.rebuild_tag_index(0, frame.nlocal());
        } else if (section == "Velocities") {
            if (!have_atoms)
                die(path, "Velocities section before Atoms");
            // LAMMPS `units real` velocity is Å/fs, identical to ATV units.
            StorageReal* __restrict vx = frame.span(kAtom_vx);
            StorageReal* __restrict vy = frame.span(kAtom_vy);
            StorageReal* __restrict vz = frame.span(kAtom_vz);
            for (long long n = 0; n < h.atoms; ++n) {
                const std::string l = in.row("Velocities");
                std::istringstream iss(l);
                long long id = 0;
                double a = 0, b = 0, c = 0;
                if (!(iss >> id >> a >> b >> c))
                    die(path, "malformed Velocities row: " + l);
                const int row = static_cast<int>(tag_of(id, "Velocities"));
                vx[row] = static_cast<StorageReal>(a);
                vy[row] = static_cast<StorageReal>(b);
                vz[row] = static_cast<StorageReal>(c);
            }
        } else if (section == "Bonds") {
            read_tuples("Bonds", h.bonds, 2, [&](const TagInt* t, int type) {
                frame.add_bond(t[0], t[1], type);
            });
        } else if (section == "Angles") {
            read_tuples("Angles", h.angles, 3, [&](const TagInt* t, int type) {
                frame.add_angle(t[0], t[1], t[2], type);
            });
        } else if (section == "Dihedrals") {
            read_tuples("Dihedrals", h.dihedrals, 4,
                        [&](const TagInt* t, int type) {
                            frame.add_dihedral(t[0], t[1], t[2], t[3], type);
                        });
        } else if (section == "Impropers") {
            read_tuples("Impropers", h.impropers, 4,
                        [&](const TagInt* t, int type) {
                            frame.add_improper(t[0], t[1], t[2], t[3], type);
                        });
        } else if (ends_with(section, "Type Labels")
                   || ends_with(section, "Coeffs")) {
            // Numeric type ids are authoritative here; parameters come from
            // the force-field components (e.g. load_lammps_coeffs), never
            // from the data file.
            in.skip_section(line);
            continue;
        } else {
            die(path, "unsupported section '" + section + "'");
        }
        if (!in.next(line))
            line.clear();
        else if (is_numeric_row(line))
            die(path, "more rows than the header declares after '" + section
                          + "': " + line);
    }
    if (!have_atoms)
        die(path, "no Atoms section");

    // ── Box: orthogonal ────────────────────────────────────────────────
    Mat3 cell;
    for (int d = 0; d < 3; ++d)
        cell(d, d) = h.hi[d] - h.lo[d];
    frame.box.set_cell(cell);

    // Build 1-2 / 1-3 / 1-4 exclusion list once topology is scattered.
    frame.build_special_list();
    return frame;
}

} // namespace cpu
} // namespace ATV
