/// @file cpu/planewave/internal/upf_reader.cpp
/// @brief UPF v1 XML-like format parser.

#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include "atomiverse/molpy.hpp"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

std::string extract_attr(const std::string& line, const std::string& attr) {
    // Match " attr=\"" (space-prefixed) to avoid matching substrings
    // like "l=\"" inside "label=\"3S\""
    std::string pat = " " + attr + "=\"";
    auto p = line.find(pat);
    if (p == std::string::npos)
        return "";
    p += pat.size(); // skip past ' attr="'
    auto q = line.find("\"", p);
    if (q == std::string::npos)
        return "";
    return line.substr(p, q - p);
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

void require_false_attribute(const std::string& line, const char* attribute) {
    auto value = trim(extract_attr(line, attribute));
    for (char& c : value)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!value.empty() && value != "f" && value != "false" && value != "0")
        throw std::invalid_argument(std::string("UPF: unsupported channel ")
                                    + attribute);
}

double number_attribute(const std::string& line, const char* attribute) {
    const auto value = trim(extract_attr(line, attribute));
    std::size_t used = 0;
    double number = 0;
    try {
        number = std::stod(value, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string("UPF: invalid numeric attribute ") + attribute);
    }
    if (used != value.size() || !std::isfinite(number))
        throw std::invalid_argument(
            std::string("UPF: invalid numeric attribute ") + attribute);
    return number;
}

int integral_attribute(const std::string& line, const char* attribute,
                       bool required) {
    if (!required && trim(extract_attr(line, attribute)).empty())
        return 0;
    const auto value = number_attribute(line, attribute);
    if (value != std::round(value) || value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            std::string("UPF: require integral attribute ") + attribute);
    return static_cast<int>(value);
}

int atomic_number(const std::string& symbol) {
    // Reuse molrs' canonical periodic table rather than maintain a second
    // symbol-to-Z mapping. This conversion runs once when admitting the file.
    const auto name = trim(symbol);
    if (name.empty())
        throw std::invalid_argument("UPF: missing element identity");
    auto frame = molrs::frame_new();
    const std::vector<rust::String> symbols{rust::String(name)};
    try {
        molrs::frame_set_column_str(*frame, "atoms", "element",
                                    {symbols.data(), symbols.size()});
        molrs::frame_fill_atomic_numbers(*frame);
        return static_cast<int>(
            molrs::frame_column_u64(*frame, "atoms", "atomic_number")[0]);
    } catch (const std::exception&) {
        throw std::invalid_argument("UPF: invalid element identity " + name);
    }
}

bool tag_open(const std::string& line, const std::string& tag,
              bool include_empty = false) {
    auto p = line.find("<" + tag);
    if (p == std::string::npos)
        return false;
    // Ensure exact tag match: next char must be '>', ' ', '\t', '.', or digit
    auto end = p + 1 + tag.size();
    if (end < line.size()) {
        char c = line[end];
        if (c != '>' && c != '/' && c != ' ' && c != '\t' && c != '\r'
            && c != '\n' && c != '.'
            && !std::isdigit(static_cast<unsigned char>(c)))
            return false;
    }
    return include_empty || line.find("/>") == std::string::npos;
}

bool tag_close(const std::string& line, const std::string& tag) {
    return line.find("</" + tag) != std::string::npos;
}

// UPF v2 writers (ONCVPSP, pslibrary) put one attribute per line in the
// opening tag. Join lines until the '>' that ends the opening tag so
// attribute extraction sees the full tag and value parsing starts after
// it. Single-line tags pass through unchanged.
std::string join_open_tag(std::istream& in, std::string line) {
    while (line.find('>') == std::string::npos) {
        std::string next;
        if (!std::getline(in, next))
            break;
        line += ' ';
        line += next;
    }
    return line;
}

std::vector<AccReal> parse_values(std::istream& in,
                                  const std::string& end_tag) {
    std::vector<AccReal> vals;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find(end_tag) != std::string::npos)
            break;
        // trim
        auto s = line;
        while (!s.empty() && (s[0] == ' ' || s[0] == '\t'))
            s.erase(0, 1);
        if (s.empty())
            continue;
        std::istringstream ss(s);
        ss >> std::ws;
        while (!ss.eof()) {
            double d;
            if (!(ss >> d) || !std::isfinite(d))
                throw std::invalid_argument("UPF: invalid numeric sample in "
                                            + end_tag);
            vals.push_back(static_cast<AccReal>(d));
            ss >> std::ws;
        }
    }
    return vals;
}

} // namespace

NcPotentialData read_upf_v1(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("UPF: cannot open " + path);

    NcPotentialData upf;
    bool header_seen = false;
    std::string line;

    while (std::getline(f, line)) {
        // Admission fixes the chemical identity and representation before
        // reading radial tables; a scalar NC operator cannot emulate NLCC,
        // spin-orbit, ultrasoft or PAW physics.
        if (line.find("<PP_HEADER") != std::string::npos) {
            line = join_open_tag(f, line);
            if (trim(extract_attr(line, "pseudo_type")) != "NC")
                throw std::invalid_argument(
                    "UPF: require an explicitly declared scalar NC potential");
            for (const char* flag :
                 {"is_ultrasoft", "is_paw", "has_so", "core_correction"})
                require_false_attribute(line, flag);
            const auto relativistic = trim(extract_attr(line, "relativistic"));
            if (!relativistic.empty() && relativistic != "no"
                && relativistic != "scalar")
                throw std::invalid_argument(
                    "UPF: spin-orbit relativistic potential is unsupported");
            upf.atomic_number = atomic_number(extract_attr(line, "element"));
            upf.z_val = integral_attribute(line, "z_valence", true);
            upf.l_max = integral_attribute(line, "l_max", false);
            header_seen = true;
        }
        if (tag_open(line, "PP_NLCC", true)
            || tag_open(line, "PP_AUGMENTATION", true)
            || tag_open(line, "PP_SPIN_ORB", true)
            || tag_open(line, "PP_Q", true))
            throw std::invalid_argument(
                "UPF: core correction, augmentation and spin-orbit channels "
                "are unsupported");
        // PP_MESH
        if (tag_open(line, "PP_MESH")) {
            while (std::getline(f, line)) {
                if (tag_open(line, "PP_R")) {
                    line = join_open_tag(f, line);
                    upf.r = parse_values(f, "PP_R");
                } else if (tag_open(line, "PP_RAB")) {
                    line = join_open_tag(f, line);
                    upf.rab = parse_values(f, "PP_RAB");
                }
                if (tag_close(line, "PP_MESH"))
                    break;
            }
        }
        // PP_LOCAL
        if (tag_open(line, "PP_LOCAL")) {
            upf.v_loc = parse_values(f, "PP_LOCAL");
        }
        // PP_RHOATOM (atomic pseudocharge — used as initial density guess)
        if (tag_open(line, "PP_RHOATOM")) {
            upf.rho_atom = parse_values(f, "PP_RHOATOM");
        }
        // PP_NONLOCAL
        if (tag_open(line, "PP_NONLOCAL")) {
            while (std::getline(f, line)) {
                if (tag_close(line, "PP_NONLOCAL"))
                    break;
                // Read PP_BETA sections
                if (tag_open(line, "PP_BETA")) {
                    line = join_open_tag(f, line);
                    const int l =
                        integral_attribute(line, "angular_momentum", true);
                    auto beta_vals = parse_values(f, "PP_BETA");
                    NcPotentialData::KBProjector kb;
                    kb.l = l;
                    kb.beta = std::move(beta_vals);
                    const auto cutoff =
                        trim(extract_attr(line, "cutoff_radius_index"));
                    const int cutoff_points =
                        cutoff.empty() ? 0
                                       : integral_attribute(
                                             line, "cutoff_radius_index", true);
                    if (!cutoff.empty() && cutoff_points < 3)
                        throw std::invalid_argument(
                            "UPF: projector cutoff must be an integral index "
                            ">= 3");
                    // No compact domain declaration means the supplied table
                    // itself declares the full integration domain.
                    kb.radial_points =
                        cutoff.empty()
                            ? kb.beta.size()
                            : static_cast<std::size_t>(cutoff_points);
                    if (kb.radial_points < 3
                        || kb.radial_points > kb.beta.size())
                        throw std::invalid_argument(
                            "UPF: PP_BETA cutoff_radius_index must describe at "
                            "least three supplied samples");
                    kb.beta.resize(kb.radial_points);
                    upf.kb.push_back(std::move(kb));
                }
                // Read PP_DIJ (after all betas) — full matrix
                if (tag_open(line, "PP_DIJ")) {
                    upf.d_ij = parse_values(f, "PP_DIJ");
                }
                if (tag_open(line, "PP_AUGMENTATION", true)
                    || tag_open(line, "PP_Q", true)
                    || tag_open(line, "PP_NLCC", true)
                    || tag_open(line, "PP_SPIN_ORB", true))
                    throw std::invalid_argument(
                        "UPF: augmentation channels are unsupported");
                // Read PP_RHOATOM (may appear inside or outside PP_NONLOCAL)
                if (tag_open(line, "PP_RHOATOM")) {
                    upf.rho_atom = parse_values(f, "PP_RHOATOM");
                }
            }
        }

        // PP_CHI (atomic wavefunctions, top-level outside PP_NONLOCAL)
        if (tag_open(line, "PP_CHI")) {
            line = join_open_tag(f, line);
            const int l_chi = integral_attribute(line, "l", true);
            const AccReal occ =
                static_cast<AccReal>(number_attribute(line, "occupation"));
            auto chi_vals = parse_values(f, "PP_CHI");
            upf.chi.push_back(std::move(chi_vals));
            upf.lchi.push_back(l_chi);
            upf.oc.push_back(occ);
            ++upf.nwfc;
        }
    }

    if (!header_seen)
        throw std::invalid_argument("UPF: missing NC header in " + path);
    // Atomic-unit canonical data begin here. The numerical operators never
    // know the file's Rydberg convention.
    for (auto& value : upf.v_loc)
        value *= AccReal(0.5);
    for (auto& value : upf.d_ij)
        value *= AccReal(0.5);
    upf.validate();
    return upf;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
