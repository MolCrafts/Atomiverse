/// @file cpu/planewave/internal/upf_reader.cpp
/// @brief UPF v1 XML-like format parser.

#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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

bool tag_open(const std::string& line, const std::string& tag) {
    auto p = line.find("<" + tag);
    if (p == std::string::npos)
        return false;
    // Ensure exact tag match: next char must be '>', ' ', '\t', '.', or digit
    auto end = p + 1 + tag.size();
    if (end < line.size()) {
        char c = line[end];
        if (c != '>' && c != ' ' && c != '\t' && c != '\r' && c != '\n'
            && c != '.' && !std::isdigit(static_cast<unsigned char>(c)))
            return false;
    }
    return line.find("/>") == std::string::npos;
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
        double d;
        while (ss >> d)
            vals.push_back(static_cast<AccReal>(d));
    }
    return vals;
}

} // namespace

void UPFData::validate_radial_mesh() const {
    if (r.size() < 3 || rab.size() != r.size())
        throw std::invalid_argument("UPF: provide matching PP_R and PP_RAB "
                                    "arrays with at least three points");
    for (std::size_t i = 0; i < r.size(); ++i)
        if (!std::isfinite(r[i]) || r[i] < 0 || !std::isfinite(rab[i])
            || rab[i] <= 0 || (i && r[i] <= r[i - 1]))
            throw std::invalid_argument(
                "UPF: PP_R must be finite, nonnegative and increasing; PP_RAB "
                "must be finite and positive");
}

AccReal UPFData::integrate_radial(std::span<const AccReal> values) const {
    // Ported from Quantum ESPRESSO qe-7.5 upflib/simpsn.f90, simpson,
    // revision 770a0b2d12928a67048e2f3da8d10d057e52179e,
    // source blob a9cfb5b33810860f53abbc30a85cb9a8a8618434.
    // Copyright (C) 2001-2012 Quantum ESPRESSO group.
    // GNU General Public License; see QE License and this repository LICENSE.
    // Even meshes use QE's endpoint correction over the full domain.
    validate_radial_mesh();
    const std::size_t n = values.size();
    if (n < 3 || n > r.size())
        throw std::invalid_argument(
            "UPF radial integral: provide at least three samples within PP_R");
    for (AccReal value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "UPF radial integral: nonfinite sample");
    AccReal sum = 0;
    for (std::size_t i = 1; i + 1 < n; ++i)
        sum += (i % 2 ? AccReal(4) : AccReal(2)) * values[i] * rab[i];
    if (n % 2)
        sum += values[0] * rab[0] + values[n - 1] * rab[n - 1];
    else
        sum += values[0] * rab[0] - AccReal(0.25) * values[n - 3] * rab[n - 3]
               + values[n - 2] * rab[n - 2]
               + AccReal(1.25) * values[n - 1] * rab[n - 1];
    return sum / AccReal(3);
}

UPFData read_upf_v1(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("UPF: cannot open " + path);

    UPFData upf;
    std::string line;

    while (std::getline(f, line)) {
        // PP_HEADER
        if (line.find("<PP_HEADER") != std::string::npos) {
            line = join_open_tag(f, line);
            auto zv = extract_attr(line, "z_valence");
            auto lm = extract_attr(line, "l_max");
            upf.pseudo_type = extract_attr(line, "pseudo_type");
            if (!zv.empty())
                upf.z_val = std::stoi(zv);
            if (!lm.empty())
                upf.l_max = std::stoi(lm);
        }
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
        // PP_NLCC
        if (tag_open(line, "PP_NLCC")) {
            upf.rho_atom = parse_values(f, "PP_NLCC");
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
                    int l = 0;
                    auto lang = extract_attr(line, "angular_momentum");
                    if (!lang.empty())
                        l = std::stoi(lang);
                    auto beta_vals = parse_values(f, "PP_BETA");
                    UPFData::KBProjector kb;
                    kb.l = l;
                    kb.beta = std::move(beta_vals);
                    const auto cutoff =
                        extract_attr(line, "cutoff_radius_index");
                    // No compact domain declaration means the supplied table
                    // itself declares the full integration domain.
                    kb.radial_points =
                        cutoff.empty()
                            ? kb.beta.size()
                            : static_cast<std::size_t>(std::stoul(cutoff));
                    if (kb.radial_points < 3
                        || kb.radial_points > kb.beta.size())
                        throw std::invalid_argument(
                            "UPF: PP_BETA cutoff_radius_index must describe at "
                            "least three supplied samples");
                    upf.kb.push_back(std::move(kb));
                }
                // Read PP_DIJ (after all betas) — full matrix
                if (tag_open(line, "PP_DIJ")) {
                    upf.d_ij = parse_values(f, "PP_DIJ");
                }
                // Read PP_Q (augmentation charges, inside PP_AUGMENTATION)
                if (tag_open(line, "PP_Q")) {
                    upf.q_ij = parse_values(f, "PP_Q");
                }
                // Read PP_RHOATOM (may appear inside or outside PP_NONLOCAL)
                if (tag_open(line, "PP_RHOATOM")) {
                    upf.rho_atom = parse_values(f, "PP_RHOATOM");
                }
            }
        }

        // PP_CHI (atomic wavefunctions, top-level outside PP_NONLOCAL)
        if (tag_open(line, "PP_CHI")) {
            line = join_open_tag(f, line);
            int l_chi = 0;
            AccReal occ = 0;
            auto lang = extract_attr(line, "l");
            auto oattr = extract_attr(line, "occupation");
            if (!lang.empty())
                l_chi = std::stoi(lang);
            if (!oattr.empty())
                occ = static_cast<AccReal>(std::stod(oattr));
            auto chi_vals = parse_values(f, "PP_CHI");
            upf.chi.push_back(std::move(chi_vals));
            upf.lchi.push_back(l_chi);
            upf.oc.push_back(occ);
            ++upf.nwfc;
        }
    }

    if (upf.r.empty())
        throw std::runtime_error("UPF: no mesh found in " + path);
    upf.validate_radial_mesh();
    if (upf.v_loc.size() != upf.r.size())
        throw std::invalid_argument("UPF: PP_LOCAL must match PP_R");
    for (const auto& projector : upf.kb)
        if (projector.beta.size() > upf.r.size())
            throw std::invalid_argument("UPF: PP_BETA exceeds PP_R");
    if (!upf.rho_atom.empty() && upf.rho_atom.size() != upf.r.size())
        throw std::invalid_argument("UPF: atomic density must match PP_R");
    for (const auto& chi : upf.chi)
        if (chi.size() != upf.r.size())
            throw std::invalid_argument(
                "UPF: atomic wavefunction must match PP_R");

    // Norm-conserving only. The plane-wave path applies no augmentation
    // charge, so an ultrasoft or PAW pseudopotential would be integrated as
    // if it were norm-conserving — E_aug still enters the total energy and
    // the result is wrong by an unbounded amount, silently. An empty
    // pseudo_type means the file predates the attribute (UPF v1); those are
    // norm-conserving by construction and stay accepted.
    if (!upf.pseudo_type.empty() && upf.pseudo_type != "NC")
        throw std::runtime_error(
            "UPF: " + path + " declares pseudo_type=\"" + upf.pseudo_type
            + "\", but the plane-wave path implements norm-conserving "
              "pseudopotentials only (no augmentation charge). Use an NC "
              "pseudopotential, e.g. an ONCVPSP one.");
    return upf;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
