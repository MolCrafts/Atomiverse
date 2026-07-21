/// @file cpu/pw/internal/upf_reader.cpp
/// @brief UPF v1 XML-like format parser.

#include "atomiverse/cpu/pw/internal/upf_reader.h"
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
            if (!zv.empty())
                upf.z_val = std::stoi(zv);
            if (!lm.empty())
                upf.l_max = std::stoi(lm);
        }
        // PP_MESH
        if (tag_open(line, "PP_MESH")) {
            // read PP_R
            while (std::getline(f, line)) {
                if (tag_open(line, "PP_R")) {
                    upf.r = parse_values(f, "PP_R");
                    break;
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
    return upf;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
