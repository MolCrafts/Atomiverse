/// @file tools/gen_gpw_tables.cpp
/// @brief One-shot generator: dumps the embedded CP2K-grammar GTH/MOLOPT data
///        as static C++ tables (basis_data/gpw_tables.h). Run once; the emitted
///        header is committed and the string parsers are deleted afterwards.
///
/// HISTORICAL: the parser sources it consumed (gpw_data.cpp, cp2k_basis.cpp,
/// cp2k_io.cpp) were deleted once the tables were committed; regenerating
/// requires checking out the commit that still carries them.
/// Build (standalone, std-only):
///   g++ -std=c++20 -I include tools/gen_gpw_tables.cpp \
///       src/cpu/gpw/basis_data/gpw_data.cpp \
///       src/cpu/gpw/internal/cp2k_basis.cpp src/cpu/gpw/internal/cp2k_io.cpp \
///       -o gen_gpw_tables && ./gen_gpw_tables > \
///       include/atomiverse/cpu/gpw/basis_data/gpw_tables.h

#include <cstdio>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/basis_data/gpw_data.h"

using namespace ATV::cpu::gpw;

namespace {

int z_of(const std::string& sym) {
    if (sym == "H")
        return 1;
    if (sym == "O")
        return 8;
    if (sym == "Na")
        return 11;
    if (sym == "Cl")
        return 17;
    if (sym == "Ar")
        return 18;
    if (sym == "Zn")
        return 30;
    std::fprintf(stderr, "unknown symbol %s\n", sym.c_str());
    std::exit(1);
}

std::string ident(const std::string& sym, const std::string& name) {
    std::string s = sym + "_" + name;
    for (char& c : s)
        if (c == '-')
            c = '_';
    return s;
}

void emit_doubles(const std::string& id, const std::vector<double>& v) {
    if (v.empty())
        return; // empty C arrays are ill-formed; rows use nullptr
    std::printf("inline constexpr double %s[] = {", id.c_str());
    for (std::size_t i = 0; i < v.size(); ++i)
        std::printf("%s%.17g", i ? ", " : "", v[i]);
    std::printf("};\n");
}

void emit_ints(const std::string& id, const std::vector<int>& v) {
    std::printf("inline constexpr int %s[] = {", id.c_str());
    for (std::size_t i = 0; i < v.size(); ++i)
        std::printf("%s%d", i ? ", " : "", v[i]);
    std::printf("};\n");
}

void emit_potential(const std::string& sym, const std::string& name) {
    const GthPotential p = builtin_gth_potential(sym, name);
    const std::string id = "kPot_" + ident(sym, name);
    emit_doubles(id + "_cloc", p.cloc);
    for (std::size_t c = 0; c < p.channels.size(); ++c)
        emit_doubles(id + "_h" + std::to_string(c), p.channels[c].h);
    if (!p.channels.empty()) {
        std::printf("inline constexpr GthChannelRow %s_ch[] = {\n", id.c_str());
        for (std::size_t c = 0; c < p.channels.size(); ++c) {
            const auto& ch = p.channels[c];
            std::printf("    {%d, %.17g, %d, ", ch.l, ch.r, ch.nproj);
            if (ch.h.empty())
                std::printf("nullptr, 0},\n");
            else
                std::printf("%s_h%zu, %zu},\n", id.c_str(), c, ch.h.size());
        }
        std::printf("};\n");
    }
    std::printf("inline constexpr GthPotentialRow %s = {%d, \"%s\", \"%s\", "
                "%d, %.17g, ",
                id.c_str(), z_of(sym), sym.c_str(), name.c_str(), p.zion,
                p.rloc);
    if (p.cloc.empty())
        std::printf("nullptr, 0, ");
    else
        std::printf("%s_cloc, %zu, ", id.c_str(), p.cloc.size());
    if (p.channels.empty())
        std::printf("nullptr, 0};\n\n");
    else
        std::printf("%s_ch, %zu};\n\n", id.c_str(), p.channels.size());
}

void emit_basis(const std::string& sym, const std::string& name) {
    const GaussianBasis b = builtin_gth_basis(sym, name);
    const std::string id = "kBas_" + ident(sym, name);
    for (std::size_t s = 0; s < b.sets.size(); ++s) {
        const auto& set = b.sets[s];
        const std::string sid = id + "_s" + std::to_string(s);
        emit_doubles(sid + "_exps", set.exps);
        emit_ints(sid + "_nshell", set.nshell);
        emit_doubles(sid + "_coef", set.coef);
    }
    std::printf("inline constexpr GaussianBasisSetRow %s_sets[] = {\n",
                id.c_str());
    for (std::size_t s = 0; s < b.sets.size(); ++s) {
        const auto& set = b.sets[s];
        const std::string sid = id + "_s" + std::to_string(s);
        std::printf("    {%d, %d, %d, %d, %s_exps, %zu, %s_nshell, %zu, "
                    "%s_coef, %zu},\n",
                    set.n, set.lmin, set.lmax, set.ncoef, sid.c_str(),
                    set.exps.size(), sid.c_str(), set.nshell.size(),
                    sid.c_str(), set.coef.size());
    }
    std::printf("};\n");
    std::printf("inline constexpr GaussianBasisRow %s = {%d, \"%s\", \"%s\", "
                "%s_sets, %zu};\n\n",
                id.c_str(), z_of(sym), sym.c_str(), name.c_str(), id.c_str(),
                b.sets.size());
}

} // namespace

int main() {
    std::printf(
        "/// @file cpu/gpw/basis_data/gpw_tables.h\n"
        "/// @brief GTH pseudopotential + MOLOPT/PADE Gaussian-basis static "
        "data"
        " tables.\n"
        "///\n"
        "/// GENERATED ONCE by tools/gen_gpw_tables.cpp from the published"
        " parameter\n"
        "/// sets and committed as plain C++ data — nothing is parsed at build"
        " or run\n"
        "/// time. Provenance: GTH-PBE — Krack, TCA 114, 145 (2005); GTH-PADE "
        "—\n"
        "/// Goedecker-Teter-Hutter, PRB 54, 1703 (1996) / Hartwigsen et al.,"
        " PRB 58,\n"
        "/// 3641 (1998); MOLOPT bases — VandeVondele & Hutter, JCP 127, 114105"
        " (2007);\n"
        "/// GTH-PADE bases — the CP2K BASIS_SET distribution of the same"
        " papers.\n"
        "/// Consumed by the GthPotential / GaussianBasis lookup constructors"
        " only.\n"
        "#pragma once\n\n"
        "namespace ATV {\nnamespace cpu {\nnamespace gpw {\n"
        "namespace basis_data {\n\n"
        "/// @brief One GTH nonlocal channel: l, r_l [Bohr], nproj, h matrix"
        " (row-major).\n"
        "struct GthChannelRow {\n"
        "    int l; double r; int nproj; const double* h; unsigned long"
        " h_len;\n"
        "};\n\n"
        "/// @brief One GTH pseudopotential entry (Z + name + local/nonlocal"
        " data).\n"
        "struct GthPotentialRow {\n"
        "    int z; const char* symbol; const char* name; int zion; double"
        " rloc;\n"
        "    const double* cloc; unsigned long n_cloc;\n"
        "    const GthChannelRow* channels; unsigned long n_channels;\n"
        "};\n\n"
        "/// @brief One contraction set: shared exponents + row-major"
        " coefficients.\n"
        "struct GaussianBasisSetRow {\n"
        "    int n; int lmin; int lmax; int ncoef;\n"
        "    const double* exps; unsigned long n_exps;\n"
        "    const int* nshell; unsigned long n_nshell;\n"
        "    const double* coef; unsigned long n_coef;\n"
        "};\n\n"
        "/// @brief One contracted Gaussian basis entry (Z + name + sets).\n"
        "struct GaussianBasisRow {\n"
        "    int z; const char* symbol; const char* name;\n"
        "    const GaussianBasisSetRow* sets; unsigned long n_sets;\n"
        "};\n\n");

    emit_potential("H", "GTH-PBE-q1");
    emit_potential("O", "GTH-PBE-q6");
    emit_potential("Na", "GTH-PBE-q9");
    emit_potential("Cl", "GTH-PBE-q7");
    emit_potential("Zn", "GTH-PBE-q12");
    emit_potential("H", "GTH-PADE-q1");
    emit_potential("Ar", "GTH-PADE-q8");

    emit_basis("H", "DZVP-MOLOPT-SR-GTH");
    emit_basis("O", "DZVP-MOLOPT-SR-GTH");
    emit_basis("Na", "DZVP-MOLOPT-SR-GTH");
    emit_basis("Cl", "DZVP-MOLOPT-SR-GTH");
    emit_basis("Zn", "DZVP-MOLOPT-SR-GTH");
    emit_basis("H", "DZVP-MOLOPT-GTH");
    emit_basis("O", "DZVP-MOLOPT-GTH");
    emit_basis("H", "DZV-GTH-PADE");
    emit_basis("Ar", "DZVP-GTH-PADE");

    std::printf(
        "inline constexpr const GthPotentialRow* kGthPotentials[] = {\n"
        "    &kPot_H_GTH_PBE_q1, &kPot_O_GTH_PBE_q6, &kPot_Na_GTH_PBE_q9,\n"
        "    &kPot_Cl_GTH_PBE_q7, &kPot_Zn_GTH_PBE_q12, &kPot_H_GTH_PADE_q1,\n"
        "    &kPot_Ar_GTH_PADE_q8,\n"
        "};\n\n"
        "inline constexpr const GaussianBasisRow* kGaussianBases[] = {\n"
        "    &kBas_H_DZVP_MOLOPT_SR_GTH, &kBas_O_DZVP_MOLOPT_SR_GTH,\n"
        "    &kBas_Na_DZVP_MOLOPT_SR_GTH, &kBas_Cl_DZVP_MOLOPT_SR_GTH,\n"
        "    &kBas_Zn_DZVP_MOLOPT_SR_GTH, &kBas_H_DZVP_MOLOPT_GTH,\n"
        "    &kBas_O_DZVP_MOLOPT_GTH, &kBas_H_DZV_GTH_PADE,\n"
        "    &kBas_Ar_DZVP_GTH_PADE,\n"
        "};\n\n"
        "} // namespace basis_data\n"
        "} // namespace gpw\n} // namespace cpu\n} // namespace ATV\n");
    return 0;
}
