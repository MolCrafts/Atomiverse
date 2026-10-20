/// @file cpu/io/lammps_coeffs.cpp
/// @brief LAMMPS `units real` coefficient-file parser.

#include "atomiverse/cpu/io/lammps_coeffs.h"
#include "atomiverse/cpu/io/lammps_real.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {

namespace {

[[noreturn]] void die(const std::string& path, const std::string& msg) {
    throw std::runtime_error("load_lammps_coeffs(" + path + "): " + msg);
}

/// Split one line into whitespace tokens, dropping `#` comments.
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> toks;
    std::istringstream iss(line);
    std::string t;
    while (iss >> t) {
        if (t[0] == '#')
            break;
        toks.push_back(t);
    }
    return toks;
}

double to_num(const std::string& path, const std::string& tok,
              const std::string& ctx) {
    try {
        std::size_t pos = 0;
        const double v = std::stod(tok, &pos);
        if (pos != tok.size())
            die(path, ctx + ": trailing characters in number '" + tok + "'");
        return v;
    } catch (const std::logic_error&) {
        die(path, ctx + ": expected a number, got '" + tok + "'");
    }
}

int to_type(const std::string& path, const std::string& tok,
            const std::string& ctx) {
    const double v = to_num(path, tok, ctx);
    const int t = static_cast<int>(v);
    if (t < 1 || static_cast<double>(t) != v)
        die(path,
            ctx + ": type id must be a positive integer, got '" + tok + "'");
    return t - 1; // 0-based
}

/// Grow `v` so index `t` is addressable, filling with `fill`.
template <typename T>
void ensure_size(std::vector<T>& v, int t, const T& fill) {
    if (static_cast<int>(v.size()) <= t)
        v.resize(static_cast<std::size_t>(t) + 1, fill);
}

} // namespace

LammpsCoeffs load_lammps_coeffs(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        die(path, "cannot open file");

    LammpsCoeffs out;
    // dihedral_style bookkeeping: which style each dihedral_coeff line uses.
    // For `hybrid`, each row names its style; otherwise the declared one.
    // "charmm" | "multi/harmonic" | "fourier" | "" (hybrid)
    std::string dihedral_default;
    bool dihedral_hybrid = false;
    auto supported_dihedral = [](const std::string& st) {
        return st == "charmm" || st == "multi/harmonic" || st == "fourier";
    };

    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const auto toks = tokenize(line);
        if (toks.empty())
            continue;
        const std::string& cmd = toks[0];
        const std::string where = "line " + std::to_string(lineno);

        auto need = [&](std::size_t n) {
            if (toks.size() < n)
                die(path, where + ": '" + cmd + "' needs at least "
                              + std::to_string(n - 1) + " arguments");
        };

        if (cmd == "pair_style") {
            need(2);
            out.pair_style = toks[1];
            if (toks[1] == "lj/cut" || toks[1] == "lj/cut/coul/long"
                || toks[1] == "lj/cut/coul/cut") {
                if (toks.size() > 2)
                    out.pair_cut_lj = to_num(path, toks[2], where);
                out.pair_cut_coul = toks.size() > 3
                                        ? to_num(path, toks[3], where)
                                        : out.pair_cut_lj;
            } else {
                throw std::invalid_argument(
                    "load_lammps_coeffs(" + path + "): " + where
                    + ": unsupported pair_style '" + toks[1]
                    + "' — supported: lj/cut, lj/cut/coul/cut, "
                      "lj/cut/coul/long");
            }
        } else if (cmd == "pair_coeff") {
            need(5);
            const int ti = to_type(path, toks[1], where);
            const int tj = to_type(path, toks[2], where);
            out.lj.push_back(LJTypeCoeff{
                ti, tj, kcal_per_mol_to_ev(to_num(path, toks[3], where)),
                to_num(path, toks[4], where)});
        } else if (cmd == "bond_style") {
            need(2);
            if (toks[1] != "harmonic")
                throw std::invalid_argument("load_lammps_coeffs(" + path
                                            + "): " + where
                                            + ": unsupported bond_style '"
                                            + toks[1] + "' — only harmonic");
        } else if (cmd == "bond_coeff") {
            need(4);
            const int t = to_type(path, toks[1], where);
            ensure_size(out.bond_K, t, 0.0);
            ensure_size(out.bond_r0, t, 0.0);
            out.bond_K[t] = kcal_per_mol_to_ev(to_num(path, toks[2], where));
            out.bond_r0[t] = to_num(path, toks[3], where);
        } else if (cmd == "angle_style") {
            need(2);
            if (toks[1] != "harmonic")
                throw std::invalid_argument("load_lammps_coeffs(" + path
                                            + "): " + where
                                            + ": unsupported angle_style '"
                                            + toks[1] + "' — only harmonic");
        } else if (cmd == "angle_coeff") {
            need(4);
            const int t = to_type(path, toks[1], where);
            ensure_size(out.angle_K, t, 0.0);
            ensure_size(out.angle_theta0, t, 0.0);
            out.angle_K[t] = kcal_per_mol_to_ev(to_num(path, toks[2], where));
            out.angle_theta0[t] = degree_to_rad(to_num(path, toks[3], where));
        } else if (cmd == "dihedral_style") {
            need(2);
            if (toks[1] == "hybrid") {
                dihedral_hybrid = true;
                for (std::size_t i = 2; i < toks.size(); ++i) {
                    if (!supported_dihedral(toks[i]))
                        throw std::invalid_argument(
                            "load_lammps_coeffs(" + path + "): " + where
                            + ": unsupported hybrid dihedral member '" + toks[i]
                            + "' — only charmm, multi/harmonic, fourier");
                }
            } else if (supported_dihedral(toks[1])) {
                dihedral_default = toks[1];
            } else {
                throw std::invalid_argument(
                    "load_lammps_coeffs(" + path + "): " + where
                    + ": unsupported dihedral_style '" + toks[1]
                    + "' — only charmm, multi/harmonic, fourier, or hybrid");
            }
        } else if (cmd == "dihedral_coeff") {
            need(3);
            const int t = to_type(path, toks[1], where);
            std::string style = dihedral_default;
            std::size_t arg0 = 2;
            if (dihedral_hybrid) {
                style = toks[2];
                arg0 = 3;
            }
            // Both tables always cover every dihedral type; the style that
            // does not own type t keeps exact-zero coefficients there.
            ensure_size(out.dihedral_charmm_K, t, 0.0);
            ensure_size(out.dihedral_charmm_n, t, 1);
            ensure_size(out.dihedral_charmm_delta, t, 0.0);
            ensure_size(out.dihedral_multi_Cn, t, {0.0});
            ensure_size(out.dihedral_fourier, t, {});
            if (style == "fourier") {
                // dihedral_coeff t m K1 n1 d1 ... Km nm dm
                if (toks.size() < arg0 + 1)
                    die(path, where + ": dihedral_coeff fourier needs m");
                const double m = to_num(path, toks[arg0], where);
                if (m < 1 || m != static_cast<int>(m))
                    die(path, where
                                  + ": fourier term count m must be a "
                                    "positive integer");
                const std::size_t nterm = static_cast<std::size_t>(m);
                if (toks.size() != arg0 + 1 + 3 * nterm)
                    die(path, where
                                  + ": dihedral_coeff fourier needs exactly m "
                                    "triples K n d");
                std::vector<TorsionTerm> terms;
                for (std::size_t k = 0; k < nterm; ++k) {
                    const std::size_t b = arg0 + 1 + 3 * k;
                    const double n = to_num(path, toks[b + 1], where);
                    if (n < 0 || n != static_cast<int>(n))
                        die(path, where
                                      + ": fourier multiplicity must be a "
                                        "non-negative integer");
                    terms.push_back(TorsionTerm{
                        kcal_per_mol_to_ev(to_num(path, toks[b], where)),
                        static_cast<int>(n),
                        degree_to_rad(to_num(path, toks[b + 2], where))});
                }
                out.dihedral_fourier[t] = std::move(terms);
                out.has_fourier = true;
            } else if (style == "charmm") {
                if (toks.size() < arg0 + 4)
                    die(path, where
                                  + ": dihedral_coeff charmm needs "
                                    "K n d weight");
                const double K =
                    kcal_per_mol_to_ev(to_num(path, toks[arg0], where));
                const double n = to_num(path, toks[arg0 + 1], where);
                const double d = to_num(path, toks[arg0 + 2], where);
                const double w = to_num(path, toks[arg0 + 3], where);
                if (w != 0.0)
                    throw std::invalid_argument(
                        "load_lammps_coeffs(" + path + "): " + where
                        + ": charmm 1-4 weight must be 0 (scaling lives on "
                          "special_bonds / the pair style)");
                if (n < 1 || n != static_cast<int>(n))
                    die(path, where
                                  + ": charmm multiplicity must be a "
                                    "positive integer");
                out.dihedral_charmm_K[t] = K;
                out.dihedral_charmm_n[t] = static_cast<int>(n);
                out.dihedral_charmm_delta[t] = degree_to_rad(d);
                out.has_charmm = true;
            } else if (style == "multi/harmonic") {
                std::vector<double> Cn;
                for (std::size_t i = arg0; i < toks.size(); ++i)
                    Cn.push_back(
                        kcal_per_mol_to_ev(to_num(path, toks[i], where)));
                if (Cn.empty())
                    die(path, where
                                  + ": dihedral_coeff multi/harmonic needs "
                                    "at least one An coefficient");
                out.dihedral_multi_Cn[t] = std::move(Cn);
                out.has_multi = true;
            } else {
                throw std::invalid_argument(
                    "load_lammps_coeffs(" + path + "): " + where
                    + ": dihedral_coeff style '" + style
                    + "' — only charmm, multi/harmonic, fourier");
            }
        } else if (cmd == "improper_style") {
            need(2);
            if (toks[1] != "cvff")
                throw std::invalid_argument("load_lammps_coeffs(" + path
                                            + "): " + where
                                            + ": unsupported improper_style '"
                                            + toks[1] + "' — only cvff");
            out.has_improper_cvff = true;
        } else if (cmd == "improper_coeff") {
            if (!out.has_improper_cvff)
                die(path,
                    where + ": improper_coeff before improper_style cvff");
            if (toks.size() != 5)
                die(path, where + ": improper_coeff cvff needs K d n");
            const int t = to_type(path, toks[1], where);
            const double d = to_num(path, toks[3], where);
            const double n = to_num(path, toks[4], where);
            if (d != 1.0 && d != -1.0)
                die(path, where + ": cvff d must be +1 or -1");
            if (n < 0 || n > 6 || n != static_cast<int>(n))
                die(path, where + ": cvff n must be an integer in [0, 6]");
            ensure_size(out.improper_K, t, 0.0);
            ensure_size(out.improper_d, t, 1);
            ensure_size(out.improper_n, t, 0);
            out.improper_K[t] =
                kcal_per_mol_to_ev(to_num(path, toks[2], where));
            out.improper_d[t] = static_cast<int>(d);
            out.improper_n[t] = static_cast<int>(n);
        } else if (cmd == "special_bonds") {
            need(2);
            out.has_special = true;
            if (toks[1] == "amber") {
                out.special_lj[0] = out.special_lj[1] = 0.0;
                out.special_lj[2] = 0.5;
                out.special_coul[0] = out.special_coul[1] = 0.0;
                out.special_coul[2] = 5.0 / 6.0;
            } else {
                // keyword triples: lj a b c / coul d e f / lj/coul a b c
                std::size_t i = 1;
                bool any = false;
                while (i < toks.size()) {
                    const std::string& kw = toks[i];
                    if (kw == "lj" || kw == "coul" || kw == "lj/coul") {
                        if (i + 3 >= toks.size())
                            die(path, where + ": special_bonds " + kw
                                          + " needs three weights");
                        double w[3];
                        for (int k = 0; k < 3; ++k)
                            w[k] = to_num(path, toks[i + 1 + k], where);
                        if (kw != "coul")
                            std::copy(w, w + 3, out.special_lj);
                        if (kw != "lj")
                            std::copy(w, w + 3, out.special_coul);
                        i += 4;
                        any = true;
                    } else {
                        die(path, where
                                      + ": unsupported special_bonds "
                                        "keyword '"
                                      + kw + "'");
                    }
                }
                if (!any)
                    die(path, where
                                  + ": special_bonds needs lj/coul "
                                    "weight triples or 'amber'");
            }
        } else if (cmd == "kspace_style") {
            need(2);
            out.kspace_style = toks[1];
            if (toks.size() > 2)
                out.kspace_accuracy = to_num(path, toks[2], where);
        } else if (cmd == "pair_modify") {
            for (std::size_t i = 1; i < toks.size(); i += 2) {
                if (i + 1 >= toks.size())
                    die(path,
                        where + ": pair_modify " + toks[i] + " needs a value");
                const std::string& key = toks[i];
                const std::string& val = toks[i + 1];
                if (key == "tail") {
                    out.tail_correction = (val == "yes");
                } else if (key == "shift") {
                    out.has_shift = true;
                    out.energy_shift = (val == "yes");
                } else if (key == "mix") {
                    if (val != "arithmetic")
                        throw std::invalid_argument(
                            "load_lammps_coeffs(" + path + "): " + where
                            + ": pair_modify mix " + val
                            + " — the LJ pair styles mix arithmetic "
                              "(Lorentz-Berthelot) only; give every i j "
                              "pair_coeff explicitly instead");
                } else {
                    die(path, where + ": unsupported pair_modify keyword '"
                                  + key + "'");
                }
            }
        } else {
            out.skipped.push_back(cmd);
        }
    }

    return out;
}

} // namespace cpu
} // namespace ATV
