/// @file regressions/forcefield/cg-electrolyte/cg_electrolyte_system.h
/// @brief Shared reader / writer of the coarse-grained polymer-electrolyte
///        single-point case (`tools/lammps_oracle/inputs/cg_electrolyte/
///        make_case.py` writes `system.txt`). Used by the CPU and CUDA
///        harnesses so both build the identical Frame and coefficient rows.
#pragma once

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/cpu/frame/frame.h"

namespace cg_electrolyte {

/// @brief Every parameter and coordinate of the case (public units).
struct System {
    double box = 0, eps0 = 0, q0 = 0, alpha = 0, rc_coul = 0, cut_factor = 0;
    int mesh = 0, order = 0;
    double fene_k = 0, fene_r0 = 0, cos_k = 0; ///< Reduced (eps0, sigma0).
    int r4_i = 0, r4_j = 0;
    double r4_s = 0, r4_sigma = 0, r4_rc = 0; ///< S reduced; sigma, rc [A].
    std::map<int, std::pair<double, double>> type; ///< t -> (sigma, eps red.)
    std::map<std::pair<int, int>, double> override_eps; ///< reduced eps.
    std::vector<int> t, z;
    std::vector<double> x, y, zc;
    std::vector<std::pair<int, int>> bonds;
    std::vector<std::array<int, 3>> angles;
    int n() const { return static_cast<int>(t.size()); }
};

inline System read(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    System s;
    std::string key;
    while (in >> key) {
        if (key[0] == '#') {
            std::getline(in, key);
            continue;
        }
        if (key == "box")
            in >> s.box;
        else if (key == "eps0")
            in >> s.eps0;
        else if (key == "q0")
            in >> s.q0;
        else if (key == "alpha")
            in >> s.alpha;
        else if (key == "rc_coul")
            in >> s.rc_coul;
        else if (key == "cut_factor")
            in >> s.cut_factor;
        else if (key == "mesh")
            in >> s.mesh;
        else if (key == "order")
            in >> s.order;
        else if (key == "fene")
            in >> s.fene_k >> s.fene_r0;
        else if (key == "cosine")
            in >> s.cos_k;
        else if (key == "r4")
            in >> s.r4_i >> s.r4_j >> s.r4_s >> s.r4_sigma >> s.r4_rc;
        else if (key == "ntypes" || key == "noverride") {
            int n;
            in >> n;
        } else if (key == "type") {
            int ty;
            double sig, eps;
            in >> ty >> sig >> eps;
            s.type[ty] = {sig, eps};
        } else if (key == "override") {
            int a, b;
            double eps;
            in >> a >> b >> eps;
            s.override_eps[{a, b}] = eps;
        } else if (key == "natoms") {
            int n;
            in >> n;
            s.t.resize(n);
            s.z.resize(n);
            s.x.resize(n);
            s.y.resize(n);
            s.zc.resize(n);
            for (int i = 0; i < n; ++i)
                in >> s.t[i] >> s.z[i] >> s.x[i] >> s.y[i] >> s.zc[i];
        } else if (key == "nbonds") {
            int n;
            in >> n;
            s.bonds.resize(n);
            for (auto& b : s.bonds)
                in >> b.first >> b.second;
        } else if (key == "nangles") {
            int n;
            in >> n;
            s.angles.resize(n);
            for (auto& a : s.angles)
                in >> a[0] >> a[1] >> a[2];
        } else
            throw std::runtime_error("system.txt: unknown key " + key);
    }
    return s;
}

/// @brief Diagonal rows with their own 2.5 sigma cutoffs plus the two
///        explicit cross rows; every other pair is left to ATV's LAMMPS
///        init_one mixing (Lorentz-Berthelot, arithmetic cutoff).
/// @param s The case.
/// @param scale Multiplies every epsilon (0 keeps the table but no LJ).
inline std::vector<ATV::LJTypeCoeff> lj_rows(const System& s,
                                             double scale = 1.0) {
    std::vector<ATV::LJTypeCoeff> rows;
    for (const auto& [ty, se] : s.type)
        rows.push_back({ty, ty, scale * se.second * s.eps0, se.first,
                        s.cut_factor * se.first});
    for (const auto& [ab, eps] : s.override_eps) {
        const double sig =
            0.5 * (s.type.at(ab.first).first + s.type.at(ab.second).first);
        rows.push_back({ab.first, ab.second, scale * eps * s.eps0, sig,
                        s.cut_factor * sig});
    }
    return rows;
}

inline ATV::R4TypeCoeff r4_row(const System& s) {
    return {s.r4_i, s.r4_j, s.r4_s * s.eps0, s.r4_sigma, s.r4_rc};
}

/// @brief The case as a CPU Frame: unit masses, tags = input order, charges
///        z q0, one bond and one angle type, special list built (all 1-2 /
///        1-3 / 1-4 weights are 1 in this model). The beads (B, E, C, A, T,
///        L) carry force-field types in `kAtom_type` and no element: the
///        Frame has no species column at all, so nothing can mistake a
///        bead type for an atomic number.
inline ATV::cpu::Frame frame(const System& s) {
    using namespace ATV;
    const int n = s.n();
    cpu::Frame f(n);
    f.block("atoms").erase(kAtom_atomic_number.key);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = s.x[i];
        f.span(kAtom_y)[i] = s.y[i];
        f.span(kAtom_z)[i] = s.zc[i];
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_type)[i] = s.t[i];
        f.span(kAtom_charge)[i] = s.z[i] * s.q0;
    }
    f.box = cpu::Box::cubic(s.box);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (const auto& b : s.bonds)
        f.add_bond(b.first, b.second, 0);
    for (const auto& a : s.angles)
        f.add_angle(a[0], a[1], a[2], 0);
    f.build_special_list();
    return f;
}

/// @brief Write energy terms, the six-component virial and per-tag forces.
inline void write(const std::string& path, ATV::cpu::Frame& f) {
    using namespace ATV;
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(out, "energy %.17g\n", f.meta.get(kEnergy));
    std::fprintf(out, "energy_pair %.17g\n", f.meta.get(kEnergyPair));
    std::fprintf(out, "energy_bond %.17g\n", f.meta.get(kEnergyBond));
    std::fprintf(out, "energy_angle %.17g\n", f.meta.get(kEnergyAngle));
    const auto& w = f.meta.get(kStress);
    std::fprintf(out, "virial %.17g %.17g %.17g %.17g %.17g %.17g\n", w[0],
                 w[1], w[2], w[3], w[4], w[5]);
    const int n = f.nlocal();
    std::vector<std::array<double, 3>> force(n);
    for (int i = 0; i < n; ++i)
        force[static_cast<std::size_t>(f.span(kAtom_tag)[i])] = {
            f.span(kAtom_fx)[i], f.span(kAtom_fy)[i], f.span(kAtom_fz)[i]};
    for (int i = 0; i < n; ++i)
        std::fprintf(out, "%d %.17g %.17g %.17g\n", i + 1, force[i][0],
                     force[i][1], force[i][2]);
    std::fclose(out);
}

} // namespace cg_electrolyte
