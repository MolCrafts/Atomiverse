/// @file core/lj_type_table.cpp
/// @brief LAMMPS `init_one` table completion for typed LJ (+ r⁻⁴).
///
/// Ported from LAMMPS `PairLJCut::init_one` (`src/pair_lj_cut.cpp`),
/// `PairLJCutCoulLong::init_one` / `coeff` (`src/KSPACE/pair_lj_cut_coul_long.cpp`)
/// and `Pair::mix_energy` / `Pair::mix_distance` (`src/pair.cpp`), LAMMPS
/// checkout `e8c4c278a8`:
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   LAMMPS development team: developers@lammps.org
///
///   Copyright (2003) Sandia Corporation.  Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software.  This software is distributed under
///   the GNU General Public License.
#include "atomiverse/core/lj_type_table.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ATV {
namespace {

std::string row_text(int i, int j) {
    return "(" + std::to_string(i) + ", " + std::to_string(j) + ")";
}

void check_lj_row(const LJTypeCoeff& c) {
    if (c.type_i < 0 || c.type_j < 0)
        throw std::invalid_argument("LJ type table: types must be >= 0, got "
                                    + row_text(c.type_i, c.type_j));
    if (!std::isfinite(c.epsilon) || c.epsilon < 0.0)
        throw std::invalid_argument(
            "LJ type table: epsilon must be finite and >= 0 for "
            + row_text(c.type_i, c.type_j));
    if (!std::isfinite(c.sigma) || c.sigma <= 0.0)
        throw std::invalid_argument(
            "LJ type table: sigma must be finite and > 0 for "
            + row_text(c.type_i, c.type_j));
    if (c.cutoff && (!std::isfinite(*c.cutoff) || *c.cutoff <= 0.0))
        throw std::invalid_argument(
            "LJ type table: a per-pair cutoff must be finite and > 0 for "
            + row_text(c.type_i, c.type_j));
}

bool same(double a, double b) {
    return std::fabs(a - b) <= 1e-12 * std::max(std::fabs(a), std::fabs(b));
}

} // namespace

LJTypeTable::LJTypeTable(std::vector<LJTypeCoeff> coeffs, double cutoff)
    : cutoff_(cutoff) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
        throw std::invalid_argument(
            "LJ type table: the global cutoff must be finite and > 0");
    if (coeffs.empty())
        throw std::invalid_argument(
            "LJ type table: at least one coefficient row is required; there "
            "is no system-wide epsilon/sigma to fall back on");
    for (const LJTypeCoeff& c : coeffs)
        check_lj_row(c);
    rows_ = std::move(coeffs);
}

void LJTypeTable::set_coeff(const LJTypeCoeff& row) {
    check_lj_row(row);
    rows_.push_back(row);
}

void LJTypeTable::set_r4(std::vector<R4TypeCoeff> rows) {
    for (const R4TypeCoeff& r : rows) {
        if (r.type_i < 0 || r.type_j < 0)
            throw std::invalid_argument(
                "r^-4 attraction: types must be >= 0, got "
                + row_text(r.type_i, r.type_j));
        if (!std::isfinite(r.strength))
            throw std::invalid_argument(
                "r^-4 attraction: strength must be finite for "
                + row_text(r.type_i, r.type_j));
        if (!std::isfinite(r.sigma) || r.sigma <= 0.0)
            throw std::invalid_argument(
                "r^-4 attraction: sigma must be finite and > 0 for "
                + row_text(r.type_i, r.type_j));
        if (!std::isfinite(r.cutoff) || r.cutoff <= 0.0)
            throw std::invalid_argument(
                "r^-4 attraction: cutoff must be finite and > 0 for "
                + row_text(r.type_i, r.type_j));
    }
    r4_ = std::move(rows);
}

double LJTypeTable::max_cutoff() const {
    double rc = 0.0;
    for (const LJTypeCoeff& c : rows_)
        rc = std::max(rc, c.cutoff.value_or(cutoff_));
    for (const R4TypeCoeff& r : r4_)
        rc = std::max(rc, r.cutoff);
    return rc;
}

int LJTypeTable::named_types() const {
    int top = -1;
    for (const LJTypeCoeff& c : rows_)
        top = std::max({top, c.type_i, c.type_j});
    for (const R4TypeCoeff& r : r4_)
        top = std::max({top, r.type_i, r.type_j});
    return top + 1;
}

void LJTypeTable::build(int n_types, bool shift) {
    const int n = std::max(n_types, named_types());
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    // LAMMPS coeff(): an explicit row sets epsilon, sigma and cut_lj
    // (cut_lj_global when the fifth argument is absent) and setflag.
    std::vector<double> eps(n2, 0.0), sig(n2, 0.0), cut(n2, 0.0);
    std::vector<char> set(n2, 0);
    auto at = [n](int a, int b) { return static_cast<std::size_t>(a) * n + b; };
    for (const LJTypeCoeff& c : rows_) {
        const double rc = c.cutoff.value_or(cutoff_);
        for (const std::size_t ab :
             {at(c.type_i, c.type_j), at(c.type_j, c.type_i)}) {
            eps[ab] = c.epsilon;
            sig[ab] = c.sigma;
            cut[ab] = rc;
            set[ab] = 1;
        }
    }

    pairs_.assign(n2, LJPairCoeff{});
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            const std::size_t ab = at(a, b);
            double e = eps[ab], s = sig[ab], rc = cut[ab];
            if (!set[ab]) {
                const std::size_t aa = at(a, a), bb = at(b, b);
                if (set[aa] && set[bb]) {
                    // init_one with mix arithmetic: mix_energy is the
                    // geometric mean, mix_distance the arithmetic mean of
                    // sigma and of the diagonal cutoffs.
                    e = std::sqrt(eps[aa] * eps[bb]);
                    s = 0.5 * (sig[aa] + sig[bb]);
                    rc = 0.5 * (cut[aa] + cut[bb]);
                } else {
                    // No diagonal to mix from: no LJ site on this pair.
                    e = 0.0;
                    s = 0.0;
                    rc = 0.0;
                }
            }
            LJPairCoeff& p = pairs_[ab];
            p.epsilon = e;
            p.sigma = s;
            p.cut_lj = rc;
            const double s6 = std::pow(s, 6.0);
            const double s12 = s6 * s6;
            p.lj1 = 48.0 * e * s12;
            p.lj2 = 24.0 * e * s6;
            p.lj3 = 4.0 * e * s12;
            p.lj4 = 4.0 * e * s6;
            p.cut_ljsq = rc * rc;
            if (shift && rc > 0.0) {
                const double ratio = s / rc;
                p.offset =
                    4.0 * e * (std::pow(ratio, 12.0) - std::pow(ratio, 6.0));
            } else {
                p.offset = 0.0;
            }
        }

    for (const R4TypeCoeff& r : r4_) {
        const double s2 = r.sigma * r.sigma;
        const double a4 = r.strength * s2 * s2;
        const double rc2 = r.cutoff * r.cutoff;
        for (const std::size_t ab :
             {at(r.type_i, r.type_j), at(r.type_j, r.type_i)}) {
            LJPairCoeff& p = pairs_[ab];
            p.r4_a = a4;
            p.r4_cutsq = rc2;
            p.r4_rc5inv = 1.0 / (rc2 * rc2 * r.cutoff);
            p.r4_e0 = 5.0 * a4 / (rc2 * rc2);
        }
    }
    n_types_ = n;
}

bool LJTypeTable::lorentz_berthelot_uniform() const {
    if (has_r4())
        return false;
    for (int a = 0; a < n_types_; ++a)
        for (int b = 0; b < n_types_; ++b) {
            const LJPairCoeff& p = (*this)(a, b);
            if (p.epsilon == 0.0)
                continue;
            const LJPairCoeff& pa = (*this)(a, a);
            const LJPairCoeff& pb = (*this)(b, b);
            if (!same(p.epsilon, std::sqrt(pa.epsilon * pb.epsilon))
                || !same(p.sigma, 0.5 * (pa.sigma + pb.sigma))
                || !same(p.cut_lj, cutoff_))
                return false;
        }
    return true;
}

// LAMMPS PairLJCut::init_one / PairLJCutCoulLong::init_one tail block.
LJTail lj_tail_pair(double epsilon, double sigma, double cut_lj, double n_i,
                    double n_j) {
    if (!(epsilon > 0.0) || !(cut_lj > 0.0))
        return {};
    constexpr double kPi = 3.14159265358979323846;
    const double sig2 = sigma * sigma;
    const double sig6 = sig2 * sig2 * sig2;
    const double rc3 = cut_lj * cut_lj * cut_lj;
    const double rc6 = rc3 * rc3;
    const double rc9 = rc3 * rc6;
    return {8.0 * kPi * n_i * n_j * epsilon * sig6 * (sig6 - 3.0 * rc6)
                / (9.0 * rc9),
            16.0 * kPi * n_i * n_j * epsilon * sig6 * (2.0 * sig6 - 3.0 * rc6)
                / (9.0 * rc9)};
}

LJTail LJTypeTable::tail(const std::vector<double>& counts) const {
    if (counts.size() < static_cast<std::size_t>(n_types_))
        throw std::invalid_argument(
            "LJ type table: tail needs one count per table type");
    // LAMMPS Pair::init: etail += etail_ij for i <= j, again for i != j —
    // the sum over ordered pairs.
    LJTail total;
    for (int a = 0; a < n_types_; ++a)
        for (int b = 0; b < n_types_; ++b) {
            const LJPairCoeff& c = (*this)(a, b);
            const LJTail t = lj_tail_pair(c.epsilon, c.sigma, c.cut_lj,
                                          counts[static_cast<std::size_t>(a)],
                                          counts[static_cast<std::size_t>(b)]);
            total.energy += t.energy;
            total.virial += t.virial;
        }
    return total;
}

} // namespace ATV
