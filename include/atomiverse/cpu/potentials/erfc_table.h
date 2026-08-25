/// @file cpu/potentials/erfc_table.h
/// @brief Tabulated erfc(αr)/r real-space Coulomb (LAMMPS-style).
///
/// Replaces per-pair `std::erfc` + `std::exp` in the hot loop with linear
/// interpolation on a precomputed table over [0, cutoff].
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cpu {
namespace md {

/// @brief Lookup table for screened Coulomb prefactors without q_i q_j.
///
/// Indexed by **r²** (no per-pair `sqrt`): for each bin centre r²,
///   e_tab[i] = erfc(α r) / r
///   f_tab[i] = (erfc(α r) + (2/√π) α r exp(−(α r)²)) / r³   (fpair: F⃗ = qq·f·d⃗)
/// so energy = qqr2e·q_i·q_j·e and fpair = qqr2e·q_i·q_j·f.
struct ErfcTable {
    static constexpr double kEwaldF = 1.1283791670955126; // 2/√π

    double alpha = 0.0;
    double cutoff = 0.0;
    double cutoff_sq = 0.0;
    double dr2 = 0.0;
    double inv_dr2 = 0.0;
    int nbin = 0;
    std::vector<double> e_tab;
    std::vector<double> f_tab;

    void clear() {
        alpha = cutoff = cutoff_sq = dr2 = inv_dr2 = 0.0;
        nbin = 0;
        e_tab.clear();
        f_tab.clear();
    }

    /// @param a   Ewald α [1/Å].
    /// @param cut Real-space cutoff [Å].
    /// @param nb  Number of bins (default 10000 ≈ LAMMPS density).
    void build(double a, double cut, int nb = 10000) {
        if (!(a > 0.0) || !(cut > 0.0) || nb < 16)
            throw std::invalid_argument("ErfcTable::build: bad alpha/cut/nbin");
        alpha = a;
        cutoff = cut;
        cutoff_sq = cut * cut;
        nbin = nb;
        dr2 = cutoff_sq / static_cast<double>(nb);
        inv_dr2 = 1.0 / dr2;
        e_tab.assign(static_cast<std::size_t>(nb + 1), 0.0);
        f_tab.assign(static_cast<std::size_t>(nb + 1), 0.0);
        for (int i = 1; i <= nb; ++i) {
            const double r2 = i * dr2;
            const double r = std::sqrt(r2);
            const double grij = a * r;
            const double expm2 = std::exp(-grij * grij);
            const double erfcv = std::erfc(grij);
            e_tab[static_cast<std::size_t>(i)] = erfcv / r;
            // LAMMPS fpair convention: force *per unit displacement vector*,
            // F_vec = qq · f · (dx,dy,dz). |F| = qq(erfc + F·αr·e^{-α²r²})/r²,
            // so f = |F|/(qq·r) = (…)/r³.
            f_tab[static_cast<std::size_t>(i)] =
                (erfcv + kEwaldF * grij * expm2) / (r2 * r);
        }
        // r→0 limit: erfc(0)/r diverges; force users skip r2==0.
        e_tab[0] = e_tab[1];
        f_tab[0] = f_tab[1];
    }

    /// @brief Interpolate e and f at squared distance r2 (0 < r2 ≤ cutoff²).
    void eval_r2(double r2, double& e, double& f) const {
        double x = r2 * inv_dr2;
        int i = static_cast<int>(x);
        if (i < 0)
            i = 0;
        if (i >= nbin)
            i = nbin - 1;
        const double t = x - static_cast<double>(i);
        const std::size_t ii = static_cast<std::size_t>(i);
        e = e_tab[ii] + t * (e_tab[ii + 1] - e_tab[ii]);
        f = f_tab[ii] + t * (f_tab[ii + 1] - f_tab[ii]);
    }

    /// @brief Interpolate at distance r (0 < r ≤ cutoff). Convenience wrapper.
    void eval(double r, double& e, double& f) const { eval_r2(r * r, e, f); }
};

} // namespace md
} // namespace cpu
} // namespace ATV
