/// @file cpu/semiempirical/internal/mio_sk_table.h
/// @brief Reader for transcribed mio-1-1 H/C/N/O Slater–Koster tables.
///
/// Internal — not a public API. Data live in `basis_data/dftb_mio_sk_tables.h`
/// (SK grid in Bohr, integrals in Hartree, DFTB+ convention). This header owns
/// the one **interpolation + unit convention** every DFTB method must share:
/// DFTB+ `polyInterU1` Neville interpolation on the uniform Bohr grid, the
/// 5th-order taper over `distFudge` past the last point, and the cubic /
/// 5th-order repulsive spline. Methods differ in what they do with H⁰/S
/// (SCC or not) — never in how the parameters are read.
///
/// Ordered pair (Z_A, Z_B) is the SKF filename left–right: `H_sps` is s on A
/// and p on B. For p_A–s_B the caller reads `hopping_sp(Z_B, Z_A, r)` and
/// applies the usual directed angular sign.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_sk_tables.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief The mio-1-1 parameter set for one ordered element pair.
///
/// Stateless w.r.t. geometry (grids are static `constexpr`); holds a pointer
/// into `kMioSkPairs`. Public units are Atomiverse — Å in, eV out.
class MioSkTable {
  public:
    /// @brief Look up the transcribed table for ordered (Z_A, Z_B).
    /// @param za Atomic number of the first (row) centre.
    /// @param zb Atomic number of the second (column) centre.
    /// @throws std::runtime_error if the pair is not among H/C/N/O.
    MioSkTable(int za, int zb) : data_(&lookup(za, zb)) {}

    /// @brief Range of the tabulated H⁰/S grid, including the DFTB+ taper.
    /// @return Separation beyond which H⁰ and S are identically zero [Å].
    double cutoff() const {
        return (static_cast<double>(data_->n_grid) * data_->grid_dist_bohr
                + basis_data::kMioSk_distFudge_Bohr)
               * basis_data::kMioSk_BohrToA;
    }

    double hopping_ss(double r_A) const {
        return grid_value(data_->H_sss, r_A * basis_data::kMioSk_AToBohr)
               * basis_data::kMioSk_HaToEV;
    }
    double hopping_sp(double r_A) const {
        return grid_value(data_->H_sps, r_A * basis_data::kMioSk_AToBohr)
               * basis_data::kMioSk_HaToEV;
    }
    double hopping_pps(double r_A) const {
        return grid_value(data_->H_pps, r_A * basis_data::kMioSk_AToBohr)
               * basis_data::kMioSk_HaToEV;
    }
    double hopping_ppp(double r_A) const {
        return grid_value(data_->H_ppp, r_A * basis_data::kMioSk_AToBohr)
               * basis_data::kMioSk_HaToEV;
    }

    double overlap_ss(double r_A) const {
        return grid_value(data_->S_sss, r_A * basis_data::kMioSk_AToBohr);
    }
    double overlap_sp(double r_A) const {
        return grid_value(data_->S_sps, r_A * basis_data::kMioSk_AToBohr);
    }
    double overlap_pps(double r_A) const {
        return grid_value(data_->S_pps, r_A * basis_data::kMioSk_AToBohr);
    }
    double overlap_ppp(double r_A) const {
        return grid_value(data_->S_ppp, r_A * basis_data::kMioSk_AToBohr);
    }

    double d_hopping_ss_dr(double r_A) const {
        return grid_derivative(data_->H_sss, r_A) * basis_data::kMioSk_HaToEV;
    }
    double d_hopping_sp_dr(double r_A) const {
        return grid_derivative(data_->H_sps, r_A) * basis_data::kMioSk_HaToEV;
    }
    double d_hopping_pps_dr(double r_A) const {
        return grid_derivative(data_->H_pps, r_A) * basis_data::kMioSk_HaToEV;
    }
    double d_hopping_ppp_dr(double r_A) const {
        return grid_derivative(data_->H_ppp, r_A) * basis_data::kMioSk_HaToEV;
    }
    double d_overlap_ss_dr(double r_A) const {
        return grid_derivative(data_->S_sss, r_A);
    }
    double d_overlap_sp_dr(double r_A) const {
        return grid_derivative(data_->S_sps, r_A);
    }
    double d_overlap_pps_dr(double r_A) const {
        return grid_derivative(data_->S_pps, r_A);
    }
    double d_overlap_ppp_dr(double r_A) const {
        return grid_derivative(data_->S_ppp, r_A);
    }

    /// @brief Pair repulsive energy from the mio-1-1 spline.
    /// @param r_A Nuclear separation [Å].
    /// @return E_rep(r) [eV]; zero at or beyond the spline cutoff.
    double repulsive(double r_A) const {
        const double rr = r_A * basis_data::kMioSk_AToBohr;
        if (rr < 1e-12 || rr >= data_->rep_cutoff_bohr)
            return 0.0;
        if (rr < data_->x_start_bohr[0]) {
            const double e_ha =
                std::exp(-data_->exp_coeffs[0] * rr + data_->exp_coeffs[1])
                + data_->exp_coeffs[2];
            return e_ha * basis_data::kMioSk_HaToEV;
        }
        std::size_t lo = 0;
        std::size_t hi = data_->n_spline;
        while (hi - lo > 1) {
            const std::size_t mid = (lo + hi) / 2;
            if (data_->x_start_bohr[mid] <= rr)
                lo = mid;
            else
                hi = mid;
        }
        const std::size_t i_match = lo;
        const double dr = rr - data_->x_start_bohr[i_match];
        double e_ha = 0.0;
        if (i_match + 1 < data_->n_spline) {
            const double* c = &data_->sp_coeffs[4 * i_match];
            e_ha = c[0] + c[1] * dr + c[2] * dr * dr + c[3] * dr * dr * dr;
        } else {
            double xh = 1.0;
            for (int ii = 0; ii < 6; ++ii) {
                e_ha += data_->sp_last_coeffs[ii] * xh;
                xh *= dr;
            }
        }
        return e_ha * basis_data::kMioSk_HaToEV;
    }

    /// @brief Radial derivative of the mio-1-1 repulsive spline.
    /// @param r_A Nuclear separation [Å].
    /// @return dE_rep/dr [eV/Å].
    double d_repulsive_dr(double r_A) const {
        const double rr = r_A * basis_data::kMioSk_AToBohr;
        if (rr < 1e-12 || rr >= data_->rep_cutoff_bohr)
            return 0.0;
        double de_drr_ha = 0.0;
        if (rr < data_->x_start_bohr[0]) {
            de_drr_ha =
                -data_->exp_coeffs[0]
                * std::exp(-data_->exp_coeffs[0] * rr + data_->exp_coeffs[1]);
        } else {
            std::size_t lo = 0;
            std::size_t hi = data_->n_spline;
            while (hi - lo > 1) {
                const std::size_t mid = (lo + hi) / 2;
                if (data_->x_start_bohr[mid] <= rr)
                    lo = mid;
                else
                    hi = mid;
            }
            const std::size_t i_match = lo;
            const double dr = rr - data_->x_start_bohr[i_match];
            if (i_match + 1 < data_->n_spline) {
                const double* c = &data_->sp_coeffs[4 * i_match];
                de_drr_ha = c[1] + 2.0 * c[2] * dr + 3.0 * c[3] * dr * dr;
            } else {
                const double* c = data_->sp_last_coeffs;
                de_drr_ha = c[1] + 2.0 * c[2] * dr + 3.0 * c[3] * dr * dr
                            + 4.0 * c[4] * dr * dr * dr
                            + 5.0 * c[5] * dr * dr * dr * dr;
            }
        }
        return de_drr_ha * basis_data::kMioSk_HaToEV
               * basis_data::kMioSk_AToBohr;
    }

  private:
    const basis_data::MioSkPairData* data_;

    static const basis_data::MioSkPairData& lookup(int za, int zb) {
        for (std::size_t i = 0; i < basis_data::kMioSkNPairs; ++i) {
            if (basis_data::kMioSkPairs[i].za == za
                && basis_data::kMioSkPairs[i].zb == zb)
                return basis_data::kMioSkPairs[i];
        }
        throw std::runtime_error("MioSkTable: no mio-1-1 table for pair Z="
                                 + std::to_string(za) + "-"
                                 + std::to_string(zb));
    }

    double grid_derivative(const double* tab, double r_A) const {
        constexpr double h = 1.0e-5; // Å
        const double rp = (r_A + h) * basis_data::kMioSk_AToBohr;
        const double rm =
            std::max(r_A - h, 1.0e-8) * basis_data::kMioSk_AToBohr;
        return (grid_value(tab, rp) - grid_value(tab, rm)) / (2.0 * h);
    }

    static double poly_interp(const double* xp, const double* yp, int nn,
                              double xx) {
        std::vector<double> X(static_cast<std::size_t>(nn + 1));
        std::vector<double> C(static_cast<std::size_t>(nn + 1));
        std::vector<double> D(static_cast<std::size_t>(nn + 1));
        for (int i = 1; i <= nn; ++i) {
            X[static_cast<std::size_t>(i)] = xp[i - 1];
            C[static_cast<std::size_t>(i)] = yp[i - 1];
            D[static_cast<std::size_t>(i)] = yp[i - 1];
        }
        int iCl =
            static_cast<int>(std::ceil((xx - X[1]) / std::abs(X[2] - X[1])));
        if (iCl < 1)
            iCl = 1;
        if (iCl > nn)
            iCl = nn;
        double yy = yp[iCl - 1];
        iCl = iCl - 1;
        for (int mm = 1; mm <= nn - 1; ++mm) {
            for (int ii = 1; ii <= nn - mm; ++ii) {
                double rTmp = X[static_cast<std::size_t>(ii)]
                              - X[static_cast<std::size_t>(ii + mm)];
                rTmp = (C[static_cast<std::size_t>(ii + 1)]
                        - D[static_cast<std::size_t>(ii)])
                       / rTmp;
                C[static_cast<std::size_t>(ii)] =
                    (X[static_cast<std::size_t>(ii)] - xx) * rTmp;
                D[static_cast<std::size_t>(ii)] =
                    (X[static_cast<std::size_t>(ii + mm)] - xx) * rTmp;
            }
            double dyy;
            if (2 * iCl < nn - mm) {
                dyy = C[static_cast<std::size_t>(iCl + 1)];
            } else {
                dyy = D[static_cast<std::size_t>(iCl)];
                iCl = iCl - 1;
            }
            yy += dyy;
        }
        return yy;
    }

    double grid_value(const double* tab, double r_bohr) const {
        constexpr int nInterNew = 8;
        constexpr int nRightInterNew = 4;
        constexpr double deltaR = 1.0e-5;
        const int leng = static_cast<int>(data_->n_grid);
        const double incr = data_->grid_dist_bohr;
        const double rMax = static_cast<double>(leng) * incr
                            + basis_data::kMioSk_distFudge_Bohr;
        const int ind = static_cast<int>(std::floor(r_bohr / incr));
        if (r_bohr >= rMax)
            return 0.0;
        if (ind < leng) {
            int iLast = std::min(leng, ind + nRightInterNew);
            iLast = std::max(iLast, nInterNew);
            double xa[nInterNew];
            double ya[nInterNew];
            for (int ii = 1; ii <= nInterNew; ++ii) {
                xa[ii - 1] = static_cast<double>(iLast - nInterNew + ii) * incr;
                ya[ii - 1] = tab[iLast - nInterNew + ii - 1];
            }
            return poly_interp(xa, ya, nInterNew, r_bohr);
        }
        const double dr = r_bohr - rMax;
        const int iLast = leng;
        double xa[nInterNew];
        double ya[nInterNew];
        for (int ii = 1; ii <= nInterNew; ++ii) {
            xa[ii - 1] = static_cast<double>(iLast - nInterNew + ii) * incr;
            ya[ii - 1] = tab[iLast - nInterNew + ii - 1];
        }
        const double y0 =
            poly_interp(xa, ya, nInterNew, xa[nInterNew - 1] - deltaR);
        const double y2 =
            poly_interp(xa, ya, nInterNew, xa[nInterNew - 1] + deltaR);
        const double y1 = ya[nInterNew - 1];
        const double y1p = (y2 - y0) / (2.0 * deltaR);
        const double y1pp = (y2 + y0 - 2.0 * y1) / (deltaR * deltaR);
        const double dx = -basis_data::kMioSk_distFudge_Bohr;
        const double invdx = -1.0 / basis_data::kMioSk_distFudge_Bohr;
        const double dx1 = y1p * dx;
        const double dx2 = y1pp * dx * dx;
        const double dd = 10.0 * y1 - 4.0 * dx1 + 0.5 * dx2;
        const double ee = -15.0 * y1 + 7.0 * dx1 - 1.0 * dx2;
        const double ff = 6.0 * y1 - 3.0 * dx1 + 0.5 * dx2;
        const double xr = dr * invdx;
        return ((ff * xr + ee) * xr + dd) * xr * xr * xr;
    }
};

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
