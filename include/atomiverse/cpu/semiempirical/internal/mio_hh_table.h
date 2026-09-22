/// @file cpu/semiempirical/internal/mio_hh_table.h
/// @brief Reader for the transcribed mio-1-1 H–H Slater–Koster tables.
///
/// Internal — not a public API. The data itself lives in
/// `basis_data/dftb_mio_hh_sk.h` (SK grid in Bohr, integrals in Hartree,
/// DFTB+ convention). This header owns the one **interpolation + unit
/// convention** every DFTB method must share when it reads that grid:
/// DFTB+ `polyInterU1` Neville interpolation on the uniform Bohr grid, the
/// 5th-order taper over `distFudge` past the last point, and the
/// cubic / 5th-order repulsive spline. Methods differ in what they do with
/// H⁰/S (SCC or not) — never in how the parameters are read.
///
/// Pinned against DFTB+ 24.1 + mio-1-1 by
/// `regressions/semiempirical/dftb2-dftbplus-energy-pin/` (SCC) and
/// `regressions/semiempirical/dftb0-h2/` (non-SCC).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_hh_sk.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

/// @brief The mio-1-1 H–H parameter table, evaluated at a nuclear separation.
///
/// Stateless (the grid is static `constexpr` data); construct one where you
/// need it. Public units are Atomiverse units — Å in, eV out — with the
/// Bohr/Hartree conversion done here so no caller repeats it.
class MioHHTable {
  public:
    MioHHTable() = default;

    /// @brief Range of the tabulated H⁰/S grid, including the DFTB+ taper.
    /// @return Separation beyond which H⁰ and S are identically zero [Å].
    double cutoff() const {
        return (static_cast<double>(basis_data::kMioHH_nGrid)
                    * basis_data::kMioHH_gridDist_Bohr
                + basis_data::kMioHH_distFudge_Bohr)
               * basis_data::kBohrToA;
    }

    /// @brief Two-centre H⁰ ssσ hopping integral.
    /// @param r_A Nuclear separation [Å].
    /// @return H_ssσ(r) [eV]; zero beyond @ref cutoff.
    double hopping_ss(double r_A) const {
        return grid_value(basis_data::kMioHH_Hss_Ha, r_A * basis_data::kAToBohr)
               * basis_data::kHaToEV;
    }

    /// @brief Two-centre overlap ssσ integral.
    /// @param r_A Nuclear separation [Å].
    /// @return S_ssσ(r) [dimensionless]; zero beyond @ref cutoff.
    double overlap_ss(double r_A) const {
        return grid_value(basis_data::kMioHH_Sss, r_A * basis_data::kAToBohr);
    }

    /// @brief Radial derivative of the ssσ hopping integral.
    ///
    /// The tabulated grid is only piecewise-polynomial, so the derivative is a
    /// central difference of the same Neville interpolation the value uses —
    /// differentiating the interpolant, not a second table.
    ///
    /// @param r_A Nuclear separation [Å].
    /// @return dH_ssσ/dr [eV/Å].
    double d_hopping_ss_dr(double r_A) const {
        return grid_derivative(basis_data::kMioHH_Hss_Ha, r_A)
               * basis_data::kHaToEV;
    }

    /// @brief Radial derivative of the ssσ overlap integral.
    /// @param r_A Nuclear separation [Å].
    /// @return dS_ssσ/dr [1/Å].
    double d_overlap_ss_dr(double r_A) const {
        return grid_derivative(basis_data::kMioHH_Sss, r_A);
    }

    /// @brief Pair repulsive energy from the mio-1-1 spline.
    /// @param r_A Nuclear separation [Å].
    /// @return E_rep(r) [eV]; zero at or beyond the spline cutoff.
    double repulsive(double r_A) const {
        const double rr = r_A * basis_data::kAToBohr;
        if (rr < 1e-12 || rr >= basis_data::kMioHH_repCutoff_Bohr)
            return 0.0;
        if (rr < basis_data::kMioHH_xStart_Bohr[0]) {
            const double e_ha = std::exp(-basis_data::kMioHH_expCoeffs[0] * rr
                                         + basis_data::kMioHH_expCoeffs[1])
                                + basis_data::kMioHH_expCoeffs[2];
            return e_ha * basis_data::kHaToEV;
        }
        std::size_t lo = 0;
        std::size_t hi = basis_data::kMioHH_nSpline;
        while (hi - lo > 1) {
            const std::size_t mid = (lo + hi) / 2;
            if (basis_data::kMioHH_xStart_Bohr[mid] <= rr)
                lo = mid;
            else
                hi = mid;
        }
        const std::size_t i_match = lo;
        const double dr = rr - basis_data::kMioHH_xStart_Bohr[i_match];
        double e_ha = 0.0;
        if (i_match + 1 < basis_data::kMioHH_nSpline) {
            const double* c = &basis_data::kMioHH_spCoeffs[4 * i_match];
            e_ha = c[0] + c[1] * dr + c[2] * dr * dr + c[3] * dr * dr * dr;
        } else {
            double xh = 1.0;
            for (int ii = 0; ii < 6; ++ii) {
                e_ha += basis_data::kMioHH_spLastCoeffs[ii] * xh;
                xh *= dr;
            }
        }
        return e_ha * basis_data::kHaToEV;
    }

    /// @brief Radial derivative of the mio-1-1 repulsive spline.
    /// @param r_A Nuclear separation [Å].
    /// @return dE_rep/dr [eV/Å]; zero at or beyond the spline cutoff.
    double d_repulsive_dr(double r_A) const {
        const double rr = r_A * basis_data::kAToBohr;
        if (rr < 1e-12 || rr >= basis_data::kMioHH_repCutoff_Bohr)
            return 0.0;
        double de_drr_ha = 0.0;
        if (rr < basis_data::kMioHH_xStart_Bohr[0]) {
            de_drr_ha = -basis_data::kMioHH_expCoeffs[0]
                        * std::exp(-basis_data::kMioHH_expCoeffs[0] * rr
                                   + basis_data::kMioHH_expCoeffs[1]);
        } else {
            std::size_t lo = 0;
            std::size_t hi = basis_data::kMioHH_nSpline;
            while (hi - lo > 1) {
                const std::size_t mid = (lo + hi) / 2;
                if (basis_data::kMioHH_xStart_Bohr[mid] <= rr)
                    lo = mid;
                else
                    hi = mid;
            }
            const std::size_t i_match = lo;
            const double dr = rr - basis_data::kMioHH_xStart_Bohr[i_match];
            if (i_match + 1 < basis_data::kMioHH_nSpline) {
                const double* c = &basis_data::kMioHH_spCoeffs[4 * i_match];
                de_drr_ha = c[1] + 2.0 * c[2] * dr + 3.0 * c[3] * dr * dr;
            } else {
                const double* c = basis_data::kMioHH_spLastCoeffs;
                de_drr_ha = c[1] + 2.0 * c[2] * dr + 3.0 * c[3] * dr * dr
                            + 4.0 * c[4] * dr * dr * dr
                            + 5.0 * c[5] * dr * dr * dr * dr;
            }
        }
        return de_drr_ha * basis_data::kHaToEV * basis_data::kAToBohr;
    }

  private:
    /// @brief Central difference of one interpolated column w.r.t. r [Å].
    /// @param tab Tabulated column on the uniform Bohr grid.
    /// @param r_A Nuclear separation [Å].
    /// @return d(column)/dr in the column's own unit per Å.
    static double grid_derivative(const double* tab, double r_A) {
        constexpr double h = 1.0e-5; // Å
        const double rp = (r_A + h) * basis_data::kAToBohr;
        const double rm = std::max(r_A - h, 1.0e-8) * basis_data::kAToBohr;
        return (grid_value(tab, rp) - grid_value(tab, rm)) / (2.0 * h);
    }

    /// @brief Neville polynomial interpolation on a uniform abscissa
    ///        (DFTB+ `polyInterU1`).
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

    /// @brief DFTB+ "new" SK grid interpolation of one integral column.
    /// @param tab Tabulated column on the uniform Bohr grid (Ha or dimensionless).
    /// @param r_bohr Nuclear separation [Bohr] — the grid's own unit.
    /// @return Interpolated value in the column's own unit; zero past the taper.
    static double grid_value(const double* tab, double r_bohr) {
        constexpr int nInterNew = 8;
        constexpr int nRightInterNew = 4;
        constexpr double deltaR = 1.0e-5;
        const int leng = static_cast<int>(basis_data::kMioHH_nGrid);
        const double incr = basis_data::kMioHH_gridDist_Bohr;
        const double rMax = static_cast<double>(leng) * incr
                            + basis_data::kMioHH_distFudge_Bohr;
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
        // Beyond the last grid point: 5th-order taper to zero over distFudge.
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
        const double dx = -basis_data::kMioHH_distFudge_Bohr;
        const double invdx = -1.0 / basis_data::kMioHH_distFudge_Bohr;
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
