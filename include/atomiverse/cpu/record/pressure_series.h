/// @file cpu/record/pressure_series.h
/// @brief One reduced-pressure row, LAMMPS thermo sign, salamander column order.
///
/// Extensive virial and kinetic tensors are eV, Voigt xx yy zz xy xz yz.
/// P* = (virial + kinetic) / V / ε with σ = 1 Å. The printed columns are
/// −P*, the same sign salamander writes as `-pxx`.
#pragma once

#include <cmath>
#include <cstdio>
#include <ostream>
#include <string>

namespace ATV {
namespace cpu {

inline constexpr char kPressureSeriesHeader[] =
    "# step strain -pxx* -pyy* -pzz* -pyz* -pxz* -pxy* lx ly lz yz xz xy\n";

struct PressureSeriesRow {
    long long step = 0;
    double strain = 0;
    /// −P* in salamander order: xx, yy, zz, yz, xz, xy.
    double minus_p_star[6]{};
    /// lx ly lz yz xz xy [Å].
    double box[6]{};
};

/// @brief Fill one row from a row-major cell and the two extensive tensors.
///
/// `cell` is xx, xy, xz, yx, yy, yz, zx, zy, zz. Extension strain is
/// (L − L0) / L0. Shear strain is the new tilt over the perpendicular
/// edge at capture (yz and xz over lz0, xy over ly0).
/// @return false when an edge, the volume, or ε is not positive.
inline bool pressure_series_row(const double stress[6], const double kinetic[6],
                                const double cell[9], const double cell0[9],
                                int direction, double epsilon_ev,
                                long long step, PressureSeriesRow& row,
                                std::string& error) {
    const double lx = cell[0];
    const double ly = cell[4];
    const double lz = cell[8];
    if (!(lx > 0.0) || !(ly > 0.0) || !(lz > 0.0) || !(epsilon_ev > 0.0)) {
        error = "pressure series: edge or epsilon is not positive";
        return false;
    }
    const double volume = lx * ly * lz;
    double strain = 0;
    if (direction == 0)
        strain = (lx - cell0[0]) / cell0[0];
    else if (direction == 1)
        strain = (ly - cell0[4]) / cell0[4];
    else if (direction == 2)
        strain = (lz - cell0[8]) / cell0[8];
    else if (direction == 3)
        strain = cell[5] / cell0[8];
    else if (direction == 4)
        strain = cell[2] / cell0[8];
    else if (direction == 5)
        strain = cell[1] / cell0[4];
    else {
        error = "pressure series: direction must be 0..5";
        return false;
    }
    if (!std::isfinite(strain)) {
        error = "pressure series: strain is not finite";
        return false;
    }
    // Voigt xx yy zz xy xz yz → salamander xx yy zz yz xz xy.
    const int src[6] = {0, 1, 2, 5, 4, 3};
    row.step = step;
    row.strain = strain;
    row.box[0] = lx;
    row.box[1] = ly;
    row.box[2] = lz;
    row.box[3] = cell[5];
    row.box[4] = cell[2];
    row.box[5] = cell[1];
    for (int i = 0; i < 6; ++i) {
        const double p_star =
            (stress[src[i]] + kinetic[src[i]]) / volume / epsilon_ev;
        row.minus_p_star[i] = -p_star;
    }
    error.clear();
    return true;
}

inline void write_pressure_series_row(std::ostream& out,
                                      const PressureSeriesRow& row) {
    char line[512];
    std::snprintf(line, sizeof(line),
                  "%lld %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e "
                  "%.16e %.16e %.16e %.16e\n",
                  static_cast<long long>(row.step), row.strain,
                  row.minus_p_star[0], row.minus_p_star[1], row.minus_p_star[2],
                  row.minus_p_star[3], row.minus_p_star[4], row.minus_p_star[5],
                  row.box[0], row.box[1], row.box[2], row.box[3], row.box[4],
                  row.box[5]);
    out << line;
}

} // namespace cpu
} // namespace ATV
