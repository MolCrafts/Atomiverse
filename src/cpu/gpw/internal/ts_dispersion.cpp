/// @file cpu/gpw/internal/ts_dispersion.cpp
/// @brief Tkatchenko–Scheffler pairwise energy + analytic forces (see header).

#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/gpw/internal/periodic.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

/// @brief Free-atom α₀, C₆, R_vdW for atomic number Z (throws if unsupported).
void free_atom_params(int Z, double& alpha, double& c6, double& rvdw) {
    if (!ts_supports_element(Z))
        throw std::runtime_error(
            "GPW TS: element Z=" + std::to_string(Z)
            + " is not in the in-tree free-atom table (no fallback)");
    const double* row = kTsFreeAtom[Z - 1];
    alpha = row[0];
    c6 = row[1];
    rvdw = row[2];
}

/// @brief TS combination rule for C6_ij (libmbd `combine_C6`).
double combine_c6(double c6i, double c6j, double ai, double aj) {
    return 2.0 * c6i * c6j / ((aj / ai) * c6i + (ai / aj) * c6j);
}

/// @brief Fermi damping f and df/dR (scalar radius).
void damping_fermi(double r, double s_vdw, double d, double& f, double& dfdR) {
    const double eta = r / s_vdw;
    const double x = -d * (eta - 1.0);
    // Avoid overflow in exp for very short / long distances.
    if (x > 60.0) {
        f = 0.0;
        dfdR = 0.0;
        return;
    }
    if (x < -60.0) {
        f = 1.0;
        dfdR = 0.0;
        return;
    }
    const double e = std::exp(x);
    f = 1.0 / (1.0 + e);
    // df/dR = d * f * (1-f) / s_vdw
    dfdR = d * f * (1.0 - f) / s_vdw;
}

} // namespace

TsResult ts_pbe(const std::vector<int>& Z,
                const std::vector<std::array<double, 3>>& pos_bohr,
                const double cell_bohr[3], bool periodic, double rcut_bohr,
                double s_r, double d_fermi,
                const std::vector<double>& vol_ratio) {
    const std::size_t n = Z.size();
    if (pos_bohr.size() != n)
        throw std::runtime_error("GPW TS: Z and pos size mismatch");
    if (!vol_ratio.empty() && vol_ratio.size() != n)
        throw std::runtime_error("GPW TS: vol_ratio size mismatch");

    std::vector<double> alpha(n), c6(n), rvdw(n);
    for (std::size_t i = 0; i < n; ++i) {
        free_atom_params(Z[i], alpha[i], c6[i], rvdw[i]);
        const double v = vol_ratio.empty() ? 1.0 : vol_ratio[i];
        if (!(v > 0.0) || !std::isfinite(v))
            throw std::runtime_error("GPW TS: volume ratio must be positive finite");
        // α ∝ v, C6 ∝ v², R ∝ v^{1/3}
        alpha[i] *= v;
        c6[i] *= v * v;
        rvdw[i] *= std::cbrt(v);
    }

    TsResult res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    int m[3] = {0, 0, 0};
    if (periodic)
        pair_image_counts(cell_bohr, true, rcut_bohr, m);

    const double rcut2 = rcut_bohr * rcut_bohr;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            for (int tx = -m[0]; tx <= m[0]; ++tx) {
                for (int ty = -m[1]; ty <= m[1]; ++ty) {
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        if (tx == 0 && ty == 0 && tz == 0 && i == j)
                            continue;
                        const double Rx =
                            pos_bohr[i][0] - pos_bohr[j][0]
                            - (periodic ? tx * cell_bohr[0] : 0.0);
                        const double Ry =
                            pos_bohr[i][1] - pos_bohr[j][1]
                            - (periodic ? ty * cell_bohr[1] : 0.0);
                        const double Rz =
                            pos_bohr[i][2] - pos_bohr[j][2]
                            - (periodic ? tz * cell_bohr[2] : 0.0);
                        const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                        if (r2 < 1e-24)
                            continue;
                        if (periodic && r2 > rcut2)
                            continue;
                        const double r = std::sqrt(r2);
                        const double cij =
                            combine_c6(c6[i], c6[j], alpha[i], alpha[j]);
                        const double s_vdw = s_r * (rvdw[i] + rvdw[j]);
                        double f = 0.0, dfdR = 0.0;
                        damping_fermi(r, s_vdw, d_fermi, f, dfdR);
                        const double inv_r6 = 1.0 / (r2 * r2 * r2);
                        double eij = -cij * f * inv_r6;
                        // Force on i along Rij: F = -dE/dR_i
                        // E = -C6 f / R^6
                        // dE/dR = -C6 (dfdR / R^6 - 6 f / R^7)
                        // unit vector u = R/r; dR_i component of pair vector.
                        const double dEdR =
                            -cij * (dfdR * inv_r6 - 6.0 * f * inv_r6 / r);
                        double fx = -dEdR * (Rx / r);
                        double fy = -dEdR * (Ry / r);
                        double fz = -dEdR * (Rz / r);
                        if (i == j) {
                            eij *= 0.5;
                            fx *= 0.5;
                            fy *= 0.5;
                            fz *= 0.5;
                        }
                        res.energy += eij;
                        res.forces[i][0] += fx;
                        res.forces[i][1] += fy;
                        res.forces[i][2] += fz;
                        if (i != j) {
                            res.forces[j][0] -= fx;
                            res.forces[j][1] -= fy;
                            res.forces[j][2] -= fz;
                        }
                    }
                }
            }
        }
    }
    return res;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
