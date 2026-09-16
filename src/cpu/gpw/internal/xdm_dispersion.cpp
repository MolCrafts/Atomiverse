/// @file cpu/gpw/internal/xdm_dispersion.cpp
/// @brief Free-atom XDM pairwise energy + analytic forces (see header).

#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/cpu/gpw/internal/periodic.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

void free_atom(int Z, double& alpha, double& m1, double& m2, double& m3) {
    if (!xdm_supports_element(Z))
        throw std::runtime_error(
            "GPW XDM: element Z=" + std::to_string(Z)
            + " is not in the in-tree free-atom table (no fallback)");
    const double* row = kXdmFreeAtom[Z - 1];
    alpha = row[0];
    m1 = row[1];
    m2 = row[2];
    m3 = row[3];
}

} // namespace

XdmResult xdm_pbe(const std::vector<int>& Z,
                  const std::vector<std::array<double, 3>>& pos_bohr,
                  const double cell_bohr[3], bool periodic, double rcut_bohr,
                  double a1, double a2_bohr) {
    const std::size_t n = Z.size();
    if (pos_bohr.size() != n)
        throw std::runtime_error("GPW XDM: Z and pos size mismatch");

    std::vector<double> alpha(n), m1(n), m2(n), m3(n);
    for (std::size_t i = 0; i < n; ++i)
        free_atom(Z[i], alpha[i], m1[i], m2[i], m3[i]);

    XdmResult res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    int m[3] = {0, 0, 0};
    if (periodic)
        pair_image_counts(cell_bohr, true, rcut_bohr, m);

    const double rcut2 = rcut_bohr * rcut_bohr;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double fac =
                alpha[i] * alpha[j]
                / (m1[i] * alpha[j] + m1[j] * alpha[i]);
            const double c6 = fac * m1[i] * m1[j];
            const double c8 =
                1.5 * fac * (m1[i] * m2[j] + m2[i] * m1[j]);
            const double c10 =
                2.0 * fac * (m1[i] * m3[j] + m3[i] * m1[j])
                + 4.2 * fac * m2[i] * m2[j];
            if (!(c6 > 0.0) || !(c8 > 0.0) || !(c10 > 0.0))
                throw std::runtime_error(
                    "GPW XDM: non-positive C6/C8/C10 for a pair");
            const double rc =
                (std::sqrt(c8 / c6) + std::pow(c10 / c6, 0.25)
                 + std::sqrt(c10 / c8))
                / 3.0;
            const double rvdw = a1 * rc + a2_bohr;
            const double d6 = std::pow(rvdw, 6.0);
            const double d8 = std::pow(rvdw, 8.0);
            const double d10 = std::pow(rvdw, 10.0);

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
                        const double r4 = r2 * r2;
                        const double r6 = r4 * r2;
                        const double r8 = r4 * r4;
                        const double r10 = r8 * r2;
                        const double den6 = r6 + d6;
                        const double den8 = r8 + d8;
                        const double den10 = r10 + d10;
                        double eij =
                            -c6 / den6 - c8 / den8 - c10 / den10;
                        // dE/dR = + 6 C6 R^5 / den6² + 8 C8 R^7 / den8²
                        //         + 10 C10 R^9 / den10²
                        // Force on i along unit vector u: F = −(dE/dR) u
                        // with R vector from j to i, so
                        // F_i = −(dE/dR)(R/r)  and dE/dR > 0 ⇒ attraction
                        // pulls i toward j when R points from j to i? 
                        // E decreases as r grows in long range? No: E is
                        // more negative at shorter r (until damping).
                        // dE/dr = +6 C6 r^5/den6² + … > 0 means energy
                        // increases with r → force F_i = −∇_i E pulls
                        // atoms together: ∇_i r = R/r so
                        // F_i = −(dE/dr)(R/r).
                        const double dEdr =
                            6.0 * c6 * r4 * r / (den6 * den6)
                            + 8.0 * c8 * r6 * r / (den8 * den8)
                            + 10.0 * c10 * r8 * r / (den10 * den10);
                        double fx = -dEdr * (Rx / r);
                        double fy = -dEdr * (Ry / r);
                        double fz = -dEdr * (Rz / r);
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
