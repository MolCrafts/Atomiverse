/// @file cpu/gpw/internal/hirshfeld_volumes.cpp
/// @brief Hirshfeld volume ratios from a GPW density grid (see header).

#include "atomiverse/cpu/gpw/internal/hirshfeld_volumes.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

double min_image_1d(double d, double L, bool periodic) {
    if (!periodic || !(L > 0.0))
        return d;
    d -= L * std::nearbyint(d / L);
    return d;
}

} // namespace

std::vector<double>
hirshfeld_volume_ratios(const UniformGrid& grid, const std::vector<double>& rho,
                        const std::vector<int>& Z,
                        const std::vector<std::array<double, 3>>& pos_bohr,
                        const double cell_bohr[3], bool periodic) {
    const std::size_t n = Z.size();
    if (pos_bohr.size() != n)
        throw std::runtime_error("GPW Hirshfeld: Z and pos size mismatch");
    if (static_cast<long>(rho.size()) != grid.npts())
        throw std::runtime_error("GPW Hirshfeld: rho size does not match grid");
    if (n == 0)
        return {};

    for (std::size_t i = 0; i < n; ++i) {
        if (!hirshfeld_supports_element(Z[i]))
            throw std::runtime_error(
                "GPW Hirshfeld: element Z=" + std::to_string(Z[i])
                + " is not in the free-atom table (no fallback)");
    }

    std::vector<double> v_free(n);
    for (std::size_t i = 0; i < n; ++i)
        v_free[i] = hirshfeld_v_free(Z[i]);

    std::vector<double> V_eff(n, 0.0);
    std::vector<double> rho_a(n);
    std::vector<double> r3(n);
    const double w = grid.weight();
    const long npts = grid.npts();
    double r[3];
    for (long g = 0; g < npts; ++g) {
        const double rho_g = rho[static_cast<std::size_t>(g)];
        if (!(rho_g > 0.0))
            continue;
        grid.point(g, r);

        // Free-atom densities at this grid point.
        double sum_free = 0.0;
        for (std::size_t a = 0; a < n; ++a) {
            const double dx =
                min_image_1d(r[0] - pos_bohr[a][0], cell_bohr[0], periodic);
            const double dy =
                min_image_1d(r[1] - pos_bohr[a][1], cell_bohr[1], periodic);
            const double dz =
                min_image_1d(r[2] - pos_bohr[a][2], cell_bohr[2], periodic);
            const double rr = std::sqrt(dx * dx + dy * dy + dz * dz);
            rho_a[a] = hirshfeld_rho_free(Z[a], rr);
            r3[a] = rr * rr * rr;
            sum_free += rho_a[a];
        }
        if (!(sum_free > 0.0))
            continue;

        for (std::size_t a = 0; a < n; ++a) {
            const double wa = rho_a[a] / sum_free;
            V_eff[a] += wa * rho_g * r3[a] * w;
        }
    }

    std::vector<double> ratios(n);
    for (std::size_t a = 0; a < n; ++a) {
        if (!(v_free[a] > 0.0))
            throw std::runtime_error("GPW Hirshfeld: V_free must be positive");
        const double v = V_eff[a] / v_free[a];
        if (!(v > 0.0) || !std::isfinite(v))
            throw std::runtime_error("GPW Hirshfeld: non-positive or "
                                     "non-finite volume ratio for atom "
                                     + std::to_string(a));
        ratios[a] = v;
    }
    return ratios;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
