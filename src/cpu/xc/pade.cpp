/// @file src/cpu/xc/pade.cpp
/// @brief Implementation of the GTH Padé LDA functional (see header).
///
/// Direct port of CP2K `src/xc/xc_pade.F` routines `pade_lsd_0` (energy) and
/// `pade_lsd_1` (potential), with the spin-interpolation function `f(x)` and
/// its first derivative from `calc_fx_*` in `xc_functionals_utilities.F`.

#include "atomiverse/cpu/xc/pade.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace xc {

namespace {

// Padé coefficients (paramagnetic set), CP2K xc_pade.F lines 47-54.
constexpr double a0 = 0.4581652932831429e+0;
constexpr double a1 = 0.2217058676663745e+1;
constexpr double a2 = 0.7405551735357053e+0;
constexpr double a3 = 0.1968227878617998e-1;
constexpr double b1 = 1.0000000000000000e+0;
constexpr double b2 = 0.4504130959426697e+1;
constexpr double b3 = 0.1110667363742916e+1;
constexpr double b4 = 0.2359291751427506e-1;

// Spin-interpolation increments (ferro - para), CP2K xc_pade.F lines 56-63.
constexpr double da0 = 0.119086804055547e+0;
constexpr double da1 = 0.6157402568883345e+0;
constexpr double da2 = 0.1574201515892867e+0;
constexpr double da3 = 0.3532336663397157e-2;
constexpr double db1 = 0.0000000000000000e+0;
constexpr double db2 = 0.2673612973836267e+0;
constexpr double db3 = 0.2052004607777787e+0;
constexpr double db4 = 0.4200005045691381e-2;

constexpr double f13 = 1.0 / 3.0;
constexpr double f23 = 2.0 / 3.0;
constexpr double f43 = 4.0 / 3.0;

// Project density cutoff (matches cpu/scf/xc_functional.cpp). NOTE: CP2K's
// xc_pade uses its own eps_rho (default ~1e-10); the difference only affects
// vanishing densities and is revisited when GPW M1 bit-parity is checked.
constexpr double kRhoMin = 1.0e-12;

/// @brief Spin-interpolation function f(x) and its derivative df/dx.
/// f(x) = ((1+x)^(4/3) + (1-x)^(4/3) - 2) / (2^(4/3) - 2),
/// with the |x| >= 1 limits taken from CP2K calc_fx (one spin density -> 0).
void spin_function(double x, double& f, double& df) {
    const double fxfac = 1.0 / (std::pow(2.0, f43) - 2.0);
    if (x <= -1.0) {
        f = 1.0;
        df = -f43 * fxfac * std::pow(2.0, f13);
    } else if (x >= 1.0) {
        f = 1.0;
        df = f43 * fxfac * std::pow(2.0, f13);
    } else {
        const double xp = std::pow(1.0 + x, f43) + std::pow(1.0 - x, f43) - 2.0;
        f = xp * fxfac;
        df = (std::pow(1.0 + x, f13) - std::pow(1.0 - x, f13)) * fxfac * f43;
    }
}

} // namespace

PadeXC pade_lsd(double rho_a, double rho_b) {
    PadeXC out;
    const double rhoab = rho_a + rho_b;
    if (rhoab < kRhoMin)
        return out;

    // rs = (3 / (4 pi rho))^(1/3).
    const double rsfac = std::cbrt(3.0 / (4.0 * M_PI));
    const double rs = rsfac * std::pow(rhoab, -f13);

    // Spin-interpolation function f(x) and derivative, x = (rhoa-rhob)/rho.
    const double x = (rho_a - rho_b) / rhoab;
    double fx1, fx2;
    spin_function(x, fx1, fx2);

    // Density-interpolated Padé coefficients (CP2K pade_lsd_*).
    const double fa0 = a0 + fx1 * da0;
    const double fa1 = a1 + fx1 * da1;
    const double fa2 = a2 + fx1 * da2;
    const double fa3 = a3 + fx1 * da3;
    const double fb1 = b1 + fx1 * db1;
    const double fb2 = b2 + fx1 * db2;
    const double fb3 = b3 + fx1 * db3;
    const double fb4 = b4 + fx1 * db4;

    const double p = fa0 + (fa1 + (fa2 + fa3 * rs) * rs) * rs;
    const double q = (fb1 + (fb2 + (fb3 + fb4 * rs) * rs) * rs) * rs;
    const double dpv = fa1 + (2.0 * fa2 + 3.0 * fa3 * rs) * rs;
    const double dq =
        fb1 + (2.0 * fb2 + (3.0 * fb3 + 4.0 * fb4 * rs) * rs) * rs;
    const double xp = da0 + (da1 + (da2 + da3 * rs) * rs) * rs;
    const double xq = (db1 + (db2 + (db3 + db4 * rs) * rs) * rs) * rs;

    const double dr = (dpv * q - p * dq) / (q * q);
    const double dx = 2.0 * (xp * q - p * xq) / (q * q) * fx2 / rhoab;
    const double dc = f13 * rs * dr - p / q;

    out.exc = -p / q;             // epade, energy per electron
    out.vrho_a = dc - dx * rho_b; // CP2K pade_lsd_1 pota
    out.vrho_b = dc + dx * rho_a; // CP2K pade_lsd_1 potb
    return out;
}

} // namespace xc
} // namespace cpu
} // namespace ATV
