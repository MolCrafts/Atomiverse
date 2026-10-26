#pragma once
#include <array>
namespace nc_example {
// Genuine QE7.5 .gth reader + analytic Fourier operators; si2.in/qe-raw.txt.
// Printed at tools/diagnostics/nc-extension/qe-raw.txt:700,715-716.
// Mermin F [Ha], full Cartesian forces [Ha/Bohr] (QE output is Ry/Bohr).
inline constexpr double energy_ha = -15.25142886 / 2;
inline constexpr std::array<std::array<double, 3>, 2> forces_ha_bohr = {
    {{{.03584083 / 2, .04163713 / 2, .03671977 / 2}},
     {{-.03584083 / 2, -.04163713 / 2, -.03671977 / 2}}}};
// Fixed before the first native acceptance run. No native output is a pin.
inline constexpr double energy_tol_ha = 1e-5;
inline constexpr double force_tol_ha_bohr = 1e-4;
} // namespace nc_example
