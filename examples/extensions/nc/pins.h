#pragma once
#include <array>
namespace nc_example {
// Genuine QE 7.5 run (commit 770a0b2d12928a67048e2f3da8d10d057e52179e-dirty,
// native .gth reader + analytic Fourier operators) of
// tools/qe_oracle/inputs/nc-extension-si2/si2.in: its printed final F and
// total-force block (capture.py there reproduces them).
// Mermin F [Ha], full Cartesian forces [Ha/Bohr] (QE output is Ry/Bohr).
inline constexpr double energy_ha = -15.25142886 / 2;
inline constexpr std::array<std::array<double, 3>, 2> forces_ha_bohr = {
    {{{.03584083 / 2, .04163713 / 2, .03671977 / 2}},
     {{-.03584083 / 2, -.04163713 / 2, -.03671977 / 2}}}};
// Fixed before the first native acceptance run. No native output is a pin.
inline constexpr double energy_tol_ha = 1e-5;
inline constexpr double force_tol_ha_bohr = 1e-4;
} // namespace nc_example
