#pragma once
#include <array>
// Generated from actual QE7.6 Hartree XML; raw input/output and hashes beside this file.
namespace pw_analytic_qe {
inline constexpr double energy_ha = -7.159845224831926;
inline constexpr std::array<std::array<double, 3>, 2> forces_ha_bohr = {{
    {0.0002016635749736829, 0.009735033562325909, -0.01182484721661555},
    {-0.0002016635749736829, -0.009735033562325909, 0.01182484721661555},
}};
inline constexpr std::array<double, 6> stress_ev_ang3 = {
    0.3514711424040171,    0.33872180521897016,   0.3542303214175935,
    0.0070914307678760224, 0.0020331405954656322, -0.006170621361208662};
} // namespace pw_analytic_qe
