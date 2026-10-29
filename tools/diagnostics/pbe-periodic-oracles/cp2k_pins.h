// Generated exclusively from actual independent CP2K output.
#pragma once
#include <array>
namespace pbe_periodic_cp2k {
struct Pin {
    const char* id;
    int functional;
    double energy_ha;
    std::array<std::array<double, 3>, 2> forces_ha_bohr;
    std::array<double, 3> diagonal_stress_gpa;
};
inline constexpr double energy_tolerance_ha = 5e-6;
inline constexpr double force_tolerance_ha_bohr = 5e-5;
inline constexpr double diagonal_stress_tolerance_ha_bohr3 = 5e-5;
inline constexpr std::array<Pin, 3> pins = {{
    {"gpw-h2-rks-pbesol",
     0,
     -1.1583851012582971,
     {{{0.20142573, -0.074758962999999998, 0.27444613899999998},
       {-0.178356757, 0.177075174, 0.19585975799999999}}},
     {5.1893974544399999, -0.37109156543200006, 1.4953606498600001}},
    {"gpw-h2-rks-revpbe",
     1,
     -1.189031220462639,
     {{{0.20635825199999999, -0.071386694299999998, 0.27401088800000001},
       {-0.18390461799999999, 0.172308185, 0.191748475}}},
     {4.9794997493000004, -0.38745681521900005, 1.4686601027299999}},
    {"gpw-h2-rks-rpbe",
     2,
     -1.190393135074038,
     {{{0.206968862, -0.071243510100000004, 0.27429318000000003},
       {-0.184410505, 0.17221535600000001, 0.191706037}}},
     {4.9661054733899999, -0.39350162354099999, 1.4639012523400001}},
}};
} // namespace pbe_periodic_cp2k
