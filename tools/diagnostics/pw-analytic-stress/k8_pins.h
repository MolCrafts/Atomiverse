// Executed QE7.6 skew Si2 LSDA eight-k reference; raw provenance in qe-k8-2026-10-28.
#pragma once
#include <array>
namespace pw_analytic_k8_qe {
inline constexpr double energy_ha = -7.7175672985039;
inline constexpr std::array<std::array<double, 3>, 2> forces_ha_bohr = {
    {{0.002702401679011487, 0.01685861726477207, -0.005140554553371032},
     {-0.002702401679011487, -0.01685861726477207, 0.005140554553371032}}};
inline constexpr std::array<double, 6> stress_ev_ang3 = {
    0.08377160546290773,   0.06884371123098748,  0.09561876004693437,
    -0.006901051094737125, -0.06626515413363718, -0.01686489535265207};
} // namespace pw_analytic_k8_qe
