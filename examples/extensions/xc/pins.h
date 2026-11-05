#pragma once
// PySCF 2.14.0 actual output: tools/pyscf_oracle/inputs/xc-extension/.
// Energies [Ha], gradients [Ha/Bohr]. See README.md for exact inputs.
namespace xc_example {
inline constexpr double water_energy = -72.4239632859654;
inline constexpr double water_gradient[][3] = {
    {8.222391183109284e-16, 3.0080501093021543e-15, -0.17581334785142078},
    {-5.1083137406993846e-17, -0.08793293389203738, 0.08791316434424123},
    {6.93195054147404e-17, 0.08793293389203938, 0.08791316434423813},
};
inline constexpr double h2plus_energy = -0.48893340293176235;
inline constexpr double h2plus_gradient[][3] = {
    {5.398412489242648e-17, 3.649422103014986e-17, 0.043151985838815576},
    {-4.366383318766491e-17, 9.762578909393011e-18, -0.04315198583881527},
};
inline constexpr double energy_tol = 1e-6;
inline constexpr double gradient_tol = 2e-5;
} // namespace xc_example
