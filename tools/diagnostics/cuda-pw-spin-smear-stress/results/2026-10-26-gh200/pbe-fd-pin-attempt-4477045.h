#pragma once
#include <array>
namespace qe_h3 {
// Actual QE7.6 output, Slurm4476891, H3 upper-triangular 4.2/.3/.2/4/.1/3.8 A
// cell, PBE GTH-PBE-q1, Gamma, 6/24 Ry, nspin2, 5 bands, FD width .02 Ry.
// Native QE analytic physical stress; not a native Atomiverse result or FD.
// tools/qe_oracle/inputs/pw-spin-smear-stress/{capture.py,H_GTH_PBE_q1.gth}.
// Root retains original input/stdout/binary hashes with the oracle capture.
inline constexpr double energy_ha = -1.45937915;
inline constexpr std::array<double, 6> stress_ev_ang3 = {
    -.0488588428984615,  .015310279350017768, -.04363268937160686,
    .015100939398485292, .006028623341062469, .022194625650853346};
inline constexpr double energy_tolerance_ha = 1e-5;
inline constexpr double stress_tolerance_ev_ang3 = 2e-3;
} // namespace qe_h3
