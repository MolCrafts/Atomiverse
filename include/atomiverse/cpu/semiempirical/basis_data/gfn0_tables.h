/// @file cpu/semiempirical/basis_data/gfn0_tables.h
/// @brief Static GFN0-xTB H/C/N/O rows transcribed from xtb/pprcht gfn0_param.
#pragma once

#include <cstddef>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief One GFN0 element (two valence shells for H/C/N/O).
struct Gfn0Element {
    int z = 0;
    int n_shell = 0;           ///< 2 for H/C/N/O.
    int pqn[2] = {0, 0};       ///< Principal quantum numbers.
    int ang[2] = {0, 0};       ///< 0=s, 1=p.
    int valence[2] = {0, 0};   ///< 1 = valence, 0 = polarisation.
    int nprim[2] = {0, 0};     ///< STO-nG primitive count.
    double zeta[2] = {0, 0};   ///< Slater exponents.
    double self_e[2] = {0, 0}; ///< Atomic levels [eV].
    double k_cn[2] = {0, 0};   ///< CN shift of levels [eV].
    double k_q_shell[2] = {0, 0};
    double k_q_atom = 0;
    double shell_poly[2] = {0, 0}; ///< s, p polynomial strengths.
    double en_ham = 0;             ///< Hamiltonian electronegativity.
    double rad_ang = 0;            ///< Atomic radius [Å] (converted internally).
    double ref_occ[2] = {0, 0};
    double rep_alpha = 0;
    double zeff = 0;
    double eeq_chi = 0;
    double eeq_gam = 0;
    double eeq_kcn = 0;
    double eeq_alp = 0;
    double srb_en = 0;
    double srb_r0 = 0;
    double srb_cnfak = 0;
    double rcov_d3_ang = 0; ///< Pyykkö covalent radius [Å] before 4/3.
};

/// Transcribed from pprcht/gfn0 `gfn0_param.f90` (xtb 6.7.1 GFN0).
inline constexpr Gfn0Element kGfn0Elements[] = {
    // H
    {1, 2, {1, 2}, {0, 0}, {1, 0}, {3, 2}, {1.2209169, 2.0165983},
     {-11.9223639, -2.8061095}, {0.07116904, 0.07116904}, {0.2591008, 0.2591008},
     0.2473983, {-3.6355142, 0.0}, 1.92, 0.32, {1.0, 0.0}, 2.1885472, 1.2455414,
     1.2500000, -0.3023159, 0.0248762, 0.7490227, 2.30085633, 0.55682207,
     0.17957827, 0.32},
    // C
    {6, 2, {2, 2}, {0, 1}, {1, 1}, {4, 3}, {1.9915841, 1.7845353},
     {-15.7545853, -9.7975356}, {-0.55477603, 0.15631408}, {0.1683176, 0.2988821},
     0.0908886, {-12.8096830, -8.82326510}, 2.48, 0.75, {1.0, 3.0}, 1.2680750,
     4.3101011, 1.3860275, 0.0308461, 0.0643920, 1.8288757, 2.49640820,
     0.98310699, 0.10514203, 0.75},
    // N
    {7, 2, {2, 2}, {0, 1}, {1, 1}, {4, 3}, {2.3225221, 2.0626548},
     {-18.8417536, -11.5390772}, {-0.02818725, 0.06027542},
     {1.6770083, -0.2856884}, -0.0630602, {-12.6808155, -8.80380760}, 2.97, 0.71,
     {1.5, 3.5}, 1.6211038, 4.5460146, 1.5341534, 0.0559522, 0.1053273, 1.9346081,
     2.81007174, 0.98423007, 0.09753494, 0.71},
    // O
    {8, 2, {2, 2}, {0, 1}, {1, 1}, {4, 3}, {2.6718248, 2.2239915},
     {-17.9332316, -11.8400363}, {-0.12537900, -0.02934837},
     {0.5947188, 0.0916715}, -0.0504985, {-16.1325184, -5.51085690}, 3.44, 0.64,
     {2.0, 4.0}, 2.1037547, 4.7850603, 1.5378836, 0.0581228, 0.1394809, 1.6974795,
     4.51078438, 0.76716063, 0.30470380, 0.63},
};

inline constexpr std::size_t kGfn0NElements =
    sizeof(kGfn0Elements) / sizeof(kGfn0Elements[0]);

// Global GFN0 Hamiltonian / repulsion / SRB scales (gfn0Globals).
inline constexpr double kGfn0Kss = 2.0000000;
inline constexpr double kGfn0Kpp = 2.4868000;
inline constexpr double kGfn0Kdiff = 1.1241000;
inline constexpr double kGfn0EnScaleS = 0.6;
inline constexpr double kGfn0EnScaleP = -0.1;
inline constexpr double kGfn0EnScale4 = 4.0;
inline constexpr double kGfn0RepEnScale = -0.0900000;
inline constexpr double kGfn0SrbShift = 0.0537000;
inline constexpr double kGfn0SrbPrefactor = -0.0129000;
inline constexpr double kGfn0SrbSteepness = 3.4847000;
inline constexpr double kGfn0SrbEnScale = 0.5097000;
inline constexpr double kGfn0D4s8 = 2.8500000;
inline constexpr double kGfn0D4a1 = 0.8000000;
inline constexpr double kGfn0D4a2 = 4.6000000;
inline constexpr double kGfn0AutoAA = 0.52917726;
inline constexpr double kGfn0EvToAu = 1.0 / 27.21138505;

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
