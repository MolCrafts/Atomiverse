/// @file cpu/semiempirical/basis_data/ipea_tables.h
/// @brief Static IPEA-xTB organic (H/C/N/O) parameters.
///
/// Transcribed from tblite `src/tblite/xtb/ipea1.f90` (LGPL-3.0-or-later),
/// which is bit-identical to xtb `param_ipea-xtb.txt` (Ásgeirsson, Bauer,
/// Grimme, Chem. Sci. 2017, DOI 10.1039/c7sc00601b). No runtime parser.
#pragma once

#include <cstddef>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief tblite eV→Hartree used when the IPEA tables were published.
inline constexpr double kIpeaEvToHa = 1.0 / 27.21138505;

/// @brief One IPEA element row (organic CHNO). Extra 3s on C/N/O.
struct IpeaElement {
    int z = 0;              ///< Atomic number.
    int n_shell = 0;        ///< Number of AO shells (2 H, 3 C/N/O).
    int ang[3] = {0, 0, 0}; ///< Angular momentum per shell (0=s, 1=p).
    int pqn[3] = {0, 0, 0}; ///< Principal quantum number per shell.
    int nprim[3] = {0, 0, 0}; ///< Stewart STO-nG primitive count per shell.
    double level[3] = {0, 0, 0};   ///< Onsite energy h_l [Hartree].
    double zeta[3] = {0, 0, 0};    ///< Slater exponent [1/Bohr].
    double kcn[3] = {0, 0, 0};     ///< CN shift subtracted from h_l [Hartree].
    double shpoly[3] = {0, 0, 0};  ///< Distance polynomial k_poly (dimensionless).
    double refocc[3] = {0, 0, 0};  ///< Reference shell occupation [e].
    double lgam[3] = {1, 1, 1};    ///< Shell hardness scale (dimensionless).
    bool valence[3] = {true, true, true}; ///< False = extra (n+1)s.
    double gam = 0;     ///< Atomic chemical hardness η [Hartree].
    double gam3 = 0;    ///< Third-order Γ [Hartree] (tblite ×0.1 of GAM3).
    double zeff = 0;    ///< Repulsive effective charge (dimensionless).
    double alpha = 0;   ///< Repulsive exponent α [1/Bohr].
    double en = 0;      ///< Pauling electronegativity (dimensionless).
    double rad = 0;     ///< Pyykkö covalent radius [Å]; converted to Bohr in Π.
    int n_val = 0;      ///< Valence electron count (refocc sum of valence shells).
    int n_ao = 0;       ///< AO count (s=1, p=3 per shell).
};

/// @brief H, C, N, O in Z order. kcn already includes tblite ×0.01 × eV→Ha.
inline constexpr IpeaElement kIpeaElements[] = {
    // H: 1s (valence) + 2s (polarization).
    {1,
     2,
     {0, 0, 0},
     {1, 2, 0},
     {4, 3, 0},
     {-10.521941 * kIpeaEvToHa, -3.560151 * kIpeaEvToHa, 0},
     {1.306199, 0.398421, 0},
     {10.5219410 * 0.01 * kIpeaEvToHa, 3.5601510 * 0.01 * kIpeaEvToHa, 0},
     {0.0, 0.0, 0.0},
     {1.0, 0.0, 0.0},
     {1.0, 1.0, 1.0},
     {true, false, false},
     0.404839,
     1.500000 * 0.1,
     1.292559,
     2.133166,
     2.20,
     0.32,
     1,
     2},
    // C: 2s + 2p + 3s (polarization).
    {6,
     3,
     {0, 1, 0},
     {2, 2, 3},
     {6, 6, 3},
     {-15.968100 * kIpeaEvToHa, -9.777107 * kIpeaEvToHa,
      -3.366990 * kIpeaEvToHa},
     {1.979793, 1.850537, 0.633095},
     {15.9681000 * 0.01 * kIpeaEvToHa, -2.9331321 * 0.01 * kIpeaEvToHa,
      3.3669900 * 0.01 * kIpeaEvToHa},
     {-2.692247 * 0.01, 8.177943 * 0.01, -2.692247 * 0.01},
     {2.0, 2.0, 0.0},
     {1.0, 1.0 - 0.0273283, 1.0},
     {true, true, false},
     0.639692,
     1.500000 * 0.1,
     4.427268,
     1.206815,
     2.55,
     0.75,
     4,
     5},
    // N: 2s + 2p + 3s (polarization).
    {7,
     3,
     {0, 1, 0},
     {2, 2, 3},
     {6, 6, 3},
     {-24.040776 * kIpeaEvToHa, -14.426520 * kIpeaEvToHa,
      -5.882278 * kIpeaEvToHa},
     {2.126160, 2.068425, 0.671040},
     {24.0407760 * 0.01 * kIpeaEvToHa, -4.3279560 * 0.01 * kIpeaEvToHa,
      5.8822780 * 0.01 * kIpeaEvToHa},
     {-10.560649 * 0.01, 0.940101 * 0.01, -10.560649 * 0.01},
     {2.0, 3.0, 0.0},
     {1.0, 1.0 + 0.0701245, 1.0},
     {true, true, false},
     0.492599,
     -1.000000 * 0.1,
     5.500005,
     1.622255,
     3.04,
     0.71,
     5,
     5},
    // O: 2s + 2p + 3s (polarization).
    {8,
     3,
     {0, 1, 0},
     {2, 2, 3},
     {6, 6, 3},
     {-26.695120 * kIpeaEvToHa, -16.014457 * kIpeaEvToHa,
      -6.622058 * kIpeaEvToHa},
     {2.575264, 2.159448, 0.958679},
     {26.6951200 * 0.01 * kIpeaEvToHa, -4.8043371 * 0.01 * kIpeaEvToHa,
      6.6220580 * 0.01 * kIpeaEvToHa},
     {-9.907396 * 0.01, 3.920125 * 0.01, -9.907396 * 0.01},
     {2.0, 4.0, 0.0},
     {1.0, 1.0 - 0.2619316, 1.0},
     {true, true, false},
     0.717333,
     -2.000000 * 0.1,
     5.178714,
     2.002624,
     3.44,
     0.63,
     6,
     5},
};

inline constexpr std::size_t kIpeaNElements =
    sizeof(kIpeaElements) / sizeof(kIpeaElements[0]);

/// @brief Global IPEA Hamiltonian / D3 / repulsion numbers (tblite ipea1.f90).
inline constexpr double kIpeaKs = 2.15;
inline constexpr double kIpeaKp = 2.30;
inline constexpr double kIpeaKsp = 2.25;
inline constexpr double kIpeaKdiff = 2.25;     ///< Polarization-shell k.
inline constexpr double kIpeaKEn = -7.0e-3;    ///< Electronegativity scale.
inline constexpr double kIpeaGExp = 1.0;       ///< Klopman–Ohno exponent.
inline constexpr double kIpeaRepKExp = 1.5;    ///< Repulsion R^kexp.
inline constexpr double kIpeaD3s6 = 1.0;
inline constexpr double kIpeaD3s8 = 2.4;
inline constexpr double kIpeaD3a1 = 0.63;
inline constexpr double kIpeaD3a2 = 5.0;
inline constexpr double kIpeaD3k1 = 16.0;      ///< D3 / exp-CN steepness.

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
