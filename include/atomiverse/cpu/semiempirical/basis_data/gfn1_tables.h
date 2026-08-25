/// @file cpu/semiempirical/basis_data/gfn1_tables.h
/// @brief Static GFN1-xTB organic (H/C/N/O) parameters.
///
/// Transcribed from tblite `src/tblite/xtb/gfn1.f90` (LGPL-3.0-or-later),
/// which is bit-identical to xtb `param_gfn1-xtb.txt` (Grimme, Bannwarth,
/// Shushkov, JCTC 2017, DOI 10.1021/acs.jctc.7b00118). No runtime parser.
#pragma once

#include <cstddef>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief tblite eV→Hartree used when the GFN1 tables were published.
inline constexpr double kGfn1EvToHa = 1.0 / 27.21138505;

/// @brief One GFN1 element row (organic CHNO).
struct Gfn1Element {
    int z = 0;            ///< Atomic number.
    int n_shell = 0;      ///< Number of AO shells (2 for H/C/N/O).
    int ang[2] = {0, 0};  ///< Angular momentum per shell (0=s, 1=p).
    int pqn[2] = {0, 0};  ///< Principal quantum number per shell.
    int nprim[2] = {0, 0};///< Stewart STO-nG primitive count per shell.
    double level[2] = {0, 0};   ///< Onsite energy h_l [Hartree].
    double zeta[2] = {0, 0};    ///< Slater exponent [1/Bohr].
    double kcn[2] = {0, 0};     ///< CN shift subtracted from h_l [Hartree].
    double shpoly[2] = {0, 0};  ///< Distance polynomial k_poly (dimensionless).
    double refocc[2] = {0, 0};  ///< Reference shell occupation [e].
    double lgam[2] = {1, 1};    ///< Shell hardness scale (dimensionless).
    bool valence[2] = {true, true}; ///< False = polarization (H 2s).
    double gam = 0;     ///< Atomic chemical hardness η [Hartree].
    double gam3 = 0;    ///< Third-order Γ [Hartree] (tblite ×0.1 of GAM3).
    double zeff = 0;    ///< Repulsive effective charge (dimensionless).
    double alpha = 0;   ///< Repulsive exponent α [1/Bohr].
    double en = 0;      ///< Pauling electronegativity (dimensionless).
    double rad = 0;     ///< Pyykkö covalent radius [Å]; converted to Bohr in Π.
    int n_val = 0;      ///< Valence electron count (refocc sum).
    int n_ao = 0;       ///< AO count (s=1, p=3 per shell).
};

/// @brief H, C, N, O in Z order.
inline constexpr Gfn1Element kGfn1Elements[] = {
    // H: 1s (valence) + 2s (polarization). shpoly s = 0.
    {1,
     2,
     {0, 0},
     {1, 2},
     {4, 3},
     {-10.923452 * kGfn1EvToHa, -2.171902 * kGfn1EvToHa},
     {1.207940, 1.993207},
     {6.5540712 * 0.01 * kGfn1EvToHa, 1.3031412 * 0.01 * kGfn1EvToHa},
     {0.0, 0.0},
     {1.0, 0.0},
     {1.0, 1.0},
     {true, false},
     0.470099,
     0.0,
     1.116244,
     2.209700,
     2.20,
     0.32,
     1,
     2},
    // C: 2s + 2p.
    {6,
     2,
     {0, 1},
     {2, 2},
     {6, 6},
     {-13.587210 * kGfn1EvToHa, -10.052785 * kGfn1EvToHa},
     {1.960324, 1.832096},
     {8.1523260 * 0.01 * kGfn1EvToHa, -3.0158355 * 0.01 * kGfn1EvToHa},
     {-7.082170 * 0.01, 0.812216 * 0.01},
     {2.0, 2.0},
     {1.0, 1.0 - 0.0471181},
     {true, true},
     0.479988,
     1.053856 * 0.1,
     4.428763,
     1.281954,
     2.55,
     0.75,
     4,
     4},
    // N: 2s + 2p.
    {7,
     2,
     {0, 1},
     {2, 2},
     {6, 6},
     {-20.058000 * kGfn1EvToHa, -12.889326 * kGfn1EvToHa},
     {2.050067, 2.113682},
     {12.0348000 * 0.01 * kGfn1EvToHa, -3.8667978 * 0.01 * kGfn1EvToHa},
     {-12.745585 * 0.01, -1.428367 * 0.01},
     {2.0, 3.0},
     {1.0, 1.0 + 0.0315090},
     {true, true},
     0.476106,
     0.042507 * 0.1,
     5.498808,
     1.727773,
     3.04,
     0.71,
     5,
     4},
    // O: 2s + 2p.
    {8,
     2,
     {0, 1},
     {2, 2},
     {6, 6},
     {-23.398376 * kGfn1EvToHa, -17.886554 * kGfn1EvToHa},
     {2.345365, 2.153060},
     {14.0390256 * 0.01 * kGfn1EvToHa, -5.3659662 * 0.01 * kGfn1EvToHa},
     {-13.729047 * 0.01, -4.453341 * 0.01},
     {2.0, 4.0},
     {1.0, 1.0 + 0.0374608},
     {true, true},
     0.583349,
     -0.005102 * 0.1,
     5.171786,
     2.004253,
     3.44,
     0.64,
     6,
     4},
};

inline constexpr std::size_t kGfn1NElements =
    sizeof(kGfn1Elements) / sizeof(kGfn1Elements[0]);

/// @brief Global GFN1 Hamiltonian / D3 / repulsion numbers (tblite gfn1.f90).
inline constexpr double kGfn1Ks = 1.85;
inline constexpr double kGfn1Kp = 2.25;
inline constexpr double kGfn1Ksp = 2.08;
inline constexpr double kGfn1Kdiff = 2.85;      ///< Polarization-shell k.
inline constexpr double kGfn1KEn = -7.0e-3;     ///< Electronegativity scale.
inline constexpr double kGfn1KHH = 0.96;        ///< H–H pair k.
inline constexpr double kGfn1KNH = 1.04;        ///< N–H pair k (tblite).
inline constexpr double kGfn1GExp = 2.0;        ///< Klopman–Ohno exponent.
inline constexpr double kGfn1RepKExp = 1.5;     ///< Repulsion R^kexp.
inline constexpr double kGfn1D3s6 = 1.0;
inline constexpr double kGfn1D3s8 = 2.4;
inline constexpr double kGfn1D3a1 = 0.63;
inline constexpr double kGfn1D3a2 = 5.0;
inline constexpr double kGfn1D3k1 = 16.0;       ///< D3 CN steepness.

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
