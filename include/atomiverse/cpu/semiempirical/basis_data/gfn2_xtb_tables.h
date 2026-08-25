/// @file cpu/semiempirical/basis_data/gfn2_xtb_tables.h
/// @brief Static GFN2-xTB element parameters (H/C/N/O). Transcribed from
///        param_gfn2-xtb.txt / tblite xtb/gfn2.f90 (Bannwarth 2019).
#pragma once

#include <cstddef>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief One GFN2-xTB element row (valence shells ≤ 2).
struct Gfn2ElementRow {
    int z = 0;            ///< Atomic number.
    int n_shell = 0;      ///< Number of valence shells.
    int l[2]{};           ///< Angular momentum per shell (0=s, 1=p).
    int pqn[2]{};         ///< Principal quantum number per shell.
    int n_prim[2]{};      ///< STO-nG primitive count per shell.
    double zeta[2]{};     ///< Slater exponent per shell (dimensionless).
    double level[2]{};    ///< Atomic level ε [eV] (converted to Eh at use).
    double kcn[2]{};      ///< CN shift of ε [eV] (tblite p_kcn, already /10).
    double shpoly[2]{};   ///< Distance polynomial (tblite, already ×0.01).
    double refocc[2]{};   ///< Reference occupation per shell [e].
    double hubbard = 0;   ///< Isotropic hardness [Eh].
    double lgam[2]{};     ///< Shell Hubbard scale (1 + Δ).
    double gam3 = 0;      ///< Third-order Hubbard derivative [Eh] (s-scale).
    double alpha_rep = 0; ///< Repulsion exponent α.
    double zeff = 0;      ///< Effective nuclear charge for repulsion.
    double dkernel = 0;   ///< Dipole AXC kernel [Eh].
    double qkernel = 0;   ///< Quadrupole AXC kernel [Eh].
    double mprad = 0;     ///< AES base damping radius [Å].
    double mpvcn = 0;     ///< AES valence CN.
    double en = 0;        ///< Pauling electronegativity.
    double rcov = 0;      ///< Pyykkö covalent radius [Å].
};

/// @brief GFN2-xTB H/C/N/O. Values from tblite `tblite_xtb_gfn2`.
inline constexpr Gfn2ElementRow kGfn2Elements[] = {
    // H
    {1,
     1,
     {0, 0},
     {1, 0},
     {3, 0},
     {1.230000, 0},
     {-10.707211, 0},
     {-0.0500000, 0},
     {-0.00953618, 0},
     {1.0, 0},
     0.405771,
     {1.0, 1.0},
     0.080000,
     2.213717,
     1.105388,
     0.05563889,
     0.00027431,
     1.4,
     1.0,
     2.20,
     0.32},
    // C
    {6,
     2,
     {0, 1},
     {2, 2},
     {4, 4},
     {2.096432, 1.800000},
     {-13.970922, -10.063292},
     {-0.0102144, 0.0161657},
     {-0.02294321, -0.00271102},
     {1.0, 3.0},
     0.538015,
     {1.0, 1.1056358},
     0.150000,
     1.247655,
     4.231078,
     -0.00411674,
     0.00213583,
     3.0,
     3.0,
     2.55,
     0.75},
    // N
    {7,
     2,
     {0, 1},
     {2, 2},
     {4, 4},
     {2.339881, 2.014332},
     {-16.686243, -12.523956},
     {-0.1955336, 0.0561076},
     {-0.08506003, -0.02504201},
     {1.5, 3.5},
     0.461493,
     {1.0, 1.1164892},
     -0.0639780,
     1.682689,
     5.242592,
     0.03521273,
     0.02026786,
     1.9,
     3.0,
     3.04,
     0.71},
    // O
    {8,
     2,
     {0, 1},
     {2, 2},
     {4, 4},
     {2.439742, 2.137023},
     {-20.229985, -15.503117},
     {0.0117826, -0.0145102},
     {-0.14955291, -0.03350819},
     {2.0, 4.0},
     0.451896,
     {1.0, 1.1497020},
     -0.0517134,
     2.165712,
     5.784415,
     -0.04935670,
     -0.00310828,
     1.8,
     2.0,
     3.44,
     0.63},
};

inline constexpr std::size_t kGfn2NElements =
    sizeof(kGfn2Elements) / sizeof(kGfn2Elements[0]);

/// @brief Global GFN2 Hamiltonian scales (tblite / param_gfn2-xtb.txt).
inline constexpr double kGfn2Ks = 1.85;
inline constexpr double kGfn2Kp = 2.23;
inline constexpr double kGfn2Wexp = 0.5;
inline constexpr double kGfn2Enscale = 0.02;
inline constexpr double kGfn2Gexp = 2.0;
inline constexpr double kGfn2Kexp = 1.5;
inline constexpr double kGfn2KexpLight = 1.0;
inline constexpr double kGfn2Rexp = 1.0;
inline constexpr double kGfn2AesDmp3 = 3.0;
inline constexpr double kGfn2AesDmp5 = 4.0;
inline constexpr double kGfn2AesShift = 1.2;
inline constexpr double kGfn2AesKexp = 4.0;
inline constexpr double kGfn2AesRmax = 5.0;
inline constexpr double kGfn2CnK = 16.0;
inline constexpr double kGfn2D4S6 = 1.0;
inline constexpr double kGfn2D4S8 = 2.7;
inline constexpr double kGfn2D4A1 = 0.52;
inline constexpr double kGfn2D4A2 = 5.0;
inline constexpr double kGfn2D4S9 = 5.0;
/// @brief xtb/tblite eV→Eh (not ATV CODATA-2018).
inline constexpr double kGfn2EvToEh = 1.0 / 27.21138505;

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
