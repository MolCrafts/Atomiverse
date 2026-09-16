/// @file cpu/gpw/basis_data/xdm_tables.h
/// @brief Free-atom XDM reference tables (polarizability + multipole moments).
///
/// Free-atom static polarizability \(\alpha\) [Bohr³] from CRC Handbook 88th Ed.
/// (same source as postg / QE `alpha_free`). Free-atom multipole moments
/// \(M_1,M_2,M_3\) [a.u.] are chosen so identical-atom XDM \(C_6,C_8,C_{10}\)
/// match literature free-atom dispersion coefficients under the Becke–Johnson
/// combination rules:
///
///   - H: Yan, Babb, Dalgarno, Drake, Phys. Rev. A 54, 2824 (1996)
///         \(C_6=6.4990267\), \(C_8=124.3991\), \(C_{10}=3285.828\)
///   - O: Chu & Dalgarno, J. Chem. Phys. 121, 4083 (2004) \(C_6=15.55\);
///         \(C_8=235\), \(C_{10}=4500\) (scale consistent with the XDM
///         combination-rule structure for free-atom O).
///
/// In-tree only — nothing is parsed at build/test/run time. Unsupported Z throws.
#pragma once

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Number of free-atom XDM rows (Z = 1 .. kXdmNFree). Only H and O are
///        populated for the ice water-dimer foothold; other Z throw.
constexpr int kXdmNFree = 10;

/// @brief Free-atom XDM parameters: [Z-1] = {α, M1, M2, M3} [a.u.].
///        Zero row ⇒ unsupported (caller must throw).
constexpr double kXdmFreeAtom[kXdmNFree][4] = {
    {4.4997895000044119e+00, 2.8885914355298747e+00, 1.8430355994782722e+01,
     2.4163646918811563e+02}, // Z=1 H
    {0.0, 0.0, 0.0, 0.0},     // Z=2
    {0.0, 0.0, 0.0, 0.0},     // Z=3
    {0.0, 0.0, 0.0, 0.0},     // Z=4
    {0.0, 0.0, 0.0, 0.0},     // Z=5
    {0.0, 0.0, 0.0, 0.0},     // Z=6
    {0.0, 0.0, 0.0, 0.0},     // Z=7
    {5.4121643356381810e+00, 5.7463147959517409e+00, 2.8947137771675433e+01,
     2.6261749605951150e+02}, // Z=8 O
    {0.0, 0.0, 0.0, 0.0},     // Z=9
    {0.0, 0.0, 0.0, 0.0},     // Z=10
};

/// @brief True when free-atom XDM tables cover element Z.
inline bool xdm_supports_element(int Z) {
    return Z >= 1 && Z <= kXdmNFree && kXdmFreeAtom[Z - 1][0] > 0.0;
}

/// @brief QE NC-PBE BJ damping a1 (dimensionless).
constexpr double kXdmPbeA1 = 0.4283;
/// @brief QE NC-PBE BJ damping a2 [Bohr] (= 2.4690 Å / a0).
constexpr double kXdmPbeA2Bohr = 4.6657338220946531;

} // namespace gpw
} // namespace cpu
} // namespace ATV
