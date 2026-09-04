/// @file cpu/semiempirical/basis_data/dftb_mio_tables.h
/// @brief Static multi-element SK parameter tables for SCC-DFTB2/3 (H/C/N/O).
///
/// In-tree data only — nothing is parsed at build or run time (GPW
/// `gpw_tables.h` pattern). Element onsite / Hubbard U and pair **SK + repulsive** for H/C/N/O
/// are transcribed mio-1-1 (`dftb_mio_sk_tables.h`). Sketch `DftbPairSkRow` /
/// `DftbPairRepRow` below are retained only as unused fallback
/// documentation — energy/force paths read `MioSkTable`. DFTB2 (Elstner et al., Phys. Rev. B 58, 7260
/// (1998), DOI 10.1103/PhysRevB.58.7260) and DFTB3 (Gaus et al., J. Chem.
/// Theory Comput. 7, 931 (2011), DOI 10.1021/ct100684s). SK H⁰/S and repulsives match mio-1-1 for every ordered pair among
/// H/C/N/O. Hubbard derivatives \(U^{\mathrm{d}}\) and collinear spin
/// constants \(W\) are **not** in this table — they are explicit user input.
/// Unsupported Z throws at lookup.
///
/// Units in this header: energy [eV], length [Å], Hubbard U [eV], valence
/// electron count dimensionless.
#pragma once

#include <cstddef>

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace basis_data {

/// @brief One supported element's onsite + Hubbard data.
struct DftbElementRow {
    int z;            ///< atomic number
    const char* sym;  ///< element symbol
    int n_ao;         ///< basis size: 1 (s) or 4 (s+px+py+pz)
    double n_val;     ///< valence electron count (Mulliken reference)
    double e_s;       ///< onsite s energy [eV]
    double e_p;       ///< onsite p energy [eV] (unused when n_ao==1)
    double hubbard_u; ///< Hubbard U [eV]
    double r_cut;     ///< pair cutoff [Å]
};

/// @brief Slater–Koster two-center integral scale for one ordered element pair.
///
/// Radial forms (R in Å):
///   V_ll'(R) = v0 * exp(-(R - r0) / d)   for H-like hoppings
///   S_ll'(R) = s0 * exp(-0.5 * ((R - r0_s) / w)^2)
/// Diagonal atom block uses onsite energies; off-diagonal SK uses these.
struct DftbPairSkRow {
    int z_a;
    int z_b;
    // Hamiltonian hoppings [eV]
    double h_ss0;
    double h_sp0;
    double h_pps0;
    double h_ppp0;
    double h_r0; ///< [Å]
    double h_d;  ///< [Å]
    // Overlap scales [dimensionless]
    double s_ss0;
    double s_sp0;
    double s_pps0;
    double s_ppp0;
    double s_r0; ///< [Å]
    double s_w;  ///< [Å]
};

/// @brief Exponential repulsive pair potential E_rep = A * exp(-B * R) [eV].
struct DftbPairRepRow {
    int z_a;
    int z_b;
    double a; ///< [eV]
    double b; ///< [1/Å]
};

// ── Element rows (H, C, N, O) ──────────────────────────────────────────
// Onsite energies and Hubbard U: transcribed mio-1-1 (Ha→eV) for **every**
// element here, not just H — e.g. C reads Es = −0.50489172 Ha and Us = 0.3647
// Ha straight from `C-C.skf`. DFTB+ uses the s-shell U when
// `OrbitalResolvedSCC = No` (its default), which is what these rows carry;
// perturbing Us in the .skf moves the DFTB+ energy and perturbing Up/Ud does
// not.
inline constexpr DftbElementRow kDftbElements[] = {
    {1, "H", 1, 1.0, -6.492647642847, 0.000000000000, 11.415176530192, 5.810366},
    {6, "C", 4, 4.0, -13.738803605321, -5.288671967091, 9.923992563912, 5.810366},
    {7, "N", 4, 5.0, -17.415287197432, -7.094769768916, 11.725386333396, 5.810366},
    {8, "O", 4, 6.0, -23.914249514572, -9.037763156895, 13.480520746262, 5.810366},
};
inline constexpr std::size_t kDftbNElements =
    sizeof(kDftbElements) / sizeof(kDftbElements[0]);

// ── Pair SK (symmetric: store both A-B and B-A when A!=B for simple lookup) ─
// Scales chosen so H–H recovers a bound H2-like curve and H–O / O–O form a
// stable water molecule under SCC-DFTB2 (framework product gate).
// Overlap peak scales must keep S positive definite (s_ss0 < 1 at all R).
// Gaussian form s0*exp(-0.5*((R-r0)/w)^2) peaks at s0 when R=r0.
inline constexpr DftbPairSkRow kDftbPairSk[] = {
    // H–H (matches legacy skeleton scale ~0.65 at R=0.74)
    {1, 1, -6.5, 0, 0, 0, 0.74, 0.50, 0.65, 0, 0, 0, 0.0, 0.80},
    // C–C
    {6, 6, -8.0, -7.0, -7.5, -3.0, 1.40, 0.55, 0.55, 0.40, 0.45, 0.20, 0.0,
     0.90},
    // N–N
    {7, 7, -9.0, -7.5, -8.0, -3.2, 1.20, 0.55, 0.50, 0.35, 0.40, 0.18, 0.0,
     0.90},
    // O–O
    {8, 8, -10.0, -8.0, -8.5, -3.5, 1.20, 0.55, 0.45, 0.30, 0.35, 0.15, 0.0,
     0.90},
    // H–C / C–H
    {1, 6, -7.0, -5.5, 0, 0, 1.09, 0.50, 0.55, 0.35, 0, 0, 0.0, 0.85},
    {6, 1, -7.0, -5.5, 0, 0, 1.09, 0.50, 0.55, 0.35, 0, 0, 0.0, 0.85},
    // H–N / N–H
    {1, 7, -7.5, -6.0, 0, 0, 1.01, 0.50, 0.50, 0.32, 0, 0, 0.0, 0.85},
    {7, 1, -7.5, -6.0, 0, 0, 1.01, 0.50, 0.50, 0.32, 0, 0, 0.0, 0.85},
    // H–O / O–H
    {1, 8, -8.0, -6.5, 0, 0, 0.96, 0.48, 0.50, 0.30, 0, 0, 0.0, 0.85},
    {8, 1, -8.0, -6.5, 0, 0, 0.96, 0.48, 0.50, 0.30, 0, 0, 0.0, 0.85},
    // C–N / N–C
    {6, 7, -8.5, -7.2, -7.8, -3.1, 1.35, 0.55, 0.50, 0.38, 0.42, 0.18, 0.0,
     0.90},
    {7, 6, -8.5, -7.2, -7.8, -3.1, 1.35, 0.55, 0.50, 0.38, 0.42, 0.18, 0.0,
     0.90},
    // C–O / O–C
    {6, 8, -9.0, -7.5, -8.0, -3.2, 1.30, 0.55, 0.48, 0.35, 0.40, 0.16, 0.0,
     0.90},
    {8, 6, -9.0, -7.5, -8.0, -3.2, 1.30, 0.55, 0.48, 0.35, 0.40, 0.16, 0.0,
     0.90},
    // N–O / O–N
    {7, 8, -9.5, -7.8, -8.2, -3.3, 1.25, 0.55, 0.45, 0.32, 0.38, 0.15, 0.0,
     0.90},
    {8, 7, -9.5, -7.8, -8.2, -3.3, 1.25, 0.55, 0.45, 0.32, 0.38, 0.15, 0.0,
     0.90},
};
inline constexpr std::size_t kDftbNPairSk =
    sizeof(kDftbPairSk) / sizeof(kDftbPairSk[0]);

// ── Pair repulsive (symmetric storage) ─────────────────────────────────
inline constexpr DftbPairRepRow kDftbPairRep[] = {
    {1, 1, 0.50, 2.0}, {6, 6, 2.5, 2.2}, {7, 7, 3.0, 2.3},  {8, 8, 3.5, 2.4},
    {1, 6, 1.2, 2.1},  {6, 1, 1.2, 2.1}, {1, 7, 1.4, 2.15}, {7, 1, 1.4, 2.15},
    {1, 8, 1.6, 2.2},  {8, 1, 1.6, 2.2}, {6, 7, 2.7, 2.25}, {7, 6, 2.7, 2.25},
    {6, 8, 2.9, 2.3},  {8, 6, 2.9, 2.3}, {7, 8, 3.2, 2.35}, {8, 7, 3.2, 2.35},
};
inline constexpr std::size_t kDftbNPairRep =
    sizeof(kDftbPairRep) / sizeof(kDftbPairRep[0]);

} // namespace basis_data
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
