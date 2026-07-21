/// @file cpu/scf/grid.cpp
/// @brief Molecular integration grid: Lebedev angular + Euler-Maclaurin radial + Becke partition.
///
/// Lebedev angular quadrature points tabulated for 6-point and 110-point grids.
/// Higher-order grids (194, 302, 590) can be added later.
///
/// Radial quadrature uses Euler-Maclaurin mapping with Bragg-Slater atomic radii.
/// Becke partitioning uses the k(μ) = 1.5μ - 0.5μ³ step function applied 3 times.

#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/lebedev_data.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

static constexpr double kPi = 3.14159265358979323846;

// ── Bragg-Slater atomic radii (Bohr) ────────────────────────────────────

/// @brief Bragg-Slater atomic radius for element Z, returned in Bohr.
///
/// Radii for Z = 1..86 (H through Rn). Values for Z = 1..36 match PySCF's
/// `pyscf.dft.radi.BRAGG_RADII` table (used by PySCF's default Mura-Knowles
/// radial grid). For Z = 37..86 we fall back to Slater (1964) crystal radii
/// converted from Angstrom.
///
/// PySCF's BRAGG_RADII is itself based on:
///   J.C. Slater, "Atomic Radii in Crystals", JCP 41, 3199 (1964),
///   DOI 10.1063/1.1725697.
/// But uses 0.35 A for H (Slater's non-bonded radius, not the 0.25 A covalent)
/// and Bondi-style radii for noble gases. Matching the PySCF table exactly
/// avoids systematic mHa-scale offsets in UKS/RKS total energies when
/// validating against PySCF references.
///
/// Throws std::invalid_argument for Z < 1 or Z > 86.
///
/// @param Z Atomic number (1 <= Z <= 86).
/// @return Bragg-Slater radius in Bohr.
double bragg_slater_radius(int Z) {
    static constexpr double kA2B = units::kAngstromInBohr;
    static constexpr int kMaxZ = 86;
    // clang-format off
    static constexpr double radii[kMaxZ + 1] = {
        0.00 * kA2B,  //  0  placeholder (unused)
        // ── Z = 1..36: match PySCF dft.radi.BRAGG_RADII exactly ─────────
        0.35 * kA2B,  //  1  H   [PySCF 0.35 A non-bonded; was Slater 0.25 A]
        1.40 * kA2B,  //  2  He  [PySCF noble-gas value]
        1.45 * kA2B,  //  3  Li
        1.05 * kA2B,  //  4  Be
        0.85 * kA2B,  //  5  B
        0.70 * kA2B,  //  6  C
        0.65 * kA2B,  //  7  N
        0.60 * kA2B,  //  8  O
        0.50 * kA2B,  //  9  F
        1.50 * kA2B,  // 10  Ne  [PySCF noble-gas value]
        1.80 * kA2B,  // 11  Na
        1.50 * kA2B,  // 12  Mg
        1.25 * kA2B,  // 13  Al
        1.10 * kA2B,  // 14  Si
        1.00 * kA2B,  // 15  P
        1.00 * kA2B,  // 16  S
        1.00 * kA2B,  // 17  Cl
        1.80 * kA2B,  // 18  Ar  [PySCF noble-gas value]
        2.20 * kA2B,  // 19  K
        1.80 * kA2B,  // 20  Ca
        1.60 * kA2B,  // 21  Sc
        1.40 * kA2B,  // 22  Ti
        1.35 * kA2B,  // 23  V
        1.40 * kA2B,  // 24  Cr
        1.40 * kA2B,  // 25  Mn
        1.40 * kA2B,  // 26  Fe
        1.35 * kA2B,  // 27  Co
        1.35 * kA2B,  // 28  Ni
        1.35 * kA2B,  // 29  Cu
        1.35 * kA2B,  // 30  Zn
        1.30 * kA2B,  // 31  Ga
        1.25 * kA2B,  // 32  Ge
        1.15 * kA2B,  // 33  As
        1.15 * kA2B,  // 34  Se
        1.15 * kA2B,  // 35  Br
        1.90 * kA2B,  // 36  Kr  [PySCF noble-gas value]
        // ── Period 5 ─────────────────────────────────────────────
        2.35 * kA2B,  // 37  Rb                [Slater 1964]
        2.00 * kA2B,  // 38  Sr                [Slater 1964]
        1.80 * kA2B,  // 39  Y                 [Slater 1964]
        1.55 * kA2B,  // 40  Zr                [Slater 1964]
        1.45 * kA2B,  // 41  Nb                [Slater 1964]
        1.45 * kA2B,  // 42  Mo                [Slater 1964]
        1.35 * kA2B,  // 43  Tc                [Slater 1964]
        1.30 * kA2B,  // 44  Ru                [Slater 1964]
        1.35 * kA2B,  // 45  Rh                [Slater 1964]
        1.40 * kA2B,  // 46  Pd                [Slater 1964]
        1.60 * kA2B,  // 47  Ag                [Slater 1964]
        1.55 * kA2B,  // 48  Cd                [Slater 1964]
        1.55 * kA2B,  // 49  In                [Slater 1964]
        1.45 * kA2B,  // 50  Sn                [Slater 1964]
        1.45 * kA2B,  // 51  Sb                [Slater 1964]
        1.40 * kA2B,  // 52  Te                [Slater 1964]
        1.40 * kA2B,  // 53  I                 [Slater 1964]
        1.08 * kA2B,  // 54  Xe                [Clementi 1967]
        // ── Period 6 ─────────────────────────────────────────────
        2.60 * kA2B,  // 55  Cs                [Slater 1964]
        2.15 * kA2B,  // 56  Ba                [Slater 1964]
        1.95 * kA2B,  // 57  La                [Slater 1964]
        1.85 * kA2B,  // 58  Ce                [Slater 1964]
        1.85 * kA2B,  // 59  Pr                [Slater 1964]
        1.85 * kA2B,  // 60  Nd                [Slater 1964]
        1.85 * kA2B,  // 61  Pm                [Slater 1964]
        1.85 * kA2B,  // 62  Sm                [Slater 1964]
        1.85 * kA2B,  // 63  Eu                [Slater 1964]
        1.80 * kA2B,  // 64  Gd                [Slater 1964]
        1.75 * kA2B,  // 65  Tb                [Slater 1964]
        1.75 * kA2B,  // 66  Dy                [Slater 1964]
        1.75 * kA2B,  // 67  Ho                [Slater 1964]
        1.75 * kA2B,  // 68  Er                [Slater 1964]
        1.75 * kA2B,  // 69  Tm                [Slater 1964]
        1.75 * kA2B,  // 70  Yb                [Slater 1964]
        1.75 * kA2B,  // 71  Lu                [Slater 1964]
        1.55 * kA2B,  // 72  Hf                [Slater 1964]
        1.45 * kA2B,  // 73  Ta                [Slater 1964]
        1.35 * kA2B,  // 74  W                 [Slater 1964]
        1.35 * kA2B,  // 75  Re                [Slater 1964]
        1.30 * kA2B,  // 76  Os                [Slater 1964]
        1.35 * kA2B,  // 77  Ir                [Slater 1964]
        1.35 * kA2B,  // 78  Pt                [Slater 1964]
        1.35 * kA2B,  // 79  Au                [Slater 1964]
        1.50 * kA2B,  // 80  Hg                [Slater 1964]
        1.90 * kA2B,  // 81  Tl                [Slater 1964]
        1.80 * kA2B,  // 82  Pb                [Slater 1964]
        1.60 * kA2B,  // 83  Bi                [Slater 1964]
        1.90 * kA2B,  // 84  Po                [Slater 1964]
        1.50 * kA2B,  // 85  At                [Cordero 2008]
        1.20 * kA2B,  // 86  Rn                [Clementi 1967]
    };
    // clang-format on
    if (Z < 1 || Z > kMaxZ) {
        throw std::invalid_argument(
            "bragg_slater_radius: unsupported element Z=" + std::to_string(Z)
            + ". Supported range: Z=1..86");
    }
    return radii[Z];
}

// ── Lebedev angular quadrature ──────────────────────────────────────────
//
// All angular grids come from the tabulated Lebedev-Laikov data in
// lebedev_data.h (orders 6, 50, 110, 194, 302, 590). The earlier
// in-source octahedral-generator code and the Gauss-Legendre x uniform-phi
// product grid that was used as a placeholder for "Lebedev-110" have
// been removed -- they produced wrong weight distributions for spherical
// harmonics. See `build_molecular_grid` below for the dispatch.

// ── Radial quadrature ───────────────────────────────────────────────────

// Euler-Maclaurin / Mura-Knowles-type log mapping.
// Maps i ∈ [1, n_radial] to r ∈ [0, ∞) with Bragg-Slater scaling.
//
// Treutler-Ahlrichs mapping (widely used, good for all-electron):
//   r_i = R_bragg * (1 + cos(θ_i))^alpha / (1 - cos(θ_i))
//
// Simpler approach: Murray-Handy-Laming (Mol. Phys. 78, 997, 1993):
//   r_i = R_bragg * (x_i / (1 - x_i))^2
//   x_i = i / (n + 1)
//   w_i = 2 R_bragg * x_i^2 / (1 - x_i)^3 * r_i^2 / (n+1)
//
// But simplest is the Becke mapping:
//   x_i = cos(π(n_radial - i + 0.5) / n_radial)   Chebyshev points
//   r_i = R * (1 + x_i) / (1 - x_i)
//   w_i = 2R / (1 - x_i)^2 * π/n_radial * sin(arccos(x_i)) * r_i^2

// RadialPoint is declared in grid.h (public API).
//
// Only Mura-Knowles is used; the earlier Becke (Chebyshev-mapped) radial
// quadrature was removed -- it left an ad-hoc radius scaling that did not
// match PySCF's reference grid and contributed to the ~mHa UKS-vs-PySCF
// offset documented in the libxc-xc-fix spec.

// ── Treutler-Ahlrichs xi scaling table (PySCF parity) ─────────────────
//
// Atom-specific xi multiplier used by the Treutler-Ahlrichs radial
// transform (JCP 102, 346 (1995)). Matches pyscf.dft.radi._treutler_
// ahlrichs_xi exactly for Z=1..36; default to xi=1.0 outside that range.
static double treutler_ahlrichs_xi(int Z) {
    // Z indexed; entry 0 unused.
    static constexpr double xi[37] = {
        0.0, // 0  (unused)
        0.8, // 1  H
        0.9, // 2  He
        1.8, // 3  Li
        1.4, // 4  Be
        1.3, // 5  B
        1.1, // 6  C
        0.9, // 7  N
        0.9, // 8  O
        0.9, // 9  F
        0.9, // 10 Ne
        1.4, // 11 Na
        1.3, // 12 Mg
        1.3, // 13 Al
        1.2, // 14 Si
        1.1, // 15 P
        1.0, // 16 S
        1.0, // 17 Cl
        1.0, // 18 Ar
        1.5, // 19 K
        1.4, // 20 Ca
        1.3, // 21 Sc
        1.2, // 22 Ti
        1.2, // 23 V
        1.2, // 24 Cr
        1.2, // 25 Mn
        1.2, // 26 Fe
        1.2, // 27 Co
        1.1, // 28 Ni
        1.1, // 29 Cu
        1.1, // 30 Zn
        1.1, // 31 Ga
        1.0, // 32 Ge
        0.9, // 33 As
        0.9, // 34 Se
        0.9, // 35 Br
        0.9, // 36 Kr
    };
    if (Z < 1 || Z > 36)
        return 1.0;
    return xi[Z];
}

// ── Treutler-Ahlrichs (M4) radial quadrature ──────────────────────────
//
// Reference: O. Treutler and R. Ahlrichs, "Efficient molecular
// numerical integration schemes", J. Chem. Phys. 102, 346 (1995),
// DOI 10.1063/1.469408 -- specifically the "M4" mapping in eq. (10).
//
// PySCF's default radial scheme. Uses Gauss-Chebyshev points on
// (-1, 1) mapped to r ∈ (0, ∞) via:
//   r = -xi/ln(2) * (1+x)^0.6 * ln((1-x)/2)
// where xi is an atom-specific scaling factor (see treutler_ahlrichs_xi).
//
// The returned RadialPoint.w already includes the r² Jacobian, so the
// 3D-space integral of f(r) over the radial axis is sum_i w_i * f(r_i)
// after multiplication by the angular weight times 4*pi (as in
// build_molecular_grid below).
static std::vector<RadialPoint> radial_grid_treutler_ahlrichs(int n, int Z) {
    std::vector<RadialPoint> pts(n);
    double xi = treutler_ahlrichs_xi(Z);
    double ln2 = xi / std::log(2.0);
    double step = kPi / (n + 1);

    // PySCF loops i=0..n-1 with x=cos((i+1)*step), then reverses
    // so the returned r is ascending. We do the same so r[0] is the
    // smallest radius.
    std::vector<double> r_tmp(n), dr_tmp(n);
    for (int i = 0; i < n; ++i) {
        double angle = (i + 1) * step;
        double x = std::cos(angle);
        double sin_angle = std::sin(angle);
        double one_plus_x = 1.0 + x;
        double one_minus_x = 1.0 - x;
        double pow_term = std::pow(one_plus_x, 0.6);
        double log_term = std::log(one_minus_x / 2.0);

        r_tmp[i] = -ln2 * pow_term * log_term;
        // dr/di including the Chebyshev quadrature weight sin(angle)*step:
        // d/dx [-ln2 (1+x)^0.6 ln((1-x)/2)]
        //     = -ln2 [0.6 (1+x)^(-0.4) ln((1-x)/2) + (1+x)^0.6 / (1-x) * (-1)]
        //     = ln2 (1+x)^0.6 [-0.6/(1+x) ln((1-x)/2) + 1/(1-x)]
        dr_tmp[i] = step * sin_angle * ln2 * pow_term
                    * (-0.6 / one_plus_x * log_term + 1.0 / one_minus_x);
    }
    // Reverse so r is ascending (PySCF returns r[::-1], dr[::-1]).
    for (int i = 0; i < n; ++i) {
        int j = n - 1 - i;
        pts[i].r = r_tmp[j];
        // dr is the radial differential -- multiply by r² to get the
        // full weight (Jacobian for 4*pi r^2 dr volume element).
        pts[i].w = r_tmp[j] * r_tmp[j] * dr_tmp[j];
    }
    return pts;
}

// ── Mura-Knowles radial quadrature ─────────────────────────────────────
// Reference: Mura & Knowles, JCP 104, 9848 (1996), equation (5).
// Uses the log3 mapping: r = -R * ln(1 - x^3), x = (i+0.5)/n.
// Matches PySCF's default radial grid (dft.gen_grid.RADII_ADJUST = none).

std::vector<RadialPoint> radial_grid_mura_knowles(int n, double R_bragg) {
    std::vector<RadialPoint> pts(n);
    for (int i = 0; i < n; ++i) {
        double x = (i + 0.5) / n;
        double x3 = x * x * x;
        double r = -R_bragg * std::log(1.0 - x3);

        // dr/dx = R * 3x^2 / (1 - x^3)
        double dr_dx = R_bragg * 3.0 * x * x / (1.0 - x3);
        // Uniform weight: 1/n
        double w_quad = 1.0 / n;

        // Total weight = r^2 * dr/dx * w_quad (4pi from angular)
        pts[i].r = r;
        pts[i].w = r * r * dr_dx * w_quad;
    }
    return pts;
}

// ── Euler-Maclaurin radial quadrature (SG-1 parent grid) ────────────────
// Reference: Gill, Johnson & Pople, Chem. Phys. Lett. 209, 506 (1993),
// using the Murray-Handy-Laming Euler-Maclaurin mapping that the SG-1
// EML-(50,194) parent grid is built on (Mol. Phys. 78, 997 (1993)):
//   r_i = R * i^2 / (n + 1 - i)^2,        i = 1..n
//   dr/di = R * 2 i (n+1) / (n + 1 - i)^3
// The integration node weight is uniform (1) on the i-grid; the returned
// RadialPoint.w already folds in the r^2 Jacobian so the 3-space integral
// of f(r) is sum_i w_i * f(r_i) after the 4*pi * angular-weight factor.

std::vector<RadialPoint> radial_grid_euler_maclaurin(int n, double R) {
    std::vector<RadialPoint> pts(n);
    const double np1 = static_cast<double>(n + 1);
    for (int idx = 0; idx < n; ++idx) {
        const double i = static_cast<double>(idx + 1); // i = 1..n
        const double denom = np1 - i;
        const double r = R * i * i / (denom * denom);
        // dr/di = R * 2 i (n+1) / (n+1-i)^3
        const double dr_di = R * 2.0 * i * np1 / (denom * denom * denom);
        pts[idx].r = r;
        pts[idx].w = r * r * dr_di; // uniform node weight = 1
    }
    return pts;
}

// ── Becke partitioning ──────────────────────────────────────────────────
//
// Audited against Becke, JCP 88, 2547 (1988). The switching function
//   s(mu) = 1/2 [1 - f(f(f(mu)))],  f(mu) = (3/2) mu - (1/2) mu^3
// iterated exactly k = 3 times is correct and unchanged.
//
// The atomic-size adjustment is grid-level dependent:
//   - GridLevel::SG0 / SG1 use the *original Becke 1988* heteronuclear
//     adjustment of the Domain basis: chi = R_A/R_B, u = (chi-1)/(chi+1),
//     a = u/(u^2-1), clamped to |a| <= 1/2. (Audited against the spec
//     formula -- no discrepancy; SG0 stays bit-for-bit identical.)
//   - GridLevel::MatchPySCF uses PySCF's default `treutler_atomic_radii_
//     adjust` (JCP 102, 346 (1995)): with rad = sqrt(BRAGG_RADII[Z]),
//     a_ij = (1/4)(rad_j/rad_i - rad_i/rad_j), clamped to |a| <= 1/2.
//     This is what PySCF's level=3 grid actually applies, so MatchPySCF
//     reproduces it exactly.
// In all cases the adjustment is nu = mu + a (1 - mu^2).

// Becke step function: f(μ) = 1.5μ - 0.5μ³ (one iteration).
static double becke_step(double mu) { return 1.5 * mu - 0.5 * mu * mu * mu; }

// Original-Becke 1988 heteronuclear size-adjustment parameter a_AB.
static double becke_size_adjust_original(int Z_a, int Z_b) {
    const double chi = bragg_slater_radius(Z_a) / bragg_slater_radius(Z_b);
    const double u = (chi - 1.0) / (chi + 1.0);
    double a = u / (u * u - 1.0);
    return std::max(-0.5, std::min(0.5, a));
}

// PySCF treutler_atomic_radii_adjust size-adjustment parameter a_AB.
static double becke_size_adjust_treutler(int Z_a, int Z_b) {
    const double rad_a = std::sqrt(bragg_slater_radius(Z_a));
    const double rad_b = std::sqrt(bragg_slater_radius(Z_b));
    double a = 0.25 * (rad_b / rad_a - rad_a / rad_b);
    return std::max(-0.5, std::min(0.5, a));
}

// Level-aware heteronuclear size-adjustment parameter a_AB. SG0/SG1 use the
// original Becke-1988 adjustment; MatchPySCF uses PySCF's treutler
// adjustment. Both are antisymmetric in (A,B). The weight-derivative port
// must use the exact same selection so it stays consistent with the
// production becke_partition_weights.
static double becke_size_adjust(int Z_a, int Z_b, GridLevel level) {
    return (level == GridLevel::MatchPySCF)
               ? becke_size_adjust_treutler(Z_a, Z_b)
               : becke_size_adjust_original(Z_a, Z_b);
}

// Becke partition weight for atom A at point r.
// Uses Becke's fuzzy-cell scheme with 3 iterations of the step function.
// The heteronuclear size adjustment depends on `level` (see the audit
// note above).
static void becke_partition_weights(const native::Atom* atoms, int natm,
                                    double rx, double ry, double rz,
                                    double* weights,
                                    GridLevel level = GridLevel::SG0) {
    if (natm == 1) {
        weights[0] = 1.0;
        return;
    }

    // Compute distances from grid point to each atom
    std::vector<double> dist(natm);
    for (int a = 0; a < natm; ++a) {
        double dx = rx - atoms[a].pos[0];
        double dy = ry - atoms[a].pos[1];
        double dz = rz - atoms[a].pos[2];
        dist[a] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    // Compute inter-atomic distances
    // R_AB = |R_A - R_B|
    // μ_AB = (r_A - r_B) / R_AB  (elliptical coordinate)

    // Initialize raw Becke weights to 1
    for (int a = 0; a < natm; ++a)
        weights[a] = 1.0;

    for (int a = 0; a < natm; ++a) {
        for (int b = a + 1; b < natm; ++b) {
            double dx = atoms[a].pos[0] - atoms[b].pos[0];
            double dy = atoms[a].pos[1] - atoms[b].pos[1];
            double dz = atoms[a].pos[2] - atoms[b].pos[2];
            double R_AB = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (R_AB < 1e-14)
                continue;

            double mu = (dist[a] - dist[b]) / R_AB;

            // Heteronuclear size adjustment, nu = mu + a_AB (1 - mu^2).
            double a_AB =
                (level == GridLevel::MatchPySCF)
                    ? becke_size_adjust_treutler(atoms[a].Z, atoms[b].Z)
                    : becke_size_adjust_original(atoms[a].Z, atoms[b].Z);
            mu += a_AB * (1.0 - mu * mu);

            // Apply step function 3 times
            double s = mu;
            s = becke_step(s);
            s = becke_step(s);
            s = becke_step(s);

            // s_AB ∈ [-1, 1], convert to cell function
            // P_A *= 0.5*(1 - s_AB),  P_B *= 0.5*(1 + s_AB)
            weights[a] *= 0.5 * (1.0 - s);
            weights[b] *= 0.5 * (1.0 + s);
        }
    }

    // Normalize
    double total = 0;
    for (int a = 0; a < natm; ++a)
        total += weights[a];
    if (total > 1e-30) {
        for (int a = 0; a < natm; ++a)
            weights[a] /= total;
    }
}

// ── Angular-order dispatch ──────────────────────────────────────────────
//
// Resolve a requested Lebedev order to the tabulated grid pointer and its
// point count. Recognizes {6, 38, 50, 86, 110, 194, 266, 302, 350, 434,
// 590, 770}; non-tabulated values round DOWN to the next supported order.
static void resolve_lebedev_grid(int order, const lebedev::Point*& pts,
                                 int& n_pts) {
    if (order >= 770) {
        pts = lebedev::grid_770;
        n_pts = lebedev::n_770;
    } else if (order >= 590) {
        pts = lebedev::grid_590;
        n_pts = lebedev::n_590;
    } else if (order >= 434) {
        pts = lebedev::grid_434;
        n_pts = lebedev::n_434;
    } else if (order >= 350) {
        pts = lebedev::grid_350;
        n_pts = lebedev::n_350;
    } else if (order >= 302) {
        pts = lebedev::grid_302;
        n_pts = lebedev::n_302;
    } else if (order >= 266) {
        pts = lebedev::grid_266;
        n_pts = lebedev::n_266;
    } else if (order >= 194) {
        pts = lebedev::grid_194;
        n_pts = lebedev::n_194;
    } else if (order >= 110) {
        pts = lebedev::grid_110;
        n_pts = lebedev::n_110;
    } else if (order >= 86) {
        pts = lebedev::grid_86;
        n_pts = lebedev::n_86;
    } else if (order >= 50) {
        pts = lebedev::grid_50;
        n_pts = lebedev::n_50;
    } else if (order >= 38) {
        pts = lebedev::grid_38;
        n_pts = lebedev::n_38;
    } else {
        pts = lebedev::grid_6;
        n_pts = lebedev::n_6;
    }
}

// SG-1 pruning alpha-quadruple (Gill-Johnson-Pople 1993). The four entries
// are alpha_1..alpha_4: the radial fractions r/R that delimit the 5
// Lebedev regions {6, 38, 86, 194, 86}.
struct SG1Alpha {
    double a1, a2, a3, a4;
};

static SG1Alpha sg1_alpha(int Z) {
    if (Z <= 2)
        return {0.25, 0.5, 1.0, 4.5}; // H, He
    if (Z <= 10)
        return {0.1667, 0.5, 0.9, 3.5}; // Li-Ne
    return {0.1, 0.4, 0.8, 2.5};        // Na-Ar (and heavier)
}

// SG-1 atom-size radius: R = 1/2 * Bragg-Slater radius, EXCEPT hydrogen,
// which keeps the full Bragg-Slater radius (no 1/2 factor).
static double sg1_atom_radius(int Z) {
    double R = bragg_slater_radius(Z);
    return (Z == 1) ? R : 0.5 * R;
}

// ── lebedev_order_for_shell ─────────────────────────────────────────────

int lebedev_order_for_shell(GridLevel level, int Z, double r, double R,
                            int radial_index, int n_radial, int uniform_order) {
    if (level == GridLevel::SG0) {
        return uniform_order;
    }

    (void)radial_index;
    (void)n_radial;

    if (level == GridLevel::SG1) {
        // 5-region pruning: orders {6, 38, 86, 194, 86} delimited by
        // alpha_1*R .. alpha_4*R, with R the SG-1 atom-size radius.
        const SG1Alpha a = sg1_alpha(Z);
        const double t = r / R;
        if (t < a.a1)
            return 6;
        if (t < a.a2)
            return 38;
        if (t < a.a3)
            return 86;
        if (t < a.a4)
            return 194;
        return 86;
    }

    // MatchPySCF: faithful port of PySCF's nwchem_prune for the level=3
    // grid (pyscf/dft/gen_grid.py).
    //
    //   leb_ngrid = LEBEDEV_NGRID[4:] = {38,50,74,86,110,146,170,194,
    //                                    230,266,302,350,434,590,770,...}
    //   leb_l     = [1, 3, idx-1, idx, idx-1]   (idx = index of n_ang)
    //   region    = sum(rads/r_atom > alphas[period])   in 0..4
    //   angs      = leb_ngrid[leb_l[region]]
    // with r_atom = BRAGG_RADII[Z] (full radius, no 1/2 factor, no
    // hydrogen exception), and the same SG-1 alpha quadruples.
    //
    // The level-3 maximum angular order n_ang comes from PySCF's
    // ANG_ORDER table (_default_ang): periods 1-2 (Z<=10) -> order 29 ->
    // 302 points; period 3 (Na-Ar) -> order 35 -> 434 points.
    //
    //   Z<=10  (periods 1-2): _default_ang lvl-3 -> order 29 -> n_ang=302
    //          idx=10 -> leb_l=[1,3,9,10,9]  -> {50, 86, 266, 302, 266}
    //   Z=11.. (periods 3+):  _default_ang lvl-3 -> order 35 -> n_ang=434
    //          idx=12 -> leb_l=[1,3,11,12,11] -> {50, 86, 350, 434, 350}
    //
    // ANG_ORDER row 3 = {29,29,35,35,35,35,35}, so every period >= 3
    // uses order 35 -> 434; Z>36 is already rejected upstream by
    // bragg_slater_radius / treutler_ahlrichs_xi.
    const SG1Alpha a = sg1_alpha(Z);
    const double t = r / R;
    int region = 0; // 0..4, = sum(t > alpha_i)
    if (t > a.a1)
        ++region;
    if (t > a.a2)
        ++region;
    if (t > a.a3)
        ++region;
    if (t > a.a4)
        ++region;

    static constexpr int light_orders[5] = {50, 86, 266, 302, 266};
    static constexpr int heavy_orders[5] = {50, 86, 350, 434, 350};
    return (Z > 10) ? heavy_orders[region] : light_orders[region];
}

// ── Grid assembly core ──────────────────────────────────────────────────
//
// Shared assembly for both build_molecular_grid overloads. `level` selects
// the radial scheme and the per-shell angular order; for GridLevel::SG0
// the legacy uniform-order Treutler-Ahlrichs path is reproduced verbatim.
static MolecularGrid assemble_grid(const native::Atom* atoms, int natm,
                                   GridLevel level, int n_radial,
                                   int n_angular) {
    MolecularGrid grid;

    // Per-level radial point count. SG0 honours the caller's n_radial;
    // SG1 uses the EML-50 parent; MatchPySCF uses PySCF's per-element
    // level-3 counts: 50 for H/He, 75 for Li-Ne, 80 for Na-Ar, 90 for
    // Z >= 19 (matches pyscf.dft.gen_grid._default_rad at level 3).
    auto radial_count = [&](int Z) -> int {
        if (level == GridLevel::SG0)
            return n_radial;
        if (level == GridLevel::SG1)
            return 50;
        if (Z <= 2)
            return 50;
        if (Z <= 10)
            return 75;
        if (Z <= 18)
            return 80;
        return 90;
    };

    grid.n_radial = radial_count(natm > 0 ? atoms[0].Z : 1);
    grid.n_angular = n_angular;

    std::vector<double> becke_w(natm);

    for (int a = 0; a < natm; ++a) {
        const int Z = atoms[a].Z;
        const int nr = radial_count(Z);

        // Radial grid: Euler-Maclaurin for SG-1, Treutler-Ahlrichs (PySCF
        // default) for SG0 and MatchPySCF.
        std::vector<RadialPoint> rad_pts;
        double R_size = 0.0;
        if (level == GridLevel::SG1) {
            R_size = sg1_atom_radius(Z);
            rad_pts = radial_grid_euler_maclaurin(nr, R_size);
        } else {
            rad_pts = radial_grid_treutler_ahlrichs(nr, Z);
            R_size = bragg_slater_radius(Z);
        }

        for (int ir = 0; ir < nr; ++ir) {
            const double r = rad_pts[ir].r;
            const double w_rad = rad_pts[ir].w;

            const int order =
                lebedev_order_for_shell(level, Z, r, R_size, ir, nr, n_angular);
            const lebedev::Point* leb_pts = nullptr;
            int actual_n_angular = 0;
            resolve_lebedev_grid(order, leb_pts, actual_n_angular);

            for (int ia = 0; ia < actual_n_angular; ++ia) {
                double px = atoms[a].pos[0] + r * leb_pts[ia].x;
                double py = atoms[a].pos[1] + r * leb_pts[ia].y;
                double pz = atoms[a].pos[2] + r * leb_pts[ia].z;

                // Becke partition weight (size adjustment depends on
                // the grid level -- see becke_partition_weights).
                becke_partition_weights(atoms, natm, px, py, pz, becke_w.data(),
                                        level);

                // Lebedev weights sum to 1.0; multiply by 4*pi for the
                // spherical integral.
                double w = 4.0 * kPi * w_rad * leb_pts[ia].w * becke_w[a];

                // Drop only genuinely negligible points. The MAGNITUDE is
                // tested, not the sign: the 266-point Lebedev rule (used
                // by the MatchPySCF level-3 prune) legitimately carries 18
                // negative weights ~1e-3, and discarding them would break
                // the angular quadrature. Becke-killed near-zero points
                // are still pruned.
                if (std::abs(w) > 1e-30) {
                    grid.points.push_back({px, py, pz, w, a});
                }
            }
        }
    }

    return grid;
}

// ── Public API ──────────────────────────────────────────────────────────

MolecularGrid build_molecular_grid(const native::Atom* atoms, int natm,
                                   GridLevel level, int n_radial,
                                   int n_angular) {
    return assemble_grid(atoms, natm, level, n_radial, n_angular);
}

void becke_partition_weights_for_test(const native::Atom* atoms, int natm,
                                      double rx, double ry, double rz,
                                      double* weights) {
    becke_partition_weights(atoms, natm, rx, ry, rz, weights);
}

// ── Becke weight derivative (grid-response) ─────────────────────────────
//
// Verbatim port of PySCF's grids_response_becke / gen_grid_partition
// (pyscf/grad/rks.py) and get_smooth_function_derivative, restricted to the
// original_becke switching scheme.

// get_smooth_function_derivative for the original_becke scheme: iterated
// polynomial f(x) = (3 - x^2) x / 2 applied three times. Returns the two
// half cell functions s_minus = (1 - p3)/2 and s_plus = (1 + p3)/2, and
// t_uab = 27/8 (1-p2^2)(1-p1^2)(1-p0^2), i.e. dp3/dmu.
//
// s_minus / s_plus are computed through the cancellation-free recurrences
//   1 - f(x) = (1-x)^2 (x+2)/2,    1 + f(x) = (1+x)^2 (2-x)/2,
// NOT as 0.5*(1 -+ p3). The naive form is catastrophically unstable for
// |mu| -> 1 (far-tail grid points): p3 rounds to exactly +-1, 1-p3
// collapses to 0, and the caller's pt = -t/s ratio — whose mathematical
// divergence is cancelled later by multiplying with pbecke ∝ s — blows up
// against the 1e-200 floor instead, leaving ~1e150 garbage in the weight
// derivative. (Observed at -O0 on znver5: dE_xc/dR ~ 4e149; -O2's FMA
// happened to keep p3 at 1-1ulp, hiding the same defect.) The factored
// recurrences keep s exact down to the denormal range, so t/s stays a
// faithful log-derivative and the downstream cancellation is restored.
static void get_smooth_function_derivative(double mu, double& s_minus,
                                           double& s_plus, double& t_uab) {
    const double p0 = mu;
    const double u0 = 1.0 - p0, v0 = 1.0 + p0;
    const double p1 = (3.0 - p0 * p0) * p0 * 0.5;
    const double u1 = 0.5 * u0 * u0 * (p0 + 2.0);
    const double v1 = 0.5 * v0 * v0 * (2.0 - p0);
    const double p2 = (3.0 - p1 * p1) * p1 * 0.5;
    const double u2 = 0.5 * u1 * u1 * (p1 + 2.0);
    const double v2 = 0.5 * v1 * v1 * (2.0 - p1);
    const double u3 = 0.5 * u2 * u2 * (p2 + 2.0);
    const double v3 = 0.5 * v2 * v2 * (2.0 - p2);
    // (1 - p_i^2) = u_i * v_i exactly in the factored form.
    t_uab = 27.0 / 8.0 * (u2 * v2) * (u1 * v1) * (u0 * v0);
    s_minus = 0.5 * u3;
    s_plus = 0.5 * v3;
}

double becke_weight_derivative(const native::Atom* atoms, int natm, double rx,
                               double ry, double rz, int parent,
                               GridLevel level, double* dw_out) {
    // Single-atom: weight is identically 1, derivative vanishes.
    for (int i = 0; i < natm * 3; ++i)
        dw_out[i] = 0.0;
    if (natm <= 1)
        return 1.0;

    const int atom_id = parent;

    // grid_dist[ia]      = |R_ia - r_g|
    // grid_norm_vec[ia]  = (R_ia - r_g) / |R_ia - r_g|  (unit vec atom<-grid)
    std::vector<double> grid_dist(natm);
    std::vector<std::array<double, 3>> grid_norm_vec(natm);
    for (int ia = 0; ia < natm; ++ia) {
        const double vx = atoms[ia].pos[0] - rx;
        const double vy = atoms[ia].pos[1] - ry;
        const double vz = atoms[ia].pos[2] - rz;
        const double n = std::sqrt(vx * vx + vy * vy + vz * vz) + 1e-200;
        grid_dist[ia] = n;
        grid_norm_vec[ia] = {vx / n, vy / n, vz / n};
    }

    // Inter-atomic distance matrix.
    std::vector<double> atm_dist(static_cast<size_t>(natm) * natm, 0.0);
    for (int ia = 0; ia < natm; ++ia) {
        for (int ib = 0; ib < natm; ++ib) {
            if (ia == ib)
                continue;
            const double dx = atoms[ia].pos[0] - atoms[ib].pos[0];
            const double dy = atoms[ia].pos[1] - atoms[ib].pos[1];
            const double dz = atoms[ia].pos[2] - atoms[ib].pos[2];
            atm_dist[ia * natm + ib] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    // get_du(ia, ib): dg/dR_ia for g = (grid_dist[ia]-grid_dist[ib])/R_ab.
    //   duab = (1/R_ab) * grid_norm_vec[ia]
    //        - (R_ia - R_ib)/R_ab^3 * (grid_dist[ia] - grid_dist[ib])
    auto get_du = [&](int ia, int ib, double out[3]) {
        const double Rab = atm_dist[ia * natm + ib];
        const double inv = 1.0 / Rab;
        const double inv3 = inv * inv * inv;
        const double dr = grid_dist[ia] - grid_dist[ib];
        for (int d = 0; d < 3; ++d) {
            const double uab_d = atoms[ia].pos[d] - atoms[ib].pos[d];
            out[d] = inv * grid_norm_vec[ia][d] - uab_d * inv3 * dr;
        }
    };

    // pbecke[ia]              -- raw (unnormalized) cell weight of atom ia
    // dpbecke[A][B][d]        -- before finalization holds d(ln pbecke[B])/dR_A
    std::vector<double> pbecke(natm, 1.0);
    std::vector<double> dpbecke(static_cast<size_t>(natm) * natm * 3, 0.0);
    auto DP = [&](int A, int B, int d) -> double& {
        return dpbecke[(static_cast<size_t>(A) * natm + B) * 3 + d];
    };

    for (int ia = 0; ia < natm; ++ia) {
        for (int ib = 0; ib < ia; ++ib) {
            const double Rab = atm_dist[ia * natm + ib];
            if (Rab < 1e-14)
                continue;

            const double g = (grid_dist[ia] - grid_dist[ib]) / Rab;
            const double a_ij =
                becke_size_adjust(atoms[ia].Z, atoms[ib].Z, level);
            // fadjust(i,j,g) = g + a*(1-g^2);  gadjust(i,j,g) = 1 - 2*a*g
            const double g_adj = g + a_ij * (1.0 - g * g);
            const double gadj = 1.0 - 2.0 * a_ij * g;

            double s_uab, s_uba, t_uab;
            get_smooth_function_derivative(g_adj, s_uab, s_uba, t_uab);
            t_uab *= 0.5 * gadj;
            // Guard only the exact-zero corner (g_adj == +-1, measure zero);
            // the stable recurrences above make this floor otherwise inert.
            s_uab += 1e-200;
            s_uba += 1e-200;
            pbecke[ia] *= s_uab;
            pbecke[ib] *= s_uba;
            const double pt_uab = -t_uab / s_uab;
            const double pt_uba = t_uab / s_uba;

            double duab[3], duba[3];
            get_du(ia, ib, duab);
            get_du(ib, ia, duba);

            if (ia == atom_id) {
                for (int d = 0; d < 3; ++d) {
                    DP(ia, ia, d) += pt_uab * duba[d];
                    DP(ia, ib, d) += pt_uba * duba[d];
                }
            } else {
                for (int d = 0; d < 3; ++d) {
                    DP(ia, ia, d) += pt_uab * duab[d];
                    DP(ia, ib, d) += pt_uba * duab[d];
                }
            }

            if (ib == atom_id) {
                for (int d = 0; d < 3; ++d) {
                    DP(ib, ib, d) -= pt_uba * duab[d];
                    DP(ib, ia, d) -= pt_uab * duab[d];
                }
            } else {
                for (int d = 0; d < 3; ++d) {
                    DP(ib, ib, d) -= pt_uba * duba[d];
                    DP(ib, ia, d) -= pt_uab * duba[d];
                }
            }

            // JCP 98, 5612 (1993) (B8)(B10) miss these terms (PySCF note).
            if (ia != atom_id && ib != atom_id) {
                for (int d = 0; d < 3; ++d) {
                    const double ua_ub =
                        (grid_norm_vec[ia][d] - grid_norm_vec[ib][d]) / Rab;
                    DP(atom_id, ia, d) -= pt_uab * ua_ub;
                    DP(atom_id, ib, d) -= pt_uba * ua_ub;
                }
            }
        }
    }

    // Finalize: dpbecke[:,ia] *= pbecke[ia]  (convert d(ln p)/dR -> dp/dR).
    for (int ia = 0; ia < natm; ++ia)
        for (int A = 0; A < natm; ++A)
            for (int d = 0; d < 3; ++d)
                DP(A, ia, d) *= pbecke[ia];

    // Normalized-weight derivative for the parent atom:
    //   z  = 1 / sum_B pbecke[B]
    //   w1[A] = dpbecke[A,parent] * z - pbecke[parent] * z^2 * sum_B dpbecke[A,B]
    double psum = 0.0;
    for (int B = 0; B < natm; ++B)
        psum += pbecke[B];
    const double z = 1.0 / psum;
    for (int A = 0; A < natm; ++A) {
        for (int d = 0; d < 3; ++d) {
            double dsum = 0.0;
            for (int B = 0; B < natm; ++B)
                dsum += DP(A, B, d);
            dw_out[A * 3 + d] =
                DP(A, atom_id, d) * z - pbecke[atom_id] * z * z * dsum;
        }
    }
    // Normalized partition weight of the parent atom at this grid point.
    return pbecke[atom_id] * z;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
