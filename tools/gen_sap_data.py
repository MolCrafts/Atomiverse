#!/usr/bin/env python3
"""One-off generator for include/atomiverse/cpu/scf/internal/sap_data.h.

Transcribes Lehtola's SAP (superposition of atomic potentials) effective
atomic potentials -- in the Gaussian-fitted error-function expansion form
shipped by PySCF -- into a static C++ data header. No file is read at
Atomiverse build, test, or run time; this script runs once, by hand.

Source data:
    <pyscf>/gto/basis/sap_grasp_large.dat   (PySCF 2.14.0; Basis Set
    Exchange entry "sap_grasp_large", "Data from S. Lehtola")

Each element block lists s-type Gaussians (exponent, coefficient). Every
Gaussian is a UNIT-normalized charge distribution scaled by its coefficient
(PySCF fakemol_for_cgtf_charge convention), so the screening potential in
real space is  V_scr(r) = -sum_k c_k erf(sqrt(alpha_k) r) / r  and the full
SAP effective atomic potential is

    V_A(r) = -( Z_A + sum_k c_k erf(sqrt(alpha_k) r) ) / r      [Hartree]

with sum_k c_k ~= -Z_A for a neutral atom (so V_A -> 0 as r -> inf).

References:
    S. Lehtola, J. Chem. Theory Comput. 15, 1593 (2019),
        DOI 10.1021/acs.jctc.8b01089  (SAP guess)
    S. Lehtola, L. Visscher, E. Engel, J. Chem. Phys. 152, 144105 (2020),
        DOI 10.1063/5.0004046  (Gaussian-fitted SAP potentials)

Regenerate with:
    python3 tools/gen_sap_data.py \
        ~/work/.x86_64/lib/python3.14/site-packages/pyscf/gto/basis/sap_grasp_large.dat \
        > include/atomiverse/cpu/scf/internal/sap_data.h
"""

import sys

SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}


def parse(path):
    blocks = {}  # Z -> [(alpha, coeff), ...]
    cur = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("BASIS"):
                continue
            parts = s.split()
            if len(parts) == 2 and parts[0] in Z_OF and parts[1] == "S":
                cur = blocks.setdefault(Z_OF[parts[0]], [])
            elif cur is not None and len(parts) == 2:
                cur.append((float(parts[0]), float(parts[1])))
    return blocks


def main():
    blocks = parse(sys.argv[1])
    max_z = max(blocks)
    offsets = [0]
    alphas, coeffs = [], []
    for z in range(1, max_z + 1):
        terms = blocks.get(z, [])
        for a, c in terms:
            alphas.append(a)
            coeffs.append(c)
        offsets.append(len(alphas))

    w = sys.stdout.write
    w("""/// @file cpu/scf/internal/sap_data.h
/// @brief Static SAP (superposition of atomic potentials) atomic-potential
///        table: Lehtola's exchange-only LDA effective atomic radial
///        potentials in the Gaussian-fitted error-function expansion form,
///        Z = 1..%d, transcribed once from PySCF 2.14.0
///        `pyscf/gto/basis/sap_grasp_large.dat` (Basis Set Exchange entry
///        "sap_grasp_large", data from S. Lehtola). GENERATED FILE -- edit
///        tools/gen_sap_data.py, not this header. Regenerate with:
///
///          python3 tools/gen_sap_data.py \\
///            <pyscf>/gto/basis/sap_grasp_large.dat \\
///            > include/atomiverse/cpu/scf/internal/sap_data.h
///
/// Each element's expansion lists unit-normalized Gaussian charge
/// distributions (exponent alpha_k [Bohr^-2], coefficient c_k [e], with
/// sum_k c_k ~= -Z for a neutral atom), so the full SAP effective atomic
/// potential at distance r [Bohr] from the nucleus is
///
///   V_A(r) = -( Z_A + sum_k c_k erf(sqrt(alpha_k) r) ) / r     [Hartree]
///
/// which tends to -Z_A/r at the nucleus and to 0 asymptotically.
///
/// References:
///   - S. Lehtola, J. Chem. Theory Comput. 15, 1593 (2019),
///     DOI 10.1021/acs.jctc.8b01089 (the SAP initial guess).
///   - S. Lehtola, L. Visscher, E. Engel, J. Chem. Phys. 152, 144105 (2020),
///     DOI 10.1063/5.0004046 (the Gaussian-fitted SAP potentials).
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Highest atomic number with a tabulated SAP potential.
inline constexpr int kSapMaxZ = %d;

/// @brief Per-element term ranges: element Z owns flat-array entries
///        [kSapOffset[Z-1], kSapOffset[Z]) of kSapAlpha / kSapCoeff.
inline constexpr int kSapOffset[kSapMaxZ + 1] = {
""" % (max_z, max_z))
    for i in range(0, len(offsets), 12):
        w("    " + ", ".join(str(o) for o in offsets[i:i + 12]) + ",\n")
    w("};\n\n")

    def dump(name, vals, unit):
        w("/// @brief Flat %s array [%s] (see kSapOffset).\n" % (name, unit))
        w("inline constexpr double kSap%s[] = {\n" % name.capitalize())
        for i in range(0, len(vals), 3):
            w("    " + ", ".join("%.16e" % v for v in vals[i:i + 3]) + ",\n")
        w("};\n\n")

    dump("alpha", alphas, "Bohr^-2")
    dump("coeff", coeffs, "e")

    w("""/// @brief Lookup row of the SAP table for one element -- an OOP lookup
///        constructor (no free-function factory). Throws on unsupported Z.
///
/// V_A(r) = -( Z + sum_k c_k erf(sqrt(alpha_k) r) ) / r  [Hartree], r [Bohr].
struct SapAtomicPotential {
    int Z;               ///< Atomic number (nuclear charge) [e].
    int n;               ///< Number of error-function expansion terms.
    const double* alpha; ///< Gaussian exponents [Bohr^-2], n entries.
    const double* coeff; ///< Charge coefficients [e], n entries (sum ~= -Z).

    /// @brief Look up the tabulated expansion for element `z`.
    /// @param z Atomic number; outside [1, kSapMaxZ] throws
    ///          std::invalid_argument (no fallback).
    explicit SapAtomicPotential(int z) {
        if (z < 1 || z > kSapMaxZ)
            throw std::invalid_argument(
                "SapAtomicPotential: no SAP potential tabulated for Z="
                + std::to_string(z) + " (supported: 1.."
                + std::to_string(kSapMaxZ) + ")");
        Z = z;
        n = kSapOffset[z] - kSapOffset[z - 1];
        alpha = kSapAlpha + kSapOffset[z - 1];
        coeff = kSapCoeff + kSapOffset[z - 1];
    }

    /// @brief Evaluate the effective atomic potential at distance `r`.
    /// @param r Distance from the nucleus [Bohr]; clamped below at 1e-12
    ///          to keep the -Z/r nuclear singularity finite on quadrature
    ///          grids (radial grids never place a point exactly at r = 0).
    /// @return V_A(r) [Hartree] (negative near the nucleus, -> 0 far away).
    double value(double r) const {
        const double rc = (r > 1e-12) ? r : 1e-12;
        double q = static_cast<double>(Z);
        for (int k = 0; k < n; ++k)
            q += coeff[k] * std::erf(std::sqrt(alpha[k]) * rc);
        return -q / rc;
    }
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
""")


if __name__ == "__main__":
    main()
