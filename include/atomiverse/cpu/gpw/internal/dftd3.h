/// @file cpu/gpw/internal/dftd3.h
/// @brief Grimme DFT-D3 dispersion correction (zero / "original" damping) for
///        the PBE reference functional, matching CP2K's `TYPE DFTD3`.
///
/// A self-contained additive correction (energy + analytic forces) on top of
/// the GPW DFT, evaluated as a pairwise sum over the (periodically imaged)
/// atoms. Method: Grimme, Antony, Ehrlich, Krieg, J. Chem. Phys. 132, 154104
/// (2010), zero damping; bit-for-bit faithful to CP2K's `qs_dispersion_d3`:
///
///   E_disp = - Σ_{A<B} Σ_{n=6,8} s_n (C_n^AB / R_AB^n) f_{d,n}(R_AB)
///   f_{d,n}(R) = 1 / (1 + 6 (R / (s_{r,n} R0_AB))^{-α_n}),  α6=14, α8=16
///   C_8^AB = 3 C_6^AB r2r4_A r2r4_B,  R0_AB from CP2K's r0ab table [Bohr]
///   C_6^AB = CN-interpolated reference C6 (k3 = -4)
///   CN_A   = Σ_{B≠A} 1/(1+exp(-k1((rcov_A+rcov_B)/R_AB - 1))) · ½(1-tanh(R-2 rcov_AB))
///
/// PBE parameters: s6 = 1.0, s8 = 0.722, sr6 = 1.217, sr8 = 1.0. A smooth
/// fifth-order cutoff (CP2K `damping_d3`) zeroes each term over [rcut-1, rcut].
/// Periodic (orthorhombic, Γ-point) lattice sums use the same image enumeration
/// as the GPW integrals; the three-body (C9 / ATM) term is not included (CP2K's
/// default with `TYPE DFTD3` for the ZnO_MD target leaves it off).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Result of a DFT-D3 evaluation.
struct D3Result {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces F_A = -dE/dR_A [Hartree/Bohr], one xyz triple per atom.
    std::vector<std::array<double, 3>> forces;
};

/// @brief PBE-D3 (zero damping) dispersion energy and forces.
///
/// @param Z        Atomic numbers, length N (1-based element identity).
/// @param pos_bohr Atom positions [Bohr], length N.
/// @param cell_bohr Orthorhombic box lengths a,b,c [Bohr] (used iff periodic).
/// @param periodic When true, sum over lattice images within `rcut_bohr`; when
///        false, a single isolated cluster (no images).
/// @param params   D3 reference parameters (`D3Params{}`, in-tree tables).
/// @param rcut_bohr Pair cutoff [Bohr]; CP2K default is 2*R_CUTOFF = 40.
/// @return Energy [Hartree] and forces [Hartree/Bohr].
D3Result dftd3_pbe(const std::vector<int>& Z,
                   const std::vector<std::array<double, 3>>& pos_bohr,
                   const double cell_bohr[3], bool periodic,
                   const D3Params& params, double rcut_bohr = 40.0);

} // namespace gpw
} // namespace cpu
} // namespace ATV
