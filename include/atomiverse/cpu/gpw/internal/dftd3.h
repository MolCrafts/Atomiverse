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
/// Optional three-body ATM / C9 (`calculate_c9`, CP2K `CALCULATE_C9_TERM`):
///
///   C9 = −√(C6_AB C6_BC C6_CA)
///   E9 contribution folded as −s9 f_d C9 ang / (R_AB² R_BC² R_CA²)^{3/2}
///   with s9 = s6, damping_d3(rabc, r0, 4/3, 16), ang from CP2K s1·s2·s3 form.
///
/// PBE parameters: s6 = 1.0, s8 = 0.722, sr6 = 1.217, sr8 = 1.0. A smooth
/// fifth-order cutoff (CP2K `damping_d3`) zeroes each term over [rcut-1, rcut].
/// Periodic (orthorhombic, Γ-point) lattice sums use the same image enumeration
/// as the GPW integrals. Default `calculate_c9 = false` matches historical
/// two-body-only pins.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief CP2K-parity pair / ABC cutoff for the GPW dispersion sum [Bohr].
///
/// CP2K's `&PAIR_POTENTIAL R_CUTOFF 20.0` (Å) is a *half* cutoff — it prints
/// `Cutoff Radius [Bohr]: 37.79` and sums pairs to **2 ×** that. Both damping
/// variants must use the same value or the energy, force and stress paths
/// differentiate different functionals. Verified against offline CP2K 2026.1 on
/// a periodic water dimer: this value reproduces `Dispersion energy` to
/// 6.7e-11 Ha, whereas the previously hard-coded 40 Bohr was off by 3.9e-7 Ha
/// (see regressions/gpw/gpw-d3-forces/REFERENCE.md).
inline constexpr double kD3Cp2kPairCutoffBohr = 2.0 * 20.0 / 0.52917720859;

/// @brief Result of a DFT-D3 evaluation.
struct D3Result {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces F_A = -dE/dR_A [Hartree/Bohr], one xyz triple per atom.
    std::vector<std::array<double, 3>> forces;
    /// @brief Unnormalized pair virial W_αβ [Hartree] in Voigt order
    ///        (xx,yy,zz,xy,xz,yz). Intensive stress is σ = W / V with V in
    ///        Bohr³ (or convert both energy and length to eV/Å and divide by
    ///        V_Å³). Includes lattice-image pair vectors when `periodic`.
    ///        Empty / zero when no pairs contributed. Sign: W = Σ r ⊗ F_pair
    ///        with F_pair the force on the first site of each pair term, so
    ///        σ = W/V matches σ = −(1/V) ∂E/∂ε for F = −∇E.
    double virial[6] = {0, 0, 0, 0, 0, 0};
};

/// @brief PBE-D3 (zero damping) dispersion energy and forces.
///
/// @param Z        Atomic numbers, length N (1-based element identity).
/// @param pos_bohr Atom positions [Bohr], length N.
/// @param cell_bohr Orthorhombic box lengths a,b,c [Bohr] (used iff periodic).
/// @param periodic When true, sum over lattice images within `rcut_bohr`; when
///        false, a single isolated cluster (no images).
/// @param params   D3 reference parameters (`D3Params{}`, in-tree tables).
/// @param rcut_bohr Pair / ABC cutoff [Bohr]; CP2K default is 2*R_CUTOFF.
/// @param calculate_c9 Include three-body ATM / C9 (CP2K CALCULATE_C9_TERM).
/// @return Energy [Hartree] and forces [Hartree/Bohr].
D3Result dftd3_pbe(const std::vector<int>& Z,
                   const std::vector<std::array<double, 3>>& pos_bohr,
                   const double cell_bohr[3], bool periodic,
                   const D3Params& params, double rcut_bohr = 40.0,
                   bool calculate_c9 = false);

/// @brief PBE-D3 with Becke–Johnson (rational) damping.
///
///   E = −Σ_{A<B} [ s6 C6/(R⁶ + f⁶) + s8 C8/(R⁸ + f⁸) ]
///   f  = a1 √(C8/C6) + a2
///
/// PBE-D3(BJ) defaults (Grimme et al., J. Comput. Chem. 32, 1456 (2011)):
/// s6 = 1.0, s8 = 0.7875, a1 = 0.4289, a2 = 4.4407 Bohr.
///
/// Coordination numbers: when `cp2k_cn` is false (default), use the original
/// Grimme form (simple-dftd3 / PySCF `disp='d3bj'` molecular path). When true,
/// use CP2K's `cnparam_d3` (extra long-range tanh factor) so GPW periodic
/// E_disp matches offline `TYPE DFTD3(BJ)`.
///
/// @param Z         Atomic numbers, length N.
/// @param pos_bohr  Atom positions [Bohr], length N.
/// @param cell_bohr Orthorhombic box lengths a,b,c [Bohr] (used iff periodic).
/// @param periodic  Lattice images within `rcut_bohr` when true; isolated else.
/// @param params    D3 reference parameters (`D3Params{}`).
/// @param rcut_bohr Pair cutoff [Bohr] (default 94.2 ≈ 50 Å; GPW/CP2K twin
///        uses 40 = 2 × CP2K `R_CUTOFF 20`).
/// @param s6        C6 scale (default 1.0 for PBE).
/// @param s8        C8 scale (default 0.7875 for PBE-BJ).
/// @param a1        BJ steepness (default 0.4289 for PBE).
/// @param a2        BJ offset [Bohr] (default 4.4407 for PBE).
/// @param cp2k_cn   Use CP2K coordination-number form (GPW parity).
/// @param calculate_c9 Include three-body ATM / C9 (CP2K CALCULATE_C9_TERM).
/// @return Energy [Hartree] and forces [Hartree/Bohr].
D3Result dftd3_pbe_bj(const std::vector<int>& Z,
                      const std::vector<std::array<double, 3>>& pos_bohr,
                      const double cell_bohr[3], bool periodic,
                      const D3Params& params, double rcut_bohr = 94.2,
                      double s6 = 1.0, double s8 = 0.7875, double a1 = 0.4289,
                      double a2 = 4.4407, bool cp2k_cn = false,
                      bool calculate_c9 = false);

} // namespace gpw
} // namespace cpu
} // namespace ATV
