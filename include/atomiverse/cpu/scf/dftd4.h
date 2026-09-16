/// @file cpu/scf/dftd4.h
/// @brief DFT-D4 with Becke–Johnson damping (PBE defaults) — molecular Frame
///        API plus a Bohr-coordinate free function used by periodic GPW.
///
/// Geometry-only additive London dispersion (Caldeweyher et al., J. Chem.
/// Phys. 150, 154122 (2019); periodic extension). Two-body BJ term always;
/// optional Axilrod–Teller–Muto three-body (`s9`, default 0 = off). Tables are
/// in-tree (Z ≤ 94).
///
/// @code
///   cpu::scf::D4 d4;                   // PBE-D4 defaults (isolated, ATM off)
///   double e_disp = d4.energy(frame);  // [Hartree]
///
///   // Periodic (orthorhombic) from Bohr positions — GPW path:
///   double ed = cpu::scf::dftd4_energy(Z, pos_bohr, cell, /*periodic=*/true);
///   // With ATM (dftd4 pbe-D4-ATM / --mbdscale 1):
///   double ed_atm = cpu::scf::dftd4_energy(Z, pos_bohr, cell, true,
///                                          /*s6=*/1.0, /*s8=*/..., /*a1=*/...,
///                                          /*a2=*/..., /*ga=*/3, /*gc=*/2,
///                                          /*wf=*/6, /*charge=*/0,
///                                          /*cn_cut=*/30, /*disp2_cut=*/60,
///                                          /*s9=*/1.0);
///   auto r = cpu::scf::dftd4(Z, pos_bohr, cell, /*periodic=*/true);
///   // r.forces[A] = −∂E_disp/∂R_A  [Hartree/Bohr]
/// @endcode
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief PBE-D4 energy and ionic forces (two-body + optional ATM).
struct D4Result {
    double energy = 0.0; ///< Dispersion energy [Hartree].
    /// @brief Forces \(F_A = -\partial E/\partial R_A\) [Hartree/Bohr].
    std::vector<std::array<double, 3>> forces;
};

/// @brief PBE-D4 energy from Bohr coordinates.
///
/// @param Z         Atomic numbers (length N).
/// @param pos_bohr  Positions [Bohr].
/// @param cell_bohr Orthorhombic a,b,c [Bohr] (used iff `periodic`).
/// @param periodic  Lattice-sum CN, Ewald EEQ, and lattice-sum dispersion.
/// @param s6,s8,a1,a2 BJ scales (PBE defaults).
/// @param ga,gc,wf  Charge / CN model parameters.
/// @param charge    Net charge for EEQ [e].
/// @param cn_cut    CN real-space cutoff [Bohr] (dftd4 default 30).
/// @param disp2_cut Two-body dispersion cutoff [Bohr] (dftd4 default 60).
/// @param s9        ATM three-body scale (0 = off; PBE-D4-ATM = 1).
/// @param disp3_cut Three-body cutoff [Bohr] (dftd4 default 40).
/// @return E_disp [Hartree].
double dftd4_energy(const std::vector<int>& Z,
                    const std::vector<std::array<double, 3>>& pos_bohr,
                    const double cell_bohr[3], bool periodic, double s6 = 1.0,
                    double s8 = 0.95948085, double a1 = 0.38574991,
                    double a2 = 4.80688534, double ga = 3.0, double gc = 2.0,
                    double wf = 6.0, double charge = 0.0, double cn_cut = 30.0,
                    double disp2_cut = 60.0, double s9 = 0.0,
                    double disp3_cut = 40.0);

/// @brief PBE-D4 energy and forces \(F = -\nabla E_{\mathrm{disp}}\).
///
/// Energy matches `dftd4_energy`. Forces are **fully analytic**: the two-body
/// gradient (BJ + CN\(_\mathrm{d4}\) chain + EEQ adjoint), and when
/// `s9 ≠ 0` the ATM three-body gradient (explicit damped Axilrod–Teller
/// triple derivatives + the C9 → neutral-C6 → CN\(_\mathrm{d4}\) chain; the
/// q = 0 reweight is geometry-independent, so the ATM leg has no EEQ
/// adjoint), giving \(F = -\nabla(E_2+E_{\mathrm{ATM}})\). Dual-gates against
/// offline dftd4 3.7.0 `--grad` (printed \(+\nabla E\); ATV stores
/// \(-\nabla E\)).
///
/// @param s9        ATM three-body scale (0 = off; PBE-D4-ATM = 1).
/// @param disp3_cut Three-body cutoff [Bohr].
/// @return Energy [Hartree] and forces [Hartree/Bohr].
D4Result dftd4(const std::vector<int>& Z,
               const std::vector<std::array<double, 3>>& pos_bohr,
               const double cell_bohr[3], bool periodic, double s6 = 1.0,
               double s8 = 0.95948085, double a1 = 0.38574991,
               double a2 = 4.80688534, double ga = 3.0, double gc = 2.0,
               double wf = 6.0, double charge = 0.0, double cn_cut = 30.0,
               double disp2_cut = 60.0, double s9 = 0.0,
               double disp3_cut = 40.0);

/// @brief Molecular Grimme DFT-D4 with Becke–Johnson (rational) damping.
///
/// Default parameters are the PBE-D4 set of Caldeweyher et al. (SAW190103
/// BJ+ATM fit; two-body only by default): s6 = 1, s8 = 0.95948085,
/// a1 = 0.38574991, a2 = 4.80688534 Bohr. Set `s9 = 1` for ATM. Charge model
/// is EEQ2019; ga = 3, gc = 2, wf = 6. Tables are in-tree (Z ≤ 94).
class D4 {
  public:
    double s6 = 1.0;        ///< C6 scale (dimensionless; PBE = 1).
    double s8 = 0.95948085; ///< C8 scale (dimensionless; PBE-D4).
    double a1 = 0.38574991; ///< BJ steepness (dimensionless; PBE).
    double a2 = 4.80688534; ///< BJ offset [Bohr]; PBE.
    double ga = 3.0;        ///< Charge-scaling height.
    double gc = 2.0;        ///< Charge-scaling steepness.
    double wf = 6.0;        ///< CN Gaussian weight factor.
    double charge = 0.0;    ///< Molecular charge for EEQ (electrons).
    /// @brief ATM three-body scale (0 = off; PBE-D4-ATM / dftd4 default = 1).
    double s9 = 0.0;

    /// @brief Dispersion energy for the Frame geometry.
    ///
    /// Reads `kAtom_type` (atomic number Z) and `kAtom_x/y/z` [Å]. Isolated
    /// (non-periodic) evaluation — the Frame box is ignored. Unsupported Z
    /// (outside the in-tree D4 table, Z > 94 or no reference states) throws.
    ///
    /// @param frame System geometry (Å, Z).
    /// @return E_disp [Hartree] (negative when dispersion is attractive).
    double energy(const Frame& frame) const;

    /// @brief Dispersion forces \(F_A = -\partial E_{disp}/\partial R_A\) for
    ///        the same functional `energy()` evaluates.
    ///
    /// Delegates to the free `dftd4()` gradient: analytic \(-\nabla E\) of the
    /// geometry-only D4 energy (BJ pair + CN\(_\mathrm{d4}\) chain + EEQ charge
    /// adjoint; analytic ATM triples when `s9 ≠ 0`). Same members, same
    /// isolated evaluation, same tables as `energy()`, so the pair is
    /// conservative. One pass — never 6N energy evaluations. Unsupported Z
    /// throws.
    ///
    /// @param frame System geometry (Å, Z).
    /// @return One xyz triple per atom [Hartree/Bohr], length `frame.nlocal()`.
    std::vector<std::array<double, 3>> forces(const Frame& frame) const;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
