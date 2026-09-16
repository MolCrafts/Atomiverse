/// @file cpu/scf/dftd3.h
/// @brief Public molecular DFT-D3 with Becke–Johnson damping (PBE defaults).
///
/// Geometry-only additive dispersion correction for the Gaussian-basis path.
/// Independent of the electronic wavefunction: evaluate on a Frame of atomic
/// numbers + positions [Å]. GPW keeps its own zero-damping `d3` flag; this is
/// the molecular / PySCF-`disp='d3bj'` twin.
///
/// @code
///   cpu::scf::D3BJ d3;                 // PBE-D3(BJ) defaults
///   double e_disp = d3.energy(frame);  // [Hartree]
///   auto f = d3.forces(frame);         // f[A] = −∂E_disp/∂R_A [Hartree/Bohr]
///
///   // Or fold into RKS total energy AND its analytic gradient:
///   cpu::scf::RKS rks(cpu::scf::XCType::PBE);
///   rks.d3bj = true;
///   rks.run(frame);
///   double ed = rks.energy_disp();     // [Hartree]
/// @endcode
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Molecular Grimme DFT-D3 with Becke–Johnson (rational) damping.
///
/// Default parameters are the PBE-D3(BJ) set of Grimme et al.,
/// *J. Comput. Chem.* **32**, 1456 (2011): s6 = 1, s8 = 0.7875, a1 = 0.4289,
/// a2 = 4.4407 Bohr. Tables are the in-tree Grimme reference C6 / r2r4 / rcov
/// set (shared with GPW). Three-body ATM is not included.
class D3BJ {
  public:
    double s6 = 1.0;         ///< C6 scale (dimensionless; PBE = 1).
    double s8 = 0.7875;      ///< C8 scale (dimensionless; PBE-BJ).
    double a1 = 0.4289;      ///< BJ steepness (dimensionless; PBE).
    double a2 = 4.4407;      ///< BJ offset [Bohr]; PBE.
    double rcut_bohr = 94.2; ///< Pair cutoff [Bohr] (~50 Å).

    /// @brief Dispersion energy for the Frame geometry.
    ///
    /// Reads `kAtom_type` (atomic number Z) and `kAtom_x/y/z` [Å]. Isolated
    /// (non-periodic) evaluation — the Frame box is ignored (gas-phase
    /// molecular path). Unsupported Z (outside the in-tree D3 table) throws.
    ///
    /// @param frame System geometry (Å, Z).
    /// @return E_disp [Hartree] (negative when dispersion is attractive).
    double energy(const Frame& frame) const;

    /// @brief Analytic dispersion forces \(F_A = -\partial E_{disp}/\partial
    ///        R_A\) for the same functional `energy()` evaluates.
    ///
    /// Two-body C6/C8 BJ terms **plus** the coordination-number chain
    /// \(\partial E/\partial C_n \cdot dC_n/dCN \cdot dCN/dR\) — the term that
    /// makes the gradient exact rather than merely dominant. Same members, same
    /// isolated (non-periodic) evaluation, same table lookups as `energy()`, so
    /// the pair is conservative by construction. Unsupported Z throws.
    ///
    /// @param frame System geometry (Å, Z).
    /// @return One xyz triple per atom [Hartree/Bohr], length `frame.nlocal()`.
    std::vector<std::array<double, 3>> forces(const Frame& frame) const;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
