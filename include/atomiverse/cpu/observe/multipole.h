/// @file cpu/observe/multipole.h
/// @brief MultipoleObserver — molecular dipole and traceless quadrupole from
///        a converged mean-field wavefunction.
///
/// Observer kind (FrameImmutable|Deferred): consumes the sibling method's
/// `scf::MeanFieldOutputs` (SlaterDeterminant + shells + optional c2s),
/// rebuilds the multipole one-electron integrals about the center of nuclear
/// charge, and writes:
///
///   - `frame.meta.get(kDipole)`          dipole [Debye], (x,y,z)
///   - `frame.meta.get(kQuadrupole)`      traceless Buckingham quadrupole
///                                        [D·Å], packed xx,xy,xz,yy,yz,zz
///   - `frame.meta.get(kMultipoleOrigin)` expansion origin [Å]
///
/// μ = Σ_A Z_A R_A − Tr(P·r); Θ_ab = ½Σ q (3 r_a r_b − r² δ_ab). The origin
/// convention (center of nuclear charge) is reported with the numbers —
/// dipoles of charged systems and quadrupoles of dipolar systems are
/// origin-bound (μ′ = μ − q_tot·d under a shift d). Szabo & Ostlund,
/// properties; integrals per Obara & Saika, J. Chem. Phys. 84, 3963 (1986).
///
/// No default bindings: bind at the host driver's end-of-step phase, like
/// `OrbitalInfoObserver`. Integrals are private scratch — never on Frame.
#pragma once

#include <array>

#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {

/// @brief Dipole + traceless quadrupole observer over MeanFieldOutputs.
class MultipoleObserver : public Observer {
  public:
    /// @brief Wire to a mean-field method's shared outputs.
    /// @param handle The method's `result_handle()` (determinant + shells +
    ///        c2s); values are read at launch time, after the method ran.
    explicit MultipoleObserver(scf::MeanFieldOutputs handle);

    /// @brief Component identifier.
    /// @return "multipole_observer".
    std::string name() const override { return "multipole_observer"; }

    /// @brief Compute μ/Θ from the current determinant and write frame.meta.
    /// @param ctx Launch context (frame + phase guard by the caller's plan).
    void launch(const LaunchContext& ctx) override;

    /// @brief Dipole from the last launch [a.u., e·Bohr].
    /// @return (x,y,z) components.
    std::array<double, 3> dipole_au() const { return dipole_au_; }

    /// @brief Dipole from the last launch [Debye].
    /// @return (x,y,z) components.
    std::array<double, 3> dipole_debye() const;

    /// @brief Traceless quadrupole from the last launch [D·Å].
    /// @return Packed xx,xy,xz,yy,yz,zz.
    std::array<double, 6> quadrupole_buckingham() const;

    /// @brief Expansion origin from the last launch [Å].
    /// @return (x,y,z), center of nuclear charge.
    std::array<double, 3> origin_angstrom() const { return origin_ang_; }

  private:
    scf::MeanFieldOutputs handle_;
    std::array<double, 3> dipole_au_{};
    std::array<double, 6> quad_au_{}; // traceless, [e·Bohr²]
    std::array<double, 3> origin_ang_{};
};

} // namespace cpu
} // namespace ATV
