/// @file cpu/dynamics/surface_tension_npt.h
/// @brief Constant surface-tension (NPγT) MTK integrator — LAMMPS
///        `fix npt … couple xy` path with \(P_\perp(L_z)\) offset.
#pragma once

#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Semi-isotropic MTK NPT with constant surface tension \(\gamma\).
///
/// Public foothold for the constant-surface-tension ensemble. Implements
/// LAMMPS `fix npt temp … x Plat … y Plat … z Pn … couple xy` with the
/// GROMACS-style / literature NPγT lateral target
///
///     P_⊥(t) = P_n − n_surf · γ / L_z(t),   P_z = P_n
///
/// (\(P\) [eV/Å³], \(\gamma\) [eV/Å²], \(L_z\) [Å]).
///
/// **Replaces** (do not add alongside) `VelocityVerlet`, `NoseHoover`,
/// `Barostat`, and plain isotropic `NoseHooverNPT`.
///
/// Instantaneous planar surface tension (diagnostic):
///
///     γ_inst = (L_z / 2) · (P_zz − (P_xx + P_yy) / 2)
///
/// with LAMMPS-standard pressure (positive ⇔ compressive).
///
/// @see NoseHooverNPT::set_couple_xy, NoseHooverNPT::set_surface_tension
/// @see Martyna, Tobias, Klein, J. Chem. Phys. 101, 4177 (1994)
class SurfaceTensionNPT : public NoseHooverNPT {
  public:
    /// @brief Construct an NPγT integrator (parameters assigned immediately).
    /// @param T_target   Target temperature [K].
    /// @param tau_T      Thermostat damping [fs] (LAMMPS Tdamp).
    /// @param P_normal   Target normal pressure \(P_n = P_{zz}\) [eV/Å³]
    ///                   (LAMMPS sign, positive compressive).
    /// @param gamma      Target surface tension [eV/Å²].
    /// @param tau_P      Barostat damping [fs] (LAMMPS Pdamp).
    /// @param dt         Timestep [fs].
    /// @param n_surfaces Number of planar interfaces (default 2).
    /// @param drag       Optional NHC drag (LAMMPS drag); 0 disables.
    /// @param mtchain    Atom NHC length (default 3).
    /// @param mpchain    Barostat NHC length (default 3).
    SurfaceTensionNPT(double T_target, double tau_T, double P_normal,
                      double gamma, double tau_P, double dt, int n_surfaces = 2,
                      double drag = 0.0, int mtchain = 3, int mpchain = 3)
        : NoseHooverNPT(T_target, tau_T, P_normal, tau_P, dt, drag, mtchain,
                        mpchain, /*couple_axis=*/{true, true, true},
                        /*T_stop=*/-1.0) {
        set_surface_tension(gamma, n_surfaces);
    }

    std::string name() const override { return "surface_tension_npt"; }
};

} // namespace md
} // namespace cpu
} // namespace ATV
