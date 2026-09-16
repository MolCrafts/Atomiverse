/// @file cpu/dynamics/fix_dfield.h
/// @brief Displacement-field finite-field MD (MetalWalls D-field):
///        vacuum foothold and self-consistent constant-D ensemble.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Apply \(\mathbf{D}\) as an external field coupling \(U=-V\mathbf{D}\cdot\mathbf{P}\),
///        optionally with Zhang–Sprik constant-D polarization feedback along z.
///
/// Default (foothold, vacuum / metal units): treat \(\mathbf{E}\equiv\mathbf{D}\)
/// for force \(F=qE\) on free charges, matching the finite-field electrolyte
/// path when \(\varepsilon=1\). For const-pot walls compose with
/// `ConstPotBO::Ex,Ey,Ez = Dx,Dy,Dz` so electrode \(Q\) sees the same field.
///
/// With `polarization_feedback = true` the z axis runs the self-consistent
/// constant-\(D_z\) ensemble (Zhang & Sprik, Phys. Rev. B 93, 144201 (2016);
/// Cox & Sprik, J. Chem. Phys. 151, 064506 (2019)): the Maxwell field
/// responds to the instantaneous itinerant polarization,
/// \(E_z = D_z - 4\pi k_e P_z\) with \(P_z = \sum_i q_i z_i^{u}/V\) from
/// unwrapped coordinates, and the electric-enthalpy term
/// \(U_D = V (D_z - 4\pi k_e P_z)^2 / (8\pi k_e)\) enters the energy so that
/// \(F_{i,z} = -\partial U_D/\partial z_i = q_i E_z\) exactly
/// (\(k_e =\) `units::coulomb` converts \(P\) [e/Å²] into a field [V/Å]).
/// x,y keep the foothold coupling — MetalWalls applies \(D\) along the cell
/// normal only. Berry-phase / electronic polarization is out of scope.
class FixDfield : public cpu::Bias {
  public:
    /// @param Dx Displacement field x [V/Å] (foothold unit match to E)
    /// @param Dy Displacement field y [V/Å]
    /// @param Dz Displacement field z [V/Å]
    FixDfield(double Dx, double Dy, double Dz) noexcept
        : Dx(Dx), Dy(Dy), Dz(Dz) {}

    double Dx = 0.0, Dy = 0.0, Dz = 0.0;

    /// @brief Enable the self-consistent constant-\(D_z\) ensemble
    ///        (\(E_z = D_z - 4\pi k_e P_z\)); default keeps the vacuum
    ///        \(E\equiv D\) foothold.
    bool polarization_feedback = false;

    std::string name() const override { return "fix_dfield"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    double last_energy() const noexcept { return last_u_; }
    /// @brief Itinerant cell polarization \(\sum q_i z_i^{u} / V\) z-component [e/Å²]
    double last_Pz() const noexcept { return last_Pz_; }
    /// @brief Maxwell field applied along z [V/Å]: \(D_z\) on the foothold
    ///        path, \(D_z - 4\pi k_e P_z\) under polarization feedback.
    double last_Ez() const noexcept { return last_Ez_; }

  private:
    double last_u_ = 0.0;
    double last_Pz_ = 0.0;
    double last_Ez_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
