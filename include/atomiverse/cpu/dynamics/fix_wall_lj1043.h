/// @file cpu/dynamics/fix_wall_lj1043.h
/// @brief CPU Steele 10-4-3 continuum wall — LAMMPS-equivalent `fix wall/lj1043`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Planar Steele 10-4-3 wall potential (metal units).
///
/// Mirrors LAMMPS `fix wall/lj1043` (`lammps/src/fix_wall_lj1043.cpp`):
/// integrated LJ continuum wall (Steele, *Surf. Sci.* **36**, 317 (1973)).
///
/// \[
/// U(r)=2\pi\varepsilon\left[
///   \tfrac{2}{5}(\sigma/r)^{10}-(\sigma/r)^{4}
///   -\frac{\sqrt{2}\,\sigma^{3}}{3\bigl(r+(0.61/\sqrt{2})\sigma\bigr)^{3}}
/// \right]-U(r_c),\qquad 0<r<r_c
/// \]
///
/// with \(r\) the perpendicular distance from the wall plane to the atom on
/// the fluid side. Energy is **shifted** so \(U(r_c)=0\) (LAMMPS offset).
/// Force is \(-\nabla U\) along the wall normal with LAMMPS sign convention.
///
/// Bound by default at `md::Phase::post_force` (LAMMPS POST_FORCE). For
/// single-point Evaluate, register explicitly:
/// `eval.on(evaluate::Phase::evaluate).add(FixWallLJ1043{…})`.
///
/// The potential is defined only on \(0 < r < r_c\): it diverges as
/// \(r\to0^+\), and its third term has a pole at \(r=-(0.61/\sqrt2)\sigma\),
/// so for \(-0.431\sigma < r < 0\) the expression is finite and *attractive*.
/// `launch` therefore throws `std::runtime_error` naming the axis, the wall
/// position and the offending coordinate when any owned atom reaches \(r\le0\),
/// before any of that launch's energy is written to the Frame.
///
/// This deliberately diverges from LAMMPS's own `fix_wall_lj1043.cpp`, which
/// skips such atoms silently — an inconsistency within LAMMPS, whose every
/// other wall style raises "Particle on or inside fix wall surface" and whose
/// documentation requires \(r>0\). We follow the documented contract.
///
/// ── Not in scope ─────────────────────────────────────────────────────
/// variable wall position, multi-wall in one object, `units lattice`,
/// `fld yes`.
class FixWallLJ1043 : public cpu::Bias {
  public:
    /// @param axis Wall normal: 0 = \(x\), 1 = \(y\), 2 = \(z\).
    /// @param side Fluid side: \(-1\) = lo (atoms with coord > wall_pos),
    ///             \(+1\) = hi (atoms with coord < wall_pos).
    /// @param wall_pos Wall plane coordinate [Å].
    /// @param epsilon Well depth \(\varepsilon\) [eV] (must be > 0).
    /// @param sigma Length \(\sigma\) [Å] (must be > 0).
    /// @param cutoff Interaction cutoff \(r_c\) [Å] (must be > 0).
    FixWallLJ1043(int axis, int side, double wall_pos, double epsilon,
                  double sigma, double cutoff);

    /// Public parameters (no Config bag) — mutable after construction.
    int axis = 2;          ///< 0=x, 1=y, 2=z
    int side = -1;         ///< -1 = lo, +1 = hi
    double wall_pos = 0.0; ///< [Å]
    double epsilon = 0.0;  ///< [eV]
    double sigma = 0.0;    ///< [Å]
    double cutoff = 0.0;   ///< [Å]

    std::string name() const override { return "fix_wall_lj1043"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After pair / bonded / kspace; before final velocity half-kick.
        return {ATV::make_binding(ATV::md::Phase::post_force, +100)};
    }

    void launch(const cpu::LaunchContext& ctx) override;

    /// @return Last wall energy contribution written this launch [eV].
    double last_energy() const noexcept { return last_u_; }

  private:
    void precompute_();

    // Precomputed coeffs (LAMMPS naming) — set in ctor / when params change
    // is not live-recomputed after mutate; call precompute_ from launch so
    // public member edits are honoured.
    double coeff1_ = 0; ///< 2π·(2/5)·ε·σ¹⁰
    double coeff2_ = 0; ///< 2π·ε·σ⁴
    double coeff3_ = 0; ///< 2π·√2/3 ·ε·σ³
    double coeff4_ = 0; ///< (0.61/√2)·σ
    double coeff5_ = 0; ///< 10·coeff1
    double coeff6_ = 0; ///< 4·coeff2
    double coeff7_ = 0; ///< 3·coeff3
    double offset_ = 0; ///< U(rc) unshifted
    double last_u_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
