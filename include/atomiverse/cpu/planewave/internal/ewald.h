/// @file cpu/planewave/internal/ewald.h
/// @brief 3D Ewald summation for periodic point charges (energy + ionic forces).
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include <span>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Ewald ion–ion energy and forces for classical point charges.
///
/// A uniform compensating background neutralizes a nonzero total charge.
/// Energy includes real, reciprocal, self and background terms [Hartree].
/// Ion-ion precision is independent of the electronic plane-wave cutoff.
/// Forces differentiate the same energy; self/background have zero ionic
/// position derivatives.
class EwaldSummation {
    AccReal e_real_ = 0, e_recip_ = 0, e_self_ = 0;
    AccReal e_background_ = 0;
    AccReal alpha_ = AccReal(1);
    AccReal real_cutoff_ = 0, reciprocal_cutoff_ = 0;
    std::vector<Vec3> positions_bohr_;
    std::vector<Vec3> forces_bohr_; ///< F = −∂E/∂R [Hartree/Bohr]

  public:
    /// @param pos_ang Positions [Å].
    /// @param charges Point charges (nuclear Z or valence, same as energy path).
    /// @param cell Simulation cell [Å].
    EwaldSummation(const std::vector<Vec3>& pos_ang,
                   const std::vector<AccReal>& charges, const cpu::Box& cell);

    AccReal total_energy() const {
        return e_real_ + e_recip_ + e_self_ + e_background_;
    }
    AccReal real_energy() const { return e_real_; }
    AccReal recip_energy() const { return e_recip_; }
    AccReal self_energy() const { return e_self_; }
    /// @brief Uniform compensating-background energy [Hartree].
    AccReal background_energy() const { return e_background_; }
    AccReal alpha() const { return alpha_; }
    /// @brief Real-space precision domain [Bohr], shared by ionic Hessians.
    AccReal real_cutoff() const { return real_cutoff_; }
    /// @brief Reciprocal precision domain [1/Bohr], shared by ionic Hessians.
    AccReal reciprocal_cutoff() const { return reciprocal_cutoff_; }
    /// @brief Primary-cell representatives [Bohr], shared by ionic Hessians.
    std::span<const Vec3> positions_bohr() const { return positions_bohr_; }

    /// @brief Ionic Ewald forces [Hartree/Bohr], one per input charge.
    const std::vector<Vec3>& forces_bohr() const { return forces_bohr_; }
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
