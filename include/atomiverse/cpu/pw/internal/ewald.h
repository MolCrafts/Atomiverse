/// @file cpu/pw/internal/ewald.h
/// @brief 3D Ewald summation for periodic point charges (energy + ionic forces).
#pragma once

#include "atomiverse/core/box.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Ewald ion–ion energy and forces for classical point charges.
///
/// Energy matches the historical PW path (`e_real + e_recip + e_self` in
/// Hartree). Forces are the negative analytic gradient of that energy
/// (self energy is position-independent; the separate charged-system G=0
/// jellium term added in `setup_pw_problem` is also R-independent).
class EwaldSummation {
    AccReal e_real_ = 0, e_recip_ = 0, e_self_ = 0;
    AccReal alpha_ = AccReal(1);
    std::vector<Vec3> forces_bohr_; ///< F = −∂E/∂R [Hartree/Bohr]

  public:
    /// @param pos_ang Positions [Å].
    /// @param charges Point charges (nuclear Z or valence, same as energy path).
    /// @param cell Simulation cell [Å].
    /// @param ecut_hartree Wavefunction cutoff [Hartree] (sets recip G cut).
    EwaldSummation(const std::vector<Vec3>& pos_ang,
                   const std::vector<AccReal>& charges, const ::ATV::Box& cell,
                   AccReal ecut_hartree);

    AccReal total_energy() const { return e_real_ + e_recip_ + e_self_; }
    AccReal real_energy() const { return e_real_; }
    AccReal recip_energy() const { return e_recip_; }
    AccReal self_energy() const { return e_self_; }
    AccReal alpha() const { return alpha_; }

    /// @brief Ionic Ewald forces [Hartree/Bohr], one per input charge.
    const std::vector<Vec3>& forces_bohr() const { return forces_bohr_; }
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
