/// @file core/dynamics_math.h
/// @brief Scalar MD laws shared by host and device execution.
#pragma once

#include "atomiverse/core/precision.h"
#include <cmath>

#if defined(__CUDACC__)
#define ATV_MD_HD __host__ __device__
#else
#define ATV_MD_HD
#endif

namespace ATV::md {

ATV_MD_HD inline BigInt degrees_of_freedom(BigInt atoms,
                                           BigInt constraints = 0) {
    const BigInt count = 3 * atoms - 3 - constraints;
    return count > 0 ? count : 0;
}

/// Frame stress stores extensive virial and kinetic contributions [eV].
template <typename T>
ATV_MD_HD inline T isotropic_pressure(const T* virial, const T* kinetic,
                                      T volume) {
    return (virial[0] + virial[1] + virial[2] + kinetic[0] + kinetic[1]
            + kinetic[2])
           / (T(3) * volume);
}

/// Berendsen et al., JCP 81, 3684 (1984), doi:10.1063/1.448118.
/// Bound the volume factor before taking its cube root.
template <typename T>
ATV_MD_HD inline T berendsen_dilation(T pressure, T target, T dt, T damping,
                                      T bulk_modulus) {
    T volume_factor =
        T(1) - dt / (damping * bulk_modulus) * (target - pressure);
    volume_factor = volume_factor < T(0.5) ? T(0.5) : volume_factor;
    volume_factor = volume_factor > T(2) ? T(2) : volume_factor;
    return cbrt(volume_factor);
}

template <typename T>
ATV_MD_HD inline T ramp_target(T start, T stop, BigInt step, BigInt first,
                               BigInt last) {
    T fraction = T(0);
    if (last > first && step > first) {
        if (step >= last)
            fraction = T(1);
        else
            // Unsigned differences avoid signed overflow and retain short
            // intervals even after the absolute clock exceeds 2^53.
            fraction = T(std::uint64_t(step) - std::uint64_t(first))
                       / T(std::uint64_t(last) - std::uint64_t(first));
    }
    return start + fraction * (stop - start);
}

/// Symmetric NHC half-step, shared by the atom and barostat thermostats.
/// Martyna et al., JCP 97, 2635 (1992), doi:10.1063/1.463940.
/// The drag is applied at every backward link, never on the forward sweep.
ATV_MD_HD inline double nhc_half_step(double* position, double* velocity,
                                      const double* mass, int count,
                                      double& kinetic, double target_twice_ke,
                                      double kT, double half_dt, double drag) {
    const double quarter_dt = 0.5 * half_dt;
    const double eighth_dt = 0.5 * quarter_dt;
    for (int k = count - 1; k >= 0; --k) {
        const double factor =
            k + 1 < count ? exp(-velocity[k + 1] * eighth_dt) : 1.0;
        velocity[k] *= factor;
        const double force =
            k == 0 ? (2.0 * kinetic - target_twice_ke) / mass[0]
                   : (mass[k - 1] * velocity[k - 1] * velocity[k - 1] - kT)
                         / mass[k];
        velocity[k] += force * quarter_dt;
        velocity[k] *= drag;
        velocity[k] *= factor;
    }
    const double scale = exp(-velocity[0] * half_dt);
    kinetic *= scale * scale;
    for (int k = 0; k < count; ++k)
        position[k] += velocity[k] * half_dt;
    for (int k = 0; k < count; ++k) {
        const double factor =
            k + 1 < count ? exp(-velocity[k + 1] * eighth_dt) : 1.0;
        velocity[k] *= factor;
        const double force =
            k == 0 ? (2.0 * kinetic - target_twice_ke) / mass[0]
                   : (mass[k - 1] * velocity[k - 1] * velocity[k - 1] - kT)
                         / mass[k];
        velocity[k] += force * quarter_dt;
        velocity[k] *= factor;
    }
    return scale;
}

} // namespace ATV::md

#undef ATV_MD_HD
