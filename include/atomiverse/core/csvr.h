/// @file core/csvr.h
/// @brief CSVR (Bussi) scalar laws and per-step noise, shared by
///        `cpu::md::FixCSVR` and `cuda::md::FixCSVR`.
///
/// Bussi, Donadio, Parrinello, "Canonical sampling through velocity
/// rescaling", J. Chem. Phys. **126**, 014101 (2007), Eq. (A7); the noise
/// sum and the group/bias semantics follow LAMMPS
/// `src/EXTRA-FIX/fix_temp_csvr.cpp` (`end_of_step`, `resamplekin`,
/// `sumnoises`) and `src/compute_temp_com.cpp` (`dof_compute`,
/// `remove_bias_all`). Port reference: e8c4c278a8.
/// Copyright (2003) Sandia Corporation. Under Contract DE-AC04-94AL85000,
/// the U.S. Government retains certain rights. GPL-2.0-only.
///
/// Both backends draw the two per-step randoms (R₁ and the χ²(N_f − 1) sum)
/// from the same counter-keyed stream: a splitmix64 sequence seeded from
/// `(seed, step)` only, Box–Muller normals and a Marsaglia–Tsang gamma
/// deviate. The stream is a pure function of `(seed, step, N_f)`, so a
/// trajectory is invariant to MPI rank count and identical (to libm
/// rounding of log/cos/exp) between the CPU and CUDA thermostats.
#pragma once

#include "atomiverse/core/dynamics_math.h"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define ATV_CSVR_HD __host__ __device__
#else
#define ATV_CSVR_HD
#endif

namespace ATV::md {

/// @brief Velocity bias removed before a group temperature is measured and
///        restored after rescaling (LAMMPS `fix_modify temp <compute>`).
enum class TemperatureBias : std::uint8_t {
    None = 0, ///< LAMMPS `compute temp`: group velocities as they are.
    Com = 1,  ///< LAMMPS `compute temp/com`: velocities relative to the
              ///< group's centre-of-mass velocity.
};

/// @brief Counter-keyed random stream of one CSVR step.
struct CsvrStream {
    std::uint64_t state;

    /// @param seed Thermostat seed.
    /// @param step MD step index (global; never a rank-local counter).
    ATV_CSVR_HD CsvrStream(std::uint32_t seed, std::int64_t step) {
        std::uint64_t s =
            static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ULL
            + static_cast<std::uint64_t>(step) * 0xBF58476D1CE4E5B9ULL;
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
        state = s ^ (s >> 31);
    }

    /// @brief splitmix64 (Steele, Lea, Flood, OOPSLA 2014).
    ATV_CSVR_HD std::uint64_t next_u64() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /// @return Uniform deviate in the open interval (0, 1).
    ATV_CSVR_HD double uniform() {
        return (static_cast<double>(next_u64() >> 11) + 0.5)
               * 1.1102230246251565e-16; // 2^-53
    }

    /// @return Standard normal deviate (Box–Muller, one per two uniforms).
    ATV_CSVR_HD double gaussian() {
        const double u1 = uniform();
        const double u2 = uniform();
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }

    /// @brief Gamma(shape a, scale 1) for a >= 1 (Marsaglia & Tsang, ACM TOMS
    ///        26, 363 (2000)).
    ATV_CSVR_HD double gamma(double a) {
        const double d = a - 1.0 / 3.0;
        const double c = 1.0 / sqrt(9.0 * d);
        for (;;) {
            double x = 0.0;
            double v = 0.0;
            do {
                x = gaussian();
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            const double u = uniform();
            const double x2 = x * x;
            if (u < 1.0 - 0.0331 * x2 * x2)
                return d * v;
            if (log(u) < 0.5 * x2 + d * (1.0 - v + log(v)))
                return d * v;
        }
    }

    /// @brief Sum of the squares of `n` independent standard normals, drawn
    ///        as LAMMPS `FixTempCSVR::sumnoises` does (χ²(n) = 2 Γ(n/2)).
    ATV_CSVR_HD double sum_noises(std::int64_t n) {
        if (n <= 0)
            return 0.0;
        if (n == 1) {
            const double r = gaussian();
            return r * r;
        }
        if (n % 2 == 0)
            return 2.0 * gamma(0.5 * static_cast<double>(n));
        const double r = gaussian();
        return 2.0 * gamma(0.5 * static_cast<double>(n - 1)) + r * r;
    }
};

/// @brief The two randoms of one CSVR step.
struct CsvrNoise {
    double r1;     ///< R₁ ~ N(0, 1).
    double sum_sq; ///< Σ_{i=2}^{N_f} R_i² ~ χ²(N_f − 1).
};

/// @brief Draw the step's randoms (LAMMPS `resamplekin`: R₁ first, then
///        `sumnoises(N_f − 1)`).
ATV_CSVR_HD inline CsvrNoise csvr_noise(std::uint32_t seed, std::int64_t step,
                                        std::int64_t n_dof) {
    CsvrStream s(seed, step);
    CsvrNoise out{};
    out.r1 = s.gaussian();
    out.sum_sq = s.sum_noises(n_dof - 1);
    return out;
}

/// @brief Squared velocity scale α² of one CSVR step (Bussi Eq. A7, LAMMPS
///        `resamplekin`).
/// @param ke        Current (bias-removed) kinetic energy of the group.
/// @param ke_target `½ N_f k_B T_target`, same unit as `ke`.
/// @param n_dof     Group degrees of freedom N_f (> 0).
/// @param dt        Time step.
/// @param tau       Coupling time, same unit as `dt`.
/// @param noise     The step's randoms.
/// @return α² ≥ 0.
ATV_CSVR_HD inline double csvr_alpha2(double ke, double ke_target, double n_dof,
                                      double dt, double tau,
                                      const CsvrNoise& noise) {
    const double c1 = exp(-dt / tau);
    const double c2 = (1.0 - c1) * ke_target / ke / n_dof;
    const double a2 = c1 + c2 * (noise.r1 * noise.r1 + noise.sum_sq)
                      + 2.0 * noise.r1 * sqrt(c1 * c2);
    return a2 > 0.0 ? a2 : 0.0;
}

/// @brief Degrees of freedom of a thermostat group (LAMMPS
///        `ComputeTemp(COM)::dof_compute`): `3 N_g − 3 − constraints`, the 3
///        being `extra_dof` for both `temp` and `temp/com`.
ATV_CSVR_HD inline BigInt csvr_group_dof(BigInt atoms, BigInt constraints) {
    return degrees_of_freedom(atoms, constraints);
}

/// @brief Per-group sums one CSVR step reduces (mass-weighted, velocity in
///        the run's units; multiply by `mvv2e` for energy).
struct CsvrSums {
    double mass = 0.0;    ///< Σ m over the group.
    double p[3] = {};     ///< Σ m v over the group.
    double mvv = 0.0;     ///< Σ m |v|² over the group.
    double mvv_all = 0.0; ///< Σ m |v|² over every atom.
};

/// @brief Twice the bias-removed group kinetic energy, Σ m |v − v_b|²
///        (`v_b` the COM velocity under `TemperatureBias::Com`, else 0).
ATV_CSVR_HD inline double csvr_group_mvv(const CsvrSums& s,
                                         TemperatureBias bias) {
    if (bias == TemperatureBias::None || !(s.mass > 0.0))
        return s.mvv;
    const double p2 = s.p[0] * s.p[0] + s.p[1] * s.p[1] + s.p[2] * s.p[2];
    return s.mvv - p2 / s.mass;
}

} // namespace ATV::md

#undef ATV_CSVR_HD
