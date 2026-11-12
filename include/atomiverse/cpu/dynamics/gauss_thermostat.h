/// @file cpu/dynamics/gauss_thermostat.h
/// @brief Thermostat settings of `cpu::md::GearPC`: Gaussian (isokinetic,
///        flow, transverse) and configurational Nosé–Hoover.
#pragma once

namespace ATV::cpu::md {

/// @brief Which kinetic temperature a Gaussian thermostat holds constant
///        (J. Petravic and J. Delhommelle, J. Chem. Phys. 118, 7477 (2003)).
enum class GaussKind {
    /// Barycentric kinetic temperature, eq. (9): every axis relative to the
    /// centre-of-mass velocity.
    Kinetic,
    /// Flow kinetic temperature, eqs. (3), (4), (10): along `axis` relative
    /// to each species' mean velocity (the species flows are not
    /// thermostatted), barycentric on the other two axes.
    Flow,
    /// "yz" temperature, eq. (11): `axis` is not thermostatted, the other two
    /// axes are barycentric.
    Transverse,
};

/// @brief Gauss least-constraint kinetic thermostat inside `GearPC`.
///
/// The constrained kinetic energy is held at
/// \(K_0 = \tfrac12 N_\mathrm{dof} k_B T\), \(N_\mathrm{dof}\) the constrained
/// degrees of freedom (`Kinetic` \(3N-3\), `Flow` \(3N-2-n_\nu\),
/// `Transverse` \(2N-2\)). The paper normalises its temperatures by \(3N\)
/// (eqs. 2, 4) instead; the two differ by \(N_\mathrm{dof}/3N\).
struct GaussThermostat {
    GaussKind kind = GaussKind::Kinetic;
    double temperature = 0.0; ///< Target T [K] (> 0).
    int axis = 0;             ///< Field axis 0..2 (`Flow`, `Transverse`).
};

/// @brief Configurational Nosé–Hoover thermostat inside `GearPC`,
///        Petravic & Delhommelle eqs. (12a–c):
///
/// \[
///   \dot r_i = v_i + \frac{\xi}{T_0}\frac{\partial T_{C1}}{\partial r_i},
///   \qquad
///   \dot\xi = -Q\,\frac{T_{C1} - T_0}{T_0},
/// \]
/// with the first-order configurational temperature
/// \(k_B T_{C1} = \sum_i F_i^2 / \sum_i \nabla_i^2 U\) (eq. 7) of the
/// interaction potential \(U\) (external drives such as `FixEfield`
/// excluded). With time in the run's unit (ps for `metal`), \(\xi\) has
/// units Å²/time and `Q` Å²/time².
struct ConfigurationalThermostat {
    double temperature = 0.0; ///< Target \(T_0\) [K] (> 0).
    double Q = 0.0;           ///< Feedback constant [Å²/time²] (> 0).
};

} // namespace ATV::cpu::md
