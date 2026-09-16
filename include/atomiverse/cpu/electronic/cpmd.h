/// @file cpu/electronic/cpmd.h
/// @brief Cpmd / CpmdCplx — Car–Parrinello propagators for the fictitious
///        electronic degrees of freedom (DOF): velocity-Verlet plus RATTLE
///        constraints, with an optional Nosé–Hoover electronic thermostat.
///
/// **The problem.** Born–Oppenheimer molecular dynamics re-converges the
/// electrons at every nuclear step — a self-consistent-field (SCF) solve, the
/// iteration that makes the electronic Hamiltonian consistent with the density
/// it produces — and that is where nearly all the cost sits. Car–Parrinello
/// molecular dynamics (CPMD) instead gives the electronic wavefunction its own
/// classical dynamics: the coefficients that expand the occupied molecular
/// orbitals (MOs) in the basis are given a *fictitious* mass \(\mu\) and are
/// integrated in time alongside the nuclei, so after one cold (from-scratch)
/// SCF solve no further SCF iteration is needed (Car & Parrinello,
/// Phys. Rev. Lett. 55, 2471 (1985)).
///
/// **Symbols used throughout this header.** \(n_b\) = number of basis
/// functions; \(n_{\mathrm{occ}}\) = number of occupied orbitals propagated;
/// \(C\) = the \(n_b \times n_{\mathrm{occ}}\) block of MO coefficients
/// (dimensionless) and \(\dot C\) its time derivative [a.u. time\(^{-1}\)];
/// \(H\) = the Kohn–Sham (KS) matrix, i.e. the effective one-electron
/// Hamiltonian in the basis [Hartree]; \(S\) = the basis-overlap matrix
/// (dimensionless); \(P\) = the one-particle density matrix; \(T_e\) = the
/// fictitious electronic kinetic energy [Hartree]; \(I\) = the identity.
///
/// **Real-valued Γ (`Cpmd`).** Γ ("Gamma") is the single \(k = 0\) sampling
/// point of a periodic cell, where every matrix may be taken real. \(C\) is
/// \(n_b\times n_{\mathrm{occ}}\) real; \(P = 2 C C^\top\) for restricted
/// Kohn–Sham (RKS — both spins share one set of orbitals), or
/// \(P_\sigma = C_\sigma C_\sigma^\top\) per spin channel \(\sigma\) for
/// unrestricted Kohn–Sham (UKS — spins get independent orbitals). The equation
/// of motion is
/// \(\mu\,\ddot C = -(I - S C C^\top)\,(2\cdot\mathrm{occ}\,H C)\): minus the
/// band-energy gradient, projected onto the directions that keep the
/// orthonormality constraint below satisfied (`linalg::band_gradient`).
///
/// **The constraint, and what RATTLE is.** The propagated orbitals must stay
/// orthonormal in the overlap metric, \(C^\top S C = I\). That is a *holonomic*
/// constraint — a condition on the coordinates themselves, not on their
/// velocities. RATTLE enforces it from inside the integrator in two stages: the
/// position stage solves for the multiplier \(\Lambda\) such that
/// \(C' = C_{\mathrm{raw}} + C\Lambda\) satisfies \(C'^\top S C' = I\)
/// exactly, and the velocity stage removes the part of \(\dot C\) that would
/// drift off the constraint surface, i.e. enforces
/// \(\mathrm{sym}(C^\top S \dot C) = 0\). It is the velocity form of SHAKE
/// (Andersen, J. Comput. Phys. 52, 24 (1983)) applied to this matrix-valued
/// electronic constraint.
///
/// **Complex multi-k (`CpmdCplx`).** In a periodic solid the orbitals are Bloch
/// states labelled by a reciprocal-space wavevector \(k\); sampling more than
/// Γ makes the coefficients complex and yields one independent propagator per
/// sampled \(k\). The constraint becomes \(C_k^\dagger S_k C_k = I\)
/// (\(\dagger\) = conjugate transpose), velocity RATTLE kills
/// \(\mathrm{Herm}(C_k^\dagger S_k \dot C_k)\), and the force is the complex
/// S-tangent gradient (`linalg::band_gradient_k`). Each instance reports its
/// own \(T_e^{(k)}\); the total is the k-weighted sum \(\sum_k w_k T_e^{(k)}\)
/// (quadrature weights \(w_k\) sum to 1), formed by the consumer, not here.
///
/// **Optional Nosé–Hoover electronic thermostat.** One extra friction variable
/// \(\xi\) is coupled to \(\dot C\) and driven so that \(T_e\) is regulated
/// toward a target \(K_0\), instead of letting the fictitious electrons heat up
/// and drag energy out of the nuclei (Blöchl & Parrinello,
/// Phys. Rev. B 45, 9413 (1992); textbook treatment in Marx & Hutter,
/// *Ab initio Molecular Dynamics*, CUP 2009). Both classes implement the same
/// half-steps and honour `thermostat`: real-Γ `Cpmd` is regression-pinned by
/// `regressions/dynamics/cpmd-electronic-thermostat`, and the multi-k GPW
/// consumer thermostats one `CpmdCplx` per (k, σ) channel
/// (`regressions/dynamics/cpmd-thermostat-kpoints`), k-weighting the
/// published \(T_e\) and \(E_{\mathrm{therm}}\) totals.
///
/// Method-agnostic electronic-DOF propagation on `cpu/linalg` matrices: the
/// only inputs are the KS matrix \(H\), the overlap \(S\) and the occupied
/// coefficients, so any mean-field method can drive it. Consumed today by
/// `cpu::gpw::GPWEvaluator` (`cpmd = true`): Γ RKS/UKS via `Cpmd`; multi-k
/// RKS/UKS via a vector of `CpmdCplx`, one per (k, σ) channel.
#pragma once

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Car–Parrinello occupied-orbital propagator with electronic RATTLE
///        and optional Nosé–Hoover electronic thermostat.
///
/// Lifecycle: construct with \(\mu > 0\), optionally enable the thermostat
/// via public members, `seed` with S-orthonormal occupied coefficients
/// (typically from a cold SCF), then once per MD force step call
/// `advance(H, S, dt_fs)` after the KS matrix at the current density has
/// been built. Electronic kinetic energy uses the **velocity-RATTLE**
/// \(\dot C\) (after NH rescaling when thermostatted).
class Cpmd {
  public:
    /// @brief Construct with the fictitious electronic mass.
    /// @param mu_au Fictitious mass \(\mu\) in atomic units (electron mass
    ///        units). Typical CPMD values are \(10^2\)–\(10^3\). Default 400.
    /// @throws std::invalid_argument when mu_au ≤ 0 or non-finite.
    explicit Cpmd(double mu_au = 400.0);

    /// @brief Fictitious electronic mass \(\mu\).
    /// @return \(\mu\) [a.u., in units of the electron mass]; always > 0.
    double mu() const { return mu_; }

    /// @brief Whether occupied coefficients have been seeded.
    /// @return true once `seed` has run and `advance` may be called.
    bool ready() const { return ready_; }

    /// @brief Enable Nosé–Hoover control of fictitious \(T_e\) (default false).
    bool thermostat = false;
    /// @brief Target electronic kinetic energy \(K_0\) [Hartree]. Must be
    ///        finite and > 0 when `thermostat` is true.
    double ekin_target_ha = 1e-5;
    /// @brief Electronic thermostat coupling time \(\tau_e\) [fs]. Must be
    ///        finite and > 0 when `thermostat` is true.
    double tau_e_fs = 50.0;

    /// @brief Current occupied MO coefficients \(C\) (\(n_b \times n_{\mathrm{occ}}\),
    ///        dimensionless).
    /// @throws std::logic_error before `seed`.
    const linalg::RealMatrix& C() const;

    /// @brief Current electronic velocity \(\dot C\) (\(n_b \times n_{\mathrm{occ}}\),
    ///        [a.u. time\(^{-1}\)]).
    /// @throws std::logic_error before `seed`.
    const linalg::RealMatrix& Cdot() const;

    /// @brief Electronic kinetic energy \(T_e = \frac12\mu\sum_i\lVert\dot c_i\rVert^2\),
    ///        the sum running over the \(n_{\mathrm{occ}}\) columns \(\dot c_i\)
    ///        of \(\dot C\) (a Frobenius norm).
    /// @return \(T_e\) [Hartree] from the last `advance`; 0 after a rest seed
    ///         and before any step. Uses the post-RATTLE (and, when
    ///         thermostatted, post-rescaling) \(\dot C\).
    double kinetic_ha() const { return ke_ha_; }

    /// @brief Nosé–Hoover thermostat energy
    ///        \(E_{\mathrm{therm}} = \frac12 Q\xi^2 + 2K_0\eta\), with
    ///        \(Q = 2K_0\tau_e^2\) the thermostat mass (\(\tau_e\) converted
    ///        from fs to a.u. time) — the bookkeeping term that keeps the total
    ///        energy conserved while the thermostat is acting.
    /// @return \(E_{\mathrm{therm}}\) [Hartree]; exactly 0 when `thermostat`
    ///         is false.
    double thermostat_energy_ha() const { return e_therm_ha_; }

    /// @brief Nosé–Hoover friction variable \(\xi\) (positive = damping
    ///        \(\dot C\), negative = pumping it).
    /// @return \(\xi\) [a.u. time\(^{-1}\)]; 0 when the thermostat is off.
    double xi() const { return xi_; }

    /// @brief Nosé–Hoover thermostat coordinate \(\eta = \int\xi\,dt\).
    /// @return \(\eta\) (dimensionless log-scale factor); 0 when the
    ///         thermostat is off.
    double eta() const { return eta_; }

    /// @brief Position-RATTLE iterations used by the last `advance`.
    /// @return Iteration count (dimensionless); 0 at rest or after `seed`.
    int last_position_iters() const { return last_pos_iters_; }

    /// @brief Largest absolute entry of the constraint residual
    ///        \(C^\top S C - I\) (an entrywise max-norm, **not** the induced
    ///        \(\infty\)-norm) after the last position stage.
    /// @return Residual (dimensionless); 0 after `seed`. `advance` throws when
    ///         it would exceed 1e-8.
    double last_constraint_residual() const { return last_sigma_inf_; }

    /// @brief Seed the propagator at rest: \(C = C_{\mathrm{occ}}\), \(\dot C = 0\),
    ///        \(T_e = 0\). Resets the Nosé–Hoover state (\(\xi = \eta = 0\)).
    /// @param C_occ Occupied coefficients [n_b × n_occ] (dimensionless),
    ///        S-orthonormal at the seed geometry (\(C^\top S C = I\)) —
    ///        typically the converged coefficients of a cold SCF solve.
    /// @throws std::invalid_argument when `C_occ` is empty.
    void seed(const linalg::RealMatrix& C_occ);

    /// @brief Seed with a prescribed electronic velocity (restart / tests).
    ///        Resets the Nosé–Hoover state and immediately reports \(T_e\)
    ///        (and \(E_{\mathrm{therm}}\)) for the supplied \(\dot C\).
    /// @param C_occ Occupied coefficients [n_b × n_occ] (dimensionless).
    /// @param Cdot_occ Electronic velocity [a.u. time\(^{-1}\)], same shape as
    ///        `C_occ`.
    /// @throws std::invalid_argument on empty / shape mismatch.
    void seed(const linalg::RealMatrix& C_occ, const linalg::RealMatrix& Cdot_occ);

    /// @brief Re-orthonormalize \(C\) under a new metric \(S\) (nuclei moved)
    ///        and project \(\dot C\) onto the velocity constraint manifold.
    ///
    /// A geometry change is not a dynamics step: \(C\) is Löwdin-transported
    /// (\(C \leftarrow C\,(C^\top S C)^{-1/2}\)) and \(\dot C\) is
    /// velocity-RATTLE projected against the new \(S\), without advancing time.
    /// \(T_e\) is recomputed from the projected \(\dot C\).
    /// @param S Overlap at the current geometry, \(n_b \times n_b\)
    ///        (dimensionless).
    /// @throws std::logic_error before `seed`.
    /// @throws std::invalid_argument when `S` does not match \(n_b\).
    /// @note A singular overlap metric is **not** diagnosed here: the real
    ///       `linalg::lowdin_orthonormalize` propagates non-finite entries
    ///       instead of throwing (its complex twin does throw).
    void align_to_overlap(const linalg::RealMatrix& S);

    /// @brief One electronic velocity-Verlet step with RATTLE (pos + vel),
    ///        optionally wrapped by Nosé–Hoover half-steps.
    ///
    /// Uses the S-tangent force
    /// \(F = -(I-SC C^\top)(2\cdot\mathrm{occ}\,H C)\) [Hartree] and
    /// \(\Delta t\) converted from fs to a.u. time. Closed-shell RKS uses
    /// default `occ = 2` (factor 4); each UKS spin channel uses `occ = 1`.
    /// Exactly one \(H\) evaluation per step: both velocity half-kicks use the
    /// same \(F\), and the half-step velocity is redefined from the
    /// constraint-corrected displacement \((C' - C)/\Delta t\) (Andersen).
    ///
    /// @param H KS matrix at the current density [Hartree], \(n_b \times n_b\).
    /// @param S Overlap matrix at the current geometry, \(n_b \times n_b\)
    ///        (dimensionless).
    /// @param dt_fs Timestep [fs] — must match the nuclear integrator dt.
    /// @param occ Orbital occupation (dimensionless; 2 closed-shell RKS,
    ///        1 per UKS spin channel).
    /// @throws std::logic_error before `seed`.
    /// @throws std::invalid_argument on non-positive dt/occ/thermostat params
    ///         or shape mismatch.
    /// @throws std::runtime_error when the position-RATTLE residual stays above
    ///         1e-8, or when the Nosé–Hoover half-step produces a non-finite
    ///         velocity-rescaling factor.
    void advance(const linalg::RealMatrix& H, const linalg::RealMatrix& S,
                 double dt_fs, double occ = 2.0);

    /// @brief Drop electronic history (geometry discontinuity / restart):
    ///        clears \(C\), \(\dot C\), the energies and the Nosé–Hoover state,
    ///        so `ready()` is false again and a fresh `seed` is required.
    void reset();

  private:
    /// Validate thermostat members when engaged; throw if invalid.
    void check_thermostat_params() const;
    /// One NH half-step of length `dt_half_au` [a.u. time] using current Te.
    void nh_half_step(double dt_half_au);
    /// Recompute \(T_e\) and \(E_{\mathrm{therm}}\) from state.
    void update_energies();

    double mu_ = 400.0;          ///< fictitious mass [a.u.].
    linalg::RealMatrix C_;       ///< occupied coefficients
    linalg::RealMatrix Cdot_;    ///< electronic velocity
    double ke_ha_ = 0.0;         ///< last T_e [Hartree]
    double e_therm_ha_ = 0.0;    ///< last E_therm [Hartree]
    double xi_ = 0.0;            ///< NH friction
    double eta_ = 0.0;           ///< NH coordinate
    int last_pos_iters_ = 0;     ///< last position-RATTLE iteration count
    double last_sigma_inf_ = 0.0;///< last ‖σ‖_∞
    bool ready_ = false;
};

/// @brief Car–Parrinello propagator on complex Bloch occupied coefficients
///        (one instance per sampled k-point).
///
/// Hermitian generalization of `Cpmd` for multi-k RKS: the constraint is
/// \(C^\dagger S C = I\) with \(\dagger\) the conjugate transpose, and velocity
/// RATTLE removes \(\mathrm{Herm}(C^\dagger S \dot C)\) instead of the real
/// symmetric part. Lifecycle mirrors `Cpmd`: construct with \(\mu > 0\), `seed`
/// after a cold (from-scratch) multi-k SCF solve, then call
/// `advance(H_k, S_k, dt_fs)` once per MD force step, after the Kohn–Sham matrix
/// at this k-point has been built at the current density.
///
/// The thermostat members are honoured exactly as in `Cpmd` (same Nosé–Hoover
/// half-steps); with real-valued inputs the complex propagator reproduces
/// `Cpmd` exactly (Γ-limit solver identity, unit-pinned). The GPW consumer
/// engages one thermostatted instance per (k, σ) channel
/// (`regressions/dynamics/cpmd-thermostat-kpoints`).
class CpmdCplx {
  public:
    /// @brief Construct with the fictitious electronic mass.
    /// @param mu_au Fictitious mass \(\mu\) [a.u., in units of the electron
    ///        mass]. Typical CPMD values are \(10^2\)–\(10^3\). Default 400.
    /// @throws std::invalid_argument when mu_au ≤ 0 or non-finite.
    explicit CpmdCplx(double mu_au = 400.0);

    /// @brief Fictitious electronic mass \(\mu\).
    /// @return \(\mu\) [a.u., electron-mass units]; always > 0.
    double mu() const { return mu_; }

    /// @brief Whether occupied Bloch coefficients have been seeded.
    /// @return true once `seed` has run and `advance` may be called.
    bool ready() const { return ready_; }

    /// @brief Enable Nosé–Hoover control of the fictitious \(T_e\) at this
    ///        (k, σ) channel (default false; see the class note above).
    bool thermostat = false;
    /// @brief Target electronic kinetic energy \(K_0\) [Hartree]. Must be
    ///        finite and > 0 when `thermostat` is true.
    double ekin_target_ha = 1e-5;
    /// @brief Electronic thermostat coupling time \(\tau_e\) [fs]. Must be
    ///        finite and > 0 when `thermostat` is true.
    double tau_e_fs = 50.0;

    /// @brief Occupied Bloch coefficients \(C_k\) (\(n_b \times n_{\mathrm{occ}}\),
    ///        dimensionless).
    /// @throws std::logic_error before `seed`.
    const linalg::CplxMatrix& C() const;

    /// @brief Electronic velocity \(\dot C_k\) (\(n_b \times n_{\mathrm{occ}}\),
    ///        [a.u. time\(^{-1}\)]).
    /// @throws std::logic_error before `seed`.
    const linalg::CplxMatrix& Cdot() const;

    /// @brief Electronic kinetic energy of **this** k-point,
    ///        \(T_e^{(k)} = \frac12\mu\sum_i\lVert\dot c_i\rVert^2\) (Frobenius
    ///        norm over the complex columns of \(\dot C\)).
    /// @return \(T_e^{(k)}\) [Hartree] from the last `advance`; 0 after a rest
    ///         seed. The consumer forms the total as \(\sum_k w_k T_e^{(k)}\).
    double kinetic_ha() const { return ke_ha_; }

    /// @brief Nosé–Hoover thermostat energy
    ///        \(E_{\mathrm{therm}} = \frac12 Q\xi^2 + 2K_0\eta\),
    ///        \(Q = 2K_0\tau_e^2\) (\(\tau_e\) in a.u. time).
    /// @return \(E_{\mathrm{therm}}\) [Hartree]; exactly 0 when `thermostat`
    ///         is false.
    double thermostat_energy_ha() const { return e_therm_ha_; }

    /// @brief Nosé–Hoover friction variable \(\xi\).
    /// @return \(\xi\) [a.u. time\(^{-1}\)]; 0 when the thermostat is off.
    double xi() const { return xi_; }

    /// @brief Nosé–Hoover thermostat coordinate \(\eta = \int\xi\,dt\).
    /// @return \(\eta\) (dimensionless); 0 when the thermostat is off.
    double eta() const { return eta_; }

    /// @brief Position-RATTLE iterations used by the last `advance`.
    /// @return Iteration count (dimensionless); 0 at rest or after `seed`.
    int last_position_iters() const { return last_pos_iters_; }

    /// @brief Largest absolute entry of \(C^\dagger S C - I\) (entrywise
    ///        max-norm, **not** the induced \(\infty\)-norm) after the last
    ///        position stage.
    /// @return Residual (dimensionless); 0 after `seed`. `advance` throws when
    ///         it would exceed 1e-8.
    double last_constraint_residual() const { return last_sigma_inf_; }

    /// @brief Seed at rest: \(C = C_{\mathrm{occ}}\), \(\dot C = 0\),
    ///        \(T_e = 0\); resets the Nosé–Hoover state.
    /// @param C_occ Occupied Bloch block [nb × n_occ] (dimensionless),
    ///        S-orthonormal at the seed geometry (\(C^\dagger S C = I\)).
    /// @throws std::invalid_argument when empty.
    void seed(const linalg::CplxMatrix& C_occ);

    /// @brief Seed with a prescribed electronic velocity (restart / tests);
    ///        reports \(T_e\) for the supplied \(\dot C\) immediately.
    /// @param C_occ Occupied Bloch block [nb × n_occ] (dimensionless).
    /// @param Cdot_occ Electronic velocity [a.u. time\(^{-1}\)], same shape as
    ///        `C_occ`.
    /// @throws std::invalid_argument on empty / shape mismatch.
    void seed(const linalg::CplxMatrix& C_occ, const linalg::CplxMatrix& Cdot_occ);

    /// @brief Löwdin-transport \(C\) under a new \(S_k\) and velocity-project
    ///        \(\dot C\) — the geometry-change (not time-step) update.
    ///
    /// \(C \leftarrow C\,(C^\dagger S C)^{-1/2}\), then velocity RATTLE against
    /// the new metric; \(T_e\) is recomputed from the projected \(\dot C\).
    /// @param S Hermitian overlap at the current geometry, nb × nb
    ///        (dimensionless).
    /// @throws std::logic_error before `seed`.
    /// @throws std::invalid_argument when `S` does not match \(n_b\).
    /// @throws std::runtime_error on a non-positive overlap-metric eigenvalue
    ///         (from `linalg::lowdin_orthonormalize_k`).
    void align_to_overlap(const linalg::CplxMatrix& S);

    /// @brief One electronic velocity-Verlet (VV) + RATTLE step on complex
    ///        Hermitian \(H, S\), optionally wrapped by Nosé–Hoover half-steps.
    ///
    /// Same scheme as `Cpmd::advance` with the Hermitian force
    /// \(F = -(I - S C C^\dagger)(2\cdot\mathrm{occ}\,H C)\) [Hartree]
    /// (`linalg::band_gradient_k`) and a single \(H\) evaluation per step.
    /// @param H KS matrix at this \(k\) [Hartree], nb × nb Hermitian.
    /// @param S Overlap at this \(k\), nb × nb Hermitian (dimensionless).
    /// @param dt_fs Timestep [fs] — must match the nuclear VV dt.
    /// @param occ Orbital occupation (dimensionless; 2 for closed-shell RKS).
    /// @throws std::logic_error before `seed`.
    /// @throws std::invalid_argument on non-positive dt/occ/thermostat params
    ///         or shape mismatch.
    /// @throws std::runtime_error when the position-RATTLE residual stays above
    ///         1e-8, or when the Nosé–Hoover half-step produces a non-finite
    ///         velocity-rescaling factor.
    void advance(const linalg::CplxMatrix& H, const linalg::CplxMatrix& S,
                 double dt_fs, double occ = 2.0);

    /// @brief Drop electronic history (geometry discontinuity / restart):
    ///        `ready()` becomes false and a fresh `seed` is required.
    void reset();

  private:
    void check_thermostat_params() const;
    void nh_half_step(double dt_half_au);
    void update_energies();

    double mu_ = 400.0;
    linalg::CplxMatrix C_;
    linalg::CplxMatrix Cdot_;
    double ke_ha_ = 0.0;
    double e_therm_ha_ = 0.0;
    double xi_ = 0.0;
    double eta_ = 0.0;
    int last_pos_iters_ = 0;
    double last_sigma_inf_ = 0.0;
    bool ready_ = false;
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
