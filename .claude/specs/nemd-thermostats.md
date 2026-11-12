# NEMD thermostats for strong-field molten salts (CPU)

Goal: the engine capability needed to run the thermostatted NEMD of Petravic &
Delhommelle, J. Chem. Phys. 118, 7477 (2003), §II: Tosi–Fumi NaCl, Ewald
(conducting boundary), a constant field along one axis, and four thermostats
integrated by a fifth-order Gear predictor–corrector. The reproduction itself is
out of scope; this spec adds engine capability only. CPU only: the
md-unification spec unifies the execution model, it does not require a CUDA
twin for every component (CSVR, NPH, PR-NPT are CPU only today). CUDA rows stay
○ in the roadmap.

## Components

1. **`cpu::md::GearPC`** — six-value (derivatives through the fifth) Gear
   predictor–corrector on the first-order system (r, v), Gear 1971 corrector
   coefficients 95/288, 1, 25/24, 35/72, 5/48, 1/120. Integrator, replaces
   `VelocityVerlet`; owns its thermostat like `NvtSllod`. Nordsieck history is
   per-atom component state keyed by tag (follows spatial sorting); spatial
   decomposition and holonomic constraints are refused. Unthermostatted form is
   eq. (8) with α ≡ 0 (adiabatic).
2. **Gaussian thermostats** (`GaussThermostat{kind, temperature, axis}`) inside
   `GearPC`: v̇ = F/m − α w with Gauss's least-constraint direction w (the
   peculiar velocity of the constrained components) and α = Σ m w·a / Σ m w².
   - `Kinetic`, eq. (9): w = v − V_cm on all three axes (barycentric).
   - `Flow`, eqs. (3), (4), (10): along `axis` w = v − u_ν, u_ν the mass-weighted
     mean of the atom's species (`kAtom_type`); barycentric on the other axes.
     The species flows are not thermostatted.
   - `Transverse`, eq. (11): `axis` excluded, barycentric on the other two.
   α is evaluated at the predicted state with the total force (a field term
   enters through `kAtom_f`). After each corrector step the constrained kinetic
   energy is restored by scaling w (nearest point on the constraint sphere in
   the mass metric); the removed relative drift is published as a diagnostic.
3. **Configurational Nosé–Hoover thermostat** (`ConfigurationalThermostat{
   temperature, Q}`) inside `GearPC`, eqs. (12a–c) with T_C1 of eq. (7):
   ṙ = v + (ξ/T0)∂T_C1/∂r, ξ̇ = −Q(T_C1 − T0)/T0, integrated by the same Gear PC.
   ∂T_C1/∂r_i = −[2 (H F)_i D + N ∇_i D]/(k_B D²), N = Σ F², D = Σ ∇²U.
4. **Curvature protocol** (`cpu/component/curvature_protocol.h`):
   `CurvatureSource` (H·v, per-atom Laplacian, gradient of the total
   Laplacian), `ExternalDrive` (a force term excluded from the interaction
   potential, e.g. `FixEfield`), `CurvatureConsumer`. The Driver hands every
   force term to the consumer before build and refuses a force term that is
   neither. Implemented by `PairBorn`, `CoulLong` (erfc real space, incl.
   special-bond Coulomb remainders), `KSpaceEwald` (reciprocal Ewald; slab
   refused), `FixEfield` (drive). `KSpacePME` / `KSpacePPPM` are refused.
   `ConfigurationalTemperature` assembles N, D, H·F and ∇T_C1 with forward
   (F) and reverse (accumulators) ghost communication.
5. **`PairBorn`**: `set_energy_shift(bool)` (default on, unchanged behaviour)
   and `set_tail_correction(bool)` (homogeneous-tail energy and pressure, the
   `PairLJCut` pattern). Wire mirror `PairBornParams` carries both.
6. **Python**: `KSpaceEwald` (unmirrored, like `KSpacePPPM`), `GearPC`,
   `GaussThermostat`, `ConfigurationalThermostat`, `PairBorn(energy_shift=)`,
   `PairBorn.set_tail_correction`.
7. `pair_born.h` type-key comment: the binding fills `kAtom_type` from
   `atoms.type_id`, not the atomic number.

## Gates (unit tests, `tests/`)

- Gear core law: global error ratio under dt halving (exact start).
- Gauss: multiplier law equals eqs. (9)/(10)/(11) incl. field term; each
  constrained temperature (computed independently) conserved to round-off
  over many steps under a strong field; per-step drift removed by the
  projection stays at truncation level (bites on an inconsistent α); the
  unconstrained motion (species flow, transverse axis, COM) is untouched.
- Curvature: H·F, per-atom Laplacian and Laplacian gradient of each source
  against central differences of its own forces / Laplacian; Coulomb sum rule
  Σ∇²(U_real + U_recip) = −4π k_e Σq²/V; ∂T_C1/∂r against differences of T_C1.
- Configurational thermostat drives T_C1 toward T0.
- Born tail energy/pressure against a numerical integral of the potential.

## Open

- Standing end-to-end gate for T_conf (eq. 5/7) = T_kin in an equilibrium
  NaCl melt: statistical, minutes of MD, so not a unit test. Needs an `md`
  method in the scientific benchmark framework (analytic reference policy).
- CUDA twins of everything above.
- PME/PPPM curvature (B-spline second derivatives / dipolar mesh pass).
- Gear start-up uses zero third-and-higher derivatives (O(dt³) one-time
  start-up error), as is customary; a self-starting procedure is not provided.
