# semiempirical/am1

One case; every section below is run by `atv_regressions --case semiempirical/am1` and keeps its own oracle record.


## Reference — AM1 water absolute energy / ΔHf vs OpenMOPAC

```text
oracle: mopac
oracle_version: MOPAC 7.00 (public domain), archived transcript
```

First **absolute OpenMOPAC pin** on the Atomiverse NDDO surface. Everything
that shipped before this gated AM1/RM1 against *itself* (analytic force vs its
own re-SCF finite difference, ΔHf inside a ±8 kcal/mol window, RM1-vs-AM1
discrimination) — nothing would have caught a systematic parameter or integral
defect that shifted the absolute energy.

### Method

AM1 (Austin Model 1), NDDO/MNDO functional form with Gaussian core-core
corrections — Dewar, Zoebisch, Healy & Stewart, *J. Am. Chem. Soc.* **107**,
3902–3909 (1985). Closed-shell RHF, `cpu::semiempirical::AM1`.

Oracle family lock: NDDO → **OpenMOPAC only**. DFTB+ is a forbidden pin source
for this row (`docs/developer/tb-method-loop.md`).

### Offline oracle run

The historical pin is a *published* MOPAC transcript, transcribed verbatim to
`tools/tb_oracle/inputs/mopac/am1-water/mopac_am1_water.out.txt` with the
reconstructed deck beside it.

Source: MOPAC 7.00 (public domain) run of `AM1 PRECISE EF T=48H` on water,
published at <https://www.cup.uni-muenchen.de/ch/compchem/energy/MOPAC_output.html>
(H. Zipse, LMU München computational-chemistry course archive, page dated
2004-11-29). The transcript prints its own parameter provenance —
`H: (AM1): M.J.S. DEWAR ET AL, J. AM. CHEM. SOC. 107 3902-3909 (1985)`, ditto
O — so the numbers and the citation are the same artifact.

### Geometry — and why *this* geometry

**A published AM1 ΔHf belongs to the AM1-optimized structure.** It cannot be
pinned at an arbitrary geometry, and the SE-only build has
`ATV_BUILD_MINIMIZER=OFF`, so the case cannot re-optimize either. The
convention chosen here is: **pin at the optimized internal coordinates printed
by the same run that printed the energy** — geometry and energy from one
self-consistent transcript, no cross-source splicing.

C2v, O at origin, H1 on +x, H2 in the xy plane (MOPAC's own frame):

| Structure | r(O–H) [Å] | ∠HOH [°] | r(H–H) [Å] | Role |
| --------- | ---------- | -------- | ---------- | ---- |
| **AM1 optimum** | 0.961267 | 103.53103 | 1.510121 | main pin |
| input | 1.000000 | 105.00000 | 1.586707 | off-stationary pin |

Cartesian coordinates of the optimum, as the case constructs them:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.0000000 | 0.0000000 | 0 |
| H | 0.9612670 | 0.0000000 | 0 |
| H | −0.2249095 | 0.9345854 | 0 |

(MOPAC printed −0.2249 / 0.9346 to 4 decimals — same structure.)
Cubic 30 Å box, isolation only. Charge 0, singlet.

Two consequences make the optimized-geometry convention a **sharper** gate, not
a weaker one:

1. **Stationarity is itself testable.** MOPAC stopped at
   `RMS GRADIENT = 0.00141` kcal/mol/Å ≈ 6×10⁻⁵ eV/Å. If ATV's AM1 is the same
   functional, that structure must be a minimum of *ATV's* energy too. The case
   asserts it (max|F| ≤ 5×10⁻³ eV/Å) — a gradient-level agreement check
   independent of the energy pin, which none of the existing AM1 regressions
   make. Measured: **8.5×10⁻⁴ eV/Å**, against 1.65 eV/Å at the un-optimized
   input geometry (a 330× separation, asserted so the gate cannot go vacuous).
2. **Coordinate truncation cannot masquerade as an error.** At a minimum the
   energy is quadratic, so the ≤5×10⁻⁷ Å / ≤10⁻⁵° rounding of the printed
   coordinates perturbs E at second order. Measured curvature along r is
   ≈ 91 eV/Å²; a parabolic fit puts ATV's own minimum 3.8×10⁻⁵ Å from the pinned
   r, worth 7×10⁻⁸ eV. Six orders of magnitude below the tolerance.

The **input-geometry pin** (MOPAC `CYCLE 1 HEAT`, a non-stationary point)
removes the remaining objection that agreement might depend on the gradient
vanishing. It does not: the residual is the same size at both geometries.

### Pins

| Quantity | Geometry | MOPAC | ATV | Δ | Tol |
| -------- | -------- | ----- | --- | - | --- |
| `TOTAL ENERGY` | optimum | **−348.56254** eV | −348.56316283 | −6.2×10⁻⁴ eV | 2×10⁻³ eV |
| `FINAL HEAT OF FORMATION` | optimum | **−59.24079** kcal/mol | −59.255061 | −1.43×10⁻² | 0.05 kcal/mol |
| `NET ATOMIC CHARGE` O | optimum | **−0.3827** e | −0.38271 | −1×10⁻⁵ e | 2×10⁻⁴ e |
| `NET ATOMIC CHARGE` H | optimum | **+0.1914** e | +0.19135 | −5×10⁻⁵ e | 2×10⁻⁴ e |
| max‖F‖ (stationarity) | optimum | RMS grad 0.00141 kcal/mol/Å | 8.51×10⁻⁴ eV/Å | — | ≤ 5×10⁻³ eV/Å |
| `CYCLE 1 HEAT` | input | **−57.64871** kcal/mol | −57.664384 | −1.57×10⁻² | 0.05 kcal/mol |
| max‖F‖ (non-vacuity) | input | — | 1.65 eV/Å | — | > 0.5 eV/Å |

ATV values measured on the CPU reference host (x86_64 epyc, GCC 14.3,
OpenBLAS), **bit-identical at `OMP_NUM_THREADS=1` and `8`**.

`ΔHf = (E_tot − Σ_A EISOL_A)·23.061 + Σ_A ΔHf_atom(A)` [kcal/mol]
(<https://openmopac.net/Manual/calc_of_hof.html>), so the ΔHf pin gates the SCF
energy **and** the thermochemistry constants; pinning the total energy
separately says which of the two moved. The constants are checked directly by
`tests/cpu/semiempirical/test_am1_thermochemistry.cpp`
(`EISOL(H) = −11.396427` eV, `EISOL(O) = −316.09952` eV,
`ΔHf_atom(H) = 52.102`, `ΔHf_atom(O) = 59.559` kcal/mol) — all equal to MOPAC's
to every printed digit, which is why the ΔHf residual (1.43×10⁻²) is exactly
the energy residual times 23.061 (6.2×10⁻⁴ × 23.061 = 1.43×10⁻²) and nothing
more.

### Tolerance rationale

The residual is **6.2×10⁻⁴ eV** (0.014 kcal/mol), which is 120× the
transcript's 5-decimal print precision. That is not noise, so it was traced
rather than absorbed.

**Cause: the physical-constant convention.** ATV's NDDO integrals use CODATA-2018
(`nddo_hcore.cpp:21-22`, `nddo_two_electron.cpp:18-19`,
`nddo_overlap.cpp:17` — `a₀ = 0.529177210903` Å, 1 Ha = `27.211386245988` eV,
so e²·k = 14.3996455 eV·Å). MOPAC froze `a₀ = 0.529167` Å, 1 Ha = `27.21` eV
and `e²·k = fpc_9 = 14.399` eV·Å — a 4.5×10⁻⁵ relative difference, and one
MOPAC does not even apply self-consistently (0.529167 × 27.21 = 14.398634,
not 14.399).

Measured, not assumed: rebuilding ATV with **MOPAC's frozen constants**
substituted for CODATA moves this energy from −348.56316 to **−348.56243** eV,
i.e. the residual collapses from 6.2×10⁻⁴ to 1.1×10⁻⁴ eV and ΔHf from
−59.25506 to **−59.23816** (residual 2.6×10⁻³ kcal/mol). **82 % of the gap is
the constant convention**; the ~10⁻⁴ eV remainder is oracle-side (MOPAC's own
internally inconsistent constants, its numerically re-derived AD/AQ/AM
multipole additives, and the 0.00141 kcal/mol/Å residual gradient of the
published structure).

ATV keeps CODATA — it is the project-wide convention (`core/units.h`) and is
physically correct; MOPAC's constants are a 1980s freeze. So the tolerance must
admit the offset while staying far below any defect class this gate exists to
catch:

| | Scale | vs `2×10⁻³` eV tolerance |
| --- | ----- | --- |
| Observed residual (constant convention) | 6.2×10⁻⁴ eV | 3.2× headroom |
| Thread-count / BLAS variation | 0 (bit-identical 1 vs 8 threads) | — |
| Transcript print precision | 5×10⁻⁶ eV | 400× tighter than the tolerance |
| **Wrong NDDO parameter set** (RM1 at the same geometry) | 3.005 eV | **1500× the tolerance** |
| **Wrong parameter set**, in ΔHf (RM1: −38.80 kcal/mol) | 20.4 kcal/mol | **410× the 0.05 tolerance** |

RM1 is the sharpest available in-tree probe of the gate: it is the *same*
functional form with reparameterized values, i.e. exactly the defect class a
parameter-transcription error would produce, and it lands 1500 tolerances away.

Charge tolerance `2×10⁻⁴` e is set by the transcript, which prints charges to 4
decimals (half-ULP 5×10⁻⁵, which is precisely the observed H residual). The
stationarity tolerance `5×10⁻³` eV/Å is 6× the measured residual and 330× below
the off-optimum gradient the case also asserts — it cannot pass by accident.

**No tolerance here was widened to make a check pass.** The energy/ΔHf
tolerances were set from the constant-convention offset *after* it was measured
and explained; the charge and stationarity tolerances were set from the
transcript's own precision.

### Perf

No fair wall comparison is claimed for this historical transcript. The archived
MOPAC run reports `COMPUTATION TIME = .000 SECONDS` on 2003 hardware — both are
below the resolution of a meaningful comparison for a 3-atom molecule.

### Gaps

- **`ELECTRONIC ENERGY` / `CORE-CORE REPULSION` are not pinned separately**
  (−493.26959 / +144.70704 eV in the transcript). `AM1Result` exposes only the
  total; the decomposition is reachable only through `AM1::state()`, which
  returns the internal `NDDOScfState`
  (`include/atomiverse/cpu/semiempirical/am1_solver.h:9,60` includes and
  re-exports `internal/nddo_solver.h`). CLAUDE.md lists that internal-type leak
  as scheduled-for-deletion debt, so this case does **not** consume it. Pinning
  the decomposition — which would localize a defect to the one- vs two-electron
  side — needs a public accessor on `AM1Result` first.
- **Koopmans IP not pinned** (12.46417 eV) — `AM1Result` exposes no orbital
  energies. Same fix, same row.
- **Dipole not pinned** (1.860 D total, point-charge 1.094 + hybrid 0.767) —
  no AM1 dipole on the public surface.
- **Constant convention.** ATV is CODATA, MOPAC 7.00 is frozen 1980s constants;
  this floor applies to the historical transcript, not modern OpenMOPAC.
  The independently executed 23.2.5 fixtures below use matching modern constants.
- **Geometry taken from the transcript, not re-optimized.** `ATV_BUILD_MINIMIZER`
  is OFF in the SE-only build, so the case cannot verify ATV's *own* AM1
  optimum coincides with MOPAC's beyond the stationarity + curvature argument
  above.


## Reference — AM1 water force

### Method

AM1 (Austin Model 1), Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985).

### What is checked

1. **Public API path:** `cpu::Evaluate` + nested `cpu::semiempirical::AM1`
   (the evaluate phase requests forces) writes a negative bound total energy [eV] and non-zero
   nuclear forces [eV/Å] on a slightly distorted water geometry; net force ≈ 0.
   The net-force gate is a plumbing check, not a physics one: the solver accumulates
   equal-and-opposite pair contributions (translational invariance), so a
   non-zero net would mean accumulation or write-back lost a component.
2. **Heat of formation:** MOPAC-style
   `ΔHf = (E_tot − Σ eisol)·23.060547830619029 + Σ E_atom` [kcal/mol] near the AM1 water
   regime (~−59 kcal/mol; expt. −57.8). See openmopac calc_of_hof.
3. **SCF gradient identity:** the published forces — analytic pair-integral
   derivatives contracted with the converged density — match a
   *re-converged* central finite difference of the total energy
   (step `5×10⁻⁴` Å) to `5×10⁻³` eV/Å. That agreement is the point: at SCF
   convergence the density-response term of `dE/dR` vanishes, and NDDO's
   orthonormal (zero-differential-overlap) basis contributes no
   basis-derivative "Pulay" term, so the fixed-density derivative is the whole
   gradient. Re-solving the SCF at each displaced geometry is the independent
   check of that claim.

This is a force/self-consistency gate, not a thermochemistry pin of ΔHf.

### Geometry

O at origin; H at (1.05, 0, 0) and (−0.26, 1.0, 0) Å (intentionally off
equilibrium so residual forces are measurable).


## nddo-sharp-discriminator — RM1 CO₂ / CO vs OpenMOPAC PARAM survey

```text
oracle: mopac
oracle_version: OpenMOPAC PARAM survey tables (github.com/openmopac/param @ 9859905,
                Apache-2.0); cross-checked against openmopac.net PM7/PM6-D3H4
                accuracy table (RM1 error column, 2021-03-11 dump)
family_oracle: mopac
```

NDDO family lock: **OpenMOPAC only**. No DFTB+ number appears in this case.
CTest never executes an oracle binary; live outputs are archived offline below.

### Why this case exists

`reg_rm1-water-mopac-energy` pins RM1 ΔHf(H₂O) = −57.86 kcal/mol against
experiment −57.80 (0.06 kcal/mol). That gate can pass if the code merely
reproduces *thermochemistry of water*, not RM1. Carbon dioxide and carbon
monoxide are the survey rows where RM1 is **12.5 kcal/mol** from experiment.

### Method

**RM1** — G. B. Rocha, R. O. Freire, A. M. Simas and J. J. P. Stewart,
*J. Comput. Chem.* **27**, 1101–1111 (2006). Same public shape as the water
pin: `cpu::semiempirical::RM1`, members, `run(frame)`.

A MOPAC ΔHf belongs to the **method's own** stationary geometry (OpenMOPAC
`data/README.md`). This case relaxes the unique C–O distance on the RM1
surface (linear CO₂ / diatomic CO) with a 1-D pattern search over `RM1::run`.

### Pin

OpenMOPAC PARAM survey `Analysis/Survey organic/RM1.heats` @ `9859905`:

```text
 Formula        Molecule                              Heat of Formation   Diff. Ref.
                                                       Exp.     Calc.
 CO            Carbon monoxide                       -26.42    -13.87    12.55     5
 CO2           Carbon dioxide                        -94.05    -81.55    12.50     5
```

`RM1.geos` (experimental column first, RM1-calculated second):

```text
 CO            Carbon monoxide     C-O    1.128   1.138
 CO2           Carbon dioxide      C-O    1.162   1.173
```

| Quantity | Published RM1 | Source | Tol |
| -------- | ------------- | ------ | --- |
| ΔHf(CO₂, relaxed) | **−81.55** kcal/mol | `RM1.heats` L829 | ±0.05 kcal/mol |
| r(C–O) CO₂ | **1.173** Å | `RM1.geos` L225 | ±0.005 Å |
| ΔHf(CO, relaxed) | **−13.87** kcal/mol | `RM1.heats` L548 | ±0.05 kcal/mol |
| r(C–O) CO | **1.138** Å | `RM1.geos` L167 | ±0.005 Å |

**Discriminator (binding):** `|ΔHf_RM1(CO₂) − (−94.05)| > 10 kcal/mol`. If a
future RM1 table accidentally tracked experiment, this case must fail — the
point is to pin the *method*, not JANAF.

Cross-check (errors, not the pin): openmopac.net accuracy table, CO₂ row,
RM1 error = +12.50 kcal/mol on ΔHf_exp = −94.05 → calc = −81.55.

### Twin

`tools/tb_oracle/inputs/mopac/rm1-co2/` — reconstructed `RM1` decks at the
experimental start geometry. The pin numbers are the PARAM survey, not a
transcript of those decks on this host.

### Tolerance rationale

PARAM survey prints two decimals (0.01 kcal/mol). The NDDO constant-convention
floor vs a live MOPAC run is ~0.02 kcal/mol (see `am1-water-mopac-energy`).
This case pins a *published survey table*, not a MOPAC 7.00 energy in eV, so
±0.05 kcal/mol is the gate (five times the print precision, twice the
convention floor). Geometry: survey prints 0.001 Å; ±0.005 Å covers the 1-D
search termination.

### Gaps

No fair ATV↔MOPAC timing comparison is recorded for the survey geometries.
Independent full-force pins are described below.


## se-analytic-hessian

**oracle:** literature (structural)
**oracle_version:** n/a — FD Hessian identity + linear H₂ mode count

### Method

`cpu::vib::Hessian` + `cpu::semiempirical::DFTB2` (MioSketch, forces on) on H₂
at 0.74 Å. Unblocks SE-loop VIB policy for electronic SE force sources.

### Gates

- Hessian 6×6, finite, symmetric to 1e-8 eV/Å²
- Linear rotor → 1 vibrational mode, ω > 100 cm⁻¹, n_imaginary = 0

### Gaps

Absolute cm⁻¹ not compared to DFTB+/experiment (foothold is API + structural).


## Executed OpenMOPAC 23.2.5 — AM1/RM1 energy, charges and full forces

The existing `am1-water-mopac-energy` section also checks six neutral,
closed-shell fixtures: distorted water, tilted CO and nonplanar ammonia with
both AM1 and RM1. These are actual runs of the unmodified official release,
with modern constants (no `OLDCON`). Source commit, binary hashes, input decks,
raw AUX/output and extraction instructions are in
[`live-23.2.5`](../../../tools/tb_oracle/inputs/mopac/live-23.2.5/README.md).
No oracle executable is required by the regression.

| Quantity | Absolute tolerance | Basis |
| --- | ---: | --- |
| Total energy | 1e-6 eV | DISP kcal/mol print rounding ≤2.17e-7 eV |
| Heat of formation | 1e-6 kcal/mol | High-precision AUX, separately checked |
| Atomic charge | 1e-7 e | Tight SCF and high-precision AUX population output |
| Every Cartesian force | 1e-5 eV/Å | Independent oracle gradient qualification |

MOPAC's printed gradient is a fixed-density pair finite difference, not an
analytic derivative. Its error was qualified using 324 independently
re-converged MOPAC single points and two Richardson refinements: maximum
refinement difference 2.28e-9 eV/Å, native-gradient discrepancy 7.85e-7 eV/Å.
No ATV value was used to set this error budget. AUX gradients are converted
with `F = -gradient / 23.060547830619029`.

Observed CPU maxima over these six fixtures: energy 2.10e-7 eV, charge
1.25e-8 e, force 7.86e-7 eV/Å. This establishes CHNO full-force evidence for
these molecular fixtures; broader element, spin and periodic coverage is not
implied. Independently executed CUDA coverage follows below.

The same `am1-water-mopac-energy` section now runs with `--device=both`.
CUDA grades energy, heat of formation, charges and every Cartesian analytic
force against these six external fixtures using the same external tolerances.
Direct CPU/CUDA differences must additionally stay below 1e-8 eV, 1e-8 e and
1e-7 eV/Å. Both use density residual 1e-10 and energy change 1e-12 eV, with
both required for convergence. The CUDA path covers FP64 neutral closed-shell
CHNO molecules without a cell.

The CUDA section checks standalone `forces()` overwrites energy, charges and
forces; energy-only mutable/const `run()` clears the method's force result and
preserves the Frame's existing force columns. Full force vectors rotate with
the geometry while translation leaves energy and charges unchanged. Default
Evaluate directly mounts AM1/RM1 and adds the method's energy/forces to an
independent contribution already present. Molecular methods do not declare
stress support; the common nested adapter applies capability vetoes rather
than requesting unavailable stress from default Evaluate.

Coordinate/identity edits, budgets, build and force-solve failures, same-parent
retry and component lifecycle remain covered. Initial/trial solves are checked
both before a first accepted guess and after an accepted guess exists. Three
subsequent normal geometry steps must match an unprobed method's iteration
count, energy and full force vector; an injected failure also leaves the
accepted guess intact. The six-fixture public-force, CPU comparison and
Evaluate checks pass on GH200 in the SE-only build. Compute-sanitizer memcheck
reports zero errors for the complete section and NDDO primitive tests;
two-center primitive racecheck reports zero hazards. Full-solve
transfer/allocation profiling remains separate acceptance work.

A closing lifecycle check injects a component destruction error after that
component has released its state. Both methods must then recover using the
same object for standalone `forces()` and a new Evaluate parent, retaining
external energy, charge and full-force agreement. Cleanup must not destroy
the freed component twice. A standalone `shutdown()` must propagate the
original destruction exception after cleanup and permit same-object `forces()`
recovery. Leaving a standalone method's scope must finish cleanup without
propagating the user's destruction exception or terminating the process.
These exception-path checks pass on GH200 together with the external numerical
checks, including complete-section memcheck with zero errors. The section
runs within its 60-second budget; cleanup recovery uses the same numerical
tolerances as normal evaluation.

When `ATV_HAS_MD` is enabled, the same section additionally runs AM1 and RM1
water with physical atomic masses and deterministic initial velocities. It
compares `MD::run(frame, 0)` energy, charges and forces with standalone
`forces()`, then compares a CUDA 2+2 step continuation with one four-step run
and CPU Velocity Verlet at the same 0.01 fs timestep. Bounds are 1e-10 Å for
coordinates, 1e-10 Å/fs for velocities, 1e-8 eV for energy, 1e-8 e for charges
and 1e-7 eV/Å for forces. This is a short composition check, not a long NVE
conservation or chaotic-trajectory claim.

When `ATV_HAS_MINIMIZER` is enabled, native CUDA Minimize mounts each method
directly. Zero iterations must prepare the same energy/charges/forces as
standalone `forces()` without committing an initial electronic guess. Three
small steepest-descent steps (`alpha=0.01`) must lower energy; the final energy
and full force vector must match standalone evaluation at that final geometry
to 1e-8 eV and 1e-7 eV/Å. This does not claim geometry convergence. Each
conditional leg prints PASS when executed or SKIP with its disabled module.
Both AM1 and RM1 pass these MD and Minimize checks in the combined GH200 build,
including initial forces, CUDA chunk equivalence, the four-step CPU comparison,
energy decrease and standalone reevaluation of the minimizer's final geometry.
The complete external-plus-composition section stays within its 60-second
budget. This acceptance is limited to the short trajectories and three descent
steps specified above; long-time NVE drift, timestep convergence and completed
geometry optimization remain separate checks.

The CPU force section also exercises direct/nested publication, accumulation,
changed controls, failed solves and retry. CPU AM1/RM1 initial-force and trial
solves leave the accepted electronic guess unchanged: six subsequent normal
steps match an unprobed method in iteration count, energy and full forces.
QM/MM parent-context and accepted-history checks are exercised in the existing
`multiscale/oniom` case. CUDA Evaluate trial-history checks are described above;
optimizer trial acceptance beyond these short composition legs remains
separate integration work.


## Fixed-duration NVE and time-step refinement

`nddo-nve` starts from the same executed OpenMOPAC distorted-water geometry
(O at the origin, H at `(0.96,0,0)` and `(-0.24,0.93,0.1)` Å). The initial
potential energy is checked against that method's external pin at `1e-6 eV`.
Masses are O=16 and H=1 amu, matching the earlier composition fixture.
H velocities are `(0.01,0.002,0)` and `(-0.003,0.008,0.001)` Å/fs; the O velocity
balances their total momentum. No random initialization, thermostat, periodic
cell or momentum-removal operation is used.

Both trajectories last **10 fs**: Velocity Verlet steps `0.1 fs` and `0.05 fs`
(100 and 200 steps). Both use neutral closed shells, FP64, a 400-iteration SCF
budget, density residual `1e-10` and energy residual `1e-12 eV`. `run(frame,0)`
prepares the initial forces and energy before the first half kick. Samples are
at identical physical times every `0.5 fs`, including zero; a failed SCF or
incomplete step budget fails the trajectory.

The conserved quantity is potential `kEnergy` plus `kKinetic`, in eV.
At every sample kinetic energy must also match an independent host calculation
`0.5 * 103.6427 * sum(m*v²)` to `1e-12 eV`. The acceptance limits were fixed
before trajectory runs: each trajectory's **maximum sampled** absolute energy
deviation must be at most `1e-3 eV` per molecule (a 1 meV absolute accuracy
budget), and halving the time step must reduce it to at most `0.4` of the coarse
value. Velocity Verlet's leading error is quadratic in the step size (ideal
ratio `0.25`); the larger ceiling admits higher-order corrections, without a
post hoc fitted noise floor. No normalization by the much larger total binding
energy is used. RMS and signed endpoint deviations are printed as diagnostics.

This tests a short trajectory spanning roughly a water stretching period;
it is neither a long-time stability claim nor a trajectory golden generated by
Atomiverse. The external initial Hamiltonian/force evidence remains separate.
The maximum refers to the specified sampled times, not every step or the
continuous trajectory. When both backends run, matching-time energy and short
final-trajectory comparisons supplement each backend's independent conservation
and refinement checks; they do not require long chaotic trajectories to coincide.

CPU/CUDA matching-time total and kinetic energy tolerances are `1e-6 eV`.
At 10 fs the short-trajectory comparison uses `1e-8 Å` positions,
`1e-9 Å/fs` velocities, `1e-7 e` charges, `1e-5 eV/Å` forces and `1e-6 eV`
potential energy. These supplement each backend's conservation test and the
existing external full-force fixtures. Exactly constant sampled energy is
rejected for this excited fixture, rather than presented as a zero refinement
ratio. Drivers are destroyed before moving or destroying their bound Frames.
