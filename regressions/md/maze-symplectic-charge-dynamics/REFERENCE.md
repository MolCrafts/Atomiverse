# maze-symplectic-charge-dynamics

oracle: metalwalls
oracle_version: **MetalWalls, git c8ad175d** (release branch, 2026-06-18), run
locally as `/nobackup/proj/disk/teoroo/personal/jicli594/work/opt/metalwalls/mw`,
`mpirun -n 1`. Supporting theory: Coretti et al., J. Chem. Phys. **152**,
194701 (2020), DOI 10.1063/5.0007192; Marin-Laflèche et al., JOSS **5**, 2373
(2020), DOI 10.21105/joss.02373.

Waves 1–5 pinned this case from the Coretti/JOSS **formulas** because no
MetalWalls binary existed on this host. The binary now exists; every pin below
has been re-checked against it. Verdict: **confirmed** — MetalWalls satisfies
every invariant this regression asserts, and reproduces Atomiverse's electrode
charges to **1.1e-6 relative** in the one configuration where the two codes
evaluate the same electrostatic model. One new pin (a real MW number) was
added; nothing was loosened.

## Unit bridge (stated explicitly — MW is atomic units, ATV is metal units)

| Quantity | MetalWalls | Atomiverse | Conversion |
| --- | --- | --- | --- |
| length | Bohr | Å | 1 Bohr = 0.529177210903 Å |
| energy | Hartree | eV | 1 Ha = 27.211386245988 eV |
| charge | e | e | identity |
| potential | Eh·e⁻¹ | V | 1 Eh/e = 27.211386245988 V |
| force | Ha/Bohr | eV/Å | 27.211386245988 / 0.529177210903 = 51.4220675 eV/Å |
| time | ħ/Eh | fs | 1 fs = 41.341373335182 a.u. |
| Gaussian width η | Bohr⁻¹ | Å⁻¹ | η[Bohr⁻¹] = η[Å⁻¹] × 0.529177210903 |

Input values actually written into MetalWalls (from `gen_inputs.py`):
η = 2.0 Å⁻¹ → **1.058354421806 Bohr⁻¹**; Ψ_top = 1 V → **0.036749322175655 Eh/e**;
dt = 0.25 fs → **10.335343333795 a.u.**; box 6 × 6 × 20 Å →
11.33835674775462 × 11.33835674775462 × 37.79452249251541 Bohr.
Dimensional check: 1 V is far below 1 Eh/e = 27.2 V, so the a.u. potential must
be ≪ 1 — it is (0.0367). η is an inverse length, so going Å⁻¹ → Bohr⁻¹
*shrinks* the number (Bohr < Å) — it does (2.0 → 1.058). MetalWalls' own
`bohr2angstrom` (`src/constants.F90`) is 0.52917721067, 4.4e-10 relative from
the value above — five orders below any tolerance here.

## Geometry (unchanged)

Parallel-plate capacitor, orthorhombic box 6 × 6 × 20 Å:

| Region | type | N | z [Å] | η [1/Å] | Ψ [V] | mass [amu] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom plate | 1 | 4 | 4 | 2.0 | 0.0 | static |
| Top plate | 2 | 4 | 16 | 2.0 | +1.0 | static |
| Ion + (+0.5 e) | 3 | 1 | 8.5 | point | — | 40.0 |
| Ion − (−0.5 e) | 3 | 1 | 11.5 | point | — | 40.0 |

Plate sites at xy ∈ {1.5, 4.5}², ions at (2.1, 2.1, 8.5) and (4.2, 4.2, 11.5) Å,
zero initial velocity. Explicit NVE velocity Verlet, dt = 0.25 fs, 400 steps
forward, velocity flip, 400 steps back (palindrome).

## What MetalWalls cannot express, and why

Atomiverse evaluates this cell in **minimum image**: `PairGaussianCoul` is a
hard-cutoff real-space pair style (12 Å > the 10.9 Å maximum min-image
distance, so every pair interacts exactly once) and `ConstPotBO` builds the
elastance with `xy_image_range = 0`, i.e. one nearest image per pair and no
reciprocal space at all.

MetalWalls has **no minimum-image Coulomb mode**. Its electrostatics is Ewald
summation in every code path (`num_pbc` 2 or 3; `src/coulomb_lr.f90`,
`src/coulomb_sr.f90`, `src/coulomb_self.f90` are the three Ewald pieces and
there is no fourth). Run on the identical 6 × 6 × 20 Å cell it therefore
answers the *fully periodic* problem, not the truncated one:

| Q on the +1 V plate, 6 × 6 × 20 Å cell | value [e] |
| --- | ---: |
| Atomiverse, minimum image (what this regression simulates) | **+0.1358218636749** |
| MetalWalls c8ad175d, 3D Ewald (`twinA_ewald_6x6x20`) | **+0.1529601174306** |
| relative discrepancy | **−11.20 %** |

MW's number is converged in its own Ewald parameters to 1.5e-13 relative
(r_cut 2.0 / 2.5 / 2.9 Å at rtol = ktol 1e-12/1e-14 all give
+0.15296011743062…), so the 11.2 % is entirely the minimum-image truncation
of the Atomiverse model — not a solver disagreement. It is **not** repinned
into the regression: that would replace the model the regression is testing.

## The kernel-matched twin — where MW and ATV compute the same thing

Take the identical 10-atom cluster and grow the cell. MetalWalls' Ewald then
converges to the **isolated pairwise Gaussian-Coulomb sum**, which is exactly
what the minimum-image path computes once the cell exceeds twice the cluster
extent. Atomiverse's answer is box-independent there (bit-identical at
L = 40, 60, 200 Å); MetalWalls approaches it monotonically as its residual
periodic images die off:

| MW cell L [Å] | MW Q_top [e] | rel Δ vs ATV | MW grand energy [eV] | rel Δ | max ion-force Δ [eV/Å] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | +0.1267873919608 | +6.301e-4 | −0.9840388431722 | +9.328e-4 | 4.38e-4 |
| 60 | +0.1267203095694 | +1.006e-4 | −0.9833929872312 | +2.759e-4 | 1.30e-4 |
| 100 | +0.1267091472474 | +1.255e-5 | −0.9831803436266 | +5.959e-5 | 2.82e-5 |
| **200** | **+0.1267076937357** | **+1.076e-6** | **−0.9831291093381** | **+7.473e-6** | **3.53e-6** |
| Atomiverse (isolated) | **+0.1267075573679** | — | **−0.9831217623464** | — | — |

Raw MetalWalls output, `twinB_dilute_200/charges.out` (already in e, no
conversion — MW writes "1 au = 1 e"):

```
# step        0
-4.234061692847945E-002      <- bottom plate, 4 sites
-3.112908052013959E-002
-3.112908052014043E-002
-2.210891576695224E-002
+2.118516355288748E-002      <- top plate (+1 V), 4 sites
+3.065933680220784E-002
+3.065933680220688E-002
+4.420385657840947E-002
```
Σ(top 4) = **+0.1267076937357 e**; Σ(all 8) = 0 to the last digit
(`charge_neutrality true electrodes`).

Raw `twinB_dilute_200/energies.out` (Hartree):

```
# step        Hamiltonian             ...            Potential energy
       0 -3.612932837933060E-002 ...            -3.612932837933060E-002
```
−3.612932837933060e-2 Ha × 27.211386245988 eV/Ha = **−0.9831291093381 eV**.
Atomiverse's grand energy for the same cluster,
E = U_pair − Σ Ψ_i Q_i = −0.8564142049784 − 1.0 × 0.1267075573679 =
**−0.9831217623464 eV** — MW's remaining +7.5e-6 relative gap is its own image
tail, which extrapolates (1/L³ Richardson on L = 100/200) to −0.9831214 eV,
4e-7 from Atomiverse.

## Hard-coded pins (asserted in the regression source)

| Pin | Value | Source | Checked against real MW |
| --- | --- | --- | --- |
| Dilute-limit Q_top | +0.1267076937357 e, rel ≤ 1e-4 | **MetalWalls c8ad175d, `twinB_dilute_200`** | measured 1.076e-6 |
| Constraint residual ‖σ‖∞, every step | ≤ 1e-10 V | Coretti 2020 §II.C | MW: 2.83e-16 V ✓ |
| Electrode electroneutrality \|ΣQ\|, every step | ≤ 1e-9 e | KKT constraint | MW: exact ✓ |
| Palindrome ion-position return (MaZe) | ≤ 1e-6 Å | Coretti 2020 time-reversibility | MW: 2.29e-7 Å ✓ |
| Palindrome electrode-charge return (MaZe) | ≤ 1e-6 e | Coretti 2020 time-reversibility | MW: 5.95e-10 e ✓ |
| MaZe return error < 0.1 × loose CG+ASPC twin | ratio | reversible vs dissipative | MW 2.29e-7 vs 1.07e-5 ✓ |
| NVE grand-energy drift (MaZe, after 5-step settle) | ≤ 1e-6 eV | Coretti 2020 no-secular-drift | MW: 6.76e-7 eV ✓ |
| MaZe drift < loose (cg_tol 1e-3) CG+ASPC twin drift | ratio | Coretti 2020 vs loose solve | — (ATV-internal twin) |
| Q_top > 0 at ΔΨ = +1 V | polarity | electrostatics sanity | MW: +0.15296 e ✓ |
| Ion min separation > 1.8 Å | validity | bound smooth trajectory | — (ATV-internal) |

Units: eV, Å, fs, amu, V, e (Atomiverse public surface).

### How each invariant was verified against the binary

* **Constraint at solver precision.** `twinA_ewald_6x6x20_md/run.out` prints,
  every step, `|maze| Maximum of the constraints: 1.04083E-17` (atomic units,
  Eh/e; the single-solve `twinA_ewald_6x6x20` case reports 5.89806E-17) —
  × 27.211386245988 V per Eh/e = **2.83e-16 V** (resp. 1.61e-15 V), five orders
  inside the 1e-10 V pin, and `|maze| number of iterations: 0` because
  `maze inversion` reaches the linear constraint in one exact solve. MetalWalls also confirms MaZe ≡
  Born–Oppenheimer: `twinB_dilute_60` (`maze inversion`) and
  `twinB_dilute_60_inv` (`matrix_inversion`) agree to 1e-17 e per site.
* **No secular drift.** `twinA_ewald_6x6x20_md`, 800 MaZe NVE steps at
  dt = 0.25 fs on the regression geometry: max |H − H(step 5)| = 2.4834e-8 Ha
  = **6.758e-7 eV**, linear slope 1.863e-11 Ha/step = 5.07e-10 eV/step
  (4.06e-7 eV over the whole run). Inside the 1e-6 eV pin. Atomiverse's
  7.57e-8 eV is smaller because its truncated forces move the ions less
  (MW's peak KE reaches 0.778 eV on the same budget); the pinned *bound* is
  what both codes must respect, and both do.
* **Time reversibility.** `twinA_pal_fwd` runs 400 steps and dumps
  `restart.000000400.out`; the velocity block is negated in place and rerun as
  `twinA_pal_back` for 400 more. Ion displacement on the forward leg is
  0.2854 Å; the palindrome returns to **2.288e-7 Å** and **5.95e-10 e**. Both
  inside the 1e-6 pins, and 47× better than the loose CG+ASPC twin's 1.07e-5 Å.
  MetalWalls' figure is an *upper bound*: its restart file is text with 15
  significant digits, so the round trip seeds ~1e-14 Bohr which this locally
  unstable ion pair amplifies over 400 steps. Atomiverse's in-memory flip has
  no such floor, hence 1.4e-14 Å.

## Reproducing

Inputs: `tools/metalwalls_oracle/inputs/metalwalls/maze-symplectic-charge-dynamics/`
(`gen_inputs.py` writes every case; `atv_twin.cpp` is the matching
out-of-library Atomiverse consumer used for the dilute comparison).

```bash
python3 gen_inputs.py
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd twinB_dilute_200 && mpirun -n 1 .../metalwalls/mw > run.stdout 2>&1'
```

Note: `electrode_charges maze <alg> <tol>` needs at least two `# charges`
sections in `data.inpt` (its Verlet history) — zeros are fine, the constraint
is linear so `inversion` lands on it from any guess. MetalWalls' random
velocity initializer divides by the instantaneous temperature, so a zero-T
start must supply an explicit all-zero `# velocities` block or MW aborts with
"Temperature is NaN or meaningless". At r_cut ≤ 2.5 Å in this cell MW's MaZe
cannot reach a 1e-14 tolerance ("M-SHAKE failed…"); use `matrix_inversion`
there, or keep r_cut 2.9 Å.

## Measured at re-pin time (2026-07-29, Release, GCC 14.3, x86_64 epyc)

| Metric | MaZe symplectic | loose (1e-3) CG+ASPC twin |
| --- | ---: | ---: |
| Grand-energy drift [eV] | 7.571e-08 | 1.755e-06 |
| Palindrome position return [Å] | 1.421e-14 | 1.068e-05 |
| Palindrome charge return [e] | 9.015e-13 | 5.092e-05 |
| Max constraint residual [V] | 9.990e-11 | — |
| Max \|ΣQ\| [e] | 5.551e-17 | 5.551e-17 |
| Q_top first solve, 6 × 6 × 20 Å [e] | +0.135822 | +0.135821 |
| Q_top dilute limit [e] | +0.1267075573679 | — |
| min ion separation [Å] | 3.798 | 3.798 |

## Runtime and the MetalWalls wall-clock comparison

Two 800-step NVE runs plus one dilute solve: **0.03 s** (≪ 60 s).

MetalWalls needs **7.6–9.0 s** for the single 800-step twin-A run
(`twinA_perf`, output every 800 steps, `mpirun -n 1`), against ≈ 0.015 s for
Atomiverse's 800-step MaZe leg — a raw ratio of ≈ 0.002. **That ratio is not
an engine benchmark and must not be quoted as one.** The two codes are not
doing the same work per step: MetalWalls converges a full 3D Ewald sum whose
k-mesh is forced very fine (the 6 Å lateral cell caps the real-space cutoff at
< 3 Å, pushing α up and the k-space mode count to O(10⁵)), while Atomiverse
evaluates 45 minimum-image pairs and no reciprocal space at all. The cost gap
is the model gap documented above, so this case carries **`perf_exempt` with a
technical reason**, not "no binary".
