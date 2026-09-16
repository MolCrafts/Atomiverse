# Reference — GPW DFT-D3 three-body ATM (C9) analytic ionic forces

## Method

Grimme **DFT-D3** (zero damping, PBE) with the **Axilrod–Teller–Muto
three-body C9** term (`CALCULATE_C9_TERM .TRUE.`) contributing to the GPW
**ionic forces**: explicit-geometry ATM gradient plus the coordination-number
chain (`accumulate_atm_c9` → `dEdCN`), the exact gradient of the same
\(E_{\mathrm{ATM}}\) the energy foothold (`gpw-d3-atm`) pins.

Public path: `cpu::gpw::GPWEvaluator` / `GpwDft` with `d3 = true`,
`d3_atm = true`, `compute_forces = true`. Implementation:
`dftd3_pbe(..., calculate_c9 = true)` matching CP2K `qs_dispersion_d3`
(`doabc`, s9 = s6, damping `damping_d3(rabc, r0, 4/3, 16)`).

## Why the pin is a difference of two CP2K runs

D3 (with or without C9) is geometry-only — it never enters the Kohn–Sham
potential, so the converged density is identical with `CALCULATE_C9_TERM`
`.TRUE.` and `.FALSE.`. Per boundary condition:

\[
F_A(\mathrm{C9}) - F_A(\mathrm{two\text{-}body})
  = -\nabla_A E_{\mathrm{ATM}},
\]

which isolates **exactly the ATM three-body gradient** this capability adds.
Both runs of each pair used identical inputs (except the C9 flag) and
identical parallelization, so the electronic force cancels to solver noise:
the y-components of the printed tables (zero dispersion force by symmetry)
differ only in the 15th digit (~2e−15 Ha/Bohr).

Two cells are pinned:

- **Isolated** (12 Å, `PERIODIC NONE`, MT): the `gpw-d3-atm` foothold
  geometry — home-cell triples (CP2K `doabc` CASE 1).
- **Periodic 6 Å** (`PERIODIC XYZ`, `R_CUTOFF 10.0`): compressed images make
  the lattice-image triple classes (CASE 2: AB home, C off-home; CASE 3:
  B and C off-home, `fac0 = 1/3`) contribute at the 1e−7 Ha/Bohr level —
  the enumeration the isolated pin never exercises.

## System

Same geometry as `gpw-d3-atm` / `gpw-d3-forces`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell (iso) | 12 × 12 × 12 Å, `PERIODIC NONE` (MT), `R_CUTOFF 20.0` → pair/ABC cutoff 2× = 75.589 Bohr |
| Cell (per) | **6 × 6 × 6 Å, `PERIODIC XYZ`**, `R_CUTOFF 10.0` → 2× = 37.794 Bohr (CP2K prints half: `Cutoff Radius [Bohr]: 18.90`) |
| Basis / PP | `DZVP-MOLOPT-SR-GTH`; O `GTH-PBE-q6`, H `GTH-PBE-q1` (`n_occ = 8`) |
| XC | PBE; D3 zero damping (s6 = 1.0, s8 = 0.722, sr6 = 1.217, sr8 = 1.0), s9 = s6 |
| Cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| SCF | iso `EPS_SCF 1e-7`; per `EPS_SCF 1e-6` (dispersion pins are geometry-only; both members of a pair share the identical SCF path) |

## Offline CP2K pins

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03` on Arrhenius) |
| Command | `mpirun -np 2 cp2k.psmp -i <input>.inp -o <input>.out`, `OMP_NUM_THREADS=2`, offline only |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-forces/water_dimer_d3_atm_iso_forces.inp` (iso, C9 T), `tools/cp2k_oracle/inputs/gpw-d3-forces/water_dimer_d3_forces.inp` (iso, C9 F, rerun), `water_dimer_d3_atm_per6_forces.inp` / `water_dimer_d3_per6_forces.inp` (per 6 Å, C9 T/F) |
| Output lines | `Dispersion energy:` + `FORCES\| Atomic forces [hartree/bohr]` table |
| Wall | iso ≈ 10.5 / 10.4 s; per ≈ 1.8 / 2.0 s (TIMING `CP2K` TOTAL TIME) |
| Date | 2026-07-31 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Iso, C9 on: total / **E_disp** | −34.338212017550674 / **−0.00081414118193** |
| Iso, C9 off: total / E_disp | −34.338211868500814 / −0.00081399213205 |
| Per 6 Å, C9 on: total / **E_disp** | −34.345965809806160 / **−0.00195809486938** |
| Per 6 Å, C9 off: total / **E_disp** | −34.345966594779618 / **−0.00195887984277** |

Iso values reproduce the `gpw-d3-atm` / `gpw-d3-forces` energy pins exactly.
ATM contribution: iso −1.4905e−7 Ha (attractive), per +7.8497e−7 Ha
(repulsive — the condensed-phase sign).

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` tables)

Isolated, `CALCULATE_C9_TERM .TRUE.`:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.13885057E−02 | 3.39382605E−08 | −7.48922277E−09 |
| H0 | 4.05547994E−02 | 1.08883799E−08 | 1.31321374E−01 |
| H1 | 4.05547997E−02 | −3.70556959E−08 | −1.31321365E−01 |
| O1 | −7.98423417E−02 | −1.52539145E−08 | 2.13908340E−08 |
| H2 | 3.73506217E−02 | 9.81842399E−09 | 1.35158535E−01 |
| H3 | 3.73506306E−02 | −1.84099620E−09 | −1.35158557E−01 |

Isolated, `CALCULATE_C9_TERM .FALSE.` (same-pair rerun of the
`gpw-d3-forces` twin; matches its 2026-07-26 table to print precision):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.13880434E−02 | 3.39382581E−08 | −7.48923114E−09 |
| H0 | 4.05546400E−02 | 1.08883804E−08 | 1.31321564E−01 |
| H1 | 4.05546403E−02 | −3.70556938E−08 | −1.31321555E−01 |
| O1 | −7.98421128E−02 | −1.52539153E−08 | 2.13908083E−08 |
| H2 | 3.73504355E−02 | 9.81842344E−09 | 1.35158735E−01 |
| H3 | 3.73504444E−02 | −1.84099650E−09 | −1.35158757E−01 |

Periodic 6 Å, `CALCULATE_C9_TERM .TRUE.`:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −7.70297833E−02 | −1.96186400E−07 | −1.29836754E−08 |
| H0 | 3.81505346E−02 | 1.23208981E−07 | 1.29913786E−01 |
| H1 | 3.81504343E−02 | 1.24270789E−07 | −1.29913785E−01 |
| O1 | −7.70296389E−02 | −2.00667422E−07 | 1.79666280E−07 |
| H2 | 3.81504401E−02 | 1.77442868E−07 | 1.29913776E−01 |
| H3 | 3.81507350E−02 | −2.61961211E−08 | −1.29913946E−01 |

Periodic 6 Å, `CALCULATE_C9_TERM .FALSE.`:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −7.70302942E−02 | −1.96186403E−07 | −1.29836767E−08 |
| H0 | 3.81508710E−02 | 1.23208981E−07 | 1.29913227E−01 |
| H1 | 3.81507708E−02 | 1.24270788E−07 | −1.29913227E−01 |
| O1 | −7.70303293E−02 | −2.00667421E−07 | 1.79666261E−07 |
| H2 | 3.81507042E−02 | 1.77442867E−07 | 1.29913227E−01 |
| H3 | 3.81509991E−02 | −2.61961207E−08 | −1.29913397E−01 |

Derived ATM force pins \(F(\mathrm{C9}) − F(\mathrm{two\text{-}body})\):
largest components **4.62e−7** Ha/Bohr (iso, O0 x) and **6.90e−7** Ha/Bohr
(per, O1 x).

## Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \(E_{\mathrm{disp}}\) (ATM on, iso + per; two-body per) | **1e−9 Ha** | Same closed-form sum, same published parameters; the iso pin already agreed to 6.7e−11 (`gpw-d3-atm`). Also gates the periodic image enumeration. |
| ATM-only force vs CP2K difference | **1e−8 Ha/Bohr** | The difference cancels the electronic force to ~2e−15 (y-components); the floor is the `FORCES\|` print resolution (9 significant digits on ~1e−1 components ⇒ ~1e−9). Signals 4.6e−7 / 6.9e−7 ⇒ ≥ 46× margin. |
| ATM signal nonzero | **≥ 1e−7 Ha/Bohr** | Foothold-style engage gate: the C9 term must actually move the forces. |
| Product wiring | **1e−7 Ha/Bohr** | Force assembly run twice on the *same* converged density (d3_atm on/off) vs the standalone ATM gradient — identical code path, zero SCF residual. |
| Total F vs CP2K (ATM run) | **1e−2 Ha/Bohr** | Bounded below by the oracle's own grid error (CP2K `FORCES\| Sum` = 5.42e−3 Ha/Bohr on this twin at CUTOFF 300) — see `gpw-d3-forces` REFERENCE for the cutoff study. |
| \(\lvert\sum_A F_A\rvert\) (iso product) | **1e−3 Ha/Bohr** | Translation invariance; guards MT placement bugs (~1 Ha/Bohr when broken). |
| Unit FD (periodic ATM) | **1e−7 Ha/Bohr** | Central FD (h = 1e−5 Bohr) of \(E(\mathrm{C9}) − E(\mathrm{2b})\) vs analytic force difference — two-body cancels exactly, so this is the pure ATM gradient identity over the image classes. |
| Wall | **≤ 60 s** | e2e tier budget. |

## Measured (2026-07-31, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| Iso \(E_{\mathrm{disp}}\)(ATM) vs CP2K | 1e−9 Ha | 6.4e−11 |
| Per \(E_{\mathrm{disp}}\)(ATM) vs CP2K | 1e−9 Ha | 2.8e−10 |
| Per \(E_{\mathrm{disp}}\)(two-body) vs CP2K | 1e−9 Ha | 2.8e−10 |
| Iso ATM force vs CP2K diff | 1e−8 Ha/Bohr | 5.0e−10 (signal 4.62e−7) |
| Per ATM force vs CP2K diff | 1e−8 Ha/Bohr | 9.8e−10 (signal 6.90e−7) |
| Product wiring | 1e−7 Ha/Bohr | 2.2e−17 |
| Total F vs CP2K | 1e−2 Ha/Bohr | 7.5e−03 |
| \(\lvert\sum F\rvert\) | 1e−3 Ha/Bohr | 2.2e−04 |
| Unit FD (isolated ATM) | 1e−7 Ha/Bohr | 4.1e−14 |
| Unit FD (periodic ATM) | 1e−7 Ha/Bohr | 2.2e−13 |
| Wall | ≤ 60 s | 34.5 s (SCF iters = 11) |

## Two defects this pin exposed (fixed with it)

Both were invisible to every earlier gate: the energy foothold is isolated
(no image triples), and the isolated FD test only differentiates the
home-cell branch. `src/cpu/gpw/internal/dftd3.cpp` (`accumulate_atm_c9`):

1. **CASE 3 double-count.** The AB pairs of the triple loop must replicate
   CP2K's *half* neighbor list — `iatom ≤ jatom`, one orientation per image
   for `i < j`, both ±T images for `i == j` (compensated by `fac = 0.5`) —
   exactly as the two-body loop already did. Enumerating off-home `i ≠ j`
   pairs in both orientations gives every lattice-image triple class 6
   anchored representatives against `fac0 = 1/3`, i.e. a 2× overcount of
   CP2K `doabc` CASE 3: on this twin the ATM energy came out +6.25e−6
   instead of +7.85e−7 Ha.
2. **pbc anchoring of the truncated triple sum.** CP2K evaluates `doabc` on
   `pbc()`-wrapped coordinates; the triple legs pass HARD `r < r_cut` gates
   over a finite ±m image window, so the *included boundary set* — and with
   it ~1e−7 Ha/Bohr of per-atom force — depends on where each atom sits in
   the cell, while the energy stays invariant (measured 4e−19). With raw
   coordinates the two molecules' ATM forces came out exactly **swapped**
   relative to CP2K (max err 1.8e−7 of a 6.9e−7 signal; each value matched
   the *other* molecule's to ~1e−10). `accumulate_atm_c9` now wraps
   positions first (round-half-away-from-zero, Fortran `ANINT` parity).
   Corollary (same in CP2K): the truncated ATM energy is discontinuous at a
   wrap seam, so the FD unit test displaces the dimer off the `x = L/2`
   boundary the CP2K coordinates sit on.

## Dual gate

- **Correctness:** the pins above, plus the analytic-vs-FD ATM gradient
  identity in `tests/cpu/gpw/test_cpu_gpw_dftd3.cpp`:
  `GpwDftD3Atm.AnalyticForcesMatchFiniteDifference` (isolated, pre-existing,
  4.1e−14) and `GpwDftD3Atm.PeriodicAtmForcesMatchFiniteDifference` (new —
  FD of the ATM-only energy difference over the periodic image classes,
  2.2e−13 Ha/Bohr).
- **Speed:** CP2K ATM iso `ENERGY_FORCE` twin ≈ 10.5 s (2 ranks × 2 OMP);
  the regression (one 300 Ry MT SCF + **two** force assemblies + four
  standalone D3 sums) measured 34.5 s at 8 threads on the same host →
  ratio ≈ **3.3**, inside the 10× fair wall. The second force assembly
  exists only for the wiring gate.

## Notes

- **ATM virial is still zero**: `accumulate_atm_c9` adds forces and the CN
  chain but no `D3Result::virial` contribution, so `d3_atm` under
  `compute_stress` misses the three-body term — the ATM **stress** pin is a
  separate queue row (as for D4 ATM).
- Energy foothold: `regressions/gpw/gpw-d3-atm` (isolated E_disp only).
- Two-body force twin: `regressions/gpw/gpw-d3-forces` (same isolated cell;
  its D3 run was rerun here so both members of the difference share one
  parallelization).
- Sibling (serialized on the force-assembly ATM fold): `gpw-d4-atm-forces`.
- The periodic pin deliberately uses `R_CUTOFF 10.0`: at the CP2K default
  (2 × 20 Å = 75.6 Bohr) the 6 Å-cell triple enumeration is ~40× more
  images for the same code coverage; both sides of the twin use the same
  doubled-cutoff convention (`Cutoff Radius [Bohr]: 18.90` printed = half).
