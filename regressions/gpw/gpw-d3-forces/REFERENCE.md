# Reference — isolated water-dimer GPW PBE-D3 (zero damping) analytic forces

## Method

Γ-point **GPW** (PBE) on an **isolated** water dimer (CP2K `PERIODIC NONE` +
`PSOLVER MT`) with additive **DFT-D3 zero ("original") damping**, PBE reference
parameters, two-body only (`CALCULATE_C9_TERM .FALSE.`).

Product ionic forces = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** the geometry-only dispersion gradient
\(F_{\mathrm{disp}} = -\nabla E_{\mathrm{disp}}^{\mathrm{D3}}\) from
`cpu::gpw::dftd3_pbe` — the exact gradient of the same energy the SCF
converged, including the coordination-number dependence of \(C_6\).

Public path: `cpu::gpw::GPWEvaluator` with `d3 = true`,
`compute_forces = true`, `nonperiodic = true`.

## Why the pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only: it adds \(E_{\mathrm{disp}}(\{R\})\) to the total
energy and does not enter the Kohn–Sham potential, so the converged density is
identical with and without it. Therefore

\[
F_A^{\mathrm{CP2K}}(\mathrm{D3}) - F_A^{\mathrm{CP2K}}(\mathrm{plain})
  = -\nabla_A E_{\mathrm{disp}}^{\mathrm{D3}}
\]

isolates exactly the quantity this capability adds, with the (much larger)
electronic force cancelling. Two CP2K `ENERGY_FORCE` runs on the identical
geometry — one with the `&VDW_POTENTIAL` block, one without — give that pin.

## System

Same geometry as `gpw-d3-atm` / `gpw-d4-forces`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, **`PERIODIC NONE`** (MT / cluster Poisson) |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3 zero damping, PBE: \(s_6=1.0\), \(s_8=0.722\), \(s_{r,6}=1.217\), \(s_{r,8}=1.0\); ATM off |
| Cutoff | 300 Ry (CP2K `CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| SCF | `EPS_SCF 1e-7` / Atomiverse `tol = 1e-7` |

Pair cutoff: `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr.
For this isolated cluster no lattice images are summed, so the value does not
affect the pin (verified: 40 and 75.589 Bohr give identical E_disp here) — but
it does affect periodic cells, which is how the old hard-coded 40 Bohr was
caught. See "Cutoff convention" below.

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-forces/water_dimer_d3_forces.inp` and `water_dimer_plain_forces.inp` |
| Wall | ≈ **12.3 s** (D3 run), ≈ **12.5 s** (plain run) |
| Date | 2026-07-26 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.337397876368769 |
| Total (D3) | −34.338211868500807 |
| **Dispersion energy** | **−0.00081399213205** |

Atomiverse `dftd3_pbe` on the same nuclei: **−8.13992196100576e−04 Ha**
(\(|\Delta| = 6.4\times10^{-11}\) Ha).

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

D3 run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.13880434E−02 | 3.39382525E−08 | −7.48922451E−09 |
| H0 | 4.05546400E−02 | 1.08883797E−08 | 1.31321564E−01 |
| H1 | 4.05546403E−02 | −3.70556942E−08 | −1.31321555E−01 |
| O1 | −7.98421128E−02 | −1.52539175E−08 | 2.13908200E−08 |
| H2 | 3.73504355E−02 | 9.81842421E−09 | 1.35158735E−01 |
| H3 | 3.73504444E−02 | −1.84099666E−09 | −1.35158757E−01 |

Plain run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.15048974E−02 | 3.39382577E−08 | −7.48922385E−09 |
| H0 | 4.06415374E−02 | 1.08883798E−08 | 1.31287871E−01 |
| H1 | 4.06415377E−02 | −3.70556926E−08 | −1.31287862E−01 |
| O1 | −8.00748724E−02 | −1.52539157E−08 | 2.13908078E−08 |
| H2 | 3.74383448E−02 | 9.81842284E−09 | 1.35165239E−01 |
| H3 | 3.74383537E−02 | −1.84099712E−09 | −1.35165261E−01 |

Derived dispersion force \(F_{\mathrm{disp}} = F(\mathrm{D3}) -
F(\mathrm{plain})\); largest component **2.328E−04 Ha/Bohr** (O1, \(x\)).
Atomiverse reproduces it to **7.8E−07 Ha/Bohr**.

## Placement (why the coordinates are shifted)

CP2K's coordinates put O0 at the cell **origin**. Atomiverse's
Martyna–Tuckerman cluster path does **not** wrap the collocation, so a Gaussian
centred near a face is clipped: the energy stops being translation invariant and
the ionic forces pick up a huge spurious boundary term while the total energy
still looks plausible. Measured on this very system: with CP2K's origin-based
coordinates, `max|F_atv − F_cp2k| = 31.9 Ha/Bohr`; after translating the dimer to
the cell centre, **7.5e−3 Ha/Bohr**. The same trap produced the garbage phonon
spectrum in `gpw-fd-phonon-si` (atom at (0,0,0), \(|\sum F| = 1.98\) Ha/Bohr).

Two consequences landed with this pin:

1. `GpwDft::build` now **throws** when `nonperiodic == true` and any nucleus is
   outside the inner 70 % of an axis, naming the fix. The silent-wrong-answer
   path is gone.
2. This regression asserts \(\lvert\sum_A F_A\rvert\) directly, so the class of
   bug is gated, not just documented.

An isolated system's absolute position is physically irrelevant, so shifting the
molecule does not invalidate the CP2K pin.

## Measured results (2026-07-26, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **6.4e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **7.82e−07** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **7.47e−03** |
| \(\lvert\sum_A F_A\rvert\) | 1e−3 Ha/Bohr | **2.22e−04** |
| Product wiring | 1e−7 Ha/Bohr | **9.1e−11** |
| Wall | ≤ 60 s | **17.8 s** |

## Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \(E_{\mathrm{disp}}\) vs CP2K | **1e−8 Ha** | Same closed-form two-body sum and same published PBE parameters; measured agreement 6.4e−11 Ha. Nothing but table transcription can move it. |
| \(F_{\mathrm{disp}}\) vs CP2K difference | **5e−6 Ha/Bohr** | The pin is a difference of two independently converged SCFs (`EPS_SCF 1e-7`), so it inherits CP2K's own force noise; measured floor 7.8e−7 against a 2.3e−4 signal (≥ 290× margin). |
| Total \(F\) vs CP2K | **1e−2 Ha/Bohr** | Bounded below by the **oracle's** grid error: CP2K's own `FORCES| Sum` is 5.42e−3 Ha/Bohr here, i.e. its force field is not translation invariant at `CUTOFF 300`. Atomiverse's raw disagreement falls 7.5e−3 → 3.2e−3 and its own \(\lvert\sum F\rvert\) 2.2e−4 → 7.6e−5 when the cutoff goes 300 → 500 Ry — discretisation on both sides, not a defect. A 1e−3 component gate would be pinning CP2K's grid noise. |
| \(\lvert\sum_A F_A\rvert\) | **1e−3 Ha/Bohr** | Translation invariance of an isolated system. Measured 2.2e−4; the placement bug it guards shows \(\sim\)1–2 Ha/Bohr, so ~5000× separation. |
| Product wiring | **1e−7 Ha/Bohr** | \(F(\mathrm{D3}) - F(\mathrm{plain})\) inside Atomiverse against its own standalone gradient — same code, so only the SCF residual enters. |
| Wall | **≤ 60 s** | e2e tier budget; measured 17.8 s. |

## Dual gate

- **Correctness:** the six gates above, plus the analytic-vs-FD gradient
  identity in `tests/cpu/gpw/test_cpu_gpw_forces.cpp`
  (`GpwForces.AnalyticMatchesFdWithD3`, measured 2.4e−6 Ha/Bohr on cheap MT
  H₂ — 6N SCFs on this cell exceed the e2e budget).
- **Speed:** CP2K twin wall ≈ 12.3 s for the same `ENERGY_FORCE`; Atomiverse
  17.8 s on the same host → ratio ≈ **1.4**, inside the 10× fair wall. The
  Atomiverse figure includes one extra plain SCF that only the wiring gate
  needs.

## Cutoff convention (divergence found and fixed with this pin)

The zero-damping path used a hard-coded **40 Bohr** pair cutoff in five places
(`gpw_solver.cpp`, `gpw_forces.cpp` Γ + k, `gpw_stress.cpp` Γ + k) while the
D3(BJ) path used CP2K's documented `2 × R_CUTOFF` convention. An isolated
cluster cannot see the difference, so no existing pin caught it. Settled against
the oracle with a dedicated **periodic** D3-zero CP2K run
(`E_disp = −8.20894676620e−04 Ha`):

| Atomiverse `rcut` [Bohr] | \(E_{\mathrm{disp}}\) [Ha] | vs CP2K |
| --- | --- | --- |
| 37.794 (= `R_CUTOFF` alone) | −8.20410639089e−04 | 4.8e−07 |
| **40 (old hard-coded)** | −8.20508087366e−04 | **3.9e−07** |
| **75.589 (= 2 × `R_CUTOFF`)** | −8.20894743608e−04 | **6.7e−11** |
| 150 | −8.20959128511e−04 | 6.4e−08 |

All five sites now share `kD3Cp2kPairCutoffBohr` (`internal/dftd3.h`), so
energy, force and stress differentiate the same functional — the failure mode
that previously shipped as a 0.53 Ha/Bohr ZnO force error.

## Notes

- Sibling dispersion-force product: `regressions/gpw/gpw-d4-forces`.
- Sibling D3 energy products: `gpw-d3-atm` (isolated, +ATM),
  `gpw-pbe-d3bj-ice` (periodic, BJ).
- Periodic BJ force twin: `regressions/gpw/gpw-d3bj-forces`.
