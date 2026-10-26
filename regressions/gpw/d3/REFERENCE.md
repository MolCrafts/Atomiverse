# gpw/d3

One case; every section below is run by `atv_regressions --case gpw/d3` and keeps its own oracle record.

The complete section budget is **60 s at its registered thread width**, including
plain/finite-difference twins and standalone checks. There are no dispersion
timing exemptions. Dated timing tables below are historical observations;
different grids, core counts and calculation boundaries do not establish a
matched-engine speedup. Numerical pins and tolerances remain unchanged.


## Reference — isolated water-dimer GPW PBE-D3 (zero damping) analytic forces

### Method

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

### Why the pin is a *difference* of two CP2K runs

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

### System

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

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-forces/water_dimer_d3_forces.inp` and `water_dimer_plain_forces.inp` |
| Wall | ≈ **12.3 s** (D3 run), ≈ **12.5 s** (plain run) |
| Date | 2026-07-26 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.337397876368769 |
| Total (D3) | −34.338211868500807 |
| **Dispersion energy** | **−0.00081399213205** |

Atomiverse `dftd3_pbe` on the same nuclei: **−8.13992196100576e−04 Ha**
(\(|\Delta| = 6.4\times10^{-11}\) Ha).

#### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

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

### Placement (why the coordinates are shifted)

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

### Measured results (2026-07-26, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **6.4e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **7.82e−07** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **7.47e−03** |
| \(\lvert\sum_A F_A\rvert\) | 1e−3 Ha/Bohr | **2.22e−04** |
| Product wiring | 1e−7 Ha/Bohr | **9.1e−11** |
| Wall | ≤ 60 s | **17.8 s** |

### Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \(E_{\mathrm{disp}}\) vs CP2K | **1e−8 Ha** | Same closed-form two-body sum and same published PBE parameters; measured agreement 6.4e−11 Ha. Nothing but table transcription can move it. |
| \(F_{\mathrm{disp}}\) vs CP2K difference | **5e−6 Ha/Bohr** | The pin is a difference of two independently converged SCFs (`EPS_SCF 1e-7`), so it inherits CP2K's own force noise; measured floor 7.8e−7 against a 2.3e−4 signal (≥ 290× margin). |
| Total \(F\) vs CP2K | **1e−2 Ha/Bohr** | Bounded below by the **oracle's** grid error: CP2K's own `FORCES| Sum` is 5.42e−3 Ha/Bohr here, i.e. its force field is not translation invariant at `CUTOFF 300`. Atomiverse's raw disagreement falls 7.5e−3 → 3.2e−3 and its own \(\lvert\sum F\rvert\) 2.2e−4 → 7.6e−5 when the cutoff goes 300 → 500 Ry — discretisation on both sides, not a defect. A 1e−3 component gate would be pinning CP2K's grid noise. |
| \(\lvert\sum_A F_A\rvert\) | **1e−3 Ha/Bohr** | Translation invariance of an isolated system. Measured 2.2e−4; the placement bug it guards shows \(\sim\)1–2 Ha/Bohr, so ~5000× separation. |
| Product wiring | **1e−7 Ha/Bohr** | \(F(\mathrm{D3}) - F(\mathrm{plain})\) inside Atomiverse against its own standalone gradient — same code, so only the SCF residual enters. |
| Wall | **≤ 60 s** | e2e tier budget; measured 17.8 s. |

### Dual gate

- **Correctness:** the six gates above.
- **Speed:** CP2K twin wall ≈ 12.3 s for the same `ENERGY_FORCE`; Atomiverse
  17.8 s on the same host → ratio ≈ **1.4**, recorded for historical context only. The
  Atomiverse figure includes one extra plain SCF that only the wiring gate
  needs.

### Cutoff convention (divergence found and fixed with this pin)

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

### Notes

- Sibling dispersion-force product: `regressions/gpw/d4`.
- Sibling D3 energy products: `gpw-d3-atm` (isolated, +ATM),
  `gpw-pbe-d3bj-ice` (periodic, BJ).
- Periodic BJ force twin: `regressions/gpw/d3`.


## Reference — periodic water-dimer GPW PBE-D3 (zero damping) analytic cell stress

### Method

Γ-point **GPW** (PBE) with additive **DFT-D3 zero ("original") damping**
(PBE reference parameters, two-body only, `CALCULATE_C9_TERM .FALSE.`) on a
**periodic** water dimer.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe` sum as the energy path,
   `kD3Cp2kPairCutoffBohr` = 2 × CP2K `R_CUTOFF 20.0` Å lattice-image cutoff).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `d3=true`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

### Why the stress pin is a *difference* of two CP2K runs

D3 is geometry-only: it adds \(E_{\mathrm{disp}}(\{R\},\mathbf{h})\) to the
total energy and does not enter the Kohn–Sham potential, so the converged
density is identical with and without it. Therefore

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3}) - \sigma^{\mathrm{CP2K}}(\mathrm{plain})
  = \sigma_{\mathrm{disp}}^{\mathrm{D3}}
\]

isolates exactly the quantity this capability adds, with the (much larger,
grid-noise-limited) electronic stress cancelling — the stress analogue of the
`gpw-d3-forces` force pin. Atomiverse isolates its own dispersion stress the
same way: analytic product with `d3=true` minus `GpwDft::stress` with
`d3=false` (its own SCF; dispersion is folded after the SCF, so the
converged density is the `d3=true` run's density).

### System

Same geometry as `gpw-d3-forces` (periodic twin) / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D3 zero damping, PBE: \(s_6=1.0\), \(s_8=0.722\), \(s_{r,6}=1.217\), \(s_{r,8}=1.0\); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr (periodic — the convention **does** move this pin; see gpw-d3-forces "Cutoff convention") |
| Cutoff | 200 Ry (ATV single grid; CP2K `CUTOFF 200, REL_CUTOFF 40, NGRIDS 4`) — `gpw-d4-stress` cost family. E_disp and Δσ_disp are geometry-only ⇒ cutoff-independent |
| SCF | CP2K `EPS_SCF 1e-7`; Atomiverse `tol = 1e-7` |
| Stress | CP2K `STRESS_TENSOR NUMERICAL`; ATV analytic fixed-P `h=2e-3` |

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`, git:5e54ba2) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-stress/water_dimer_d3_stress.inp` and `water_dimer_plain_stress.inp` |
| Wall | **33.6 s** (D3 run), **38.9 s** (plain run) — full NUMERICAL stress (many strained SCFs) |
| Date | 2026-07-31 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.338293290245112 |
| Total (D3) | −34.339114184921726 |
| **Dispersion energy** | **−0.00082089467662** |

(Consistent with the periodic D3-zero energy pin −8.20894676620e−04 Ha taken
at CUTOFF 300 for `gpw-d3-forces` — geometry-only, cutoff-independent.)

#### Numerical stress tensors (CP2K prints **bar**; diagonal)

| Axis | σ(D3) [bar] | σ(plain) [bar] | Δσ = D3 − plain [bar] |
| ---- | ----------- | -------------- | --------------------- |
| xx | 5.44389060708×10³ | 5.44891104191×10³ | **−5.02043483** |
| yy | −1.06513889194×10² | −1.06247934255×10² | **−0.26595494** |
| zz | 1.28102626350×10⁴ | 1.28091614698×10⁴ | **+1.10116520** |

Unit conversion (CP2K internal table):
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

**Hard Δσ pin [Ha/Bohr³]:** xx **−1.70641140246×10⁻⁸**,
yy **−9.03962616420×10⁻¹⁰**, zz **+3.74278507120×10⁻⁹**.

**Soft total σ(D3) pin [Ha/Bohr³]:** xx 1.85034111988×10⁻⁵,
yy −3.62033411836×10⁻⁷, zz 4.35412050331×10⁻⁵.

#### Re-run (offline only — never ctest)

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
cd tools/cp2k_oracle/inputs/gpw-d3-stress
OMP_NUM_THREADS=2 mpirun -np 2 cp2k.psmp -i water_dimer_d3_stress.inp -o d3.out
OMP_NUM_THREADS=2 mpirun -np 2 cp2k.psmp -i water_dimer_plain_stress.inp -o plain.out
```

### Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − pin\| | **1×10⁻⁸ Ha** | Same closed-form two-body sum, same published PBE parameters, same 2×R_CUTOFF image convention; `gpw-d3-forces` measured 6.4×10⁻¹¹ agreement. |
| max\|Δσ_atv − Δσ_cp2k\| diag | **2×10⁻⁹ Ha/Bohr³** | Both sides are differences of two runs sharing one density fixed point. CP2K-side noise: NUMERICAL FD with EPS_SCF 1e-7 → ≲10⁻¹¹ Ha/Bohr³ (off-diagonals of the printed tensors, ~10⁻⁶ bar, bound it). ATV-side noise: two independently converged SCFs (tol 1e-7) enter the fixed-P Lagrangian → measured residual below; gate leaves ≥5× headroom over the xx signal 1.7×10⁻⁸ while still 8× above the measured floor. |
| max\|σ_an − σ_fd\| diag (d3=true) | **2×10⁻⁵ Ha/Bohr³** | Internal dual gate, `gpw-d4-stress` precedent: fixed-P Lagrangian vs full re-SCF differ by SCF residual + O(h²). |
| off-diagonal product | **1×10⁻⁸** | Orthorhombic cell: shear identically 0 by construction. |
| max\|σ_diag\| floor | **>1×10⁻¹⁰ Ha/Bohr³** | Stress actually computed, not zero-filled. |
| max\|σ_tot − σ_cp2k\| diag | **1×10⁻⁴ Ha/Bohr³** (sanity) | ATV single 200 Ry grid vs CP2K NGRIDS 4 / REL_CUTOFF 40 differ at the grid-discretisation level (several ×10⁻⁵ Ha/Bohr³ ≈ kbar on this cell — same class as the `gpw-d3-forces` total-force gate being bounded by CP2K's own grid noise). The gate catches unit/sign/blow-up errors only; the decisive stress physics is the Δσ pin above. |

### Atomiverse numbers (first green, 2026-07-31)

`reg_gpw-d3-stress` (OMP 8, epyc login node), 11 SCF iters,
\(E_{\mathrm{tot}}\) = −34.331152296855 Ha:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \|E_disp − pin\| | 1e−8 Ha | **5.53e−11** (E_disp = −8.20894731919e−04 Ha) |
| diag \|Δσ_atv − Δσ_cp2k\| | 2e−9 Ha/Bohr³ | **2.27e−10** (xx), 2.26e−12 (yy), 1.92e−11 (zz) |
| dual gate max\|σ_an − σ_fd\| diag | 2e−5 Ha/Bohr³ | **7.39e−07** (xx; yy 2.46e−8, zz 1.24e−9) |
| max\|σ_tot − σ_cp2k\| diag (sanity) | 1e−4 Ha/Bohr³ | 4.02e−05 |
| off-diagonal product | 1e−8 | ~0 (pass) |
| max\|σ_diag\| floor | >1e−10 | 5.83e−5 |
| Wall (product + plain twin + all-axes FD twin) | ≤ 60 s | **18.5 + 14.9 + 14.3 = 47.7 s** |

Product analytic diagonals [Ha/Bohr³]: xx −2.16997707×10⁻⁵,
yy +2.00723305×10⁻⁵, zz +5.82684354×10⁻⁵.

The Δσ residual (2.27e−10 on xx) sits at the SCF-noise floor of the
**two-full-products** isolation — both analytic runs re-converge to the same
density fixed point only to `tol = 1e−7`, and the residual difference enters
the fixed-\(P\) Lagrangian FD as δE/(2hΩ). The sibling `gpw-d3bj-stress`
reaches 3.9e−14 because its Δσ side is a strain-FD of the standalone
(no-SCF) dispersion sum; this case deliberately pays the noise to gate the
**product path fold** end-to-end — σ_an(d3) − σ_an(plain) runs
`GPWEvaluator`/`GpwDft::stress` twice. Signal separation stays decisive:
xx signal 1.71e−8 = 75× the measured residual, tolerance 2e−9 keeps
8.8× headroom.

### Dual gate

- **Correctness:** the six gates above, including the section's complete
  analytic-fixed-density versus reconverged strain-FD comparison.
- **Speed:** CP2K NUMERICAL stress twin wall **33.6 s** (np 2 × OMP 2); the
  historical ATV product wall is SCF + fixed-P stress —
  **18.5 s** (OMP 8) → ratio **0.55**, recorded for historical context only. The
  regression's extra plain/FD twins (14.9 + 14.3 s) exist only for the
  difference pin and the dual gate, not the product path. Settings caveat:
  cutoffs and core counts differ as stated (ATV single 200 Ry grid, OMP 8 vs
  CP2K NGRIDS 4, 2×2) — both walls deliver the same observable (this cell's
  stress), which is what is compared.

### Notes

- Sibling dispersion-stress products: `regressions/gpw/d4`
  (D4, dual-gate only — no CP2K DFTD4 build), `gpw-mbd-stress`.
- Sibling D3-zero pins: `gpw-d3-forces` (isolated forces + periodic E_disp),
  `gpw-d3-atm` (isolated, +ATM energy).
- D3(BJ) stress is the separate queue row `gpw-d3bj-stress` (shared fold —
  serialized per queue surface note).


## Reference — GPW DFT-D3 three-body ATM (C9)

### Method

Grimme **DFT-D3** (zero damping, PBE) with the **Axilrod–Teller–Muto
three-body C9** term, as an additive correction on Γ-point **GPW**.

Public API:

- `cpu::gpw::GpwDft::d3 = true`
- `cpu::gpw::GpwDft::d3_atm = true`  (CP2K `CALCULATE_C9_TERM .TRUE.`)

Dispersion energy is on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Implementation: `dftd3_pbe(..., calculate_c9=true)` matching
CP2K `qs_dispersion_d3` (`s9 = s6`, CN-dependent \(C_6\), damping
`damping_d3(rabc, r0, 4/3, 16)`).

### Reference value (hard gate)

**E_disp (PBE-D3 + ATM, isolated water dimer) = −0.00081414118193 Ha**

Source: offline **CP2K 2026.1** (`2026.1-gmpflf-2026.03` on Arrhenius), twin
`tools/cp2k_oracle/inputs/gpw-d3-atm/water_dimer_d3_atm.inp`.

```text
Dispersion energy:                                           -0.00081414118193
```

Two-body-only baseline on the same geometry (ATM off):
**−0.00081399213205 Ha** (existing unit-test pin). ATM Δ ≈ **−1.49×10⁻⁷ Ha**.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC NONE** (isolated / MT).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Oracle: CP2K 2026.1 (offline — never launched from ctest)

```bash
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d3-atm
mpirun -np 1 cp2k.psmp -i water_dimer_d3_atm.inp -o water_dimer_d3_atm.out
## rg 'Dispersion energy' water_dimer_d3_atm.out
```

CP2K wall (this pin, 1 rank × 4 OMP): **T_cp2k ≈ 10.8 s** (TIMING `CP2K`
TOTAL TIME).

### Tolerance

**1 × 10⁻⁹ Ha** on E_disp.

### What is checked

`GpwDft` with `d3 = true` and `d3_atm = true` returns
`last_result().e_disp` within 1e-9 Ha of −0.00081414118193 Ha. Unit tests
check analytic ATM forces vs FD and that default `d3_atm = false` keeps the
two-body pin.

### Perf gate

The complete section must finish within 60 s. CP2K's full QS wall and a
dispersion-only sum have different calculation boundaries, so they do not
supply a matched performance comparison.


## Reference — GPW DFT-D3 three-body ATM (C9) analytic ionic forces

### Method

Grimme **DFT-D3** (zero damping, PBE) with the **Axilrod–Teller–Muto
three-body C9** term (`CALCULATE_C9_TERM .TRUE.`) contributing to the GPW
**ionic forces**: explicit-geometry ATM gradient plus the coordination-number
chain (`accumulate_atm_c9` → `dEdCN`), the exact gradient of the same
\(E_{\mathrm{ATM}}\) the energy foothold (`gpw-d3-atm`) pins.

Public path: `cpu::gpw::GPWEvaluator` / `GpwDft` with `d3 = true`,
`d3_atm = true`, `compute_forces = true`. Implementation:
`dftd3_pbe(..., calculate_c9 = true)` matching CP2K `qs_dispersion_d3`
(`doabc`, s9 = s6, damping `damping_d3(rabc, r0, 4/3, 16)`).

### Why the pin is a difference of two CP2K runs

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

### System

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

### Offline CP2K pins

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03` on Arrhenius) |
| Command | `mpirun -np 2 cp2k.psmp -i <input>.inp -o <input>.out`, `OMP_NUM_THREADS=2`, offline only |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-forces/water_dimer_d3_atm_iso_forces.inp` (iso, C9 T), `tools/cp2k_oracle/inputs/gpw-d3-forces/water_dimer_d3_forces.inp` (iso, C9 F, rerun), `water_dimer_d3_atm_per6_forces.inp` / `water_dimer_d3_per6_forces.inp` (per 6 Å, C9 T/F) |
| Output lines | `Dispersion energy:` + `FORCES\| Atomic forces [hartree/bohr]` table |
| Wall | iso ≈ 10.5 / 10.4 s; per ≈ 1.8 / 2.0 s (TIMING `CP2K` TOTAL TIME) |
| Date | 2026-07-31 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Iso, C9 on: total / **E_disp** | −34.338212017550674 / **−0.00081414118193** |
| Iso, C9 off: total / E_disp | −34.338211868500814 / −0.00081399213205 |
| Per 6 Å, C9 on: total / **E_disp** | −34.345965809806160 / **−0.00195809486938** |
| Per 6 Å, C9 off: total / **E_disp** | −34.345966594779618 / **−0.00195887984277** |

Iso values reproduce the `gpw-d3-atm` / `gpw-d3-forces` energy pins exactly.
ATM contribution: iso −1.4905e−7 Ha (attractive), per +7.8497e−7 Ha
(repulsive — the condensed-phase sign).

#### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` tables)

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

### Tolerances (physical justification)

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

### Measured (2026-07-31, this pin)

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

### Two defects this pin exposed (fixed with it)

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

### Dual gate

- **Correctness:** the pins above, plus the analytic-vs-FD ATM gradient
  identity in `tests/cpu/gpw/test_cpu_gpw_dftd3.cpp`:
  `GpwDftD3Atm.AnalyticForcesMatchFiniteDifference` (isolated, pre-existing,
  4.1e−14) and `GpwDftD3Atm.PeriodicAtmForcesMatchFiniteDifference` (new —
  FD of the ATM-only energy difference over the periodic image classes,
  2.2e−13 Ha/Bohr).
- **Speed:** CP2K ATM iso `ENERGY_FORCE` twin ≈ 10.5 s (2 ranks × 2 OMP);
  the regression (one 300 Ry MT SCF + **two** force assemblies + four
  standalone D3 sums) measured 34.5 s at 8 threads on the same host →
  ratio ≈ **3.3**, recorded for historical context only. The second force assembly
  exists only for the wiring gate.

### Notes

- The standalone `D3Result::virial` omits the direct three-body ATM term.
  GPW cell stress differentiates the complete dispersion energy under strain;
  its existing `gpw-d3-atm-stress` section checks that path. Those checks do
  not qualify the standalone ATM virial field.
- Energy foothold: `regressions/gpw/d3` (isolated E_disp only).
- Two-body force twin: `regressions/gpw/d3` (same isolated cell;
  its D3 run was rerun here so both members of the difference share one
  parallelization).
- Sibling (serialized on the force-assembly ATM fold): `gpw-d4-atm-forces`.
- The periodic pin deliberately uses `R_CUTOFF 10.0`: at the CP2K default
  (2 × 20 Å = 75.6 Bohr) the 6 Å-cell triple enumeration is ~40× more
  images for the same code coverage; both sides of the twin use the same
  doubled-cutoff convention (`Cutoff Radius [Bohr]: 18.90` printed = half).


## Reference — GPW PBE-D3 three-body ATM (C9) analytic cell stress

### Method

Γ-point **GPW** (PBE) on the **periodic 6 Å** water-dimer cell of
`gpw-d3-atm-forces`, with Grimme **DFT-D3** (zero damping, PBE) including the
**Axilrod–Teller–Muto three-body C9** term (`CALCULATE_C9_TERM .TRUE.`)
folded into the **cell stress**.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`cpu::gpw::GPWEvaluator` with `d3 = true`, `d3_atm = true`,
`compute_stress = true`, `stress_use_fd = false`):

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian \(E_L[P;W;\varepsilon]\)
   under orthorhombic diagonal strain at fixed fractional coordinates, with
   the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (`dftd3_pbe(..., calculate_c9 = true)`, the
   same call as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

This regression also ships the configurable **`d3_rcut_bohr`** member
(`GpwDft` / `GPWEvaluator`): CP2K's `&PAIR_POTENTIAL R_CUTOFF` (doubled
convention, stored in Bohr; default = CP2K default 20 Å → 75.59 Bohr =
`kD3Cp2kPairCutoffBohr`). The pinned twin uses `R_CUTOFF 10.0` → 37.79 Bohr,
so the product path folds **exactly** the oracle's functional; energy, force
and stress all read the one member.

Sibling twins: `gpw-d3-atm-forces` (same geometry, ATM force pins — including
the CASE-3 image-triple and pbc-anchoring fixes this stress fold inherits),
`gpw-d3bj-stress` (two-body BJ stress pin, 12 Å cell — the pattern this case
follows), `gpw-d4-atm-forces` (D4 ATM forces; no CP2K D4 stress pin —
Arrhenius CP2K lacks libdftd4).

### Why the pin is a *difference* of two CP2K runs

DFT-D3 — with or without C9 — is geometry-only and never enters the
Kohn–Sham potential, so at **every strained cell** the converged density is
identical with `CALCULATE_C9_TERM` `.TRUE.` and `.FALSE.`, and the two-body
dispersion energy is identical too. CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{C9})-\sigma^{\mathrm{CP2K}}(\mathrm{2b})
 = -\Omega^{-1}\,\partial E_{\mathrm{ATM}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure three-body ATM stress: the (three-orders-larger) electronic stress,
the two-body dispersion stress, **and** the shared parts of CP2K's FD scheme
cancel in the difference. Total-energy difference of the two runs reproduces
\(E_{\mathrm{disp}}(\mathrm{C9})-E_{\mathrm{disp}}(\mathrm{2b})\) to
4.7e−15 Ha — the electronic cancellation is exact to solver noise.

### System

Verbatim the merged `gpw-d3-atm-forces` periodic twin:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **6 × 6 × 6 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3 zero damping, PBE: s6 = 1.0, s8 = 0.722, sr6 = 1.217, sr8 = 1.0; s9 = s6; ATM per run (`CALCULATE_C9_TERM` T/F) |
| Pair/ABC cutoff | `R_CUTOFF 10.0` Å → doubled 37.794 Bohr (`d3_rcut_bohr`; CP2K prints half: 18.90) |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 150 Ry (both dispersion gates are geometry-only ⇒ cutoff-independent; the dual gate is grid-consistent by construction) |
| SCF | CP2K `EPS_SCF 1e-8` (tightened vs the 1e-6 forces twin — lowers the FD-difference noise floor); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); in-regression ATM-only FD at **exactly CP2K's step** (`&NUMERICAL_STRESS` default `DX 0.001` Bohr ⇒ \(\varepsilon = DX/L = 8.8196\times10^{-5}\)) — see the step-sensitivity note below |

Strain at fixed fractional coordinates leaves the `accumulate_atm_c9` ANINT
wrap decisions invariant (positions and cell scale together), and the
truncated-boundary-set anchoring moves the ATM **energy** only at ~4e−19 Ha
(`gpw-d3-atm-forces` REFERENCE) ⇒ ≲3e−18 Ha/Bohr³ in the FD — far below
every gate here.

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-atm-stress/water_dimer_d3_atm_per6_stress.inp` (C9 T) and `water_dimer_d3_per6_stress.inp` (C9 F) |
| Output lines | final `Dispersion energy:` + `STRESS| Numerical stress tensor [bar]` table (20 dispersion evaluations per run: reference + 18 perturbed + final) |
| Wall | 54.5 s (C9 run), 18.0 s (two-body run) — CP2K internal timer |
| Date | 2026-07-31 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (C9, per6) | −34.345965809817606 |
| Total (two-body, per6) | −34.345966594791001 |
| **E_disp (C9)** (printed) | **−0.00195809486938** |
| **E_disp (two-body)** (printed) | **−0.00195887984277** |

Both reproduce the `gpw-d3-atm-forces` periodic energy pins exactly (print
precision). ATM contribution +7.8497339e−7 Ha (repulsive — condensed-phase
sign); the total-energy difference reproduces it to 4.7e−15 Ha.

#### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | C9 run | two-body run | Δ = C9 − 2b [bar] | Δ [Ha/Bohr³] |
| --------- | ------ | ------------ | ----------------- | ------------- |
| xx | 4.69756890548E+04 | 4.69807342693E+04 | −5.04521450 | **−1.7148335571E−08** |
| yy | 2.34423739140E+03 | 2.33148843324E+03 | +12.74895816 | **+4.3332828109E−08** |
| zz | 1.00716094815E+05 | 1.00713419919E+05 | +2.67489600 | **+9.0917867267E−09** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). Off-diagonals
of both tables are ≤ 2.1e−5 bar (≈ 7e−14 Ha/Bohr³) — the numerical-stress
noise floor. Note the ATM stress is strongly anisotropic (yy largest) even
though the dimer axis is x: the compressed 6 Å cell's image triples dominate.

### Measured results (2026-07-31, this pin; `run_reg` target, OMP 8, loaded login node)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| E_disp (C9) vs CP2K (standalone) | 1e−9 Ha | **2.79e−10** |
| E_disp (two-body) vs CP2K | 1e−9 Ha | **2.83e−10** |
| E_disp (C9) product (`GPWEvaluator`) | 1e−9 Ha | **1.18e−10** |
| diag σ_ATM vs CP2K Δ | 5e−12 Ha/Bohr³ | **1.13e−13** (xx; yy 1.14e−14, zz 3.68e−14) |
| ATM engage floor max \|σ_ATM\| | ≥ 1e−9 Ha/Bohr³ | 4.33e−8 |
| dual gate \|σ_an − σ_fd\| (xx in e2e) | 2e−5 Ha/Bohr³ | **2.38e−7** |
| off-diagonal product | 1e−8 Ha/Bohr³ | ~0 (pass) |
| max \|σ_diag\| floor | > 1e−10 Ha/Bohr³ | 2.26e−4 |
| Wall (product + x-only FD twin) | ≤ 60 s | **40.8 s + 7.8 s** (9 SCF iters; load ≈ 22/48 cores) |

Product analytic diagonals (150 Ry, `tol = 1e−6`,
\(E_{\mathrm{tot}}\) = −34.301294491640 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | +6.92003753×10⁻⁵ |
| \(\sigma_{yy}\) | −1.57391519×10⁻⁴ |
| \(\sigma_{zz}\) | +2.25590855×10⁻⁴ |

### The ATM strain-FD is step-locked (re-pin warning)

The two-body D3 sum passes a **smooth** fifth-order cutoff, so its strain-FD
is step-independent to ~1e−13 (`gpw-d3bj-stress` matched CP2K at a different
step). The ATM triple enumeration instead passes **hard** gates —
`r² < r_cut²` on each leg plus CP2K's empirical scaling gate
`(r²_AB r²_BC r²_CA) > 0.01 r_cut⁶` — so the included boundary set jumps by
~5e−11 Ha at discrete strains. An FD window that differs from the oracle's
straddles different gate crossings: at \(h = 10^{-4}\) the xx/zz components
were off by 1.6e−10 Ha/Bohr³ (staircase, not noise), while at exactly CP2K's
`DX 0.001` Bohr step the pin reproduces to 1.1e−13. Any re-pin must use the
oracle's `DX` (and note it if CP2K ever changes the default).

### Tolerance justification

- **E_disp 1e−9 Ha:** closed-form D3 + ATM sum with published parameters and
  the CP2K CN convention; the same cell measured 2.8e−10 in the
  `gpw-d3-atm-forces` pin. Only table transcription can move it. Also gates
  the periodic image-triple enumeration (CASE 1/2/3) under the stress input.
- **σ_ATM 5e−12 Ha/Bohr³:** the pin is a difference of two independently
  converged CP2K numerical-stress runs at `EPS_SCF 1e−8` sharing one FD
  scheme — electronic + two-body parts cancel to the print floor (~7e−14,
  the off-diagonal residuals). The residual at the shared `DX` step measured
  1.13e−13 (xx) against the 9.1e−9…4.3e−8 signals; 5e−12 sits ~40× above
  the measured residual and keeps ~1800× separation from the smallest
  signal, while absorbing plausible gate-crossing jitter of a re-pin on a
  different host (≲1e−12, the observed near-step spread).
- **Dual gate 2e−5 Ha/Bohr³:** same bound as `gpw-d3bj-stress` (measured
  7.4e−7 there, 2.38e−7 here): analytic fixed-P Lagrangian vs a fully
  re-converged SCF at `tol = 1e−6` — the residual is SCF-convergence noise,
  not the ATM fold.

### Dual gate

- **Correctness:** the pins above and the section's analytic-fixed-density
  versus reconverged strain-FD comparison on the same `d3 + d3_atm` method
  along x.
- **Speed:** CP2K numerical-stress C9 twin 54.5 s (18 perturbed SCFs +
  reference, 2 MPI × 2 OMP, 300 Ry); Atomiverse product analytic pass
  **40.8 s** (one SCF + 7 fixed-P grid evaluations, OMP 8, 150 Ry, load ≈
  22/48 login node) → ratio ≈ **0.75**, recorded for historical context only. Settings
  caveat: cutoffs and core counts differ as stated — both produce the same
  deliverable (the ATM-folded cell stress of this system), which is the wall
  being compared.

### Notes

- The ATM term still contributes **no** `D3Result::virial` entry; the
  fixed-P stress path never consumes the virial — it rebuilds the dispersion
  energy at each strained cell, which is the fold this pin certifies. A
  future virial-path consumer (e.g. MD barostat coupling through the pair
  virial) would be its own queue row.
- `gpw-d4-atm-stress` is the D4 sibling (no CP2K pin possible — libdftd4
  absent; dftd4-CLI strained-cell energies policy, like `gpw-d4-atm-forces`).
- The periodic pin deliberately keeps `R_CUTOFF 10.0` from the forces twin:
  at the CP2K default (2 × 20 Å = 75.6 Bohr) the 6 Å-cell triple enumeration
  is ~40× more images for the same code coverage; both members of the pair
  and the ATV product use the same doubled-cutoff convention via
  `d3_rcut_bohr`.


## Reference — periodic water-dimer GPW PBE-D3(BJ) analytic ionic forces

### Method

Γ-point **GPW** (PBE) on a **periodic** water dimer (`PERIODIC XYZ`) with
additive **DFT-D3 Becke–Johnson (rational) damping**, PBE parameters, two-body
only (`CALCULATE_C9_TERM .FALSE.`).

\[
E_{\mathrm{disp}} = -\sum_{A<B}\Bigl[\frac{s_6 C_6^{AB}}{R^6 + f^6}
  + \frac{s_8 C_8^{AB}}{R^8 + f^8}\Bigr],\qquad
f = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2
\]

Product ionic forces = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** the geometry-only dispersion gradient
\(-\nabla E_{\mathrm{disp}}\) from `cpu::gpw::dftd3_pbe_bj`, summed over lattice
images inside the CP2K-parity pair cutoff.

Public path: `cpu::gpw::GPWEvaluator` with `d3bj = true`,
`compute_forces = true`, `nonperiodic = false`.

Sibling: `gpw-d3-forces` (isolated cluster, zero damping). Together they cover
both damping functions × both boundary conditions for the dispersion force.

### Why the pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so the
converged density is identical with and without it. Hence

\[
F_A^{\mathrm{CP2K}}(\mathrm{D3BJ}) - F_A^{\mathrm{CP2K}}(\mathrm{plain})
  = -\nabla_A E_{\mathrm{disp}},
\]

which isolates exactly the quantity this capability adds; the much larger
electronic force cancels.

### System

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, **`PERIODIC XYZ`** |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN form (`cp2k_cn = true`); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr |
| Cutoff | 300 Ry (CP2K `CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| SCF | `EPS_SCF 1e-7` / Atomiverse `tol = 1e-7` |

Placement is verbatim CP2K here: a periodic cell wraps by minimum image, so the
absolute position is immaterial. (The isolated MT sibling must be centred — see
`gpw-d3-forces/REFERENCE.md` "Placement".)

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-forces/water_dimer_d3bj_forces.inp` and `water_dimer_plain_forces.inp` |
| Wall | ≈ **11.9 s** (D3BJ run), ≈ **13.5 s** (plain run) |
| Date | 2026-07-26 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain, periodic) | −34.338251977942306 |
| Total (D3BJ, periodic) | −34.339628255637152 |
| **Dispersion energy** | **−0.00137627769481** |

Atomiverse `dftd3_pbe_bj` on the same nuclei: **−1.37627778350509e−03 Ha**
(\(|\Delta| = 8.9\times10^{-11}\) Ha).

#### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

D3(BJ) run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.18231267E−02 | 3.13886473E−08 | −7.96369314E−09 |
| H0 | 4.08872662E−02 | 1.13178277E−08 | 1.31270247E−01 |
| H1 | 4.08872656E−02 | −3.52097593E−08 | −1.31270237E−01 |
| O1 | −8.05124430E−02 | −1.50606809E−08 | 2.18547987E−08 |
| H2 | 3.75636473E−02 | 1.02088685E−08 | 1.35137187E−01 |
| H3 | 3.75636570E−02 | −2.12332662E−09 | −1.35137209E−01 |

Plain run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.19227425E−02 | 3.13886469E−08 | −7.96370458E−09 |
| H0 | 4.08225109E−02 | 1.13178281E−08 | 1.31264307E−01 |
| H1 | 4.08225103E−02 | −3.52097601E−08 | −1.31264297E−01 |
| O1 | −8.04237663E−02 | −1.50606797E−08 | 2.18548003E−08 |
| H2 | 3.76338722E−02 | 1.02088694E−08 | 1.35133732E−01 |
| H3 | 3.76338819E−02 | −2.12332661E−09 | −1.35133754E−01 |

Derived dispersion force largest component **9.962E−05 Ha/Bohr**; Atomiverse
reproduces it to **1.21E−06 Ha/Bohr**.

### Measured results (2026-07-26, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **8.9e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **1.21e−06** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **4.30e−03** |
| \(\lvert\sum_A F_A\rvert\) (translation invariance) | 3e−3 Ha/Bohr | **1.34e−03** |
| Product wiring | 1e−7 Ha/Bohr | **5.9e−12** |
| Wall | ≤ 60 s | **32.2 s** |

### Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** closed-form two-body sum with published
  PBE-BJ parameters and the same CN convention; measured 8.9e−11. Only table
  transcription can move it.
- **\(F_{\mathrm{disp}}\) 5e−6 Ha/Bohr:** the pin is a difference of two
  independently converged CP2K SCFs, so it inherits CP2K's force noise;
  measured floor 1.2e−6 against a 1.0e−4 signal.
- **Total \(F\) 1e−2 Ha/Bohr:** bounded below by the *oracle's* grid error —
  CP2K's own `FORCES| Sum` is 5.43e−3 Ha/Bohr on this twin, so a tighter
  absolute comparison is not physically meaningful at `CUTOFF 300`.
- **\(\lvert\sum F\rvert\) 3e−3 Ha/Bohr:** exact in the continuum, broken by
  grid collocation (egg-box). Atomiverse's 1.34e−3 is 4× tighter than the
  oracle's 5.43e−3 on the same system. The gate's purpose is to catch the
  boundary-condition placement bug class (signature |Σ F| ~ 1–2 Ha/Bohr, see
  `gpw-d3-forces/REFERENCE.md`), so ~500× separation is retained.
- **Wiring 1e−7 Ha/Bohr:** same code on both sides of the difference, so only
  the SCF residual enters; measured 5.9e−12.

### Dual gate

- **Correctness:** the six gates above, including every pinned total-force
  component and the product-minus-plain dispersion-force comparison.
- **Speed:** CP2K `ENERGY_FORCE` twin ≈ 11.9 s; Atomiverse product pass 32.2 s
  on the same host → ratio ≈ **2.7**, recorded for historical context only. Note the
  Atomiverse figure includes one extra plain SCF for the wiring gate.

### Notes

- Sibling dispersion-force products: `gpw-d3-forces` (isolated, zero damping),
  `gpw-d4-forces` (periodic, D4/BJ).
- Sibling D3(BJ) energy product: `gpw-pbe-d3bj-ice`.
- The shared `kD3Cp2kPairCutoffBohr` constant (`internal/dftd3.h`) was
  introduced with this pin: the zero-damping path previously used a hard-coded
  40 Bohr, which disagreed with CP2K by 3.9e−7 Ha on a periodic cell. Verified
  against an offline periodic D3-zero CP2K run (`E_disp = −8.20894676620e−04`;
  75.589 Bohr reproduces it to 6.7e−11, 40 Bohr to 3.9e−7).


## Reference — periodic water-dimer GPW PBE-D3(BJ) analytic cell stress

### Method

Γ-point **GPW** (PBE) on a **periodic** water dimer with additive **DFT-D3
Becke–Johnson (rational) damping**, PBE parameters, two-body only
(`CALCULATE_C9_TERM .FALSE.`).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`cpu::gpw::GPWEvaluator` with `d3bj = true`, `compute_stress = true`,
`stress_use_fd = false`):

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian \(E_L[P;W;\varepsilon]\)
   under orthorhombic diagonal strain at fixed fractional coordinates, with the
   geometry-only \(E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd3_pbe_bj` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Sibling twins: `gpw-d3bj-forces` (same geometry, force pins),
`gpw-d4-stress` (same stress path, D4 — no CP2K stress pin there because the
Arrhenius CP2K build lacks libdftd4; **this case carries the real oracle
stress pin** for the dispersion stress fold), `gpw-pbe-d3bj-ice` (D3(BJ)
energy pin).

### Why the stress pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so at
**every strained cell** the converged density with and without D3 is
identical, and \(E^{\mathrm{D3BJ}}(\varepsilon)=E^{\mathrm{plain}}(\varepsilon)
+E_{\mathrm{disp}}(\varepsilon)\) exactly. CP2K's `STRESS_TENSOR NUMERICAL`
re-converges the SCF at each perturbed cell, hence

\[
\sigma^{\mathrm{CP2K}}(\mathrm{D3BJ})-\sigma^{\mathrm{CP2K}}(\mathrm{plain})
 = -\Omega^{-1}\,\partial E_{\mathrm{disp}}/\partial\varepsilon
 \Big|_{\mathrm{CP2K\ FD}},
\]

the pure D3(BJ) dispersion stress: the (three-orders-larger) electronic
stress **and** CP2K's own FD truncation cancel in the difference. This
isolates exactly the quantity this capability adds — the same construction as
the `gpw-d3bj-forces` force pin.

### System

Verbatim the shipped `gpw-d3bj-forces` twin geometry:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN form; ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å = 75.589 Bohr |
| CP2K cutoff | 300 Ry (`CUTOFF 300`, `REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 200 Ry (both dispersion gates are geometry-only ⇒ cutoff-independent; the dual gate is grid-consistent by construction) |
| SCF | CP2K `EPS_SCF 1e-8` (tightened vs the 1e-7 forces twin — lowers the FD-difference noise floor); ATV `tol = 1e-6` |
| Stress step | ATV analytic/FD \(h=2\times10^{-3}\); in-regression D3-only FD \(h=10^{-4}\); CP2K numerical-stress default cell-perturbation FD |

### Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2, EasyBuild `2026.1-gmpflf-2026.03`) |
| Host | Arrhenius epyc login, offline only (`mpirun -np 2`, `OMP_NUM_THREADS=2`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-stress/water_dimer_d3bj_stress.inp` and `water_dimer_plain_stress.inp` |
| Wall | 83.8 s (D3BJ run), 64.5 s (plain run) — CP2K internal timer |
| Date | 2026-07-31 |

#### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain, periodic) | −34.338251977942591 |
| Total (D3BJ, periodic) | −34.339628255637400 |
| **Dispersion energy** (printed) | **−0.00137627769481** |

(The total-energy difference reproduces the printed dispersion energy to
1.4e−16 Ha.)

#### Numerical stress tensors (`STRESS| Numerical stress tensor [bar]`)

| Component | D3BJ run | plain run | Δ = D3BJ − plain [bar] | Δ [Ha/Bohr³] |
| --------- | -------- | --------- | --------------------- | ------------- |
| xx | 5.50901740298E+03 | 5.54281007140E+03 | −33.79266842 | **−1.1485894562E−07** |
| yy | 1.88563460724E+01 | 1.91191341799E+01 | −0.26278811 | **−8.9319862445E−10** |
| zz | 1.28131586907E+04 | 1.28128745905E+04 | +0.28410020 | **+9.6563695465E−10** |

Conversion: 1 Ha/Bohr³ = 2.9421015697×10⁸ bar (CODATA 2018). The dimer axis
is x, so \(\Delta\sigma_{xx}\) carries the ~1.1e−7 signal; yy/zz are the
lattice-image / H-offset residuals (~1e−9), still ~30× above the pin floor.

### Measured results (2026-07-31, this pin)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K (standalone + product) | 1e−8 Ha | **7.6e−12** |
| diag \(\sigma_{\mathrm{disp}}\) vs CP2K Δ | 1e−10 Ha/Bohr³ | **3.9e−14** (xx), 2.7e−14 (yy), 3.2e−14 (zz) |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | 2e−5 Ha/Bohr³ | **7.4e−07** (pinning ran all axes: yy 2.5e−8, zz 1.2e−9) |
| off-diagonal product | 1e−8 Ha/Bohr³ | ~0 (pass) |
| max \(|\sigma_{\mathrm{diag}}|\) floor | > 1e−10 Ha/Bohr³ | 5.83e−5 |
| Wall (product + x-only FD twin) | ≤ 60 s | **28.4 s + 4.8 s** (warm-started FD; wave-1 loaded node — 3-axis cold-measure pinning FD was 79.4 s) |

Product analytic diagonals (first green, 200 Ry, `tol = 1e−6`, 11 SCF iters,
\(E_{\mathrm{tot}}\) = −34.331707679811 Ha):

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −2.17975632×10⁻⁵ |
| \(\sigma_{yy}\) | +2.00723385×10⁻⁵ |
| \(\sigma_{zz}\) | +5.82656483×10⁻⁵ |

The e2e FD twin re-SCFs the **x axis only** (dimer axis — carries the
1.15e−7 dispersion-stress signal **and** the largest measured analytic↔FD
residual). The full tensor is compared to the external pins; this native
FD supplement covers x only.

### Tolerance justification

- **\(E_{\mathrm{disp}}\) 1e−8 Ha:** closed-form two-body sum with published
  PBE-BJ parameters and the CP2K CN convention; same pin class as
  `gpw-d3bj-forces` (measured 8.9e−11 there). Only table transcription can
  move it.
- **\(\sigma_{\mathrm{disp}}\) 1e−10 Ha/Bohr³:** the pin is a difference of
  two independently converged CP2K numerical-stress runs at `EPS_SCF 1e−8`,
  so it inherits (SCF energy noise)/(h·Ω) ≲ 1e−11 Ha/Bohr³ plus the C6/CN
  table parity; measured 3.9e−14 against the 1.15e−7 xx signal — the gate
  keeps ~1000× separation from the signal and even holds the tiny yy/zz
  components (~9e−10) to ~30× above tolerance floor.
- **Dual gate 2e−5 Ha/Bohr³:** same bound as `gpw-d4-stress` (measured
  7.4e−7 there): the analytic fixed-P Lagrangian vs six fully re-converged
  SCFs at `tol = 1e−6` — the residual is SCF-convergence noise, not the
  dispersion fold.

### Dual gate

- **Correctness:** the pins above and the section's analytic-fixed-density
  versus reconverged strain-FD comparison on the same `d3bj = true` method.
- **Speed:** CP2K numerical-stress D3BJ twin 83.8 s (18 perturbed SCFs +
  reference, 2 MPI × 2 OMP, 300 Ry); Atomiverse product analytic pass
  **31.1 s** (one SCF + 7 fixed-P grid evaluations, OMP 8, 200 Ry, wave-1
  loaded login node) → ratio ≈ **0.37**, recorded for historical context only.
  Settings caveat: the cutoffs and core counts differ as stated — both
  produce the same deliverable (the cell stress of this system), which is
  the wall being compared.

### Notes

- The FD-twin wall in this binary (multi-SCF `gpw_stress_fd`) is gate
  overhead, not the product path; the product path is the analytic pass.
- Siblings: `gpw-d3-stress` (zero damping, same fold) is a separate queue row;
  the fold itself (`energy_fixed_P` / `energy_fixed_P_k` in
  `src/cpu/gpw/internal/gpw_stress.cpp`) predates this pin — this case adds
  the missing oracle evidence, the unit twin, and the e2e product proof.


## Reference — periodic water-dimer GPW PBE-D3(BJ) dispersion energy

### Method

Grimme **DFT-D3 with Becke–Johnson (rational) damping** as an additive
correction on top of Γ-point **GPW** (PBE), for a periodic water dimer cell
(ice-relevant H-bonded pair; not a full ice Ih supercell).

Public API: `cpu::gpw::GpwDft::d3bj = true` (and `GPWEvaluator::d3bj`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Coordination numbers use the **CP2K** form (tanh long-range factor)
so E_disp matches offline `TYPE DFTD3(BJ)`. Molecular `cpu::scf::D3BJ` keeps
the Grimme CN (PySCF twin) and is unchanged.

PBE-D3(BJ) parameters (Grimme et al., *J. Comput. Chem.* **32**, 1456 (2011);
CP2K `REFERENCE_FUNCTIONAL PBE`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.7875 |
| \(a_1\) | 0.4289 |
| \(a_2\) | 4.4407 Bohr |

Pair cutoff in Atomiverse: **2 × R_CUTOFF 20 Å ≈ 75.59 Bohr**. CP2K prints
`Cutoff Radius [Bohr]: 37.79` for `R_CUTOFF 20` (Å→Bohr); the energy lattice
sum matches the pin when the engine rcut is **twice** that (same 2× convention
as the zero-damping GPW D3 pin). Three-body ATM off.

### Reference value (hard gate)

**E_disp (PBE-D3(BJ), periodic water dimer, 12 Å cell) = −0.00137627769481 Ha**

Source: offline **CP2K 2026.1** (`git:5e54ba2`, module build
`2026.1-gmpflf-2026.03` on Arrhenius), twin input
`tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice/water_dimer_d3bj.inp`.

Printed line:

```text
Dispersion energy:                                           -0.00137627769481
```

#### Context (not the hard gate)

CP2K total QS energy for the same twin (PBE, DZVP-MOLOPT-SR-GTH, GTH-PBE,
CUTOFF 300 Ry, OT): **−34.33962825563683 Ha**. Absolute GPW e_tot parity is
out of scope for this slug (OT/grid/DIIS differences); the regression asserts
**E_disp** and that `e_total` includes it.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.

Basis / PP (CP2K twin and Atomiverse tables): `DZVP-MOLOPT-SR-GTH` +
`GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Oracle: CP2K 2026.1 (offline — never launched from ctest)

```bash
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$MPICH/bin:$PATH   # MPICH 4.3.2 + dep libs on Arrhenius
cd tools/cp2k_oracle/inputs/gpw-pbe-d3bj-ice
mpirun -np 1 cp2k.psmp -i water_dimer_d3bj.inp -o water_dimer_d3bj.out
## rg 'Dispersion energy' water_dimer_d3bj.out
```

CP2K wall (this pin run, 1 rank × 4 OMP): **T_cp2k ≈ 13.4 s** total (TIMING
table `CP2K` TOTAL TIME).

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp. With matching geometry, PBE-BJ parameters, CP2K CN,
and rcut 40 Bohr, Atomiverse is expected to track CP2K dispersion to
nano-Hartree.

### What is checked

`GpwDft` with `d3bj = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −0.00137627769481 Ha. Unit tests also
check analytic BJ forces vs FD and that `e_total(d3bj) − e_total(off) = e_disp`.

### Perf gate

D3 evaluation itself is \(O(N^2)\) over a 6-atom cell (microseconds). Full GPW
SCF wall is dominated by the electronic structure, not D3. Fair ratio for a
**dispersion-only** microbenchmark vs CP2K is not meaningful; full twin SCF
ratio depends on cutoff/OT choices. The complete section still has a 60 s
budget. CP2K wall includes full QS (~13 s), while the hard-gate regression
uses a short SCF to materialize geometry-only `e_disp`; these boundaries do
not establish a matched-engine performance result.
