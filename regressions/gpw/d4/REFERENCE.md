# gpw/d4

One case; every section below is run by `atv_regressions --case gpw/d4` and keeps its own oracle record.


## Reference — periodic water-dimer GPW PBE-D4 dispersion energy

### Method

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson (rational) damping as an
additive two-body correction on top of Γ-point **GPW** (PBE), for a periodic
water dimer cell (same geometry as `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::d4 = true` (and `GPWEvaluator::d4`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`. Three-body ATM is **off** (`s9 = 0`).

PBE-D4 parameters (SAW190103 BJ set; dftd4 `pbe-D4`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.95948085 |
| \(a_1\) | 0.38574991 |
| \(a_2\) | 4.80688534 Bohr |

Cutoffs (dftd4 defaults): CN 30 Bohr, disp2 60 Bohr. Periodic EEQ uses an
Ewald-split A-matrix (multicharge-style).

### Reference value (hard gate)

**E_disp (PBE-D4, two-body, periodic water dimer, 12 Å cell) = −0.00099458842745846774 Ha**

Source: offline **dftd4 3.7.0** (`/software/.../dftd4/3.7.0-gfbf-2025b` on
Arrhenius), `dftd4 -f pbe --mbdscale 0.0` on the POSCAR twin below.

Printed:

```text
Dispersion energy:      -9.9458842745847E-04 Eh
```

JSON: `{"version":"3.7.0","energy":-9.9458842745846774E-04}`.

#### Why not CP2K QS?

**CP2K 2026.1** (`2026.1-gmpflf-2026.03`) on Arrhenius was built **without**
the DFTD4 library: `TYPE DFTD4` aborts with
`CP2K build without DFTD4` (`qs_dispersion_d4.F`). The twin input is retained
under `tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts that link libdftd4.
The hard pin is therefore **libdftd4 / dftd4 3.7.0**, which is the same
backend CP2K uses when DFTD4 is enabled (CP2K banner would report
`DFT-D4(Version 3.6.0)`-class).

#### Context (not the hard gate)

Isolated (non-periodic) two-body on the same nuclear geometry:
**−9.8816586433966×10⁻⁴ Ha** (dftd4 3.7.0, `--mbdscale 0`).

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

Basis / PP for the GPW regression (E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Oracle: dftd4 3.7.0 (offline — never launched from ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
## + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR
## Dispersion energy: -9.9458842745847E-04 Eh
```

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp. Matching geometry, PBE-D4 parameters, EEQ2019,
and lattice sums are expected to track dftd4 to nano-Hartree on this cell.

### What is checked

`GpwDft` with `d4 = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −0.00099458842745846774 Ha.
Unit tests also check the isolated two-body pin on the same nuclei.

### Perf gate

D4 evaluation is \(O(N^2 \times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds–milliseconds). Full GPW SCF wall is dominated by the electronic
structure, not D4. **perf_exempt = true** for this slug: reason — primary
deliverable is E_disp pin parity; CP2K QS wall is unavailable (no DFTD4 build)
and a full twin SCF ratio is unfair for a geometry-only hard gate (same policy
as `gpw-pbe-d3bj-ice`).


## Reference — periodic water-dimer GPW PBE-D4 analytic ionic forces

### Method

Γ-point **GPW** (PBE) with additive **DFT-D4** (Becke–Johnson, two-body,
PBE parameters; ATM off) on a periodic water dimer.

**Product ionic forces** = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** geometry-only
\(F_{\mathrm{disp}}^{\mathrm{D4}} = -\nabla E_{\mathrm{disp}}\) from
`cpu::scf::dftd4` (exact gradient of the same two-body energy as
`dftd4_energy` / `gpw-d4-water`).

Oracle twins:

1. **Hard (geometry-only):** offline **dftd4 3.7.0** `-g` gradients on the
   same nuclei (force = −gradient).
2. **Wiring (product):** \(F^{\mathrm{d4}}-F^{\mathrm{plain}}\approx F_{\mathrm{disp}}\)
   (geometry-only D4 does not change the SCF density).
3. **Unit dual-gate:** multi-SCF `gpw_forces_fd` with `d4=true` on cheap MT
   H₂ lives in `tests/cpu/gpw/test_cpu_gpw_forces.cpp` (6N SCFs on the water
   dimer exceed the ≤60 s regression budget).

Public path: `cpu::gpw::GPWEvaluator` with `d4=true`, `compute_forces=true`.

### System

Same geometry as `gpw-d4-water` / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| Cutoff | 200 Ry |

### Offline pins

#### Hard: E_disp (dftd4 3.7.0)

**E_disp = −0.00099458842745846774 Ha** (two-body, periodic, `--mbdscale 0`).

Source: offline **dftd4 3.7.0** on Arrhenius (same as `gpw-d4-water`).

#### Hard: D4 gradient (dftd4 3.7.0 `-g`)

Printed **gradient** \(g = \partial E/\partial R\) [Ha/Bohr]. Product force
contribution is \(F = -g\).

Remapped from POSCAR order (O O H H H H) to ATV order **O H H O H H**:

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3738961271913915e−05 | ~0 | ~0 |
| H0 | −5.9550148958160894e−05 | ~0 | −1.0810592129949165e−05 |
| H1 | −5.9550148958160921e−05 | ~0 | +1.0810592129949182e−05 |
| O1 | +1.3609823922940861e−04 | ~0 | ~0 |
| H2 | +1.8370509979413634e−05 | ~0 | −8.5497124425277334e−06 |
| H3 | +1.8370509979413539e−05 | ~0 | +8.5497124425278096e−06 |

Gradient norm ≈ **1.719e−04** Ha/Bohr.

#### Why not CP2K QS ENERGY_FORCE?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4` aborts).
Twin input retained under `tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts
with libdftd4. Hard pin is **libdftd4 / dftd4 3.7.0**.

#### Optional: re-run pin (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR
## note POSCAR atom order O O H H H H — remap to O H H O H H for ATV
```

### Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_d4 + g_dftd4\| | **1×10⁻⁷** Ha/Bohr |
| max\|(F_d4_on − F_plain) − F_d4\| | **1×10⁻⁷** Ha/Bohr (wiring; SCF residual) |
| Unit dual-gate max\|F_an − F_fd\| | **5×10⁻⁵** Ha/Bohr (H₂, not this e2e) |

### Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| --- | --- |
| **E (Γ PBE-D4 200 Ry)** | **−34.3313259463 Ha** (SCF 11 iters) |
| E_disp | **−9.9458849003×10⁻⁴ Ha** (\|d pin\| = 6.3×10⁻¹¹) |
| max\|F_d4 + g_dftd4\| | **3.0×10⁻¹¹** Ha/Bohr |
| max\|(F_d4 − F_plain) − F_standalone\| | **2.9×10⁻⁹** Ha/Bohr |
| max\|F_product\| | **0.136** Ha/Bohr |
| T_product (OMP=8, Release) | **17.6 s** |
| T_plain twin (OMP=8) | **17.1 s** |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K build lacks DFTD4 (no fair QS ENERGY_FORCE wall). Primary hard gate is dftd4 E_disp + gradient pin parity + product wiring (same policy as `gpw-d4-water` / `gpw-d4-stress`). |

### What is checked

1. Product analytic forces with `d4=true` finish without throw.
2. `e_disp` within 1e-8 Ha of the dftd4 pin.
3. Standalone D4 forces match offline −gradient pin.
4. Additive wiring: product(d4) − product(plain) ≈ standalone D4.
5. Unit suite dual-gate (H₂ + D4 analytic vs multi-SCF FD).


## Reference — periodic water-dimer GPW PBE-D4 analytic cell stress

### Method

Γ-point **GPW** (PBE) with additive **DFT-D4** (Becke–Johnson, two-body,
PBE parameters; ATM off) on a periodic water dimer.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\) and energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P;W;\varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, with geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})\)
   rebuilt at each strained cell (same `dftd4_energy` as the energy path).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same `d4=true`
method (full re-SCF includes D4 via `fold_dispersion`).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `d4=true`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

### System

Same geometry as `gpw-d4-water` / `gpw-pbe-d3bj-ice`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); ATM off |
| Cutoff | 200 Ry |
| Stress step | \(h=2\times 10^{-3}\) |

### Offline pins

#### Hard: E_disp (dftd4 3.7.0)

**E_disp = −0.00099458842745846774 Ha** (two-body, periodic, `--mbdscale 0`).

Source: offline **dftd4 3.7.0** on Arrhenius
(`/software/.../dftd4/3.7.0-gfbf-2025b`), same POSCAR as
`tools/cp2k_oracle/inputs/gpw-d4-water/POSCAR`. See sibling
`regressions/gpw/d4/REFERENCE.md`.

#### Decisive: dual-gate analytic vs multi-SCF FD

No absolute CP2K `STRESS_TENSOR` pin: Arrhenius **CP2K 2026.1** is built
**without** DFTD4 (`TYPE DFTD4` aborts). Twin input retained under
`tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts with libdftd4.

Decisive gate is **internal dual-gate** \(\max_a|\sigma_a^{\mathrm{an}}
-\sigma_a^{\mathrm{fd}}|\).

#### Optional: CP2K / dftd4 re-run (offline only — never ctest)

```bash
## E_disp pin
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
cd tools/cp2k_oracle/inputs/gpw-d4-water
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR
```

### Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|\σ_an − σ_fd\| diag | **2×10⁻⁵** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|\σ_diag\| floor | > **1×10⁻¹⁰** Ha/Bohr³ |

### Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| --- | --- |
| **E (Γ PBE-D4 200 Ry)** | **−34.3313259463 Ha** (SCF 11 iters) |
| E_disp | **−9.9458849003×10⁻⁴ Ha** (\|d pin\| = 6.3×10⁻¹¹) |
| σ_xx analytic | −2.17585158e-05 Ha/Bohr³ |
| σ_yy analytic | 2.00723219e-05 Ha/Bohr³ |
| σ_zz analytic | 5.82669568e-05 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **7.39e-07** Ha/Bohr³ |
| T_product (OMP=8, Release) | **21.4 s** |
| T_fd_twin (OMP=8) | **42.3 s** |
| T_dual_gate wall | **~64 s** |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K build lacks DFTD4 (no fair QS stress wall). Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Primary hard gate is E_disp pin parity + analytic↔FD residual (same policy as `gpw-d4-water` / `gpw-dftu-stress`). |

### What is checked

1. Product analytic stress with `d4=true` finishes without throw.
2. `e_disp` within 1e-8 Ha of the dftd4 pin.
3. Dual-gate diagonal residual ≤ 2e-5 Ha/Bohr³.
4. Off-diag product ≈ 0; Frame `kStress` unit conversion.


## Reference — periodic water-dimer GPW PBE-D4 + ATM (s9)

### Method

Grimme / Caldeweyher **DFT-D4** with Becke–Johnson two-body damping **plus**
the Axilrod–Teller–Muto **three-body** term at \(s_9 = 1\) (dftd4
`pbe-D4-ATM`), as an additive correction on top of Γ-point **GPW** (PBE).

Same geometry and cell as `gpw-d4-water` (two-body-only twin).

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/200.0, XcKind::PBE};
gpw.d4 = true;
gpw.d4_atm = true;   // s9 = 1
```

`GPWEvaluator::d4_atm` mirrors the solver flag. Dispersion energy is on
`GpwResult::e_disp` [Hartree] and folded into `e_total`.

PBE-D4 parameters (SAW190103 BJ set; dftd4 `pbe-D4-ATM`):

| param | value |
| ----- | ----- |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.95948085 |
| \(s_9\) | **1.0** (ATM on) |
| \(a_1\) | 0.38574991 |
| \(a_2\) | 4.80688534 Bohr |
| alp (ATM) | 16 |

Cutoffs (dftd4 defaults): CN 30 Bohr, disp2 60 Bohr, **disp3 40 Bohr**.
ATM uses **neutral-charge** C6 reweight (`q = 0`), matching simple-dftd4
`disp.f90`.

### Reference value (hard gate)

**E_disp (PBE-D4 + ATM, periodic water dimer, 12 Å cell) = −9.9554475834240288×10⁻⁴ Ha**

Source: offline **dftd4 3.7.0**
(`/software/.../dftd4/3.7.0-gfbf-2025b` on Arrhenius),

```text
dftd4 -f pbe --json pin_periodic_atm.json POSCAR
## Rational (Becke-Johnson) damping: pbe-D4-ATM
##   s9         1.0000
## Dispersion energy:      -9.9554475834240E-04 Eh
```

JSON: `{"version":"3.7.0","energy":-9.9554475834240288E-04}`.

Context (not the hard gate):

| Case | E_disp [Ha] |
| ---- | ----------- |
| Periodic two-body only (`--mbdscale 0`) | −9.9458842745846774×10⁻⁴ |
| Isolated ATM (`s9=1`) | −9.8819741533958208×10⁻⁴ |

ATM contribution on this cell ≈ −9.56×10⁻⁷ Ha.

#### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius was built **without** the DFTD4 library
(`CP2K build without DFTD4`). Hard pin is **libdftd4 / dftd4 3.7.0**, the
same backend CP2K uses when DFTD4 is linked. Twin input retained under
`tools/cp2k_oracle/inputs/gpw-d4-water/` for hosts that link libdftd4;
ATM-specific pin files live under `tools/cp2k_oracle/inputs/gpw-d4-atm/`.

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

Basis / PP for the GPW regression (E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Oracle: dftd4 3.7.0 (offline — never launched from ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
## + FlexiBLAS / GCCcore from buildenv-gcc module
cd tools/cp2k_oracle/inputs/gpw-d4-atm
dftd4 -f pbe --json pin_periodic_atm.json POSCAR
## Dispersion energy: -9.9554475834240E-04 Eh
```

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp.

### What is checked

`GpwDft` with `d4 = true` and `d4_atm = true` returns
`last_result().e_disp` within 1e-8 Ha of −9.9554475834240288×10⁻⁴ Ha.
Unit tests also pin isolated + periodic free-function ATM energies.

### Perf gate

ATM is \(O(N^3 \times N_{\mathrm{images}}^2)\) on a 6-atom cell (still
milliseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; CP2K QS
wall is unavailable (no DFTD4 build); same policy as `gpw-d4-water`.


## Reference — periodic water-dimer GPW PBE-D4+ATM analytic ionic forces

### Method

Γ-point **GPW** (PBE) with additive **DFT-D4** including the
Axilrod–Teller–Muto **three-body** term (\(s_9 = 1\), dftd4 `pbe-D4-ATM`) on
a periodic water dimer — the force companion of the `gpw-d4-atm` energy
foothold and the ATM extension of `gpw-d4-forces`.

**Product ionic forces** = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) **plus** geometry-only
\(F_{\mathrm{disp}} = -\nabla(E_2 + E_{\mathrm{ATM}})\) from
`cpu::scf::dftd4` with `s9 = 1`. The ATM gradient is **analytic**: explicit
damped Axilrod–Teller triple derivatives plus the
C9 → neutral-C6 → CN\(_\mathrm{d4}\) chain (the \(q = 0\) reweight of dftd4
`disp.f90` is geometry-independent, so the ATM leg has no EEQ adjoint).

Public path: `cpu::gpw::GPWEvaluator` with `d4 = true`, `d4_atm = true`,
`compute_forces = true`.

### System

Same geometry as `gpw-d4-atm` / `gpw-d4-forces`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); **ATM on**: \(s_9=1\), alp = 16, disp3 cutoff 40 Bohr, neutral-charge C6 reweight |
| Cutoff | 200 Ry |

### Offline pins (dftd4 3.7.0)

#### Hard: E_disp with ATM

**E_disp = −9.9554475834240310×10⁻⁴ Ha** (`pin_periodic_atm_grad.json`;
matches the `gpw-d4-atm` energy pin −9.9554475834240288×10⁻⁴ to 2×10⁻²⁰).

#### Hard: D4+ATM gradient (dftd4 3.7.0 `--grad`, s9 = 1)

Printed gradient \(g = \partial E/\partial R\) [Ha/Bohr]; product force
contribution is \(F = -g\). Remapped from POSCAR order (O O H H H H) to ATV
order **O H H O H H** (`pin_periodic_atm_grad.json`, produced 2026-07-31):

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3595427766884764e−05 | ~0 | ~0 |
| H0 | −5.9463596930597288e−05 | ~0 | −1.0397154303254306e−05 |
| H1 | −5.9463596930597301e−05 | ~0 | +1.0397154303254314e−05 |
| O1 | +1.3610085695186597e−04 | ~0 | ~0 |
| H2 | +1.8210882338106763e−05 | ~0 | −8.1642912818385456e−06 |
| H3 | +1.8210882338106668e−05 | ~0 | +8.1642912818386235e−06 |

The ATM-only force increment vs the two-body `--mbdscale 0` pin
(`gpw-d4-forces` REFERENCE) has max component **4.05×10⁻⁷ Ha/Bohr** — that is
the physics this case gates.

Isolated twin (same nuclei, no cell; ATV atom order directly):
`pin_isolated_atm_grad.json`, E = −9.8819741533958229×10⁻⁴ Ha — pinned at
**1×10⁻⁹ Ha/Bohr** in `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp` (ATV measured
1.7×10⁻¹²).

#### Oracle finding: dftd4 3.7.0's periodic ATM gradient is non-conservative

Checked offline (2026-07-31), all numbers printed by dftd4 3.7.0:

- **Isolated**: central difference of its s9=1 energy (h = 10⁻³ Å, O0 x)
  matches its printed gradient to **2.6×10⁻¹⁰** Ha/Bohr — conservative.
- **Periodic (12 Å cell)**: the same FD gives −5.3616541919×10⁻⁵ while its
  printed \(g_x\)(O0) is −5.3595427767×10⁻⁵ — a **2.1×10⁻⁸ Ha/Bohr**
  discrepancy against its *own* energy (max over components ~2.4×10⁻⁸).
- ATV's s9=1 periodic energy equals dftd4's at the base **and** displaced
  geometries to a constant 3.8×10⁻¹² Ha (Bohr-conversion constant; cancels
  in differences), so the two codes share one energy functional, and ATV's
  analytic force is conservative against it to 3×10⁻¹² (self-FD test).

A conservative gradient of the shared functional therefore cannot match the
printed periodic pin beyond that oracle defect. Gates: the **direct** pin
comparison at **5×10⁻⁸** (absorbs the defect, still 8× below the ATM
increment, so it fails if the ATM gradient is dropped), plus a **sharp
1×10⁻⁹ conservative-parity** gate against the FD over the oracle's printed
displaced energies (truncation ≈ 3×10⁻¹⁰):

| Pin | dftd4 3.7.0 s9=1 energy [Ha] |
| --- | --- |
| O0 x = +0.001 Å (`pin_periodic_atm_dispO0x_plus.json`) | −9.9564614957741663×10⁻⁴ |
| O0 x = −0.001 Å (`pin_periodic_atm_dispO0x_minus.json`) | −9.9544350841746518×10⁻⁴ |

#### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4`
aborts). The hard pin is **libdftd4 / dftd4 3.7.0**, the same backend CP2K
uses when DFTD4 is linked — same policy as `gpw-d4-atm` / `gpw-d4-forces`.

#### Re-run (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
cd tools/cp2k_oracle/inputs/gpw-d4-atm
dftd4 -f pbe --grad grad_periodic_atm.txt --json pin_periodic_atm_grad.json POSCAR
## POSCAR atom order O O H H H H — remap to O H H O H H for ATV
dftd4 -f pbe --grad grad_isolated_atm.txt --json pin_isolated_atm_grad.json water_dimer.xyz
## Displaced-energy pins (edit POSCAR O0 x to ±0.001 before each run):
dftd4 -f pbe --json pin_periodic_atm_dispO0x_plus.json POSCAR   # x=+0.001 Å
dftd4 -f pbe --json pin_periodic_atm_dispO0x_minus.json POSCAR  # x=−0.001 Å
```

### Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (standalone and product) | **1×10⁻⁸** Ha |
| max\|F_d4atm + g_pin\| (direct, periodic) | **5×10⁻⁸** Ha/Bohr (oracle defect ~2.4×10⁻⁸; ATM increment 4.05×10⁻⁷) |
| \|F_x(O0) − FD(oracle E±)\| (conservative parity) | **1×10⁻⁹** Ha/Bohr |
| ATM increment max\|(F_atm − F_2b) − Δpin\| | **5×10⁻⁸** Ha/Bohr (same oracle defect); non-vacuity max\|dF\| > 1×10⁻⁸ |
| max\|(F_on − F_plain) − F_standalone\| | **1×10⁻⁷** Ha/Bohr (wiring; SCF residual) |
| Isolated max\|F + g_pin\| (unit suite) | **1×10⁻⁹** Ha/Bohr |
| Unit dual-gate max\|F_an − F_fd\| (s9=1 functional) | **1×10⁻⁷** Ha/Bohr (FD-limited; unit suite) |

### Atomiverse numbers (first green, 2026-07-31)

| Quantity | Value |
| --- | --- |
| E_disp (standalone s9=1) | \|d pin\| = 6.3×10⁻¹¹ Ha (Bohr-conversion constant) |
| max\|F + g_pin\| (periodic, direct) | 2.394×10⁻⁸ Ha/Bohr (= the oracle defect) |
| \|F_x(O0) − FD(oracle E±)\| | 2.65×10⁻¹⁰ Ha/Bohr |
| Isolated max\|F + g_pin\| (unit) | 1.7×10⁻¹² Ha/Bohr |
| ATM increment max\|dF − Δpin\| | 2.391×10⁻⁸ Ha/Bohr (same defect); max\|dF\| = 4.05×10⁻⁷ |
| Self-FD (analytic vs FD of own s9=1 energy, unit) | 3.0×10⁻¹² Ha/Bohr |
| Molecular D4 s9=1 analytic-vs-FD (unit) | 1.9×10⁻¹¹ Ha/Bohr |
| Product E (Γ PBE-D4+ATM 200 Ry) | −34.331326946882 Ha (SCF 11 iters); e_disp \|d pin\| = 2.8×10⁻¹⁴ |
| max\|wiring\| | 1.52×10⁻⁸ Ha/Bohr |
| Wall (product SCF+forces, OMP=8) | 15.7 s |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K lacks DFTD4 — no fair QS ENERGY_FORCE wall. Hard gate is dftd4 gradient-pin parity (same policy as `gpw-d4-water` / `gpw-d4-forces` / `gpw-d4-atm`). The analytic ATM path replaces the previous 6N-FD ATM leg, strictly faster. |

### What is checked

1. Standalone `dftd4(s9=1)` energy and forces match the offline pins
   (1e−8 Ha / 1e−9 Ha/Bohr).
2. The ATM-only force increment matches the oracle's pin delta (1e−9
   Ha/Bohr) and is non-vacuous (>1e−8).
3. Product GPW `d4 + d4_atm` converges; `e_disp` matches the ATM pin.
4. Additive wiring: product(d4_atm) − product(plain) ≈ standalone (1e−7).
5. Unit suite: isolated + periodic s9=1 pins, ATM increment, and
   analytic-vs-FD of the s9=1 functional
   (`tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`,
   `tests/cpu/scf/test_dispersion_gradient.cpp`).


## Reference — periodic water-dimer GPW PBE-D4+ATM analytic cell stress

### Method

Γ-point **GPW** (PBE) with additive **DFT-D4** including the
Axilrod–Teller–Muto **three-body** term (\(s_9 = 1\), dftd4 `pbe-D4-ATM`) on
a periodic water dimer — the stress companion of `gpw-d4-atm-forces` and the
ATM extension of `gpw-d4-stress`.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path
(`GpwDft::stress`): one SCF → \(P, W\); central FD of the non-self-consistent
Lagrangian under orthorhombic diagonal strain at fixed fractional
coordinates, with geometry-only
\(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
(**s9 = 1** when `d4_atm`) rebuilt at each strained cell;
\(\sigma_{aa} = -\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
(CP2K `STRESS_TENSOR` sign).

Public path: `cpu::gpw::GPWEvaluator` with `d4 = true`, `d4_atm = true`,
`compute_stress = true` → Frame `kStress` [eV/Å³];
`last_stress_ha_bohr3()` [Ha/Bohr³].

### System

Same geometry as `gpw-d4-atm` / `gpw-d4-atm-forces` / `gpw-d4-stress`:

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms** (order **O H H O H H**) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Dispersion | DFT-D4 BJ (PBE: \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)); **ATM on**: \(s_9=1\), alp = 16, disp3 cutoff 40 Bohr |
| Cutoff | 200 Ry |
| Stress step | \(h = 2\times10^{-3}\) |

### Offline pins (dftd4 3.7.0)

#### Hard + decisive: strained-cell dispersion energies

The ATM stress contribution is **~10⁻¹⁰ Ha/Bohr³** on this system — three
orders below the GPW analytic-vs-FD dual-gate residual (~7×10⁻⁷), so only
dedicated energy pins at the exact strain samples resolve it. For each axis,
cell row a **and** the Cartesian a-components scaled by \(1 \pm h\) (the
`scale_axis` fixed-fractional convention), dftd4 3.7.0 printed energies
(`tools/cp2k_oracle/inputs/gpw-d4-atm/pin_strain_*_{atm,2b}.json`):

| Strain | s9=1 [Ha] | two-body (`--mbdscale 0`) [Ha] |
| ------ | --------- | ------------------------------ |
| x, +h | −9.9375767649976510e−04 | −9.9281871131614977e−04 |
| x, −h | −9.9733212469993015e−04 | −9.9635829706912161e−04 |
| y, +h | −9.9553758220206315e−04 | −9.9456763760422703e−04 |
| y, −h | −9.9555200940363935e−04 | −9.9460945924979903e−04 |
| z, +h | −9.9558989265599194e−04 | −9.9463925734105045e−04 |
| z, −h | −9.9549659214095867e−04 | −9.9453454831018983e−04 |

Base-cell s9=1 pin (product `e_disp` gate): **−9.9554475834240290e−04 Ha**
(`pin_periodic_atm.json`, shared with `gpw-d4-atm`).

Derived oracle dispersion stress, σ_aa = −[E(+h) − E(−h)]/(2hΩ),
Ω = 11661.119488 Bohr³ — the ATM-only part is the physics this case gates:

| Axis | σ^disp(s9=1) [Ha/Bohr³] | σ^ATM = σ(s9=1) − σ(2b) |
| ---- | ----------------------- | ----------------------- |
| xx | −7.6631754863e−08 | **−7.474078e−10** |
| yy | −3.0930138378e−10 | **+5.873030e−10** |
| zz | +2.0002478134e−09 | **−2.445845e−10** |

All gates are on the oracle's **printed energies** — never its gradient
(dftd4 3.7.0's periodic ATM gradient is non-conservative by ~2.4×10⁻⁸
against its own energy; documented in `gpw-d4-atm-forces/REFERENCE.md`).

#### Why not CP2K QS?

**CP2K 2026.1** on Arrhenius is built **without** DFTD4 (`TYPE DFTD4`
aborts). The hard pin is **libdftd4 / dftd4 3.7.0**, the same backend CP2K
uses when DFTD4 is linked — same policy as `gpw-d4-atm` /
`gpw-d4-atm-forces` / `gpw-d4-stress`.

#### Re-run (offline only — never ctest)

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
## For each axis a and sign ±: scale lattice row a AND Cartesian a-components
## of tools/cp2k_oracle/inputs/gpw-d4-atm/POSCAR by (1 ± 2e-3), save as
## POSCAR in its own directory (dftd4 detects format by filename), then:
dftd4 -f pbe --json pin_strain_<a>_<pm>_atm.json POSCAR
dftd4 -f pbe --mbdscale 0.0 --json pin_strain_<a>_<pm>_2b.json POSCAR
```

### Tolerances

| Check | Tol |
| ----- | --- |
| strained \|E_disp − pin\| (s9=1 and 2b, 12 values) | **1×10⁻⁹** Ha |
| \|σ^disp(s9=1) − FD(oracle E)\| and \|σ^ATM − Δpin\| per axis | **1×10⁻¹²** Ha/Bohr³ |
| ATM non-vacuity max\|σ^ATM_pin\| | > **5×10⁻¹⁰** Ha/Bohr³ (pin 7.47×10⁻¹⁰) |
| product \|e_disp − ATM base pin\| | **1×10⁻⁸** Ha |
| dual gate \|σ_xx^an − σ_xx^fd\| (multi-SCF strain-FD twin) | **2×10⁻⁵** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|σ_diag\| floor | > **1×10⁻¹⁰** Ha/Bohr³ |
| wiring \|(σ_xx(ATM) − σ_xx(2b)) − σ^ATM_xx\| | **3×10⁻¹⁰** Ha/Bohr³ (noise 7×10⁻¹¹ from independently converged reference densities at SCF tol 10⁻⁶; dropping the s9 fold leaves 7.5×10⁻¹⁰ → FAIL) |

### Atomiverse numbers (first green, 2026-07-31)

| Quantity | Value |
| --- | --- |
| strained-energy parity max\|d pin\| (12 values) | **6.30×10⁻¹¹ Ha** (Bohr-conversion constant) |
| σ^disp(s9=1) parity max | 5.19×10⁻¹⁵ Ha/Bohr³ |
| σ^ATM parity max | **3.65×10⁻¹⁶ Ha/Bohr³** |
| E (Γ PBE-D4+ATM 200 Ry) | −34.331326946882 Ha |
| e_disp product \|d pin\| | 2.76×10⁻¹¹ Ha |
| σ_xx analytic (d4+ATM) | −2.17593330×10⁻⁵ Ha/Bohr³ |
| σ_yy analytic | +2.00729249×10⁻⁵ Ha/Bohr³ |
| σ_zz analytic | +5.82666928×10⁻⁵ Ha/Bohr³ |
| dual gate \|σ_xx^an − σ_xx^fd\| | **7.392×10⁻⁷** Ha/Bohr³ (same class as `gpw-d4-stress` 7.39×10⁻⁷) |
| wiring \|Δσ_xx − σ^ATM_xx\| | 7.05×10⁻¹¹ Ha/Bohr³ |
| walls (OMP=8, Release) | product 24.4 s; FD twin (x, warm) 4.9 s; 2b analytic (x, warm) 6.1 s |

Unit twin: `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`
`WaterDimerPeriodicStrainedAtmMatchesDftd4` (strained x+h s9=1 pin at 1×10⁻⁹;
measured 3.8×10⁻¹²).

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | Arrhenius CP2K lacks DFTD4 — no fair QS `STRESS_TENSOR` wall exists. Hard gate is dftd4 printed-energy parity at the strain samples (same policy as `gpw-d4-atm` / `gpw-d4-atm-forces` / `gpw-d4-stress`). Case wall ~36 s is dominated by the one product SCF + FD twin, not by the ATM term. |

### What is checked

1. ATV `dftd4_energy(s9=1)` and `(s9=0)` at all 6 strained cells match the
   dftd4 3.7.0 printed energies (1×10⁻⁹ Ha).
2. The dispersion cell stress and its **ATM-only contribution** from those
   energies match the FD over the oracle's printed energies (1×10⁻¹²
   Ha/Bohr³), and the ATM contribution is non-vacuous (>5×10⁻¹⁰ pin).
3. Product GPW `d4 + d4_atm` analytic fixed-P stress: converges, `e_disp`
   matches the ATM base pin, off-diag ≈ 0, `kStress` unit conversion.
4. Dual gate: analytic vs multi-SCF strain-FD twin on the same d4+ATM
   method (σ_xx, warm-started).
5. Wiring: σ_xx(d4_atm) − σ_xx(d4-only) equals the standalone σ^ATM_xx —
   the `s9` fold inside the strained-cell dispersion rebuild of
   `GpwDft::stress` is exercised, not just the energy path.
