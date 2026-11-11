# gpw/tkatchenko-scheffler

One case; every section below is run by `atv_regressions --case gpw/tkatchenko-scheffler` and keeps its own oracle record.


## Reference — periodic water-dimer GPW PBE-TS dispersion energy

### Method

**Tkatchenko–Scheffler (TS)** pairwise dispersion as an additive correction on
top of Γ-point **GPW** (PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\)
(free-atom limit of TS). Ice-relevant H-bonded water dimer in a periodic
12 Å cell (same geometry as `gpw-pbe-d3bj-ice` / `gpw-d4-water`).

Public API: `cpu::gpw::GpwDft::ts = true` (and `GPWEvaluator::ts`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009);
libMBD `get_ts_energy` real-space path):

\[
E_{\mathrm{disp}} = -\sum_{i\le j}\sum_{\mathbf T}'
  C_{6,ij}\,f_{\mathrm{damp}}(R)\,R^{-6}
\]

\[
C_{6,ij}=\frac{2\,C_{6,i}C_{6,j}}
  {(\alpha_j/\alpha_i)\,C_{6,i}+(\alpha_i/\alpha_j)\,C_{6,j}},\quad
R_{0,ij}=R_i+R_j
\]

\[
f_{\mathrm{damp}}=\bigl(1+\exp[-d\,(R/(s_R R_{0,ij})-1)]\bigr)^{-1}
\]

PBE damping: \(s_R=0.94\), \(d=20\). Free-atom \(\alpha_0\), \(C_6\), \(R_{\mathrm{vdW}}\)
from libMBD `ts_vdw_params` (in-tree `basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------- | ------------------ | -------------------------- |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

Pair cutoff: **50 Å = 94.561… Bohr** (pure real-space orthorhombic lattice sum;
no Ewald). Hirshfeld density scaling and MBD are **out of scope** (see
`gpw-mbd-ice`).

### Reference value (hard gate)

**E_disp (PBE-TS free-atom, periodic water dimer, 12 Å cell, rcut 50 Å) =
−6.099630773386013×10⁻⁴ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| **libMBD** C API `cmbd_ts_energy` (git `88069c6`, built 2026-07-25) | isolated | **−5.9764872295816794×10⁻⁴** |
| Dual pure-Python lattice sum (same formula/tables) | isolated | −5.9764872295816815×10⁻⁴ |
| **libMBD** Fortran `get_ts_energy`, pure-RS branch (`ewald_on = false`, `dipole_cutoff = 50 Å`; driver `tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90`, 2026-07-31) | periodic 12 Å, rcut 50 Å | **−6.0996307733860822×10⁻⁴** |
| Atomiverse `ts_pbe` (pair_image_counts shell) | periodic 12 Å, rcut 50 Å | −6.099630773386013×10⁻⁴ |
| libMBD with Ewald (`ewald_cutoff_scaling=2.5`) | periodic 12 Å | −6.100334450881942×10⁻⁴ |

Isolated libMBD matches Atomiverse to \({\sim}10^{-19}\) Ha (unit-test hard pin).
The periodic pure-RS pin is anchored by the genuine libMBD no-Ewald run
(agreement 7×10⁻¹⁸ Ha, `gpw-ts-stress` pin driver). Periodic pure-RS and
libMBD Ewald differ by \({\sim}7\times10^{-8}\) Ha (long-range \(1/R^6\) tail +
Ewald vs hard cutoff) — documented; regression hard gate is the pure-RS value
above.

#### Why not CP2K QS?

**CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. libMBD is the
canonical offline TS/MBD oracle (also the backend FHI-aims / QE / DFTB+ use).
Twin input retained under `tools/cp2k_oracle/inputs/gpw-ts-ice/` for hosts that
may later expose TS via an external library.

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

Basis / PP (GPW regression; E_disp is basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Oracle: libMBD (offline — never launched from ctest)

```bash
## Build libmbd (once); pin isolated free-atom TS on the water dimer
## (see tools/cp2k_oracle/inputs/gpw-ts-ice/pin_libmbd_ts.c for a sketch).
## Isolated: Dispersion energy = -5.9764872295816794e-04 Eh
## Periodic pure-RS pin: dual Python lattice sum (REFERENCE.md table).
```

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use tighter checks on the
standalone `ts_pbe` kernel.

### What is checked

`GpwDft` with `ts = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −6.099630773386013e-04 Ha.
Unit tests also pin isolated libMBD and analytic forces vs FD.

### Perf gate

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; full twin SCF
ratio is unfair for a geometry-only hard gate (same policy as
`gpw-pbe-d3bj-ice` / `gpw-d4-water`).


## Reference — water-dimer GPW PBE-TS ionic forces

### Method

**Tkatchenko–Scheffler (TS)** ionic forces on top of Γ-point **GPW** (PBE),
free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Same water dimer as
`gpw-ts-ice` / `gpw-mbd-forces`.

Product path:

1. `ts_pbe` evaluates \(E_{\mathrm{disp}}\) matching libMBD
   (`cmbd_ts_energy` / pure real-space).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only energy (analytic product kernel; dual-gated vs central
   FD, \(h=10^{-5}\) Bohr).
3. `assemble_gpw_forces` with `GpwDft::ts = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms.

Public API: `cpu::gpw::GpwDft::ts = true` with `dft.forces(frame)` (or a
`GPWEvaluator` wrapping that `GpwDft` with `compute_forces=true`).

PBE: \(s_R = 0.94\), Fermi \(d = 20\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).
Pair cutoff (periodic): **50 Å**.

### Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp isolated** | libMBD Fortran `cmbd_ts_energy` (git `88069c6`, ATV Å→Bohr) | **−5.9764872295816794×10⁻⁴ Ha** |
| **E_disp periodic** | libMBD Fortran `get_ts_energy` pure-RS (`ewald_on = false`, rcut 50 Å; `gpw-ts-stress` pin driver) | **−6.0996307733860822×10⁻⁴ Ha** (= `gpw-ts-ice` pin to 7×10⁻¹⁸) |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |
| Wiring | \(F(\mathrm{ts})-F(\mathrm{plain})\) vs standalone | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |

#### Offline force sample (isolated free-atom, product kernel)

Printed by the regression (geometry-only; not a CP2K absolute pin).

Cross-check: pure-Python / libMBD analytic gradients agree with ATV product
forces to \(O(h^2)\) of the FD step (offline — never in ctest).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. Hard pin is
**libMBD energy** + **force dual-gate**, same policy as `gpw-ts-ice` /
`gpw-mbd-forces`. Twin notes under `tools/cp2k_oracle/inputs/gpw-ts-forces/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF + periodic TS pin): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force dual-gate uses the same nuclear coordinates without lattice
images.

Basis / PP (GPW short SCF; E_disp and F_TS are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (isolated / periodic) | **1×10⁻⁸** Ha (reg) |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |
| max\|F(ts)−F(plain)−F_geom\| | **1×10⁻⁷** Ha/Bohr |

### Perf gate

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-ts-ice` / `gpw-mbd-forces`). CP2K has no stock TS twin for
a fair wall ratio.


## Reference — Γ GPW analytic cell stress with free-atom TS dispersion

### Method

**Tkatchenko–Scheffler (TS)** free-atom pairwise dispersion
(\(V/V_{\mathrm{free}} = 1\); Tkatchenko & Scheffler, *Phys. Rev. Lett.*
**102**, 073005 (2009)) folded into the Γ-point **GPW** (PBE) analytic
fixed-\(P\) cell stress. TS is geometry-only, so the stress Lagrangian
rebuilds \(E_{\mathrm{disp}}^{TS}(\{\mathbf R\},\mathbf h)\) at every
strained cell (same `ts_pbe` call as the energy path, pure real-space
lattice sum, pair cutoff **50 Å**, \(s_R = 0.94\), Fermi \(d = 20\)):

1. One SCF at the unstrained cell → \(P, W\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[P, W; \varepsilon]\) under orthorhombic diagonal strain at fixed
   fractional coordinates, \(h = 2\times10^{-3}\).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; Hartree/Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `ts = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

### Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **libMBD Fortran** `get_ts_energy`, pure-RS branch (`ewald_on = .false.`, `dipole_cutoff = 50 Å`, ATV Å→Bohr) — the genuine-oracle anchor of the `gpw-ts-ice` periodic pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh offline libMBD pins at each strained cell (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd`): full re-SCF at each strained cell, TS folded through the energy path |
| D. TS actually folded | \(\sigma(\mathrm{ts}) - \sigma(\mathrm{plain})\) equals the libMBD strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock free-atom `TYPE TS`,
hence no CP2K stress twin. libMBD is the canonical offline TS oracle (the
backend FHI-aims / QE / DFTB+ use) — same policy as `gpw-ts-ice` /
`gpw-ts-forces`, strengthened here by gate B/D external pins on the strained
cells (structure of `gpw-mbd-kpoint-stress`).

### System

Same geometry as `gpw-ts-ice` / `gpw-ts-forces`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ = 8`) |
| XC | PBE, Γ point |
| Dispersion | TS free-atom, pure-RS rcut 50 Å, \(s_R = 0.94\), \(d = 20\) |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

### Pins

Source: offline **libMBD Fortran** `get_ts_energy` (libmbd git `88069c6`,
GCC 14.3 / OpenBLAS 0.3.32), pure real-space branch, free-atom
\(\alpha_0/C_6/R_{\mathrm{vdW}}\) from libMBD `ts_vdw_params`, Å→Bohr =
1/0.52917720859. Reproduce:
`tools/cp2k_oracle/inputs/gpw-ts-stress/pin_ts_strain.f90` (never launched
at ctest time). Output lines (2026-07-31):

```
E_disp isolated             =  -5.9764872295816794E-04 Ha
E_disp unstrained           =  -6.0996307733860822E-04 Ha
E_disp x +h            =  -6.1278297256613687E-04 Ha
E_disp x -h            =  -6.0709134569744607E-04 Ha
sigma_TS_xx (FD, libMBD)  =   1.2202142278E-07 Ha/Bohr3
E_disp y +h            =  -6.0992605267944588E-04 Ha
E_disp y -h            =  -6.1000059619617259E-04 Ha
sigma_TS_yy (FD, libMBD)  =  -1.5981205691E-09 Ha/Bohr3
E_disp z +h            =  -6.0996696060335368E-04 Ha
E_disp z -h            =  -6.0995897233070745E-04 Ha
sigma_TS_zz (FD, libMBD)  =   1.7125866055E-10 Ha/Bohr3
```

Provenance cross-checks at pin time: the isolated value is **bit-identical**
to the shipped `cmbd_ts_energy` C-API pin (`gpw-ts-ice`); the unstrained
periodic value agrees with the shipped periodic pure-RS pin
(−6.099630773386013×10⁻⁴ Ha) to **7×10⁻¹⁸ Ha**, retro-anchoring that pin to a
genuine libMBD run.

### Tolerances

| Check | Tol |
| ----- | --- |
| A/B \(\lvert E_{\mathrm{disp}} - \mathrm{libMBD}\rvert\) (unstrained + 6 strained) | 1×10⁻⁸ Ha |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (sibling dispersion-stress tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{ts}}-\sigma_{\mathrm{plain}}) - \sigma^{TS}_{\mathrm{libMBD}}\rvert\) | 2×10⁻⁹ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

### Perf gate

CP2K 2026.1 stock has no free-atom TS pair potential, so no fair oracle wall
exists for a TS stress run: **`perf_exempt`** — same policy as `gpw-ts-ice` /
`gpw-ts-forces` / `gpw-mbd-stress`.
