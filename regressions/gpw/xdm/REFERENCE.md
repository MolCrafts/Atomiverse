# gpw/xdm

One case; every section below is run by `atv_regressions --case gpw/xdm` and keeps its own oracle record.


## Reference — periodic water-dimer GPW free-atom XDM dispersion energy

### Method

**Exchange-hole dipole moment (XDM)** pairwise dispersion as an additive
correction on top of Γ-point **GPW** (PBE), free-atom multipole moments and
polarizabilities (free-atom limit of XDM — no Hirshfeld volume scaling).
Ice-relevant H-bonded water dimer in a periodic 12 Å cell (same geometry as
`gpw-ts-ice` / `gpw-pbe-d3bj-ice`).

Public API: `cpu::gpw::GpwDft::xdm = true` (and `GPWEvaluator::xdm`).
Dispersion energy is reported on `GpwResult::e_disp` [Hartree] and folded into
`e_total`.

Formula (Becke & Johnson, *J. Chem. Phys.* **127**, 154108 (2007);
postg / QE `xdm_dispersion` combination rules + BJ damping):

\[
\mathrm{fac}=\frac{\alpha_i\alpha_j}{M_{1i}\alpha_j+M_{1j}\alpha_i},\quad
C_6=\mathrm{fac}\,M_{1i}M_{1j},
\]

\[
C_8=\tfrac{3}{2}\,\mathrm{fac}\,(M_{1i}M_{2j}+M_{2i}M_{1j}),
\]

\[
C_{10}=2\,\mathrm{fac}\,(M_{1i}M_{3j}+M_{3i}M_{1j})
+\tfrac{21}{5}\,\mathrm{fac}\,M_{2i}M_{2j},
\]

\[
R_c=\frac{1}{3}\Bigl(\sqrt{C_8/C_6}+(C_{10}/C_6)^{1/4}+\sqrt{C_{10}/C_8}\Bigr),
\quad R_{\mathrm{vdW}}=a_1 R_c+a_2,
\]

\[
E_{\mathrm{disp}}=-\sum_{i\le j}\sum_{\mathbf T}'
\Biggl[
\frac{C_6}{R^6+R_{\mathrm{vdW}}^6}
+\frac{C_8}{R^8+R_{\mathrm{vdW}}^8}
+\frac{C_{10}}{R^{10}+R_{\mathrm{vdW}}^{10}}
\Biggr].
\]

PBE damping: **QE NC-PBE** \(a_1=0.4283\), \(a_2=2.4690\) Å
(Otero-de-la-Roza & Johnson / postg `xdm.param`). Free-atom \(\alpha\) from
CRC Handbook 88th Ed. (same as postg/QE). Free-atom \(M_1,M_2,M_3\) inverted
from literature homo \(C_n\) under the XDM combination rules:

| Z | \(\alpha\) [Bohr³] | \(M_1\) | \(M_2\) | \(M_3\) | Source \(C_n\) |
| - | ------------------ | ------- | ------- | ------- | -------------- |
| H | 4.49979 | 2.88859 | 18.4304 | 241.636 | Yan et al. PRA 54, 2824 |
| O | 5.41216 | 5.74631 | 28.9471 | 262.617 | Chu–Dalgarno JCP 121, 4083 (\(C_6\)); consistent \(C_8,C_{10}\) |

Pair cutoff: **50 Å** pure real-space orthorhombic lattice sum
(`pair_image_counts`). Density-dependent Hirshfeld/BR moments are **out of
scope** (free-atom foothold; same policy as free-atom TS).

### Reference value (hard gate)

**E_disp (free-atom XDM, PBE BJ, periodic water dimer, 12 Å cell, rcut 50 Å) =
−2.114815761287903×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Geometry | E_disp [Ha] |
| ------ | -------- | ----------- |
| Dual pure-Python lattice sum (this REFERENCE) | isolated | **−2.102043599851622×10⁻³** |
| Dual pure-Python lattice sum | periodic 12 Å, rcut 50 Å | **−2.114815761287903×10⁻³** |
| Atomiverse `xdm_pbe` | same | matches to \(\sim 10^{-16}\) Ha |

#### Why not CP2K QS?

**CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`. The
canonical offline XDM oracles are postg / critic2 / QE (density-dependent
full XDM). This slug ships the **free-atom** foothold with a dual pure-Python
pin (same pattern as free-atom TS when stock CP2K is unavailable). Twin note
retained under `tools/cp2k_oracle/inputs/gpw-xdm-ice/`.

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

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests use tighter checks on the
standalone `xdm_pbe` kernel.

### What is checked

`GpwDft` with `xdm = true` on the pin geometry returns
`last_result().e_disp` within 1e-8 Ha of −2.114815761287903e-03 Ha.
Unit tests also pin isolated dual-Python and analytic forces vs FD.

### Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall is dominated by the electronic structure.
**perf_exempt = true**: primary deliverable is E_disp pin parity; full twin SCF
ratio is unfair for a geometry-only hard gate (same policy as
`gpw-ts-ice` / `gpw-pbe-d3bj-ice`).


## Reference — water-dimer GPW free-atom XDM ionic forces

### Method

**Exchange-hole dipole moment (XDM)** free-atom pairwise dispersion ionic
forces on top of Γ-point **GPW** (PBE). Same water dimer as `gpw-xdm-ice` /
`gpw-ts-ice`. Free-atom multipole moments and polarizabilities — no Hirshfeld
volume scaling (foothold; density-dependent XDM is out of scope).

Product path:

1. `xdm_pbe` evaluates \(E_{\mathrm{disp}}\) matching the dual pure-Python
   lattice sum (`gpw-xdm-ice` pin).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only kernel (analytic; dual-gated by central FD, \(h=10^{-5}\)
   Bohr).
3. `assemble_gpw_forces` with `GpwDft::xdm = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms (no longer energy-only).

Public API: `cpu::gpw::GpwDft::xdm = true` (and `GPWEvaluator::xdm`) with
`compute_forces=true` / `gpw_forces_*`.

PBE BJ damping: QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å. Free-atom
\(\alpha,M_1,M_2,M_3\) from in-tree `basis_data/xdm_tables.h`. Periodic
lattice sum: pure real-space, rcut **50 Å**, `pair_image_counts`.

### Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (periodic)** | dual pure-Python / `gpw-xdm-ice` | **−2.1148157612879029×10⁻³ Ha** |
| **E_disp (isolated)** | dual pure-Python | **−2.1020435998516220×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

#### Offline force sample (isolated free-atom, product kernel)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

Cross-check: dual pure-Python analytic gradient agrees with ATV `xdm_pbe`
forces to \(\sim 10^{-16}\) Ha/Bohr (offline — never in ctest).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`. Hard pin is
**dual pure-Python energy + force dual-gate**, same policy as `gpw-xdm-ice` /
`gpw-mbd-forces`. Twin notes under `tools/cp2k_oracle/inputs/gpw-xdm-forces/`.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF only): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force pin uses the same nuclear coordinates without lattice images.
Periodic E_disp pin uses rcut 50 Å.

Basis / PP (GPW short SCF; E_disp and F_XDM are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

### Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |
| max\|F − F_dual_python sample\| | **1×10⁻¹⁰** Ha/Bohr |

### Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-xdm-ice` / `gpw-mbd-forces`). No fair CP2K ENERGY_FORCE
wall exists without stock XDM.


## Reference — Γ GPW analytic cell stress with free-atom XDM

### Method

Γ-point **GPW** (PBE) on a periodic water dimer with **free-atom XDM**
(exchange-hole dipole moment) pairwise dispersion — Becke–Johnson combination
rules, QE NC-PBE BJ damping \(a_1=0.4283\), \(a_2=2.4690\) Å, pure real-space
lattice sum with rcut **50 Å** (same kernel as `gpw-xdm-ice` /
`gpw-xdm-forces`).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One SCF at the unstrained cell → \(P\), energy-weighted \(W\).
2. Central FD of the non-self-consistent Lagrangian \(E_L[P;W;\varepsilon]\)
   under orthorhombic diagonal strain at fixed fractional coordinates, with
   the geometry-only \(E_{\mathrm{disp}}^{\mathrm{XDM}}(\{\mathbf R\},\mathbf h)\)
   **rebuilt at every strained cell** (same `xdm_pbe` call + rcut as the
   energy path).
3. \(\sigma_{aa} = -\Omega_0^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) in Hartree, \(\Omega_0\) in Bohr³).

Rebuilding at each strain is the whole point: holding \(E_{\mathrm{disp}}\)
fixed would differentiate a different functional than the SCF converged (the
0.53 Ha/Bohr ZnO failure mode). The fold itself shipped with the
dispersion-stress plumbing; this case adds the **dedicated pins** (queue row:
energy folds, dedicated pin missing).

Public path: `cpu::gpw::GPWEvaluator` with `xdm = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

### Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **dual pure-Python** free-atom XDM lattice sum — bit-identical provenance with the shipped `gpw-xdm-ice` pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis, fixed fractional coords, rcut 50 Å) | fresh dual pure-Python pins at each of the 6 strained cells (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd`): full re-SCF at each strained cell, XDM folded through the energy path |
| D. XDM actually folded, with the pinned magnitude | \(\sigma(\mathrm{xdm}) - \sigma(\mathrm{plain})\) equals the dual-python strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega_0)\) per axis |

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`, so no CP2K
stress twin exists. Hard pins are the dual pure-Python energy sums (same
policy and oracle family as `gpw-xdm-ice` / `gpw-xdm-forces`), plus the
multi-SCF strain-FD dual gate for the tensor — same structure as
`gpw-mbd-stress` / `gpw-mbd-kpoint-stress`.

### System

Same geometry as `gpw-xdm-ice` / `gpw-mbd-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Dispersion | free-atom XDM, QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å, rcut 50 Å |
| Cutoff | 200 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

### Pins

Source: offline **dual pure-Python** free-atom XDM lattice sum
(`tools/cp2k_oracle/inputs/gpw-xdm-stress/pin_xdm_strain.py`, python3 stdlib
only; independent implementation sharing only the cited free-atom data:
CRC 88th Ed. α, homoatomic \(C_6/C_8/C_{10}\) from Yan et al. PRA **54**,
2824 and Chu & Dalgarno JCP **121**, 4083). Never launched at ctest time.
Output lines (2026-07-31):

```
E_disp unstrained            = -2.1148157612879029e-03 Ha
E_disp x +h              = -2.1110849131940963e-03 Ha
E_disp x -h              = -2.1185431863446984e-03 Ha
sigma_XDM_xx (FD, dual-py)  = -1.5989612853e-07 Ha/Bohr3
E_disp y +h              = -2.1147774136190569e-03 Ha
E_disp y -h              = -2.1148546246483736e-03 Ha
sigma_XDM_yy (FD, dual-py)  = -1.6553087314e-09 Ha/Bohr3
E_disp z +h              = -2.1146664855098696e-03 Ha
E_disp z -h              = -2.1149651549478578e-03 Ha
sigma_XDM_zz (FD, dual-py)  = -6.4031024180e-09 Ha/Bohr3
```

Provenance cross-check at pin time: the unstrained value **bit-reproduces**
the shipped `gpw-xdm-ice` periodic pin (−2.1148157612879029×10⁻³ Ha).

### Tolerances

| Check | Tol |
| ----- | --- |
| B \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (6 strained cells, direct `xdm_pbe` + pin Å→Bohr) | 1×10⁻¹² Ha (deterministic geometry-only sum) |
| A \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (product SCF via Frame) | 1×10⁻⁸ Ha (sibling `gpw-xdm-ice` / `gpw-xdm-forces` tier) |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (Γ dispersion-stress reg tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{xdm}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{XDM}}_{\mathrm{dual\text{-}py}}\rvert\) | 1×10⁻⁸ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

### Tolerance justification

- **B 1e−12 Ha:** the in-tree `xdm_pbe` kernel and the dual pure-Python sum
  are two implementations of the same deterministic geometry-only formula on
  shared cited data, evaluated at the pin Å→Bohr constant
  (`1/0.52917720859`); measured residual ~1e−18.
- **A 1e−8 Ha:** product path converts Frame Å geometry with
  `units::kAngstromInBohr` (CODATA 2006), which differs from the pin constant
  at ~4e−9 relative — enough to move \(E_{\mathrm{disp}}\) by ~1e−11 Ha.
  Same product-path tier as `gpw-xdm-ice` / `gpw-xdm-forces` (1e−8 Ha); the
  decisive kernel identity is gate B.
- **C 2e−5 Ha/Bohr³:** analytic fixed-\(P\) vs multi-SCF FD differ by the
  frozen density-matrix approximation plus SCF residual at `tol = 1e-6`, not
  by the dispersion treatment — same tier as `gpw-d4-stress` /
  `gpw-mbd-stress`.
- **D 1e−8 Ha/Bohr³:** both sides are central FDs at the same \(h\) of the
  same dispersion energy (XDM does not enter the Fock operator, so the
  electronic parts cancel); the gate only has to exclude "silently not
  folded" (σ_XDM_xx ≈ 1.6e−7) and a wrong strained-cell rebuild. Residual
  also absorbs the Frame vs pin Å→Bohr constant difference on the pin side.

### Dual gate

- **Correctness:** the gates above, plus unit gates in
  `tests/cpu/gpw/test_cpu_gpw_stress.cpp` — `AnalyticMatchesFdWithXdm`
  (cheap MT H₂, sibling parity with D4/D3BJ/MBD) and
  `XdmChangesGammaDiagonal`.
- **Speed:** no XDM-capable oracle binary exists to time against (CP2K stock
  has no XDM), so this case is **perf_exempt** — same policy as
  `gpw-xdm-ice` / `gpw-xdm-forces` / `gpw-mbd-stress`.

### Measured results (2026-07-31, first green)

`reg_gpw-xdm-stress` (OMP 8, epyc login node), 11 SCF iters:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs dual-py (6 cells) | 1e−12 Ha | **8.67e−19** |
| A \(E_{\mathrm{disp}}\) product vs pin | 1e−8 Ha | **8.62e−12** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **7.39e−07** (xx; yy 2.46e−8, zz 1.22e−9) |
| D shift vs dual-py FD term | 1e−8 Ha/Bohr³ | **1.25e−10** (xx; yy 1.39e−11, zz 5.68e−12) |
| E off-diagonals | exactly 0 | ~0 (pass) |
| E `kStress` conversion | 1e−9 | pass |
| Product wall | ≤ 60 s | **22.1 s** (FD twin 32.3 s, not product budget) |

Product analytic diagonals [Ha/Bohr³]: xx −2.18425974×10⁻⁵,
yy +2.00715789×10⁻⁵, zz +5.82582894×10⁻⁵.

### Notes

- Sibling XDM products: `gpw-xdm-ice` (energy), `gpw-xdm-forces` (ionic
  forces). Sibling dispersion stress: `gpw-d3bj-stress`, `gpw-d4-stress`,
  `gpw-mbd-stress`, `gpw-mbd-kpoint-stress`.
- Multi-k XDM stress folds through the same k-summed Lagrangian
  (`energy_fixed_P_k` dispersion branch) but has no dedicated pin case yet.


## Reference — multi-k GPW analytic forces with free-atom XDM

### Method

**Monkhorst–Pack multi-k GPW** ionic forces with additive **free-atom XDM**
pairwise dispersion (geometry-only multipole moments + CRC \(\alpha\)).
Closes the dedicated multi-k force pin residual: Γ
`assemble_gpw_forces` and multi-k `assemble_gpw_forces_kpoints` both fold
XDM; this regression hard-gates the multi-k wiring against dual pure-Python
energy/force pins (same policy as `gpw-xdm-forces` / `gpw-xdm-kpoint-stress`).

Product path:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (the `GpwDft`
   multi-k force path).
3. Geometry-only \(F_{\mathrm{XDM}} = -\nabla E_{\mathrm{XDM}}\) from
   `xdm_pbe` is **added** on the multi-k path (identical switch as Γ:
   pure real-space lattice sum, rcut 50 Å, QE NC-PBE BJ damping).

Public API: `GpwDft::xdm = true` + multi-k mesh + `GpwDft::forces(frame)` /
`GPWEvaluator(compute_forces=true)`. No new symbols.

PBE BJ damping: QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å. Free-atom
\(\alpha,M_1,M_2,M_3\) from in-tree `basis_data/xdm_tables.h`.

### Reference values (hard gates)

#### A — Geometry-only free-atom XDM (water dimer)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (isolated)** | dual pure-Python / `gpw-xdm-ice` | **−2.1020435998516220×10⁻³ Ha** |
| **E_disp (periodic)** | dual pure-Python / `gpw-xdm-ice` | **−2.1148157612879029×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |
| Isolated F sample | dual pure-Python ≡ ATV tables | max\|Δ\| ≤ **1×10⁻¹⁰** Ha/Bohr |

Same nuclei / pin policy as `gpw-xdm-forces` / `gpw-xdm-ice`. XDM is
geometry-only — independent of the electronic k mesh.

#### Offline force sample (isolated free-atom)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

#### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-mbd-kpoint-forces`):
PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, free-atom XDM (periodic cell, rcut 50 Å).

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty; Γ has `n_kpts==0`) | hard |
| \(F^{k+\mathrm{XDM}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{XDM}}^{\mathrm{geom}}\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr |
| \(\|F_{\mathrm{XDM},0x}^{\mathrm{geom}}\|\) nontrivial | > **1×10⁻⁸** Ha/Bohr |

Geometry-only XDM does not change the SCF density, so the force difference
must match the standalone `xdm_pbe` forces within tol.

#### Why not CP2K QS?

**CP2K 2026.1** (`cp2k.psmp` Source code revision git:5e54ba2) stock
`PAIR_POTENTIAL TYPE` is only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is
**no** stock `TYPE XDM` (`input_cp2k_xc.F` usage string). Hard pin is
**dual pure-Python energy + force dual-gate + multi-k wiring**, same policy
as `gpw-xdm-forces` / `gpw-xdm-kpoint-stress`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-xdm-kpoint-forces/`. Never fake `oracle: cp2k`.

### Geometry

#### Water dimer (Å) — XDM pin

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (periodic E_disp pin): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force pin uses the same nuclear coordinates without lattice images.
Periodic E_disp / force dual-gate uses rcut **50 Å**.

#### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell (H₂ SCF): **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**.

### Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (water dimer) | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F_XDM\| | **1×10⁻⁸** Ha/Bohr |
| max\|F − F_dual_python sample\| | **1×10⁻¹⁰** Ha/Bohr |
| max\|F_k+XDM − F_k − F_geom\| | **1×10⁻⁸** Ha/Bohr |

### Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Multi-k H₂ force assembly is a single gradient pass after
one SCF. Full CP2K QS wall is not a fair twin (no stock XDM pair type).
**`perf_exempt = true`** — same policy as `gpw-xdm-forces` /
`gpw-xdm-kpoint-stress` / `gpw-mbd-kpoint-forces`. Oracle family:
`literature` / dual pure-Python (never fake `oracle: cp2k`).


## Reference — multi-k GPW analytic cell stress with free-atom XDM

### Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with **free-atom XDM** (exchange-hole dipole moment) pairwise
dispersion — Becke–Johnson combination rules, QE NC-PBE BJ damping
\(a_1=0.4283\), \(a_2=2.4690\) Å, pure real-space lattice sum with rcut
**50 Å** (same kernel as `gpw-xdm-ice` / `gpw-xdm-stress` / `gpw-xdm-forces`).

Closes the residual that the Γ fixed-\(P\) stress Lagrangian already rebuilt
\(E_{\mathrm{disp}}^{\mathrm{XDM}}\) at every strained cell
(`gpw-xdm-stress`) while the multi-k path needed its dedicated offline pin
(product fold in `energy_fixed_P_k` already ships geometry-only XDM via
`xdm_pbe`).

Product cell stress is the Hellmann–Feynman fixed-\(P_k\) path:

1. One multi-k SCF at the unstrained cell → \(\{P_k, W_k\}\).
2. Central FD of the non-self-consistent k-summed Lagrangian
   \(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain at
   fixed fractional coordinates, with the geometry-only
   \(E_{\mathrm{disp}}^{\mathrm{XDM}}(\{\mathbf R\},\mathbf h)\) **rebuilt at
   every strained cell** (same `xdm_pbe` call + rcut as the energy path).
3. \(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; \(E\) Hartree, \(\Omega\) Bohr³).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `xdm = true`,
`compute_stress = true`, `stress_use_fd = false` → Frame `kStress` [eV/Å³];
compared here via `last_stress_ha_bohr3()`.

### Oracles

| Gate | Oracle |
| ---- | ------ |
| A. \(E_{\mathrm{disp}}\) unstrained | offline **dual pure-Python** free-atom XDM lattice sum — the same pin `gpw-xdm-ice` / `gpw-xdm-stress` use; XDM is independent of the electronic k mesh, so the multi-k run must reproduce the Γ pin |
| B. \(E_{\mathrm{disp}}\) **under strain** (±h per axis) | fresh dual pure-Python pins at each strained geometry (below) — the strain derivative being folded is externally validated point by point |
| C. \(\sigma\) | **multi-SCF strain-FD** (`gpw_stress_fd`): full re-SCF at each strained cell on the same 1×1×2 mesh, XDM folded through the energy path |
| D. XDM actually folded | \(\sigma(\mathrm{xdm}) - \sigma(\mathrm{plain})\) equals the dual-python strain-FD term \(-\Delta E_{\mathrm{disp}}/(2h\Omega)\) per axis |

#### Why not CP2K QS?

**CP2K 2026.1** (`cp2k.psmp` git:5e54ba2) stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE XDM`, so no CP2K
stress twin exists. Hard pins are the dual pure-Python energy sums (same
policy and oracle family as `gpw-xdm-ice` / `gpw-xdm-forces` /
`gpw-xdm-stress`), plus the multi-SCF strain-FD dual gate for the tensor —
same structure as `gpw-mbd-kpoint-stress`.

### System

Same geometry as `gpw-xdm-ice` / `gpw-xdm-stress`:

| Item | Value |
|------|-------|
| System | water dimer, **6 atoms**, order **O H H O H H** |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ = 8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2**, time-reversal folded |
| Dispersion | free-atom XDM, QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å, rcut 50 Å |
| Cutoff | 120 Ry (E_disp is geometry-only — cutoff does not enter the pins) |
| Strain step | \(h = 2\times10^{-3}\) |

### Pins

Source: offline **dual pure-Python** free-atom XDM lattice sum
(`tools/cp2k_oracle/inputs/gpw-xdm-kpoint-stress/pin_xdm_strain.py`, python3
stdlib only; independent implementation sharing only the cited free-atom
data: CRC 88th Ed. α, homoatomic \(C_6/C_8/C_{10}\) from Yan et al. PRA
**54**, 2824 and Chu & Dalgarno JCP **121**, 4083). Never launched at ctest
time. Output lines (2026-08-01 re-run; geometry-only bit-identical to the
`gpw-xdm-stress` pin set):

```
E_disp unstrained            = -2.1148157612879029e-03 Ha
E_disp x +h              = -2.1110849131940963e-03 Ha
E_disp x -h              = -2.1185431863446984e-03 Ha
sigma_XDM_xx (FD, dual-py)  = -1.5989612853e-07 Ha/Bohr3
E_disp y +h              = -2.1147774136190569e-03 Ha
E_disp y -h              = -2.1148546246483736e-03 Ha
sigma_XDM_yy (FD, dual-py)  = -1.6553087314e-09 Ha/Bohr3
E_disp z +h              = -2.1146664855098696e-03 Ha
E_disp z -h              = -2.1149651549478578e-03 Ha
sigma_XDM_zz (FD, dual-py)  = -6.4031024180e-09 Ha/Bohr3
```

Provenance cross-check at pin time: the unstrained value **bit-reproduces**
the shipped `gpw-xdm-ice` periodic pin (−2.1148157612879029×10⁻³ Ha).

### Tolerances

| Check | Tol |
| ----- | --- |
| B \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (6 strained cells, direct `xdm_pbe` + pin Å→Bohr) | 1×10⁻¹² Ha (deterministic geometry-only sum) |
| A \(\lvert E_{\mathrm{disp}} - \mathrm{pin}\rvert\) (product multi-k SCF via Frame) | 1×10⁻⁸ Ha (sibling `gpw-xdm-ice` / `gpw-xdm-stress` tier) |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2×10⁻⁵ Ha/Bohr³ (sibling dispersion-stress tier) |
| C \(\lvert\sigma\rvert\) nonzero | > 1×10⁻¹⁰ |
| D \(\lvert(\sigma_{\mathrm{xdm}}-\sigma_{\mathrm{plain}}) - \sigma^{\mathrm{XDM}}_{\mathrm{dual\text{-}py}}\rvert\) | 1×10⁻⁸ Ha/Bohr³ |
| E off-diagonals (orthorhombic) | exactly 0 |
| E `kStress` unit conversion | 1×10⁻⁹ |
| Product wall | ≤ 60 s |

### Tolerance justification

- **B 1e−12 Ha:** the in-tree `xdm_pbe` kernel and the dual pure-Python sum
  are two implementations of the same deterministic geometry-only formula on
  shared cited data, evaluated at the pin Å→Bohr constant
  (`1/0.52917720859`); measured residual ~1e−18.
- **A 1e−8 Ha:** product path converts Frame Å geometry with
  `units::kAngstromInBohr` (CODATA 2006), which differs from the pin constant
  at ~4e−9 relative. Same product-path tier as `gpw-xdm-stress`.
- **C 2e−5 Ha/Bohr³:** analytic fixed-\(P_k\) vs multi-SCF FD differ by the
  frozen density-matrix approximation plus SCF residual at `tol = 1e-6`, not
  by the dispersion treatment — same tier as `gpw-mbd-kpoint-stress` /
  `gpw-xdm-stress`.
- **D 1e−8 Ha/Bohr³:** both sides are central FDs at the same \(h\) of the
  same dispersion energy (XDM does not enter the Fock operator, so the
  electronic parts cancel); residual also absorbs the Frame vs pin Å→Bohr
  constant difference on the pin side.

### Perf gate

CP2K 2026.1 stock has no XDM pair potential, so no fair oracle wall exists
for a multi-k XDM stress run: **`perf_exempt`** — same policy as
`gpw-xdm-stress` / `gpw-xdm-forces` / `gpw-mbd-kpoint-stress`.

### Measured results (2026-08-01, first green)

`reg_gpw-xdm-kpoint-stress` (OMP 8, epyc login node), 11 SCF iters,
TR-folded nk = 1 from MP 1×1×2:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| B \(E_{\mathrm{disp}}\) under strain vs dual-py (6 cells) | 1e−12 Ha | **8.67e−19** |
| A \(E_{\mathrm{disp}}\) multi-k vs Γ pin | 1e−8 Ha | **8.62e−12** |
| C \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **1.37e−09** |
| D shift vs dual-py FD term | 1e−8 Ha/Bohr³ | **1.10e−15** |
| E off-diagonals | exactly 0 | **0** |
| E `kStress` conversion | 1e−9 | exact |
| Product wall | ≤ 60 s | **17.8 s** (FD twin 103.8 s; not product budget) |

Product analytic diagonals [Ha/Bohr³]: xx −3.621e−05, yy −8.570e−05,
zz −3.089e−06. XDM shifts: xx −1.59896e−07, yy −1.655e−09,
zz −6.403e−09 Ha/Bohr³ — matching the dual-python FD terms per axis.

### Notes

- Sibling XDM products: `gpw-xdm-ice` (energy), `gpw-xdm-forces` (ionic
  forces), `gpw-xdm-stress` (Γ stress). Sibling multi-k dispersion stress:
  `gpw-mbd-kpoint-stress`.
